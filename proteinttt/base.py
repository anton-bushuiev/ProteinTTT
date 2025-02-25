import copy
import time
import typing as T
from omegaconf import OmegaConf
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import torch
from lora_diffusion.lora import inject_trainable_lora

from proteinttt.utils.io import setup_logger
from proteinttt.utils.torch import preserve_model_state, get_optimal_window
from proteinttt.utils.msa import read_msa, MSAServer


@dataclass
class TTTConfig:
    """
    Configuration for test-time training (TTT).

    TODO
    """
    lr: float = 4e-4
    ags: int = 16
    steps: int = 30
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_target_replace_module: str = 'MultiheadAttention'
    optimizer: str = 'sgd'  # T.Literal['adamw', 'sgd']
    momentum: float = 0.0
    weight_decay: float = 0.0
    batch_size: int = 2
    mask_ratio: float = 0.15  # Used for ESM2 / ESMFold, SaProt, ProSST pre-training
    crop_size: int = 1024  # Used for ESM2 / ESMFold, SaProt, ProSST pre-training
    bert_leave_prob: float = 0.1
    bert_replace_prob: float = 0.1
    loss_kind: str = 'cross_entropy'  # T.Literal['cross_entropy', 'unnormalized_cross_entropy', 'msa_soft_labels' TODO]
    msa: T.Optional[bool] = False
    msa_cache_dir: Path = Path.home() / '.cache' / 'ttt'
    score_seq_kind: T.Optional[str] = None  # T.Optional[T.Literal['pseudo_perplexity', 'gordon2024', 'none']] = None
    score_seq_steps_list: T.Any = None  # T.Optional[int | list[int]]. None to use all steps
    perplexity_early_stopping: T.Optional[float] = None
    eval_each_step: bool = True
    initial_state_reset: bool = True
    automatic_best_state_reset: bool = True
    seed: T.Optional[int] = 0  # None means using environment seed
    log_file_path: T.Optional[str] = None
    log_name: str = 'ttt_log'
    debug: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'TTTConfig':
        """Load TTTConfig from a YAML file using OmegaConf."""
        default_conf = OmegaConf.structured(cls)
        file_conf = OmegaConf.load(yaml_path)
        conf = OmegaConf.merge(default_conf, file_conf)
        OmegaConf.resolve(conf)
        return cls(**OmegaConf.to_container(conf))
    
    def verify(self) -> None:
        """Verify the configuration."""
        if self.score_seq_kind == 'none':
            self.score_seq_kind = None

        if self.score_seq_steps_list is not None:
            if isinstance(self.score_seq_steps_list, int):
                self.score_seq_steps_list = [self.score_seq_steps_list]
            elif not isinstance(self.score_seq_steps_list, list):
                raise ValueError("score_seq_steps_list must be None, an integer, or a list of integers")
            if not all(isinstance(x, int) for x in self.score_seq_steps_list):
                raise ValueError("All elements in score_seq_steps_list must be integers")

        if self.perplexity_early_stopping is not None and self.score_seq_kind is None:
            raise ValueError("perplexity_early_stopping can only be used if score_seq_kind is not None")

        if self.loss_kind == 'msa_soft_labels' and not self.msa:
            raise ValueError("msa_soft_labels loss kind can only be used if msa=True")


class TTTModule(torch.nn.Module, ABC):
    ttt_default_cfg: T.Optional[TTTConfig] = None

    def __init__(
        self,
        ttt_cfg: T.Optional[TTTConfig | Path | str] = None
    ):
        ABC.__init__(self)  # no torch.nn.Module init because it is already done in child class 

        # Init TTTConfig
        self.ttt_cfg = ttt_cfg or TTTConfig()
        if isinstance(ttt_cfg, Path) or isinstance(ttt_cfg, str):
            self.ttt_cfg = TTTConfig.from_yaml(ttt_cfg)
        self.ttt_cfg.verify()

        # Set MSA server to automatically build MSA if needed
        self.msa_server = MSAServer(self.ttt_cfg.msa_cache_dir)

        # Set random seed if specified, otherwise use environment seed
        self.ttt_generator = torch.Generator()
        if self.ttt_cfg.seed is not None:
            self.ttt_generator.manual_seed(self.ttt_cfg.seed)

        # Init logger
        self.ttt_logger = setup_logger(
            log_file_path=self.ttt_cfg.log_file_path, 
            log_name=self.ttt_cfg.log_name,
            debug=self.ttt_cfg.debug
        )
        self.ttt_logger.debug(f"TTTConfig: {self.ttt_cfg}")

        # Store initial state of trainable modules
        self._ttt_initial_state = None
        if self.ttt_cfg.initial_state_reset:
            self._ttt_initial_state = self._ttt_get_state()

    @classmethod
    def ttt_from_pretrained(
        cls, 
        model: torch.nn.Module,
        ttt_cfg: T.Optional[TTTConfig] = None,
        **kwargs
    ) -> 'TTTModule':
        # Use default TTTConfig if not provided
        if ttt_cfg is None:
            ttt_cfg = cls.ttt_default_cfg or TTTConfig()

        # Initialize instance without pretrained state
        instance = cls(ttt_cfg, **kwargs)

        # Copy state from pretrained model
        for key, value in model.__dict__.items():
            setattr(instance, key, value)

        # Store initial state of trainable modules after initializing weights
        if instance.ttt_cfg.initial_state_reset:
            instance._ttt_initial_state = instance._ttt_get_state()

        return instance

    @preserve_model_state
    def ttt(
        self,
        seq: T.Optional[str] = None,
        msa_pth: T.Optional[Path] = None,
        **kwargs
    ) -> dict[str, T.Any]:
        """
        Run TTT loop. After calling this method, the model will be customized to the input protein
        via test-time training (TTT).

        Args:
            seq: Input amino acid sequence to customize the model to.
            msa_pth: Optional path to MSA file. If None, MSA will be built on-the-fly from scratch. This argument will
            be used only if TTTConfig.msa_mode is not None.
            **kwargs: Keyword arguments to forward of the original model

        Returns:
            A dictionary containing the results of the TTT loop.
        """
        # Tokenize input sequence
        x = self._ttt_tokenize(seq, **kwargs)  # [bs=1, seq_len]

        # Build MSA if needed and tokenize it
        msa = None
        if self.ttt_cfg.msa:
            # Build MSA from scratch if needed
            if msa_pth is None:
                msa_pth = self.msa_server.get(seq)
            self.ttt_logger.debug(f"MSA path: {msa_pth}.")

            # Read MSA by replacing all insertions with padding tokens, then tokenize each sequence, and stack them
            msa = []
            for seq in read_msa(msa_pth, replace_inserstions=self._ttt_token_to_str(self._ttt_get_padding_token())):
                msa.append(self._ttt_tokenize(seq, **kwargs).squeeze(0))
            msa = torch.stack(msa)  # [msa_len, seq_len]

            # Check the MSA contains the target sequence as the first sequence
            assert torch.all(x[0, :] == msa[0, :]), "First sequence in MSA must be the same as the input sequence"

            # Set x to MSA to sample sequences from
            # - except for MSA soft labels where MSA is only used for loss calculation
            if not self.ttt_cfg.loss_kind == 'msa_soft_labels':
                x = msa

        # Get trainable parameters and optimizer
        parameters = self._ttt_get_parameters()
        optimizer = self._ttt_get_optimizer(parameters)
        optimizer.zero_grad()

        # Initialize dictionaries to store results and metrics each TTT step
        df = []
        ttt_step_data = defaultdict(dict)
        score_seq_time = None
        eval_step_time = None

        # Initialize best state and confidence for storing the best weights
        if self.ttt_cfg.automatic_best_state_reset:
            best_confidence = 0
            best_state = None

        # Run TTT loop
        x = x.to(next(self.parameters()).device)
        loss = None
        self.eval()
        for step in range(self.ttt_cfg.steps * self.ttt_cfg.ags + 1):
            # Sample batch
            batch_masked, targets, mask, start_indices = self._ttt_sample_batch(x)  # [bs, seq_len]

            # Score sequence with the updated model (predict log probs for each position) and evaluate TTT step
            # We make prediction before training in each iteration to store predictions before TTT (epoch 0)
            # The final predictions will be stored after the last iteration outside the TTT loop
            if (step) % self.ttt_cfg.ags == 0:
                # Measure time since the beginning of the last TTT step
                if step == 0:
                    last_step_time = time.time()
                ttt_step_time = time.time() - last_step_time

                # Score sequence
                all_log_probs, perplexity = None, None
                should_score = (
                    self.ttt_cfg.score_seq_kind is not None and (
                        self.ttt_cfg.score_seq_steps_list is None or
                        (step // self.ttt_cfg.ags) in self.ttt_cfg.score_seq_steps_list
                    )
                )
                if should_score:
                    score_seq_start_time = time.time()
                    all_log_probs, perplexity = self._ttt_score_seq(x, **kwargs)
                    score_seq_time = time.time() - score_seq_start_time
                    all_log_probs = [x.detach().cpu() for x in all_log_probs]
                    ttt_step_data[step // self.ttt_cfg.ags]['all_log_probs'] = all_log_probs
                else:
                    all_log_probs, perplexity = None, None
                    score_seq_time = 0.0
            
                # Evaluate TTT step
                if self.ttt_cfg.eval_each_step:
                    eval_step_start_time = time.time()
                    eval_step_preds, eval_step_metric_dict, confidence = self._ttt_eval_step(
                        step=step // self.ttt_cfg.ags,
                        loss=loss.item() if loss is not None else None,
                        perplexity=perplexity,
                        all_log_probs=all_log_probs,
                        seq=seq,
                        msa_pth=msa_pth,
                        **kwargs
                    )
                    eval_step_time = time.time() - eval_step_start_time

                    # Update best state and confidence
                    if self.ttt_cfg.automatic_best_state_reset and confidence is not None:
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_state = self._ttt_get_state()
                else:
                    eval_step_metric_dict = {}
                ttt_step_data[step // self.ttt_cfg.ags]['eval_step_preds'] = eval_step_preds
                
                # Store all metrics in a row
                row = dict(
                    step=step // self.ttt_cfg.ags,
                    accumulated_step=step,
                    loss=loss.item() if loss is not None else None,
                    perplexity=perplexity,
                    ttt_step_time=ttt_step_time,
                    score_seq_time=score_seq_time,
                    eval_step_time=eval_step_time,
                    **eval_step_metric_dict
                )
                df.append(row)

                # Log
                log_row = ', '.join([
                    f'{k}: {v:.5f}' if isinstance(v, float) else f'{k}: {v}'
                    for k, v in row.items()
                ])
                self.ttt_logger.info(log_row)

                last_step_time = time.time()

                # Early stopping
                if self.ttt_cfg.perplexity_early_stopping is not None and perplexity is not None:
                    if perplexity < self.ttt_cfg.perplexity_early_stopping:
                        self.ttt_logger.info(f"Early stopping at step {step} with perplexity {perplexity}")
                        break

            # Last step is just for logging
            if step == self.ttt_cfg.steps * self.ttt_cfg.ags:
                break

            # Forward pass
            self.train()
            logits = self._ttt_predict_logits(batch_masked, start_indices, **kwargs)
            
            # Calculate loss
            if self.ttt_cfg.loss_kind == 'cross_entropy':
                loss = self._ttt_cross_entropy_loss(logits, targets, mask)
            elif self.ttt_cfg.loss_kind == 'unnormalized_cross_entropy':
                loss = self._ttt_unnormalized_cross_entropy_loss(logits, targets, mask)
            elif self.ttt_cfg.loss_kind == 'msa_soft_labels':
                loss = self._ttt_msa_soft_labels_loss(logits, targets, mask, msa, start_indices)
            else:
                raise ValueError(f"Loss kind {self.ttt_cfg.loss_kind} not supported")

            # Backward pass
            loss.backward()
            if (step + 1) % self.ttt_cfg.ags == 0:
                optimizer.step()
                optimizer.zero_grad()
            self.eval()  

        # Reset to best state to have the most confident model after TTT
        if self.ttt_cfg.automatic_best_state_reset and best_state is not None:
            self._ttt_set_state(best_state)

        df = pd.DataFrame(df)
        return dict(
            ttt_step_data=ttt_step_data,
            df=df
        )

    def ttt_reset(self) -> None:
        if self._ttt_initial_state is None:
            raise ValueError("Initial state is not set. Make sure initial_state_reset=True in TTTConfig.")
        self._ttt_set_state(self._ttt_initial_state)

    @abstractmethod
    def _ttt_tokenize(self, seq: T.Optional[str] = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement _ttt_tokenize method")

    @abstractmethod
    def _ttt_predict_logits(self, batch: torch.Tensor, start_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Predict logits for a batch of sequences.

        Args:
            batch: Batch of sequences to predict logits for.
            start_indices: Starting indices of sequences in the batch with respect to the 
                original input sequence used for TTT customization. This argument may be needed
                as a result of cropping.
        """
        raise NotImplementedError("Subclass must implement _ttt_predict_logits method")
    
    @abstractmethod
    def _ttt_mask_token(self, token: int) -> int:
        raise NotImplementedError("Subclass must implement _ttt_mask_token method")
    
    @abstractmethod
    def _ttt_get_non_special_tokens(self) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement _ttt_get_non_special_tokens method")

    @abstractmethod
    def _ttt_get_padding_token(self) -> int:
        raise NotImplementedError("Subclass must implement _ttt_get_padding_token method")

    @abstractmethod
    def _ttt_token_to_str(self, token: int) -> str:
        raise NotImplementedError("Subclass must implement _ttt_token_to_str method")

    def _ttt_get_trainable_modules(self) -> list[torch.nn.Module]:
        """
        Return a list of modules to train. _ttt_get_frozen_modules is called after this function, so
        the returned modules can contain parameters that will be frozen.
        """
        return [self]
    
    def _ttt_get_frozen_modules(self) -> list[torch.nn.Module]:
        """
        Return a list of modules to freeze.
        """
        return []

    def _ttt_get_parameters(self) -> T.Iterator[torch.nn.Parameter]:
        """
        Configures and returns trainable parameters for TTT.

        If lora_rank > 0, injects LoRA layers into modules within 
        _ttt_get_trainable_modules() and makes only LoRA parameters trainable. Otherwise, 
        makes parameters trainable in _ttt_get_trainable_modules() excluding _ttt_get_frozen_modules().

        Returns:
            Iterator of parameters requiring gradients
        """
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Get modules to train
        module_list = self._ttt_get_trainable_modules()

        # Unfreeze parameters to train
        if self.ttt_cfg.lora_rank > 0:  # Train only LoRA parameters
            require_grad_param_groups = []
            for module in module_list:
                require_grad_params, names = inject_trainable_lora(
                    module,
                    target_replace_module=self.ttt_cfg.lora_target_replace_module,
                    r=self.ttt_cfg.lora_rank,
                    scale=self.ttt_cfg.lora_alpha
                )
                require_grad_param_groups.append(require_grad_params)
            for param_groups in require_grad_param_groups:
                for param_group in param_groups:
                    for param in param_group:
                        param.requires_grad = True
        else:  # Train all specified parameters
            for module in module_list:
                for param in module.parameters():
                    param.requires_grad = True
            for module in self._ttt_get_frozen_modules():
                for param in module.parameters():
                    param.requires_grad = False

        self.ttt_logger.debug("Parameters to be trained during TTT:")
        num_trainable_params = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                self.ttt_logger.debug(f"{name} {p.shape}")
                num_trainable_params += p.numel()
        self.ttt_logger.debug(f"Total trainable parameters: {num_trainable_params:,}")

        return filter(lambda p: p.requires_grad, self.parameters())

    def _ttt_get_optimizer(self, parameters: T.Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        if self.ttt_cfg.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                parameters, 
                lr=self.ttt_cfg.lr, 
                momentum=self.ttt_cfg.momentum, 
                weight_decay=self.ttt_cfg.weight_decay
            )
        elif self.ttt_cfg.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                parameters, 
                lr=self.ttt_cfg.lr, 
                weight_decay=self.ttt_cfg.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.ttt_cfg.optimizer} not supported")
        return optimizer

    def _ttt_get_state(self) -> T.Any:
        """Creates a deep copy of all child modules' states.

        The whole modules rather than parameters are saved to avoid support changing modules such
        as in the case of LoRA.

        TODO Optimize memory by only saving modules from self._ttt_get_trainable_modules()

        Returns:
            A dictionary mapping module names to their copied states.
        """
        state = {}
        for name, module in self.named_children():
            state[name] = copy.deepcopy(module)
        return state

    def _ttt_set_state(self, state: T.Any) -> None:
        """Restores model to a previously saved state.

        Args:
            state: Dictionary of module states from _ttt_get_state()
        """
        for k, v in state.items():
            if hasattr(self, k):
                delattr(self, k)
            self.add_module(k, copy.deepcopy(v))

    def _ttt_sample_batch(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, seq_len = x.shape
        batch_size = self.ttt_cfg.batch_size
        crop_size = self.ttt_cfg.crop_size

        # Create batch of unmasked and uncropped sequences
        if x.shape[0] == 1:
            # If only one sequence, replicate it batch_size times
            x_expanded = x.expand(batch_size, -1)
        elif x.shape[0] >= batch_size:
            # If multiple sequences available, randomly sample batch_size sequences
            indices = torch.randint(0, x.shape[0], (batch_size,), generator=self.ttt_generator)
            x_expanded = x[indices]
        else:  # 1 < x.shape[0] < batch_size
            # If fewer sequences than batch_size, replicate sequences up to batch_size
            num_repeats = batch_size // x.shape[0] + 1
            x_repeated = x.repeat(num_repeats, 1)
            indices = torch.randperm(x_repeated.shape[0], generator=self.ttt_generator)[:batch_size]
            x_expanded = x_repeated[indices]

        # Sample crop_size-tokens cropped subsequences
        if seq_len < crop_size:
            start_indices = torch.zeros(batch_size, dtype=torch.long)
            crop_size = seq_len
        else:
            start_indices = torch.randint(0, seq_len - crop_size + 1, (batch_size,), generator=self.ttt_generator).to(torch.long)
        batch_cropped = torch.stack([x_expanded[i, start:start + crop_size] for i, start in enumerate(start_indices)])

        # Get non-special tokens
        non_special_tokens = self._ttt_get_non_special_tokens()
        non_special_tokens_set = set(non_special_tokens)

        # Apply BERT masking only to non-special tokens
        mask = torch.zeros((batch_size, crop_size), dtype=torch.bool)
        for i in range(batch_size):
            non_special_positions = [j for j in range(crop_size) if batch_cropped[i,j].item() in non_special_tokens_set]
            if len(non_special_positions) > 0:
                num_to_mask = int(len(non_special_positions) * self.ttt_cfg.mask_ratio)
                if num_to_mask > 0:
                    positions_to_mask = torch.tensor(non_special_positions)[torch.randperm(len(non_special_positions), generator=self.ttt_generator)[:num_to_mask]]
                    mask[i, positions_to_mask] = True

        batch_masked = batch_cropped.clone()
        for i in range(batch_size):
            for idx in torch.nonzero(mask[i], as_tuple=True)[0]:
                if self.ttt_cfg.bert_leave_prob + self.ttt_cfg.bert_replace_prob > 0:
                    prob = torch.rand(1, generator=self.ttt_generator).item()
                    if prob < 1 - self.ttt_cfg.bert_leave_prob - self.ttt_cfg.bert_replace_prob:  # 80% random chance to mask token
                        batch_masked[i, idx] = self._ttt_mask_token(batch_masked[i, idx])
                    elif prob < 1 - self.ttt_cfg.bert_leave_prob:  # 10% chance to change to random token
                        batch_masked[i, idx] = non_special_tokens[torch.randint(0, len(non_special_tokens), (1,), generator=self.ttt_generator).item()]
                    else:  # 10% chance to keep current token
                        pass
                else:
                    # 100% change to mask token
                    batch_masked[i, idx] = self._ttt_mask_token(batch_masked[i, idx])

        # Targets are the original cropped sequences
        targets = batch_cropped

        return batch_masked, targets, mask, start_indices

    def _ttt_cross_entropy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        assert logits.ndim == 3, "Logits must be a 3D tensor [bs, seq_len, vocab_size]"
        bs, seq_len, vocab_size = logits.shape

        # Flatten logits and targets
        logits_reshaped = logits.view(-1, vocab_size)  # [bs*seq_len, vocab_size]
        if targets.ndim == 3:  # class probabilities are passed
            targets_reshaped = targets.view(-1, vocab_size)  # [bs*seq_len, vocab_size]
        else:  # class indices are passed
            targets_reshaped = targets.view(-1)  # [bs*seq_len]
        mask_reshaped = mask.view(-1).bool()  # [bs*seq_len]

        # Calculate cross-entropy loss over masked tokens
        loss = torch.nn.functional.cross_entropy(
            logits_reshaped[mask_reshaped],
            targets_reshaped[mask_reshaped],
            reduction='none'
        )  # [bs*seq_len]

        # Split loss back into per-sequence chunks and average each sequence separately
        masked_tokens_per_seq = mask.sum(dim=1)  # [bs]
        loss_split = torch.split(loss, masked_tokens_per_seq.tolist())  # List of [n_masked] tensors
        seq_losses = torch.stack([chunk.mean() for chunk in loss_split])  # [bs]
        loss = seq_losses.mean()
        return loss
    
    def _ttt_unnormalized_cross_entropy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        assert logits.ndim == 3, "Logits must be a 3D tensor [bs, seq_len, vocab_size]"
        bs, seq_len, vocab_size = logits.shape

        # Flatten logits and targets
        logits_reshaped = logits.view(-1, vocab_size)  # [bs*seq_len, vocab_size]
        targets_reshaped = targets.view(-1)  # [bs*seq_len]

        return torch.nn.functional.cross_entropy(logits_reshaped, targets_reshaped)

    def _ttt_msa_soft_labels_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        msa: torch.Tensor,
        start_indices: torch.Tensor
    ) -> torch.Tensor:
        # TODO Optimize by precomputing soft labels once after reading MSA

        # Get token sets
        all_tokens = self._ttt_get_all_tokens()
        non_special_tokens = self._ttt_get_non_special_tokens()
        vocab_size = len(all_tokens)
        bs = logits.shape[0]
        seq_len = msa.shape[1]

        # Initialize soft labels tensor
        msa_soft_labels = torch.zeros(seq_len, vocab_size)

        # For each position, count frequency of each non-special token across MSA sequences
        for pos in range(seq_len):
            col_tokens, counts = torch.unique(msa[:, pos], return_counts=True)
            for t, c in zip(col_tokens, counts):
                if t in non_special_tokens:
                    msa_soft_labels[pos, t] = c.item()

            # Convert counts to probabilities (nans for <bos> and <eos> columns)
            msa_soft_labels[pos] = msa_soft_labels[pos] / msa_soft_labels[pos].sum()

        # Duplicate msa_soft_labels bs times to match logits shape
        msa_soft_labels = msa_soft_labels.unsqueeze(0).expand(bs, seq_len, vocab_size)  # [bs, seq_len, vocab_size]

        # Apply start indices to get subsequences of length crop_size
        cropped_labels = []
        for i in range(bs):
            start = start_indices[i]
            end = start + self.ttt_cfg.crop_size
            cropped_labels.append(msa_soft_labels[i, start:end])
        msa_soft_labels = torch.stack(cropped_labels)  # [bs, crop_size, vocab_size]
        msa_soft_labels = msa_soft_labels.contiguous().to(logits.device)

        loss = self._ttt_cross_entropy_loss(logits, msa_soft_labels, mask)
        return loss

    def _ttt_score_seq(self, x: torch.Tensor, **kwargs) -> tuple[list[torch.Tensor], float]:
        """
        Score a sequence using TTT. 

        If the sequence is a multiple sequence alignment (MSA), only the first sequence is 
        used for scoring. The function handles special tokens by skipping them for perplexity
        calculation and putting zeros for log-probabilities. The function also handles the case
        when the sequence length is larger than the model context size (crop_size) by using the
        optimal window selection from ProteinGym when masking tokens.

        Returns:
            all_log_probs: Log probabilities for each token in the sequence when masked.
            perplexity: Perplexity of the sequence.
        """
        # Check input shape
        assert x.ndim == 2, "Input must be a 2D tensor"
        assert x.shape[0] == 1, "Input batch size must be 1"

        if self.ttt_cfg.score_seq_kind == 'pseudo_perplexity':
            all_log_probs, perplexity =  self._ttt_score_seq_pseudo_perplexity(x, **kwargs)
        elif self.ttt_cfg.score_seq_kind == 'gordon2024':
            all_log_probs, perplexity = self._ttt_score_seq_gordon2024(x, **kwargs)
        else:
            raise ValueError(f"Invalid score_seq_kind: {self.ttt_cfg.score_seq_kind}")

        return all_log_probs, perplexity

    def _ttt_score_seq_pseudo_perplexity(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> tuple[list[torch.Tensor], float]:
        # Get model-specific token sets
        all_tokens = self._ttt_get_all_tokens()
        non_special_tokens = self._ttt_get_non_special_tokens()

        all_log_probs = []  # [seq_len, vocab_size]
        wt_log_probs = []  # [seq_len]. Only for non-special tokens to calculate perplexity

        for i in range(x.size(-1)):

            # Check if the token is a special token
            i_special = False
            token = x[0, i]
            if token not in non_special_tokens:
                i_special = True

            # Mask current token
            x_masked = x.clone().to(x.device)
            x_masked[0, i] = self._ttt_mask_token(x_masked[0, i])
            
            # If sequence length is larger than the model context size, use the optimal window selection
            if x.size(-1) > self.ttt_cfg.crop_size:
                start, end = get_optimal_window(
                    mutation_position_relative=i,
                    seq_len_wo_special=x.size(-1),  # len(args.sequence)+2 in ProteinGym
                    model_window=self.ttt_cfg.crop_size
                )
                x_masked = x_masked[..., start:end]
            else:
                start=0

            # Predict logs for each token (amino acid) at the position
            with torch.no_grad():
                start_indices = torch.tensor([start], device=x.device)
                logits = self._ttt_predict_logits(x_masked, start_indices, **kwargs)
                token_log_probs = torch.log_softmax(logits, dim=-1)
                all_log_probs.append(token_log_probs[:, i-start])  # [1, vocab size]

            # Skip appending wild-type log-probabilities for special tokens (used later for perplexity calculation)
            if not i_special:
                wt_log_probs.append(token_log_probs[0, i-start, x[0, i-start]].item())

        # Stack log probabilities into a single tensor [seq_len, vocab_size]
        all_log_probs = torch.cat(all_log_probs, dim=0)

        # Calculate perplexity from wild-type log-probabilities
        perplexity = torch.exp(-torch.mean(torch.tensor(wt_log_probs))).item()

        return all_log_probs, perplexity

    def _ttt_score_seq_gordon2024(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> tuple[list[torch.Tensor], float]:
        """
        Score a sequence in a single forward pass using a method from Gordon et al. 2024 
        (https://openreview.net/forum?id=UvPdpa4LuV).

        Work in progress.
        """
        # Get all probabilities in a single forward pass without masking
        with torch.no_grad():
            logits = self._ttt_predict_logits(x, **kwargs)  # [bs=1, seq_len, vocab_size]
        all_probs = torch.softmax(logits, dim=-1)  # [bs=1, seq_len, vocab_size]

        # Calculate modified probabilities
        alpha = self.ttt_cfg.bert_replace_prob
        beta = self.ttt_cfg.bert_leave_prob
        eps = 1e-3
        all_probs = ((alpha + beta) / alpha) * all_probs - beta / alpha
        all_probs = all_probs.masked_fill(all_probs < eps, eps)  # [bs=1, seq_len, vocab_size]

        # Get log-probabilities
        all_log_probs = torch.log(all_probs)
        wt_log_probs = torch.tensor([all_log_probs[0, i, t] for i, t in enumerate(x[0, :])]).to(x.device)

        # Zero out log probabilities for special tokens and find the number of non-special tokens
        non_special_tokens = self._ttt_get_non_special_tokens()
        special_token_mask = torch.ones_like(x[0, :], dtype=torch.bool)
        special_token_mask[torch.isin(x[0, :], torch.tensor(non_special_tokens, device=x.device))] = False
        wt_log_probs = wt_log_probs.masked_fill(special_token_mask, 0)
        num_non_special_tokens = torch.sum(~special_token_mask).item()

        # Calculate perplexity
        perplexity = torch.exp(-torch.mean(wt_log_probs) / num_non_special_tokens).item()

        return all_log_probs, perplexity

    def _ttt_eval_step(
        self,
        step: int,
        loss: torch.Tensor,
        perplexity: float,
        all_log_probs: torch.Tensor,
        seq: str,
        msa_pth: Path,
        **kwargs
    ) -> tuple[dict, dict, T.Optional[float]]:
        return {}, {}, None
