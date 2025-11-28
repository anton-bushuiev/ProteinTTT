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

try:
    from lora_diffusion.lora import inject_trainable_lora
except ImportError:
    inject_trainable_lora = None

from proteinttt.utils.io import setup_logger
from proteinttt.utils.torch import preserve_model_state, get_optimal_window
from proteinttt.utils.msa import read_msa, MSAServer


@dataclass
class TTTConfig:
    """Configuration for protein test-time training (ProteinTTT).

    This class contains all the hyperparameters and settings needed to customize a protein
    language model to a specific protein sequence.

    Args:
        lr: Learning rate for optimization
        ags: Number of gradient accumulation steps
        steps: Number of optimization steps
        lora_rank: Rank for LoRA adaptation (0 to disable)
        lora_alpha: Scaling factor for LoRA layers
        lora_target_replace_module: Class of modules to apply LoRA to
        optimizer: Optimizer type ('adamw' or 'sgd')
        momentum: Momentum for SGD optimizer
        weight_decay: Weight decay regularization factor
        batch_size: Number of sequences per batch
        mask_ratio: Fraction of tokens to mask for MLM training
        crop_size: Maximum sequence length to process
        bert_leave_prob: Probability to leave token unchanged when masking
        bert_replace_prob: Probability to replace token with random token
        loss_kind: Type of loss function to use
        msa: Whether to use multiple sequence alignment
        msa_cache_dir: Directory to cache MSAs
        score_seq_kind: Method for sequence scoring
        score_seq_steps_list: Steps at which to score sequence
        perplexity_early_stopping: Early stopping threshold for perplexity
        eval_each_step: Whether to evaluate customization after each step (if _ttt_eval_step is implemented)
        initial_state_reset: Whether to save initial model state for resetting
        automatic_best_state_reset: Whether to automatically save best state
        seed: Random seed (None uses environment seed)
        log_file_path: Path to save log file
        log_name: Name for logger
        logger_level: Logging verbosity level
        tmalign_path: Path to TMalign executable
    """

    lr: float = 4e-4
    ags: int = 16
    steps: int = 30
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_target_replace_module: str = "MultiheadAttention"
    optimizer: str = "sgd"  # T.Literal['adamw', 'sgd']
    momentum: float = 0.0
    weight_decay: float = 0.0
    batch_size: int = 2
    mask_ratio: float = (
        0.15  # Used for ESM2 / ESMFold, SaProt, ProSST pre-training
    )
    crop_size: int = (
        1024  # Used for ESM2 / ESMFold, SaProt, ProSST pre-training
    )
    bert_leave_prob: float = 0.1
    bert_replace_prob: float = 0.1
    loss_kind: str = "cross_entropy"  # T.Literal['cross_entropy', 'unnormalized_cross_entropy', 'msa_soft_labels' TODO]
    model_kind: str = "bidirectional"  # T.Literal['bidirectional', 'autoregressive']
    msa: T.Optional[bool] = False
    msa_cache_dir: Path = Path.home() / ".cache" / "ttt"
    score_seq_kind: T.Optional[
        str
    ] = None  # T.Optional[T.Literal['pseudo_perplexity', 'gordon2024', 'none']] = None
    score_seq_steps_list: T.Any = (
        None  # T.Optional[int | list[int]]. None to use all steps
    )
    perplexity_early_stopping: T.Optional[float] = None
    eval_each_step: bool = True
    initial_state_reset: bool = True
    automatic_best_state_reset: bool = True
    seed: T.Optional[int] = 0  # None means using environment seed
    log_file_path: T.Optional[str] = None
    log_name: str = "ttt_log"
    logger_level: str = (
        "INFO"  # T.Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    )
    tmalign_path: T.Optional[Path] = None

    @classmethod
    def from_yaml(cls, yaml_path: T.Union[str, Path]) -> "TTTConfig":
        """Load TTTConfig from a YAML file using OmegaConf.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            TTTConfig instance initialized from YAML file
        """
        default_conf = OmegaConf.structured(cls)
        file_conf = OmegaConf.load(yaml_path)
        conf = OmegaConf.merge(default_conf, file_conf)
        OmegaConf.resolve(conf)
        return cls(**OmegaConf.to_container(conf))

    def verify(self) -> None:
        """Verify the configuration is valid.

        Checks that:
        - score_seq_kind is valid
        - score_seq_steps_list has correct format
        - perplexity_early_stopping is used correctly
        - loss_kind and msa settings are compatible
        - LoRA requirements are met if enabled

        Raises:
            ValueError: If configuration is invalid
            ImportError: If LoRA is enabled but package not installed
        """
        if self.score_seq_kind == "none":
            self.score_seq_kind = None

        if self.score_seq_steps_list is not None:
            if isinstance(self.score_seq_steps_list, int):
                self.score_seq_steps_list = [self.score_seq_steps_list]
            elif not isinstance(self.score_seq_steps_list, list):
                raise ValueError(
                    "score_seq_steps_list must be None, an integer, or a list of integers"
                )
            if not all(isinstance(x, int) for x in self.score_seq_steps_list):
                raise ValueError(
                    "All elements in score_seq_steps_list must be integers"
                )

        if (
            self.perplexity_early_stopping is not None
            and self.score_seq_kind is None
        ):
            raise ValueError(
                "perplexity_early_stopping can only be used if score_seq_kind is not None"
            )

        if self.loss_kind == "msa_soft_labels" and not self.msa:
            raise ValueError(
                "msa_soft_labels loss kind can only be used if msa=True"
            )

        if self.lora_rank > 0 and inject_trainable_lora is None:
            raise ImportError(
                "lora_diffusion is not installed. Please install it with "
                "`pip install git+https://github.com/cloneofsimo/lora.git`."
            )

        if self.model_kind not in ['bidirectional', 'autoregressive']:
            raise ValueError(
                f"Invalid model kind: {self.model_kind}. Valid model kinds are 'bidirectional' and 'autoregressive'"
            )

        if self.model_kind == "autoregressive":
            if self.loss_kind != "unnormalized_cross_entropy":
                raise ValueError(
                    "Only loss kind 'unnormalized_cross_entropy' is supported for autoregressive models"
                )
            if self.batch_size != 1:
                raise ValueError(
                    "Batch size must be 1 for autoregressive models"
                )
            if self.score_seq_kind is not None:
                raise ValueError(
                    "Scoring for autoregressive models is not implemented yet"
                )

        if self.tmalign_path is not None and not self.tmalign_path.exists():
            raise FileNotFoundError(f"TMalign executable not found at {self.tmalign_path}")


class TTTModule(torch.nn.Module, ABC):
    """Base class for test-time training modules.

    This abstract class provides the core functionality for test-time training of protein
    language models. It handles model state management, optimization, evaluation, and
    various training strategies like MLM and MSA-based training.

    Child classes must implement several abstract methods to define model-specific
    behavior.

    Attributes:
        ttt_default_cfg: Default configuration to use if none provided
        ttt_cfg: Active configuration for this instance
        msa_server: Server for handling multiple sequence alignments
        ttt_generator: Random number generator for reproducibility
        ttt_logger: Logger for tracking customization progress
        _ttt_initial_state: Saved initial model state for resetting
    """

    ttt_default_cfg: T.Optional[TTTConfig] = None

    def __init__(self, ttt_cfg: T.Optional[T.Union[TTTConfig, Path, str]] = None):
        """Initialize TTTModule.

        Args:
            ttt_cfg: Configuration for test-time training. Can be:
                - TTTConfig instance
                - Path to YAML config file
                - None to use default config
        """
        ABC.__init__(
            self
        )  # no torch.nn.Module init because it is already done in child class

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
            logger_level=self.ttt_cfg.logger_level,
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
        **kwargs,
    ) -> "TTTModule":
        """Create TTTModule instance from a pretrained model.

        Args:
            model: Pretrained model to adapt
            ttt_cfg: Configuration for test-time training
            **kwargs: Additional arguments passed to the pretrained model's class constructor

        Returns:
            TTTModule instance initialized from pretrained model
        """
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
        **kwargs,
    ) -> T.Dict[str, T.Any]:
        """Run test-time training loop to customize model to input protein.

        Performs iterative optimization to adapt the model's parameters to better fit
        the input protein sequence. Uses masked language modeling on the input sequence and
        optionally MSA.

        Args:
            seq: Input amino acid sequence to customize model to
            msa_pth: Path to MSA file (optional)
            **kwargs: Additional arguments passed to model forward pass

        Returns:
            Dictionary containing training metrics and results
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
            for seq in read_msa(
                msa_pth,
                replace_inserstions=self._ttt_token_to_str(
                    self._ttt_get_padding_token()
                ),
            ):
                msa.append(self._ttt_tokenize(seq, **kwargs).squeeze(0))
            msa = torch.stack(msa)  # [msa_len, seq_len]

            # Check the MSA contains the target sequence as the first sequence
            assert torch.all(
                x[0, :] == msa[0, :]
            ), "First sequence in MSA must be the same as the input sequence"

            # Set x to MSA to sample sequences from
            # - except for MSA soft labels where MSA is only used for loss calculation
            if not self.ttt_cfg.loss_kind == "msa_soft_labels":
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
        loss = None
        self.eval()
        for step in range(self.ttt_cfg.steps * self.ttt_cfg.ags + 1):
            # Sample batch
            batch_masked, targets, mask, start_indices = self._ttt_sample_batch(
                x
            )  # [bs, seq_len]

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
                should_score = self.ttt_cfg.score_seq_kind is not None and (
                    self.ttt_cfg.score_seq_steps_list is None
                    or (step // self.ttt_cfg.ags)
                    in self.ttt_cfg.score_seq_steps_list
                )
                if should_score:
                    score_seq_start_time = time.time()
                    # Take only the input sequence for scoring
                    seq_to_score = x[0:1, :]
                    all_log_probs, perplexity = self._ttt_score_seq(seq_to_score, **kwargs)
                    
                    score_seq_time = time.time() - score_seq_start_time
                    all_log_probs = [prob.detach().cpu() for prob in all_log_probs]
                    ttt_step_data[step // self.ttt_cfg.ags][
                        "all_log_probs"
                    ] = all_log_probs
                else:
                    all_log_probs, perplexity = None, None
                    score_seq_time = 0.0

                # Evaluate TTT step
                if self.ttt_cfg.eval_each_step:
                    eval_step_start_time = time.time()
                    (
                        eval_step_preds,
                        eval_step_metric_dict,
                        confidence,
                    ) = self._ttt_eval_step(
                        step=step // self.ttt_cfg.ags,
                        loss=loss.item() if loss is not None else None,
                        perplexity=perplexity,
                        all_log_probs=all_log_probs,
                        seq=seq,
                        msa_pth=msa_pth,
                        **kwargs,
                    )
                    eval_step_time = time.time() - eval_step_start_time

                    # Update best state and confidence
                    if (
                        self.ttt_cfg.automatic_best_state_reset
                        and confidence is not None
                    ):
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_state = self._ttt_get_state()
                else:
                    eval_step_metric_dict = {}
                ttt_step_data[step // self.ttt_cfg.ags][
                    "eval_step_preds"
                ] = eval_step_preds

                # Store all metrics in a row
                row = dict(
                    step=step // self.ttt_cfg.ags,
                    accumulated_step=step,
                    loss=loss.item() if loss is not None else None,
                    perplexity=perplexity,
                    ttt_step_time=ttt_step_time,
                    score_seq_time=score_seq_time,
                    eval_step_time=eval_step_time,
                    **eval_step_metric_dict,
                )
                df.append(row)

                # Log
                log_row = ", ".join(
                    [
                        f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in row.items()
                    ]
                )
                self.ttt_logger.info(log_row)

                last_step_time = time.time()

                # Early stopping
                if (
                    self.ttt_cfg.perplexity_early_stopping is not None
                    and perplexity is not None
                ):
                    if perplexity < self.ttt_cfg.perplexity_early_stopping:
                        self.ttt_logger.info(
                            f"Early stopping at step {step} with perplexity {perplexity}"
                        )
                        break

            # Last step is just for logging
            if step == self.ttt_cfg.steps * self.ttt_cfg.ags:
                break

            # Move the sampled batch to the GPU
            device = next(self.parameters()).device
            batch_masked = batch_masked.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            if start_indices is not None:
                start_indices = start_indices.to(device)

            # Forward pass
            self.train()
            logits = self._ttt_predict_logits(
                batch_masked, start_indices, **kwargs
            )

            # Calculate loss
            if self.ttt_cfg.loss_kind == "cross_entropy":
                loss = self._ttt_cross_entropy_loss(logits, targets, mask)
            elif self.ttt_cfg.loss_kind == "unnormalized_cross_entropy":
                if self.ttt_cfg.model_kind == "bidirectional":
                    loss = self._ttt_unnormalized_cross_entropy_loss(
                        logits, targets, mask
                    )
                elif self.ttt_cfg.model_kind == "autoregressive":
                    seq_start = start_indices[0]  # [0] because batch size is always 1 for autoregressive models
                    seq_end = seq_start + self.ttt_cfg.crop_size
                    targets = x[:, seq_start + 1:seq_end]  # + 1 for teacher forcing, logits should be shifted by - 1 in TTTModule._ttt_predict_logits
                    loss = self._ttt_unnormalized_cross_entropy_loss(
                        logits, targets, None
                    )
            elif self.ttt_cfg.loss_kind == "msa_soft_labels":
                loss = self._ttt_msa_soft_labels_loss(
                    logits, targets, mask, msa, start_indices
                )
            else:
                raise ValueError(
                    f"Loss kind {self.ttt_cfg.loss_kind} not supported"
                )

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
        return dict(ttt_step_data=ttt_step_data, df=df)

    def ttt_reset(self) -> None:
        """Reset model to initial state that was before ProteinTTT started.

        Raises:
            ValueError: If initial state was not saved during initialization
        """
        if self._ttt_initial_state is None:
            raise ValueError(
                "Initial state is not set. Make sure initial_state_reset=True in TTTConfig."
            )
        self._ttt_set_state(self._ttt_initial_state)

    @abstractmethod
    def _ttt_tokenize(
        self, seq: T.Optional[str] = None, **kwargs
    ) -> torch.Tensor:
        """Convert input sequence to tensor of token indices.

        Args:
            seq: Input amino acid sequence
            **kwargs: Additional arguments for tokenization

        Returns:
            Tensor of token indices [batch_size, sequence_length]
        """
        raise NotImplementedError(
            "Subclass must implement _ttt_tokenize method"
        )

    @abstractmethod
    def _ttt_predict_logits(
        self, batch: torch.Tensor, start_indices: torch.Tensor = None
    ) -> torch.Tensor:
        """Predict logits for each position in input sequences.

        Args:
            batch: Batch of token indices [batch_size, sequence_length]
            start_indices: Starting indices for cropped sequences

        Returns:
            Logits tensor [batch_size, sequence_length, vocab_size]
        """
        raise NotImplementedError(
            "Subclass must implement _ttt_predict_logits method"
        )

    @abstractmethod
    def _ttt_mask_token(self, token: int) -> int:
        """Convert token to its masked version.

        Args:
            token: Token index to mask

        Returns:
            Index of mask token
        """
        raise NotImplementedError(
            "Subclass must implement _ttt_mask_token method"
        )

    @abstractmethod
    def _ttt_get_non_special_tokens(self) -> T.List[int]:
        """Get indices of non-special tokens (e.g. 20 standard amino acids).

        Returns:
            List of token indices
        """
        raise NotImplementedError(
            "Subclass must implement _ttt_get_non_special_tokens method"
        )

    def _ttt_get_token_replacement_candidates(self, token: int) -> T.List[int]:
        """Get candidates for token replacement.

        For some models (e.g. DPLM2), the replacement candidates differ for sequence and
        structure tokens. So, simply returning all non-special tokens may not be sufficient
        and a different implementation may be needed.

        Returns:
            List of token indices
        """
        return self._ttt_get_non_special_tokens()

    def _ttt_get_padding_token(self) -> int:
        """Get index of padding token.

        This abstrcact method is optional. The error will be raised if the implementation is needed.

        Returns:
            Padding token index
        """
        raise NotImplementedError(
            "Subclass must implement _ttt_get_padding_token method"
        )

    def _ttt_token_to_str(self, token: int) -> str:
        """Convert token index to string representation.

        This abstrcact method is optional. The error will be raised if the implementation is needed.

        Args:
            token: Token index

        Returns:
            String representation of token
        """
        raise NotImplementedError(
            "Subclass must implement _ttt_token_to_str method"
        )

    def _ttt_get_trainable_modules(self) -> T.List[torch.nn.Module]:
        """Get list of modules to train.

        Note that some parameters in these modules may still be frozen by
        _ttt_get_frozen_modules.

        Returns:
            List of PyTorch modules
        """
        return [self]

    def _ttt_get_frozen_modules(self) -> T.List[torch.nn.Module]:
        """Get list of modules to freeze during training.

        Returns:
            List of PyTorch modules
        """
        return []

    def _ttt_get_parameters(self) -> T.Iterator[torch.nn.Parameter]:
        """Configure and return trainable parameters for test-time training.

        If LoRA is enabled, injects LoRA layers and makes only those parameters
        trainable. Otherwise makes parameters trainable in modules from
        _ttt_get_trainable_modules excluding those in _ttt_get_frozen_modules.

        TODO: If one wants to further customize an already customized model without resetting,
        this currently fails with `ValueError: optimizer got an empty parameter list` when LoRA is
        enabled. This is not an intended use case, but supporting it in the future could be 
        interesting.

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
            if inject_trainable_lora is None:
                raise ImportError(
                    "lora_diffusion is not installed. Please install it with `pip install git+https://github.com/cloneofsimo/lora.git`."
                )

            require_grad_param_groups = []
            for module in module_list:
                require_grad_params, names = inject_trainable_lora(
                    module,
                    target_replace_module=self.ttt_cfg.lora_target_replace_module,
                    r=self.ttt_cfg.lora_rank,
                    scale=self.ttt_cfg.lora_alpha,
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
        self.ttt_logger.debug(
            f"Total trainable parameters: {num_trainable_params:,}"
        )

        return filter(lambda p: p.requires_grad, self.parameters())

    def _ttt_get_optimizer(
        self, parameters: T.Iterator[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:
        """Create optimizer for test-time training.

        Args:
            parameters: Iterator of parameters to optimize

        Returns:
            PyTorch optimizer

        Raises:
            ValueError: If optimizer type not supported
        """
        if self.ttt_cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.ttt_cfg.lr,
                momentum=self.ttt_cfg.momentum,
                weight_decay=self.ttt_cfg.weight_decay,
            )
        elif self.ttt_cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.ttt_cfg.lr,
                weight_decay=self.ttt_cfg.weight_decay,
            )
        else:
            raise ValueError(
                f"Optimizer {self.ttt_cfg.optimizer} not supported"
            )
        return optimizer

    def _ttt_get_state(self) -> T.Any:
        """Create deep copy of all child modules' states.

        The whole modules rather than parameters are saved to avoid support changing
        modules such as in the case of LoRA.

        TODO: Optimize memory by only saving modules from _ttt_get_trainable_modules()

        Returns:
            Dictionary mapping module names to their copied states
        """
        state = {}

        for name, module in self.named_children():
            state[name] = copy.deepcopy(module)

        return state

    def _ttt_set_state(self, state: T.Any) -> None:
        """Restore model to a previously saved state.

        Args:
            state: Dictionary of module states from _ttt_get_state()
        """
        for k, v in state.items():
            if hasattr(self, k):
                delattr(self, k)
            self.add_module(k, copy.deepcopy(v))

    def _ttt_sample_batch(
        self, x: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample and mask a batch of sequences for training.

        Args:
            x: Input sequences [num_sequences, sequence_length]

        Returns:
            Tuple containing:
            - Masked sequences [batch_size, crop_size]
            - Target sequences [batch_size, crop_size]
            - Mask indicating masked positions [batch_size, crop_size]
            - Starting indices for cropped sequences [batch_size]
        """
        _, seq_len = x.shape
        batch_size = self.ttt_cfg.batch_size
        crop_size = self.ttt_cfg.crop_size

        # Create batch of unmasked and uncropped sequences
        if x.shape[0] == 1:
            # If only one sequence, replicate it batch_size times
            x_expanded = x.expand(batch_size, -1)
        elif x.shape[0] >= batch_size:
            # If multiple sequences available, randomly sample batch_size sequences
            indices = torch.randint(
                0, x.shape[0], (batch_size,), generator=self.ttt_generator
            )
            x_expanded = x[indices]
        else:  # 1 < x.shape[0] < batch_size
            # If fewer sequences than batch_size, replicate sequences up to batch_size
            num_repeats = batch_size // x.shape[0] + 1
            x_repeated = x.repeat(num_repeats, 1)
            indices = torch.randperm(
                x_repeated.shape[0], generator=self.ttt_generator
            )[:batch_size]
            x_expanded = x_repeated[indices]

        # Sample crop_size-tokens cropped subsequences
        if seq_len < crop_size:
            start_indices = torch.zeros(batch_size, dtype=torch.long)
            crop_size = seq_len
        else:
            start_indices = torch.randint(
                0,
                seq_len - crop_size + 1,
                (batch_size,),
                generator=self.ttt_generator,
            ).to(torch.long)
        batch_cropped = torch.stack(
            [
                x_expanded[i, start : start + crop_size]
                for i, start in enumerate(start_indices)
            ]
        )

        # Get non-special tokens
        non_special_tokens = self._ttt_get_non_special_tokens()
        non_special_tokens_set = set(non_special_tokens)

        # Apply BERT masking only to non-special tokens
        mask = torch.zeros((batch_size, crop_size), dtype=torch.bool)
        for i in range(batch_size):
            non_special_positions = [
                j
                for j in range(crop_size)
                if batch_cropped[i, j].item() in non_special_tokens_set
            ]
            if len(non_special_positions) > 0:
                num_to_mask = int(
                    len(non_special_positions) * self.ttt_cfg.mask_ratio
                )
                if num_to_mask > 0:
                    positions_to_mask = torch.tensor(non_special_positions)[
                        torch.randperm(
                            len(non_special_positions),
                            generator=self.ttt_generator,
                        )[:num_to_mask]
                    ]
                    mask[i, positions_to_mask] = True

        batch_masked = batch_cropped.clone()
        for i in range(batch_size):
            for idx in torch.nonzero(mask[i], as_tuple=True)[0]:
                orig_token = batch_masked[i, idx].item()
                if (
                    self.ttt_cfg.bert_leave_prob
                    + self.ttt_cfg.bert_replace_prob
                    > 0
                ):
                    prob = torch.rand(1, generator=self.ttt_generator).item()
                    if (
                        prob
                        < 1
                        - self.ttt_cfg.bert_leave_prob
                        - self.ttt_cfg.bert_replace_prob
                    ):  # 80% random chance to mask token
                        batch_masked[i, idx] = self._ttt_mask_token(
                            batch_masked[i, idx]
                        )
                    elif (
                        prob < 1 - self.ttt_cfg.bert_leave_prob
                    ):  # 10% chance to change to random token
                        cands = self._ttt_get_token_replacement_candidates(orig_token)
                        batch_masked[i, idx] = cands[
                            torch.randint(
                                0,
                                len(cands),
                                (1,),
                                generator=self.ttt_generator,
                            ).item()
                        ]
                    else:  # 10% chance to keep current token
                        pass
                else:
                    # 100% change to mask token
                    batch_masked[i, idx] = self._ttt_mask_token(
                        batch_masked[i, idx]
                    )

        # Targets are the original cropped sequences
        targets = batch_cropped

        return batch_masked, targets, mask, start_indices

    def _ttt_cross_entropy_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculate cross entropy loss over masked tokens.

        Computes per-sequence cross entropy loss by first flattening the inputs, calculating
        loss only on masked tokens, then splitting back into per-sequence chunks and averaging.

        Args:
            logits: Model predictions [batch_size, sequence_length, vocab_size]
            targets: Ground truth tokens [batch_size, sequence_length] or [batch_size, sequence_length, vocab_size]
            mask: Boolean mask indicating positions to calculate loss for [batch_size, sequence_length]

        Returns:
            Mean cross entropy loss across sequences
        """
        assert (
            logits.ndim == 3
        ), "Logits must be a 3D tensor [bs, seq_len, vocab_size]"
        bs, seq_len, vocab_size = logits.shape

        # Flatten logits and targets
        logits_reshaped = logits.view(
            -1, vocab_size
        )  # [bs*seq_len, vocab_size]
        if targets.ndim == 3:  # class probabilities are passed
            targets_reshaped = targets.view(
                -1, vocab_size
            )  # [bs*seq_len, vocab_size]
        else:  # class indices are passed
            targets_reshaped = targets.view(-1)  # [bs*seq_len]
        mask_reshaped = mask.view(-1).bool()  # [bs*seq_len]

        # Calculate cross-entropy loss over masked tokens
        loss = torch.nn.functional.cross_entropy(
            logits_reshaped[mask_reshaped],
            targets_reshaped[mask_reshaped],
            reduction="none",
        )  # [bs*seq_len]

        # Split loss back into per-sequence chunks and average each sequence separately
        masked_tokens_per_seq = mask.sum(dim=1)  # [bs]
        loss_split = torch.split(
            loss, masked_tokens_per_seq.tolist()
        )  # List of [n_masked] tensors
        seq_losses = torch.stack([chunk.mean() for chunk in loss_split])  # [bs]
        loss = seq_losses.mean()
        return loss

    def _ttt_unnormalized_cross_entropy_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculate unnormalized cross entropy loss.

        Similar to cross entropy loss but without normalizing by sequence length.

        Args:
            logits: Model predictions [batch_size, sequence_length, vocab_size]
            targets: Ground truth token indices [batch_size, sequence_length]
            mask: Boolean mask indicating positions to calculate loss for [batch_size, sequence_length]

        Returns:
            Unnormalized cross entropy loss
        """
        assert (
            logits.ndim == 3
        ), "Logits must be a 3D tensor [bs, seq_len, vocab_size]"
        bs, seq_len, vocab_size = logits.shape

        # Flatten logits and targets
        logits_reshaped = logits.view(
            -1, vocab_size
        )  # [bs*seq_len, vocab_size]
        targets_reshaped = targets.view(-1)  # [bs*seq_len]

        return torch.nn.functional.cross_entropy(
            logits_reshaped, targets_reshaped
        )

    def _ttt_msa_soft_labels_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        msa: torch.Tensor,
        start_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate loss using soft labels derived from MSA frequencies.

        For each position, computes amino acid frequencies from the MSA and uses these as
        soft labels for cross entropy loss calculation. See Gong et al. 2024 "Evolution-Inspired
        Loss Functions for Protein Representation Learning"
        (https://proceedings.mlr.press/v235/gong24e.html) for details on soft labels.

        Args:
            logits: Model predictions [batch_size, sequence_length, vocab_size]
            targets: Ground truth tokens [batch_size, sequence_length]
            mask: Boolean mask indicating positions to calculate loss for [batch_size, sequence_length]
            msa: Multiple sequence alignment [num_sequences, sequence_length]
            start_indices: Starting indices for cropped sequences [batch_size]

        Returns:
            Cross entropy loss using MSA-derived soft labels
        """
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
            msa_soft_labels[pos] = (
                msa_soft_labels[pos] / msa_soft_labels[pos].sum()
            )

        # Duplicate msa_soft_labels bs times to match logits shape
        msa_soft_labels = msa_soft_labels.unsqueeze(0).expand(
            bs, seq_len, vocab_size
        )  # [bs, seq_len, vocab_size]

        # Apply start indices to get subsequences of length crop_size
        cropped_labels = []
        for i in range(bs):
            start = start_indices[i]
            end = start + self.ttt_cfg.crop_size
            cropped_labels.append(msa_soft_labels[i, start:end])
        msa_soft_labels = torch.stack(
            cropped_labels
        )  # [bs, crop_size, vocab_size]
        msa_soft_labels = msa_soft_labels.contiguous().to(logits.device)

        loss = self._ttt_cross_entropy_loss(logits, msa_soft_labels, mask)
        return loss

    def _ttt_score_seq(
        self, x: torch.Tensor, **kwargs
    ) -> T.Tuple[T.List[torch.Tensor], float]:
        """Score a sequence.

        If the sequence is a multiple sequence alignment (MSA), only the first sequence is
        used for scoring. The function handles special tokens by skipping them for perplexity
        calculation and putting zeros for log-probabilities. The function also handles the case
        when the sequence length is larger than the model context size (crop_size) by using the
        optimal window selection from ProteinGym when masking tokens.

        Args:
            x: Input sequence tensor [1, sequence_length]
            **kwargs: Additional arguments passed to model forward pass

        Returns:
            tuple containing:
                - all_log_probs: Log probabilities for each token in the sequence when masked
                - perplexity: Perplexity of the sequence
        """
        # Check input shape
        assert x.ndim == 2, "Input must be a 2D tensor"
        assert x.shape[0] == 1, "Input batch size must be 1"

        if self.ttt_cfg.score_seq_kind == "pseudo_perplexity":
            all_log_probs, perplexity = self._ttt_score_seq_pseudo_perplexity(
                x, **kwargs
            )
        elif self.ttt_cfg.score_seq_kind == "gordon2024":
            all_log_probs, perplexity = self._ttt_score_seq_gordon2024(
                x, **kwargs
            )
        else:
            raise ValueError(
                f"Invalid score_seq_kind: {self.ttt_cfg.score_seq_kind}"
            )

        return all_log_probs, perplexity

    def _ttt_score_seq_pseudo_perplexity(
        self, x: torch.Tensor, **kwargs
    ) -> T.Tuple[T.List[torch.Tensor], float]:
        """Score sequence using pseudo-perplexity.

        Calculates pseudo-perplexity by masking each token one at a time and computing
        the model's prediction probability for the original token.

        Args:
            x: Input sequence tensor [1, sequence_length]
            **kwargs: Additional arguments passed to model forward pass

        Returns:
            tuple containing:
                - all_log_probs: Log probabilities for each token in the sequence when masked
                - perplexity: Pseudo-perplexity of the sequence
        """
        # Get model-specific token sets
        all_tokens = self._ttt_get_all_tokens()
        non_special_tokens = self._ttt_get_non_special_tokens()

        all_log_probs = []  # [seq_len, vocab_size]
        wt_log_probs = (
            []
        )  # [seq_len]. Only for non-special tokens to calculate perplexity

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
                    seq_len_wo_special=x.size(
                        -1
                    ),  # len(args.sequence)+2 in ProteinGym
                    model_window=self.ttt_cfg.crop_size,
                )
                x_masked = x_masked[..., start:end]
            else:
                start = 0

            # Predict logs for each token (amino acid) at the position
            with torch.no_grad():
                start_indices = torch.tensor([start], device=x.device)
                logits = self._ttt_predict_logits(
                    x_masked, start_indices, **kwargs
                )
                token_log_probs = torch.log_softmax(logits, dim=-1)
                all_log_probs.append(
                    token_log_probs[:, i - start]
                )  # [1, vocab size]

            # Skip appending wild-type log-probabilities for special tokens (used later for perplexity calculation)
            if not i_special:
                wt_log_probs.append(
                    token_log_probs[0, i - start, x[0, i - start]].item()
                )

        # Stack log probabilities into a single tensor [seq_len, vocab_size]
        all_log_probs = torch.cat(all_log_probs, dim=0)

        # Calculate perplexity from wild-type log-probabilities
        perplexity = torch.exp(-torch.mean(torch.tensor(wt_log_probs))).item()

        return all_log_probs, perplexity

    def _ttt_score_seq_gordon2024(
        self, x: torch.Tensor, **kwargs
    ) -> T.Tuple[T.List[torch.Tensor], float]:
        """Score sequence using method from Gordon et al. 2024.

        Implements sequence scoring method from Gordon et al. 2024
        (https://openreview.net/forum?id=UvPdpa4LuV) that requires only a single
        forward pass without masking.

        Args:
            x: Input sequence tensor [1, sequence_length]
            **kwargs: Additional arguments passed to model forward pass

        Returns:
            tuple containing:
                - all_log_probs: Log probabilities for each token in the sequence
                - perplexity: Perplexity calculated using Gordon et al. method

        Note:
            This is a work in progress implementation.
        """
        # Get all probabilities in a single forward pass without masking
        with torch.no_grad():
            logits = self._ttt_predict_logits(
                x, **kwargs
            )  # [bs=1, seq_len, vocab_size]
        all_probs = torch.softmax(logits, dim=-1)  # [bs=1, seq_len, vocab_size]

        # Calculate modified probabilities
        alpha = self.ttt_cfg.bert_replace_prob
        beta = self.ttt_cfg.bert_leave_prob
        eps = 1e-3
        all_probs = ((alpha + beta) / alpha) * all_probs - beta / alpha
        all_probs = all_probs.masked_fill(
            all_probs < eps, eps
        )  # [bs=1, seq_len, vocab_size]

        # Get log-probabilities
        all_log_probs = torch.log(all_probs)
        wt_log_probs = torch.tensor(
            [all_log_probs[0, i, t] for i, t in enumerate(x[0, :])]
        ).to(x.device)

        # Zero out log probabilities for special tokens and find the number of non-special tokens
        non_special_tokens = self._ttt_get_non_special_tokens()
        special_token_mask = torch.ones_like(x[0, :], dtype=torch.bool)
        special_token_mask[
            torch.isin(
                x[0, :], torch.tensor(non_special_tokens, device=x.device)
            )
        ] = False
        wt_log_probs = wt_log_probs.masked_fill(special_token_mask, 0)
        num_non_special_tokens = torch.sum(~special_token_mask).item()

        # Calculate perplexity
        perplexity = torch.exp(
            -torch.mean(wt_log_probs) / num_non_special_tokens
        ).item()

        return all_log_probs, perplexity

    def _ttt_eval_step(
        self,
        step: int,
        loss: torch.Tensor,
        perplexity: float,
        all_log_probs: torch.Tensor,
        seq: str,
        msa_pth: Path,
        **kwargs,
    ) -> T.Tuple[dict, dict, T.Optional[float]]:
        """Evaluate model during test-time training (e.g., to select the optimal step).

        Base implementation that returns empty dictionaries. Child classes should override
        this to implement model-specific evaluation.

        Args:
            step: Current training step
            loss: Training loss at current step
            perplexity: Perplexity at current step
            all_log_probs: Log probabilities for each token
            seq: Input amino acid sequence
            msa_pth: Path to MSA file
            **kwargs: Additional arguments passed to model forward pass

        Returns:
            tuple containing:
                - Dictionary of predictions
                - Dictionary of evaluation metrics
                - Optional confidence score
        """
        return {}, {}, None
