import typing as T

import torch
from byprot.models.dplm2 import DPLM2Bit

from proteinttt.base import TTTModule, TTTConfig


DEFAULT_DPLM2BIT_TTT_CFG = TTTConfig(
    lr=4e-4, batch_size=4, ags=2, steps=10
)


class DPLM2BitTTT(TTTModule, DPLM2Bit):
    """
    ProteinTTT wrapper for DPLM2Bit.

    Example usage:
    ```python
    model = DPLM2BitTTT.ttt_from_pretrained(
        model, cfg=model.cfg, net=model.net, ttt_cfg=DEFAULT_DPLM2BIT_TTT_CFG
    )
    ```
    """
    ttt_default_cfg = DEFAULT_DPLM2BIT_TTT_CFG

    # see https://github.com/bytedance/dplm/blob/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d/src/byprot/models/dplm2/dplm2_bit.py#L80
    LAST_SEQ_TOKEN = 32

    def __init__(self, ttt_cfg: TTTConfig, **kwargs):
        DPLM2Bit.__init__(self, **kwargs)
        TTTModule.__init__(self, ttt_cfg=ttt_cfg)

    def _ttt_tokenize(self, seq: T.Optional[str] = None, **kwargs) -> torch.Tensor:
        if seq is not None:
            raise NotImplementedError(
                "Please pass already tokenized 'input_tokens' instead of 'seq' to the .ttt method."
            )
        assert "input_tokens" in kwargs, "Tokenized 'input_tokens' must be provided."
        x = kwargs["input_tokens"]
        assert isinstance(x, torch.Tensor), "'input_tokens' must be a tensor"
        return x

    def _ttt_get_trainable_modules(self) -> list[torch.nn.Module]:
        return [self.net.esm.encoder]

    def _ttt_mask_token(self, token: int) -> int:
        if token <= self.LAST_SEQ_TOKEN:
            return self.tokenizer.token_to_id(self.tokenizer.aa_mask_token)
        else:
            return self.tokenizer.token_to_id(self.tokenizer.struct_mask_token)

    def _ttt_get_all_tokens(self) -> list[int]:
        return [self.tokenizer.token_to_id(t) for t in self.tokenizer.all_tokens]

    def _ttt_get_non_special_tokens(self) -> list[int]:
        """
        Note: 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>' are also treated as non-special tokens
        """
        return [
            self.tokenizer.token_to_id(t)
            for t in self.tokenizer.all_tokens
            if t not in self.tokenizer.all_special_tokens
        ]

    def _ttt_get_token_replacement_candidates(self, token: int) -> list[int]:
        """
        Return either all non-special sequence tokens or all non-special structure tokens.
        """
        if token <= self.LAST_SEQ_TOKEN:
            return [t for t in self._ttt_get_non_special_tokens() if t <= self.LAST_SEQ_TOKEN]
        else:
            return [t for t in self._ttt_get_non_special_tokens() if t > self.LAST_SEQ_TOKEN]

    def _ttt_predict_logits(
        self, batch: torch.Tensor, start_indices: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """
        Note: Currently this implementation of _ttt_predict_logits restricts DPLM2BitTTT to
        customization via the sequence track. It can further be extended to support customization
        via the structure track by adding the structure logits to the output (the rest of the code
        should already support customization via the structure track). It should be done in
        some smart way because out["aatype_logits"] and out["struct_logits"] have different vocab
        sizes, and cannot be straightforwardly concatenated.
        """
        out = self(batch)  # [bs, seq_len] -> [bs, seq_len, vocab_size]
        logits = out["aatype_logits"]
        # Generate dummy logits for the structure track, these will be ignored by the loss function
        nan_logits = torch.full_like(logits, float("nan"), device=logits.device)
        return torch.cat([nan_logits, logits], dim=1)
