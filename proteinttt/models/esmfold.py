import typing as T
from pathlib import Path

import torch
import esm
from esm.esmfold.v1.esmfold import ESMFold

from proteinttt.base import TTTModule, TTTConfig


DEFAULT_ESMFOLD_TTT_CFG = TTTConfig(
    lr=4e-4, batch_size=4, ags=4, steps=30, lora_rank=8, lora_alpha=32.0
)


class ESMFoldTTT(TTTModule, ESMFold):
    ttt_default_cfg = DEFAULT_ESMFOLD_TTT_CFG

    def __init__(self, ttt_cfg: TTTConfig, **kwargs):
        ESMFold.__init__(self, **kwargs)
        TTTModule.__init__(self, ttt_cfg=ttt_cfg)
        self.ttt_alphabet = esm.Alphabet.from_architecture(
            "ESM-1b"
        )  # ESM2 uses ESM-1b alphabet
        self.ttt_batch_converter = self.ttt_alphabet.get_batch_converter()

    def _ttt_tokenize(self, seq: str, **kwargs) -> torch.Tensor:
        _, _, x = self.ttt_batch_converter([(None, seq)])
        return x

    def _ttt_get_trainable_modules(self) -> list[torch.nn.Module]:
        return [self.esm]

    def _ttt_get_frozen_modules(self) -> list[torch.nn.Module]:
        return [self.esm.embed_tokens]

    def _ttt_mask_token(self, token: int) -> int:
        return self.ttt_alphabet.mask_idx

    def _ttt_get_padding_token(self) -> int:
        return self.ttt_alphabet.padding_idx

    def _ttt_token_to_str(self, token: int) -> str:
        return self.ttt_alphabet.all_toks[token]

    def _ttt_get_all_tokens(self) -> list[int]:
        return [
            self.ttt_alphabet.tok_to_idx[t] for t in self.ttt_alphabet.all_toks
        ]

    def _ttt_get_non_special_tokens(self) -> list[int]:
        return [
            self.ttt_alphabet.tok_to_idx[t]
            for t in self.ttt_alphabet.standard_toks
        ]

    def _ttt_predict_logits(
        self, batch: torch.Tensor, start_indices: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        return self.esm(batch)[
            "logits"
        ]  # [bs, seq_len] -> [bs, seq_len, vocab_size]

    def _ttt_eval_step(
        self,
        step: int,
        loss: torch.Tensor,
        perplexity: float,
        all_log_probs: torch.Tensor,
        seq: str,
        msa_pth: Path,
        **kwargs,
    ) -> tuple[dict, dict, T.Optional[float]]:
        # Predict structure
        with torch.no_grad():
            output = self.infer(seq, masking_pattern=None)
        
        pdb_str = output["pdb_str"][0]
        plddt = output["plddt"][0].mean().item()

        # Store predictions
        eval_step_preds = {"pdb": pdb_str}
        eval_step_metric_dict = {"plddt": plddt}
        confidence = plddt

        return eval_step_preds, eval_step_metric_dict, confidence
