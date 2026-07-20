import typing as T
from pathlib import Path
import tempfile

import torch
import esm
from esm.esmfold.v1.esmfold import ESMFold

from proteinttt.base import TTTModule, TTTConfig
from proteinttt.utils.structure import calculate_tm_score, lddt_score



DEFAULT_ESMFOLD_TTT_CFG = TTTConfig(
    lr=4e-4, batch_size=4, ags=4, steps=30, lora_rank=8, lora_alpha=32.0
)

GRAD_CLIP_ESMFOLD_TTT_CFG = TTTConfig(
    lr=0.002, batch_size=4, ags=4, steps=30, lora_rank=128, lora_alpha=256.0, gradient_clip=True, gradient_clip_max_norm=1.0
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
        # Reusable temp file path to avoid creating new temp files each eval step
        self._ttt_temp_pdb_path = None

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

    def ttt_reset(self) -> None:
        """Reset model and cleanup temporary files."""
        super().ttt_reset()
        # Clean up temporary PDB file if it exists
        if self._ttt_temp_pdb_path is not None and self._ttt_temp_pdb_path.exists():
            self._ttt_temp_pdb_path.unlink()
            self._ttt_temp_pdb_path = None

    def _ttt_eval_step(
        self,
        step: int,
        loss: torch.Tensor,
        perplexity: float,
        all_log_probs: torch.Tensor,
        seq: str,
        msa_pth: Path,
        correct_pdb_path: T.Optional[Path] = None,
        **kwargs,
    ) -> T.Tuple[dict, dict, T.Optional[float]]:
        # Predict structure
        with torch.no_grad():
            try:
                output = self.infer(seq, masking_pattern=None)
            except IndexError:
                # compute_tm in openfold crashes when the model produces NaN
                # logits (NaN != NaN so .nonzero() returns empty tensor).
                # Return sentinel values so TTT can continue.
                eval_step_preds = {"pdb": None}
                eval_step_metric_dict = {
                    "plddt": 0.0,
                    "tm_score": None,
                    "lddt": None,
                }
                return eval_step_preds, eval_step_metric_dict, 0.0

        pdb_str = self.output_to_pdb(output)
        plddt = output["mean_plddt"].item()

        tm_score = None
        lddt = None
        if correct_pdb_path is not None:
            if self._ttt_temp_pdb_path is None:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pdb') as tmp_file:
                    self._ttt_temp_pdb_path = Path(tmp_file.name)
            
            pdb_str_to_write = pdb_str[0] if isinstance(pdb_str, list) else pdb_str
            with open(self._ttt_temp_pdb_path, 'w', buffering=8192) as f:
                f.write(pdb_str_to_write)
            
            if self.ttt_cfg.tmalign_path is not None:
                tm_score = calculate_tm_score(self._ttt_temp_pdb_path, correct_pdb_path)
            lddt = lddt_score(correct_pdb_path, self._ttt_temp_pdb_path)

        eval_step_preds = {"pdb": pdb_str}
        eval_step_metric_dict = {
            "plddt": plddt,
            "tm_score": tm_score,
            "lddt": lddt,
        }
        confidence = plddt

        return eval_step_preds, eval_step_metric_dict, confidence
