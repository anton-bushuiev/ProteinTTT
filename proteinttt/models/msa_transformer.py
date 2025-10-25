import typing as T
from pathlib import Path

import torch
import esm
from esm.model.msa_transformer import MSATransformer

from proteinttt.base import TTTModule, TTTConfig


DEFAULT_MSA_TRANSFORMER_TTT_CFG = TTTConfig(
    lr=1e-4,
    batch_size=1,
    ags=8,
    steps=30,
)


class MSATransformerTTT(TTTModule, MSATransformer):
    ttt_default_cfg = DEFAULT_MSA_TRANSFORMER_TTT_CFG

    def __init__(self, ttt_cfg: TTTConfig, **kwargs):
        MSATransformer.__init__(self, **kwargs)
        TTTModule.__init__(self, ttt_cfg=ttt_cfg)
        self.ttt_alphabet = esm.Alphabet.from_architecture("msa_transformer")
        self.ttt_batch_converter = self.ttt_alphabet.get_batch_converter()
    
    def _ttt_tokenize(self, seq: str, **kwargs) -> torch.Tensor:
        # Check that MSA is provided
        assert "msa" in kwargs, "MSA must be provided"
        msa_input = kwargs["msa"]

        if isinstance(msa_input, torch.Tensor):
            msa = msa_input
        elif isinstance(msa_input, (str, Path)):
            msa_data = esm.data.read_msa(str(msa_input))
            # Check that the first sequence in the MSA file matches the input seq
            msa_ref_seq_ungapped = msa_data[0][1].replace("-", "")
            assert (
                msa_ref_seq_ungapped == seq
            ), "First sequence in MSA file (ungapped) does not match input sequence"
            # Tokenize the MSA data
            _, _, msa = self.ttt_batch_converter(
                [msa_data]
            )  # [msa_data] makes it a batch of 1
        else:
            raise TypeError(
                "MSA must be a torch.Tensor or a file path (str or Path)"
            )

        assert isinstance(msa, torch.Tensor), "MSA must be a tensor"
        assert (
            msa.ndim == 3
        ), "MSA must be a 3D tensor with shape [bs, msa_len, seq_len]"
        assert msa.shape[0] == 1, "Only one MSA should be provided"

        # Check that first sequence in MSA tensor is the same as the target sequence
        _, _, seq_tokens = self.ttt_batch_converter([("seq", seq)])
        assert seq_tokens.shape[1] == msa.shape[2], (
            f"Tokenized seq length ({seq_tokens.shape[1]}) does not match "
            f"MSA sequence length ({msa.shape[2]})"
        )
        assert torch.all(
            seq_tokens[0, :] == msa[0, 0, :]
        ), "First sequence in MSA tensor must be the same as the input sequence"

        return msa  # [1, msa_len, seq_len]  

    def _ttt_get_frozen_modules(self) -> list[torch.nn.Module]:
        return [
            self.embed_tokens,
            self.embed_positions,
            self.emb_layer_norm_before,
            self.emb_layer_norm_after,
        ]

    def _ttt_mask_token(self, token: int) -> int:
        return self.ttt_alphabet.mask_idx

    def _ttt_get_all_tokens(self) -> list[int]:
        return [
            self.ttt_alphabet.tok_to_idx[t] for t in self.ttt_alphabet.all_toks
        ]

    def _ttt_get_non_special_tokens(self) -> list[int]:
        return [
            self.ttt_alphabet.tok_to_idx[t]
            for t in self.ttt_alphabet.standard_toks
            if t
            != "-"  # Exclude MSA gap token so that it is not masked or used for scoring
        ]

    def _ttt_get_padding_token(self) -> int:
        return self.ttt_alphabet.padding_idx

    def _ttt_token_to_str(self, token: int) -> str:
        return self.ttt_alphabet.all_toks[token]

    def _ttt_predict_logits(
        self, batch: torch.Tensor, start_indices: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        return self(batch)["logits"]  # [1, msa_len, seq_len, vocab_size]

    def _ttt_sample_batch(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get the reference sequence and the rest of MSA as homologous sequences
        x_ref = x[0, [0], :]
        x_hom = x[0, 1:, :]

        # Sample 1 mask for the reference sequence
        (
            batch_masked_ref,
            targets_ref,
            mask_ref,
            start_indices_ref,
        ) = super()._ttt_sample_batch(x_ref)
        batch_masked_ref = batch_masked_ref[[0], :]
        targets_ref = targets_ref[[0], :]
        mask_ref = mask_ref[[0], :]
        start_indices_ref = start_indices_ref[[0]]

        # Sample bs - 1 masks for random homologous sequences
        (
            batch_masked_hom,
            targets_hom,
            mask_hom,
            start_indices_hom,
        ) = super()._ttt_sample_batch(x_hom)
        batch_masked_hom = batch_masked_hom[1:, :]
        targets_hom = targets_hom[1:, :]
        mask_hom = mask_hom[1:, :]
        start_indices_hom = start_indices_hom[1:]

        # Concatenate the results
        batch_masked = torch.cat([batch_masked_ref, batch_masked_hom], dim=0)
        targets = torch.cat([targets_ref, targets_hom], dim=0)
        mask = torch.cat([mask_ref, mask_hom], dim=0)
        start_indices = torch.cat([start_indices_ref, start_indices_hom], dim=0)

        # Reshape back forward inputs. targets and mask are not reshaped because they will be used
        # for loss calculation next. start_indices is always in [bs] shape.
        batch_masked = batch_masked.unsqueeze(0)
        return batch_masked, targets, mask, start_indices

    def _ttt_cross_entropy_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        logits = logits[0, :, :, :]
        return super()._ttt_cross_entropy_loss(logits, targets, mask)

    def _ttt_score_seq(
        self, x: torch.Tensor, **kwargs
    ) -> tuple[list[torch.Tensor], float]:
        raise NotImplementedError(
            "Scoring for MSA Transformer is not implemented."
        )
        # TODO Extend_ttt_predict_logits to check the number of dimenstions. If the number is 2, then
        # a single sequence is passed and it should be concatenated with the rest of MSA.

        # Get only the first sequence from MSA and reshape to [bs=1, seq_len]
        # x = x[0, 0, :].unsqueeze(0)
        # return super()._ttt_score_seq(x, **kwargs)
