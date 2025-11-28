import torch
import typing as T
from tokenizers import Tokenizer

# TODO: explain installation in a comment
from .progen.progen2.models.progen.modeling_progen import ProGenForCausalLM

from proteinttt.base import TTTModule, TTTConfig

DEFAULT_PROGEN2_TTT_CFG = TTTConfig(
    lr=4e-4,
    batch_size=1,
    ags=4,
    steps=15,
    model_kind="autoregressive",
    loss_kind="unnormalized_cross_entropy",
    lora_rank=0,
    lora_alpha=0.0,
    lora_target_replace_module="ProGenAttention"
)


class ProGen2TTT(TTTModule, ProGenForCausalLM):
    """
    ProGen2TTT is a TTTModule that uses the ProGen2 model.
    """
    ttt_default_cfg = DEFAULT_PROGEN2_TTT_CFG

    def __init__(self, ttt_cfg: TTTConfig, tokenizer: Tokenizer, **kwargs):
        ProGenForCausalLM.__init__(self, **kwargs)
        TTTModule.__init__(self, ttt_cfg=ttt_cfg)
        self.ttt_tokenizer = tokenizer

    def _ttt_tokenize(self, seq: T.Optional[str], **kwargs) -> torch.Tensor:
        ids = torch.tensor(self.ttt_tokenizer.encode(seq).ids).to(self.device)
        ids = ids.unsqueeze(0)  # For consistency with bidirectional models where batch size is > 1
        return ids  # [1, seq_len]

    # NOTE: It works without freezing the embedding layers but maybe worths trying in the future for consistency
    # def _ttt_get_frozen_modules(self) -> list[torch.nn.Module]:
    #     return [self.esm.embeddings]

    def _ttt_mask_token(self, token: int) -> int:
        return ValueError("Mask token is not supposed to be used with ProGen2 (an autoregressive model)")

    def _ttt_get_padding_token(self) -> int:
        return 0  # 0 for '<|pad|>'

    def _ttt_token_to_str(self, token: int) -> str:
        return self.ttt_tokenizer.decode([token])

    def _ttt_get_all_tokens(self) -> list[int]:
        return list(self.ttt_tokenizer.get_vocab().values())

    def _ttt_get_non_special_tokens(self) -> list[int]:
        return [
            t
            for t in self._ttt_get_all_tokens()
            if t < 3  # 0, 1, 2 for '<|pad|>', '<|bos|>', '<|eos|>' respectively
        ]

    def _ttt_predict_logits(
        self, batch: torch.Tensor, start_indices: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        return self(batch[:, :-1]).logits  # [bs, seq_len - 1, vocab_size], seq_len - 1 for teacher forcing, labels are shifted by one position to the right in the base model class
