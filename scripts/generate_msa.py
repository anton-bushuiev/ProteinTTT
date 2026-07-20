"""Generate MSA .a3m files via MMseqs2 (Boltz-1 / OpenFold pipeline), with caching."""
from pathlib import Path
from typing import Union

from proteinttt.utils.msa import MSAServer


def generate_msa(seq: str, seq_id: str, cache_dir: Union[str, Path]) -> Path:
    """Generate (or fetch from cache) an .a3m MSA for the given sequence.

    The file is written to ``cache_dir / f"{seq_id}.a3m"`` and reused on
    subsequent calls.
    """
    server = MSAServer(cache_dir=cache_dir)
    return server.get(seq=seq, seq_id=seq_id)
