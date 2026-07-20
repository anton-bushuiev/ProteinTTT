import hashlib
from pathlib import Path
from Bio import SeqIO
from typing import Optional, Union
from proteinttt.utils.boltz1_mmseqs2 import run_mmseqs2


def process_msa_seq(seq: str, replace_inserstions: Optional[str] = None, delete_lowercase: bool = True) -> str:
    seq = seq.replace("\x00", "")
    if delete_lowercase:
        for char in seq:
            if char.islower():
                seq = seq.replace(char, "")

    seq = seq.upper()
    seq = seq.replace(".", "-")
    if replace_inserstions is not None:
        seq = seq.replace("-", replace_inserstions)
    return seq


def read_msa(pth: Path, replace_inserstions: Optional[str] = None, delete_lowercase: bool = True) -> list[str]:
    """Reads an .a2m MSA file and returns a list of sequences."""
    msa = [
        process_msa_seq(str(s.seq), replace_inserstions, delete_lowercase)
        for s in SeqIO.parse(pth, "fasta")
    ]
    return msa


class MSAServer:
    """
    Given a target directory, this class will automatically build MSA .a3m files for query sequences. If the .a3m was
    previously built, it will be read from the cache. Otherwise, it will be built from scratch using the
    Boltz-1/OpenFold code (https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/msa/mmseqs2.py).
    """

    def __init__(self, cache_dir: Union[Path, str]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, seq: str, seq_id: Optional[str] = None) -> Path:
        """
        Get a path to the .a3m file for a given sequence and optional sequence ID used to create a unique file name.
        If seq_id is not provided, it will be created from the first 10 characters of the sequence and the MD5 hash of
        the sequence.
        """
        if seq_id is None:
            seq_id = f"{seq[:10]}_{self._hash_seq(seq)}"
        if not self._exists_in_cache(seq_id):
            self._fetch_from_server(seq, seq_id)
        return self._read_from_cache(seq_id)

    def _fetch_from_server(self, seq: str, seq_id: str) -> Path:
        a3m_lines = run_mmseqs2(
            x=seq, prefix=self.cache_dir / f"mmseqs2_{seq_id}"
        )
        a3m_pth = self._seq_id_to_a3m_pth(seq_id)
        with open(a3m_pth, "w") as f:
            for line in a3m_lines:
                f.write(line)
        return a3m_pth

    def _exists_in_cache(self, seq_id: str) -> bool:
        return self._seq_id_to_a3m_pth(seq_id).exists()

    def _read_from_cache(self, seq_id: str) -> Path:
        return self._seq_id_to_a3m_pth(seq_id)

    def _seq_id_to_a3m_pth(self, seq_id: str) -> Path:
        return self.cache_dir / f"{seq_id}.a3m"

    def _hash_seq(self, seq: str) -> str:
        return hashlib.md5(seq.encode()).hexdigest()[:10]
