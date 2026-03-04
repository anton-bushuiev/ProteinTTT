<div align="center">

# ProteinTTT

[![arXiv badge](https://img.shields.io/badge/arXiv-2411.02109-b31b1b.svg)](https://arxiv.org/abs/2411.02109)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-ProteinTTT-blue)](https://huggingface.co/spaces/pimenova/ProteinTTT)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-ProteinTTT-blue)](https://colab.research.google.com/drive/1l_h7cw82SQpW9PvYzSQeYS4TXzS1QJ8o)

</div>

<p align="center">
  <img src="assets/folding_example.gif" width="600" alt="Example of TTT applied to protein folding">
</p>

ProteinTTT enables customizing protein language models to one protein at a time for enhanced performance on challenging targets.

## Demo on Hugging Face Spaces 🤗 and Google Colab

To try ESMFold+ProteinTTT in your browser (no installation), see the [Hugging Face Space](https://huggingface.co/spaces/pimenova/ProteinTTT) and the [Google Colab notebook](https://colab.research.google.com/drive/1l_h7cw82SQpW9PvYzSQeYS4TXzS1QJ8o).

![Hugging Face Space](assets/hf_space.png)

## Installation

To install locally, please first install the model you are planning to use with ProteinTTT (for example, if you want to use ESMFold+ProteinTTT, install ESMFold according to the [official guide](https://github.com/facebookresearch/esm?tab=readme-ov-file#quick-start-) first) and then install the `proteinttt` package itself:

```bash
git clone https://github.com/anton-bushuiev/ProteinTTT
cd ProteinTTT
pip install -e .
```

<details>
<summary><b>Developer installation and running tests</b></summary>

Install in editable mode with dev dependencies (includes `pytest`):

```bash
git clone https://github.com/anton-bushuiev/ProteinTTT
cd ProteinTTT
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -ra
```

</details>

## Usage

In the following example, we use ESMFold+ProteinTTT to predict the structure of a protein. Here, customizing ESMFold with ProteinTTT leads to structure prediction with twice higher pLDDT.

```python
import torch
import esm
import biotite.structure.io as bsio
from proteinttt.models.esmfold import ESMFoldTTT, DEFAULT_ESMFOLD_TTT_CFG

# Set your sequence
sequence = "GIHLGELGLLPSTVLAIGYFENLVNIICESLNMLPKLEVSGKEYKKFKFTIVIPKDLDANIKKRAKIYFKQKSLIEIEIPTSSRNYPIHIQFDENSTDDILHLYDMPTTIGGIDKAIEMFMRKGHIGKTDQQKLLEERELRNFKTTLENLIATDAFAKEMVEVIIEE"

# Load model
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

def predict_structure(model, sequence):
    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open("result.pdb", "w") as f:
        f.write(output)

    struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
    print('pLDDT:', struct.b_factor.mean())

predict_structure(model, sequence)
# pLDDT: 38.43025

# ============ ProteinTTT =============
ttt_cfg = DEFAULT_ESMFOLD_TTT_CFG
ttt_cfg.steps = 10  # This is how you can modify config
model = ESMFoldTTT.ttt_from_pretrained(model, ttt_cfg=ttt_cfg, esmfold_config=model.cfg)
model.ttt(sequence)
# =====================================

predict_structure(model, sequence)
# pLDDT: 78.69619

# Reset model to original state (after this model.ttt can be called again on another protein)
# ============== ProteinTTT ===========
model.ttt_reset()
# =====================================
```

The package currently implements ProteinTTT for the following protein language models:

| Model name       | Reference                                                                                  |
|------------------|-------------------------------------------------------------------------------------------|
| DPLM2 Bit-based  | [Hsieh et al., 2025](https://arxiv.org/abs/2504.11454)                          |
| ProSST           | [Li et al., 2024](https://www.biorxiv.org/content/10.1101/2024.04.15.589672v3)                          |
| SaProt             | [Su et al., 2023](https://www.biorxiv.org/content/10.1101/2023.10.01.560349v5)                          |
| ESMFold          | [Lin et al., 2023](https://doi.org/10.1126/science.ade2574)                          |
| ESM2             | [Lin et al., 2023](https://doi.org/10.1126/science.ade2574)                          |
| ProGen2          | [Nijkamp et al., 2023](https://www.cell.com/cell-systems/fulltext/S2405-4712(23)00272-7)         |
| MSA Transformer  | [Rao et al., 2021](https://doi.org/10.1101/2021.02.12.430858)                          |


See `notebooks/demo.ipynb` for more usage examples.

## Implementing ProteinTTT for your custom model

The abstract class `proteinttt.base.TTTModule` implements all core functionality of ProteinTTT, including batch construction, optimization, and logging. To implement ProteinTTT for your custom model, you can subclass `TTTModule` and implement several methods that specify your model's interface (the alphabet and logic for predicting logits). Optionally, you can also implement logic for evaluating the model during test-time training to select the optimal step. For an example implementation, see how ProteinTTT is implemented for ESMFold in `ProteinTTT/proteinttt/models/esmfold.py`. The submodule `ProteinTTT/proteinttt/models` contains other examples.

## References

If you use ProteinTTT in your research, please cite the following paper:

```bibtex
@article{bushuiev2025one,
  title={One protein is all you need},
  author={Bushuiev, Anton and Bushuiev, Roman and Pimenova, Olga and Zadorozhny, Nikola and Samusevich, Raman and Manaskova, Elisabet and Kim, Rachel Seongeun and St{\"a}rk, Hannes and Sedlar, Jiri and Steinegger, Martin and Tom{\'a}{\v{s}} and Sivic, Josef},
  journal={arXiv preprint arXiv:2411.02109},
  url={https://arxiv.org/abs/2411.02109},
  doi={10.48550/arXiv.2411.02109},
  year={2025}
}
```
