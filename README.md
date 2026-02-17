# Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation

This is the official code for the Quartet II NVFP4 training paper [![arXiv](https://img.shields.io/badge/arXiv-2601.22813-b31b1b.svg)](https://arxiv.org/abs/2601.22813) 

<img width="1286" height="521" alt="image" src="https://github.com/user-attachments/assets/2925e164-9998-4f43-9b6e-b806b8b5964b" />


## Quickstart 

Create a conda environment and install dependencies (we recommend Python 3.11):

```bash
conda create -n env python=3.11
conda activate env
```

```bash
pip install -r requirements.txt
```

Reproduce Quartet II sweeps in SLURM:
```bash
cd scripts
sbatch quartetv2_sweep.sh
```

Inspect the scheme implementation at:
```
[quartet_2.py](./src/models/quantization/schemes/quartet_2.py)
```

## NVFP4 Kernels

We provide the kernels tuned for RTX 5090 (`sm120a`) and B200 (`sm100`) in `./kernels`.

The kernels were tested with **CUDA 13.0**, **PyTorch 2.10** and **Python 3.11**. Install them with

```bash
cd kernels
pip install --no-build-isolation .
```

You can then use the provided drop-in NVFP4 `nn.Linear` replacement as follows:
```python
from quartet2.linear import Quartet_II_linear

linear = Quartet_II_linear(
    in_dim,
    out_dim,
    device="cuda",
    dtype=torch.bfloat16,
    four_over_six=True, # Enables/Disables 4/6 on the forward pass. On by default
)
...
```

You can further benchmark the kernels agains BF16, FP8 and [Quartet](https://arxiv.org/abs/2505.14669) with

```bash
cd test
python bench_linear.py
```


## Cite This Work

```
@misc{panferov2026quartetiiaccuratellm,
      title={Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation}, 
      author={Andrei Panferov and Erik Schultheis and Soroush Tabesh and Dan Alistarh},
      year={2026},
      eprint={2601.22813},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.22813}, 
}
```
