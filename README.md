# Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation

This is the official code for the Quartet II NVFP4 training paper [![arXiv](https://img.shields.io/badge/arXiv-2601.22813-b31b1b.svg)](https://arxiv.org/abs/2601.22813) 

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

Coming soon...

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
