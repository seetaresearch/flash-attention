# FlashAttention for Dragon

This repository extends [FlashAttention](https://github.com/Dao-AILab/flash-attention) and other Transformer operators for [Dragon](https://github.com/seetaresearch/dragon).

Following the design principle of Dragon, this repository devotes to unify the modeling of Transformers for NVIDIA GPUs, Apple Silicon processors, Cambricon MLUs and more AI accelerators.

## Installation

### Installing Package

Clone this repository to local disk and install:

```bash
cd flash-attention && mkdir build
cd build && cmake .. && make install -j $(nproc)
pip install ..
```

## License
[BSD 2-Clause license](LICENSE)

## Acknowledgement

We thank the repositories: [FlashAttention](https://github.com/Dao-AILab/flash-attention).
