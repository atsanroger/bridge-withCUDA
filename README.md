This project is a previous version used for testing CUDA on Bridge++. 
Because the current version of Bridge++ lacks CUDA support, I separated this into an independent branch to coexist alongside RutticeQCD (https://github.com/atsanroger/RutticeQCD/tree/main), serving as a study of two different CUDA development philosophies.

The main research topics include:

1.AMG for MDWF 
(primarily implemented in RutticeQCD, with plans to consider porting it back to Bridge++)

2.Emulation of Float-Doublet to Accelerate Compute-Bound SpMV Solvers in Lattice QCD

The primary objective is to investigate whether Ozaki's float-doublet scheme can achieve performance acceleration on RTX and AI-focused GPUs.

Based on the idea from (https://arxiv.org/pdf/2510.13536) D,Mukunoki and K,Ozaki.

### Bug Reports
If you discover any bugs, numerical discrepancies, or unexpected behavior, please:
- email the author directly: `wlchenba@gmail.com`

## Acknowledgements

This project draws heavily on [Bridge++](https://bridge.kek.jp/Lattice-code/)
If you use this repo to publish, please cite Bridge++ as its developers ask of its own
users: J. Phys. Conf. Ser. 523 (2014) 012046 and J. Phys. Conf. Ser. 2207 (2022) 012053, and
its website, <https://bridge.kek.jp/Lattice-code/>.

## License

The code is GPL-3.0-or-later; see [LICENSE](LICENSE). The multigrid note in
`docs/mg_background/` (text, PDF, figures made here, data) is CC BY 4.0, also in the
earlier releases, except three figures reproduced from PoS LATTICE2024 (2025) 414, which
keep their CC BY-NC-ND 4.0; see [docs/mg_background/LICENSE](docs/mg_background/LICENSE).

