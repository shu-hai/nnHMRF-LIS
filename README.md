# FDRhmrf

This package is designed for the false discovery rate (FDR) controlling procedure proposed in [1]. This FDR procedure is based on the two-parameter Ising model (a classical hidden Markov random field) and the local index of significance (LIS; [2,3]), which aims to minimize the false nondiscovery rate (FNR) while controlling FDR at a given level especially for 3D spatial data, e.g., neuroimaging data.

## Usage
C++ version of the package is available under `cpu/` directory, and can be called from Matlab by MEX. Examples are provided, and please see [README.pdf](https://github.com/shu-hai/FDRhmrf/blob/master/README.pdf) for details.

GPU utilized Pytorch (built with Python 3.12.4, Pytorch 2.6.0) package is available under `gpu/` directory. The FDR procedure can be called as follows:
```bash
python gem.py --labelpath {optional groundtruth file if using for simulation}
              --datapath {input test statistics file}
              --savepath {directory path for saving results}
```

Please cite the article [1] for this package, which is available [here](https://deepblue.lib.umich.edu/handle/2027.42/113759).

## References
[1] Shu, H., Nan, B., and Koeppe, R. (2015). "Multiple Testing for Neuroimaging via Hidden Markov Random Field". Biometrics, 71, 741-750.
```
@article{shu2015multiple,
  title={Multiple testing for neuroimaging via hidden Markov random field},
  author={Shu, Hai and Nan, Bin and Koeppe, Robert},
  journal={Biometrics},
  volume={71},
  number={3},
  pages={741--750},
  year={2015},
  publisher={Wiley Online Library}
}
```

[2] Sun, W., and Cai. T.T. (2009). "Large‐scale multiple testing under dependence". Journal of the Royal Statistical Society, Series B, 71, 393-424.

[3] Wei, Z., Sun, W., Wang, K., and Hakonarson, H. (2009). "Multiple testing in genome-wide association studies via hidden
Markov models". Bioinformatics 25, 2802–2808.





