[![arXiv](https://img.shields.io/badge/arXiv-2610.29530-b31b1b.svg)](https://arxiv.org/abs/2610.29530)

# Loscar: Location-Scale RESIT

Loscar extends RESIT (regression with subsequent independence test, Peters et al. 2014) to location-scale noise models (LSNMs), using in-context quantile regression with TabPFN. This repository contains the `loscar` package, the data generation and the experiments.

## Project Structure

```bash
Loscar/
├── src/loscar/             # the `loscar` package (Loscar, Loscar_bivariate, etc.)
├── data/                   # data generation (dgp.py), dataset loaders and the datasets (not in git)
├── experiments/            # benchmark scripts
├── tutorial.py             # Loscar on a simulated graph and a cause-effect pair
├── environment.yaml        # conda environment for Loscar
├── setupR.R                # R packages for the SID metric
├── LICENSE.txt             # license of this repository
└── LICENSES/               # licenses of adapted code, data and TabPFN
```

## Installation & Quick-Start

The following contains instructions to install the `loscar` environment, as well as the packages required to run the experiments and compute the performance metrics. To test the environment and get an example for running Loscar, please have a look at the `tutorial.py` file. **Note** that for all code that uses **TabPFN**, the corresponding licenses provided by [Prior Labs GmbH](https://priorlabs.ai) need to be accepted.

### 1. Create the environment

The file [environment.yaml](environment.yaml) contains everything needed for the tutorial and the Loscar experiments, including the R tool chain for the SID metric.

```bash
$ conda env create -f environment.yaml
$ conda activate loscar-env
```

### 2. Install Loscar

From the repository root, install the `loscar` package:

```bash
$ pip install -e .
```

### 3. Install the R packages for SID

The SID metric is computed in R via `cdt`. Install the R packages `pcalg` and `SID` with:

```bash
$ Rscript setupR.R
```

Without R, the experiments still run; the SID columns are then reported as NaN. `cdt` prints warnings about missing R packages when imported, which can be ignored if you do not need the SID metric.

### 4. Run the tutorial

Run it from the repository root (it imports `data.dgp`). On the first run, TabPFN downloads its model weights and asks you to accept their license in the browser (see [TabPFN license](#tabpfn-license)).

```bash
$ python tutorial.py
```

## Datasets

The datasets are not part of the repository. Download them into `data/` as described below; `data/datasets.py` and `data/real_world.py` expect exactly these folder names.

- **Cause-effect pairs:** All pairs are taken from the `data` folder of the [skewd repository](https://github.com/causality-lab/skewd). These include datasets from Mooij et al. (2016), Goudet et al. (2018), Tagasovska et al. (2020), and Klippert and Marx (2026).
- **SACHS** (Sachs et al., 2005): observational data available at the [BCD-Nets repository](https://github.com/ermongroup/BCD-Nets).
- **LightTunnel** (Causal Chambers, Gamella et al., 2025): downloaded automatically to `data/causal_chambers/` on first use via the `causalchamber` package. The data is licensed under CC BY 4.0, see `LICENSES/causal-chambers-4.0.txt`.
- **Synthetic graphs:** All synthetic graphs are generated on the fly with `data/dgp.py`.

## Running the Experiments

The scripts to reproduce the experiments involving all Loscar and RESIT variants are provided in the `experiments` folder.

## License and Third Party Code

The code in this repository is licensed under the MIT License, see [LICENSE.txt](LICENSE.txt). Code and data adapted from other projects keep their original licenses, see [LICENSES/](LICENSES/) and the notes at the top of the respective files. In particular, this project builds on the following resources:

- [https://github.com/cdt15/lingam](https://github.com/cdt15/lingam) for the base RESIT algorithm and the HSIC implementation.
- [https://github.com/causality-lab/skewd](https://github.com/causality-lab/skewd) for the bivariate experiments pipeline.
- [https://github.com/xunzheng/notears](https://github.com/xunzheng/notears) as a basis for the synthetic multivariate experiments.

#### TabPFN license

Loscar uses [TabPFN](https://github.com/PriorLabs/TabPFN) by Prior Labs GmbH for in-context quantile regression. TabPFN is not included in this repository. It is installed as a dependency and its weights are downloaded on first use.

- **TabPFN package (code):** Apache License 2.0, see [LICENSES/tabpfn-Apache-2.0.txt](LICENSES/tabpfn-Apache-2.0.txt).
- **TabPFN-3 weights** (the default, `tabpfn-v3-regressor-v3_default.ckpt`): TABPFN-3 Non-Commercial License, see [LICENSES/tabpfn-3-weights-license.txt](LICENSES/tabpfn-3-weights-license.txt) or the [original](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE).
- **TabPFN-3.5 weights** (optional, `--model-version v35`): TABPFN-3.5 Non-Commercial License, see [LICENSES/tabpfn-3.5-weights-license.txt](LICENSES/tabpfn-3.5-weights-license.txt) or the [original](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE).

The weight licenses allow only non-commercial use (e.g. academic research and benchmarking); this also applies to their outputs. Every TabPFN-based method in this repository (Loscar, Loscar-GL, RESIT-TabPFN, `Loscar_bivariate`) is therefore restricted to non-commercial use, unless you obtain a commercial license from Prior Labs. Loscar-FGLS and RESIT-spline do not use TabPFN.

## Citation

```bibtex
@article{klippert:26:loscar,
  author = {
    Klippert, Daniel and
    Huynh, Quang Nhat and
    Marx, Alexander
  },
  title = {Residual-Based Causal Discovery via Pretrained In-Context Quantile Regression},
  year = {2026}
}
```

## References

- Gamella, Peters, and Bühlmann, P. *Causal chambers as a real-world physical testbed for ai methodology*. Nature Machine Intelligence, 2025
- Goudet et al. *Learning functional causal models with generative neural networks*. Explainable and Interpretable Models in Computer Vision and Machine Learning, 2018
- Klippert and Marx. *Skewness-robust causal discovery in location-scale noise models*. ICML, 2026
- Mooij et al. *Distinguishing cause from effect using observational data: Methods and benchmarks*. JMLR, 2016
- Peters et al. *Causal discovery with continuous additive noise models*. JMLR, 2014
Sachs et al. *Causal protein-signaling networks derived from multiparameter single-cell data*. Science, 2005
- Tagasovska, Chavez-Demoulin and Vatter. *Distinguishing cause from effect using quantiles: Bivariate quantile causal discovery*. ICML, 2020
