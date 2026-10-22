"""Dataset classes for bivariate causal discovery.

This file has been adapted from https://github.com/aleximmer/loci
and from https://github.com/causality-lab/skewd
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Dataset files are expected in data/, next to this file.
DATA_DIR = Path(__file__).resolve().parent

class CausalDataset:
    """Skew and ANLSMN parent."""

    pass

class skewANsLSs(CausalDataset):
    """Skew dataset generator."""

    n_datasets = 100

    def __init__(
        self,
        dataset_type,  # 'ANs' or 'LSs'
        distribution,  # 'AGN' or 'SN'
        skewness_level,  # 'extreme', 'strong', or 'med'
        pair_id=None,
        path=DATA_DIR,
        double=False,
        preprocessor=None,
    ):
        assert dataset_type in ["ANs", "LSs"]
        assert distribution in ["AGN", "SN"]
        if distribution == "AGN":
            assert skewness_level in ["med", "strong", "extreme"]
        elif distribution == "SN":
            assert skewness_level in ["med", "strong"]
        assert pair_id in list(range(1, self.n_datasets + 1))

        # Construct folder name
        folder_parts = []
        folder_parts.append("pure")
        folder_parts.append(dataset_type)
        folder_parts.append(distribution)
        folder_parts.append(skewness_level)
        inner_folder_name = "_".join(folder_parts)

        directory = Path(path) / self.folder_name / inner_folder_name
        df = pd.read_csv(directory / f"pair_{pair_id}.txt", delimiter=",")
        ground_truth = (
            pd.read_csv(directory / "pairs_gt.txt", header=None)
            .iloc[pair_id - 1]
            .values[0]
        )

        x = np.array(df.iloc[:, 1].values).reshape(-1, 1)
        y = np.array(df.iloc[:, 2].values).reshape(-1, 1)
        if ground_truth == 1:
            cause, effect = x, y
        elif ground_truth == 0:
            cause, effect = y, x
        else:
            raise ValueError("Invalid ground truth label")

        if preprocessor is not None:
            cause = preprocessor.fit_transform(cause)
            effect = preprocessor.fit_transform(effect)

        cause, effect = torch.from_numpy(cause), torch.from_numpy(effect)
        if double:
            cause, effect = cause.double(), effect.double()
        else:
            cause, effect = cause.float(), effect.float()

        self.cause = cause
        self.effect = effect

    @property
    def folder_name(self):
        return "skew"


class ANLSMN(CausalDataset):
    "ANSLMN dataset generator."
    n_datasets = 100

    def __init__(
        self, pair_id=None, path=DATA_DIR, double=False, preprocessor=None
    ):
        assert pair_id in list(range(1, self.n_datasets + 1))
        directory = Path(path) / self.folder_name
        df = pd.read_csv(directory / f"pair_{pair_id}.txt", delimiter=",")
        ground_truth = (
            pd.read_csv(directory / "pairs_gt.txt", header=None)
            .iloc[pair_id - 1]
            .values[0]
        )
        if ground_truth == 1:  # true order
            cause, effect = df.iloc[:, 1:2].values, df.iloc[:, 2:3].values
        elif ground_truth == 0:  # wrong order
            cause, effect = df.iloc[:, 2:3].values, df.iloc[:, 1:2].values
        else:
            raise ValueError()

        if preprocessor is not None:
            cause = preprocessor.fit_transform(cause)
            effect = preprocessor.fit_transform(effect)

        cause, effect = torch.from_numpy(cause), torch.from_numpy(effect)
        if double:  # default double
            cause, effect = cause.double(), effect.double()
        else:
            cause, effect = cause.float(), effect.float()
        self.cause, self.effect = cause, effect

    @property
    def folder_name(self):
        return Path("ANLSMN_pairs")


class AN(ANLSMN):
    """Read AN dataset."""

    @property
    def folder_name(self):
        return super().folder_name / "AN"


class ANs(ANLSMN):
    """Read ANs dataset."""

    @property
    def folder_name(self):
        return super().folder_name / "AN-s"


class LS(ANLSMN):
    """Read LS dataset."""

    @property
    def folder_name(self):
        return super().folder_name / "LS"


class LSs(ANLSMN):
    """Read LSs dataset."""

    @property
    def folder_name(self):
        return super().folder_name / "LS-s"


class MNU(ANLSMN):
    """Read MNU dataset."""

    @property
    def folder_name(self):
        return super().folder_name / "MN-U"


class Tuebingen(CausalDataset):
    """Tuebingen datasets class."""

    n_datasets = 108

    def __init__(
        self, pair_id=None, path=DATA_DIR, double=False, preprocessor=None
    ):
        assert pair_id in list(range(1, self.n_datasets + 1))
        directory = Path(path) / self.folder_name
        df = pd.read_csv(
            directory / f"pair{pair_id:04d}.txt",
            sep=r'\s+',
            header=None,
        )
        # TODO: to use weight need to cast weight column to float..
        meta = pd.read_csv(
            directory / "pairmeta.txt",
            sep=r'\s+',
            header=None,
            names=[
                "id",
                "cause_start",
                "cause_end",
                "effect_start",
                "effect_end",
                "weight",
            ],
            index_col=0,
        ).astype(int)
        cause = df.iloc[
            :,
            meta.loc[pair_id, "cause_start"]
            - 1 : meta.loc[pair_id, "cause_end"],
        ].values
        effect = df.iloc[
            :,
            meta.loc[pair_id, "effect_start"]
            - 1 : meta.loc[pair_id, "effect_end"],
        ].values

        if preprocessor is not None:
            cause = preprocessor.fit_transform(cause)
            effect = preprocessor.fit_transform(effect)

        cause, effect = torch.from_numpy(cause), torch.from_numpy(effect)
        if double:  # default double
            cause, effect = cause.double(), effect.double()
        else:
            cause, effect = cause.float(), effect.float()

        self.cause, self.effect = cause, effect

    @property
    def folder_name(self):
        return "Tuebingen"


class BenchmarkSimulated(Tuebingen):
    """BenchmarkSimulated class."""

    n_datasets = 100

    @property
    def folder_name(self):
        return "Benchmark_simulated"


class SIM(BenchmarkSimulated):
    """SIM class."""

    @property
    def folder_name(self):
        return Path(super().folder_name) / "SIM"


class SIMc(BenchmarkSimulated):
    """SIMc class."""

    @property
    def folder_name(self):
        return Path(super().folder_name) / "SIM-c"


class SIMG(BenchmarkSimulated):
    """SIMG class."""

    @property
    def folder_name(self):
        return Path(super().folder_name) / "SIM-G"


class SIMln(BenchmarkSimulated):
    """SIMln class."""

    @property
    def folder_name(self):
        return Path(super().folder_name) / "SIM-ln"


class Dataverse(CausalDataset):
    """Dataverse class."""

    n_datasets = 300

    def __init__(
        self, pair_id=None, path=DATA_DIR, double=False, preprocessor=None
    ):
        assert pair_id in list(range(1, self.n_datasets + 1))
        directory = Path(path) / self.folder_name / self.file_name
        pairs = pd.read_csv(f"{directory}_pairs.csv")
        targets = pd.read_csv(f"{directory}_targets.csv")
        target = targets.loc[pair_id - 1, "Target"]
        if target == 1:
            cause = self.to_numpy(pairs.loc[pair_id - 1, "A"])
            effect = self.to_numpy(pairs.loc[pair_id - 1, "B"])
        elif target == -1:
            cause = self.to_numpy(pairs.loc[pair_id - 1, "B"])
            effect = self.to_numpy(pairs.loc[pair_id - 1, "A"])
        else:
            raise ValueError("Invalid target:", target)

        if preprocessor is not None:
            cause = preprocessor.fit_transform(cause)
            effect = preprocessor.fit_transform(effect)

        cause, effect = torch.from_numpy(cause), torch.from_numpy(effect)
        if double:  # default double
            cause, effect = cause.double(), effect.double()
        else:
            cause, effect = cause.float(), effect.float()
        self.cause, self.effect = cause, effect

    @staticmethod
    def to_numpy(data_string):
        return np.array(
            [float(e) for e in data_string.strip().split(" ")]
        ).reshape(-1, 1)

    @property
    def folder_name(self):
        return "Dataverse_pairs"

    @property
    def file_name(self):
        raise NotImplementedError()


class Cha(Dataverse):
    """CE-Cha class."""

    @property
    def file_name(self):
        return "CE-Cha"


class Multi(Dataverse):
    """CE-Multi class."""

    @property
    def file_name(self):
        return "CE-Multi"


class Net(Dataverse):
    """CE-Net class."""

    @property
    def file_name(self):
        return "CE-Net"
