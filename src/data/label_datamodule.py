# LabelDataModule for measurement data
import os
from typing import Any, Dict, Optional, Tuple

import torch
import pandas as pd
import numpy as np

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def downsample(df_sensor, target_res, interp=False):
    downsampled_list = []
    for tag, group in df_sensor.groupby("tag_id"):
        # Make datetime the index for resampling
        group = group.set_index("datetime")
        # Take first value in each bin (instantaneous)
        group_resampled = group.resample(target_res).first()

        if interp:
            # Interpolate missing values linearly
            group_resampled["value"] = group_resampled["value"].interpolate(
                method="linear"
            )
            group_resampled["label"] = (
                group_resampled["label"].ffill().bfill()
            )  # forward and backward fill labels
            group_resampled["sensor"] = group["sensor"].iloc[0]  # keep sensor type
            group_resampled["conceptdoi"] = group["conceptdoi"].iloc[0]
            group_resampled["scientific_name"] = group["scientific_name"].iloc[0]

        group_resampled["tag_id"] = tag  # keep tag_id as column

        # Check for NaNs
        if np.any(group_resampled["value"].isna()):
            print(
                f"Warning: NaN values present after downsampling for tag {tag}. Tag will be skipped."
            )
        else:
            downsampled_list.append(group_resampled.reset_index())
    return pd.concat(downsampled_list, ignore_index=True)


class MeasurementDataset(Dataset):
    """
    Dataset for measurement data. Expects a pre-processed DataFrame.
    """

    def __init__(
        self,
        data,
        window: int = 72,  # 6 hours if 5min resolution
    ):
        self.data = data.sort_values(["tag_id", "datetime"]).reset_index(drop=True)
        self.window = window

        # Store as numpy for speed
        self.select = np.where(data["select"])[0]
        self.X = data[["value_act", "value_pres"]].to_numpy(dtype=np.float32)
        self.y = data["flight"].to_numpy(dtype=np.float32)[:, np.newaxis]

    def __len__(self):
        return len(self.select)

    def __getitem__(self, idx1):
        # Get the actual index from the boolean array
        idx = self.select[idx1]

        # Window around idx
        start = max(0, idx - self.window)
        end = min(idx + self.window + 1, len(self.data))

        X = self.X[start:end]

        # Pad if needed
        pad_left = max(0, self.window - idx)
        pad_right = max(0, (idx + self.window + 1) - len(self.data))
        if pad_left > 0 or pad_right > 0:
            X = np.pad(X, ((pad_left, pad_right), (0, 0)), mode="edge")

        y = self.y[idx]
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class LabelDataModule(LightningDataModule):
    def balanced_subsample(self, df, max_samples, flight_ratio=0.4, random_state=42):
        # Get positive and negative indices
        pos = df[df["flight"]].copy()
        neg = df[~df["flight"]].copy()
        n_pos = int(max_samples * flight_ratio)
        n_neg = max_samples - n_pos
        # If not enough positives/negatives, take all
        pos_sample = pos.sample(n=min(n_pos, len(pos)), random_state=random_state)
        neg_sample = neg.sample(n=min(n_neg, len(neg)), random_state=random_state)
        df_sub = (
            pd.concat([pos_sample, neg_sample])
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )
        return df_sub

    def __init__(
        self,
        data_dir: str = "../data",
        act: bool = True,
        act_res: str = "5min",
        window: int = 72,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        split_samples: dict = None,
        flight_ratio: float = 0.2,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.data_dir = data_dir
        self.window = int(window)
        self.act_res = act_res
        self.act = act
        self.batch_size_per_device = batch_size

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Accept split_samples as dict or OmegaConf DictConfig
        if split_samples is None:
            split_samples = {"train": 1000, "val": 200, "test": 200}
        if not isinstance(split_samples, dict):
            split_samples = dict(split_samples)
        self.split_samples = split_samples
        self.flight_ratio = float(flight_ratio)
        self.flight_ratio = flight_ratio

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

    def read_data(self):
        import time

        dtype_dict = {
            "tag_id": str,
            "sensor": str,
            "datetime": str,
            "value": float,
            "label": str,
        }

        t0 = time.time()
        df = pd.read_csv(
            os.path.join(self.data_dir, "labels.csv"),
            na_values=["NaN"],  # treat "NaN" as NaN
            keep_default_na=False,  # do not treat default strings (like empty) as NaN
            dtype=dtype_dict,
            low_memory=False,
            engine="c",  # ensure fast C engine
        )
        log.info(f"pd.read_csv took {time.time() - t0:.2f} seconds")

        # Convert to datetime after reading
        df["datetime"] = pd.to_datetime(df["datetime"])

        t1 = time.time()
        pressure = downsample(df[df["sensor"] == "pressure"], self.act_res, interp=True)
        log.info(f"Downsampling pressure took {time.time() - t1:.2f} seconds")

        if self.act:
            t2 = time.time()
            activity = downsample(
                df[
                    (df["sensor"] == "activity")
                    & (df["tag_id"].isin(pressure["tag_id"].unique()))
                ],
                self.act_res,
                interp=True,
            )
            log.info(f"Downsampling activity took {time.time() - t2:.2f} seconds")
            # Keep overlapping tags
            tags = set(activity["tag_id"]) & set(pressure["tag_id"])
            activity = activity[activity["tag_id"].isin(tags)]
            pressure = pressure[pressure["tag_id"].isin(tags)]

            t3 = time.time()
            data = pd.merge_asof(
                activity.sort_values("datetime"),
                pressure.sort_values("datetime"),
                on="datetime",
                by=["tag_id", "scientific_name", "conceptdoi"],
                suffixes=("_act", "_pres"),
                direction="nearest",
            ).drop(columns=["sensor_act", "sensor_pres"])
            log.info(
                f"Merging activity and pressure took {time.time() - t3:.2f} seconds"
            )

            data["flight"] = data["label_act"] == "flight"
        else:
            log.info("Activity data not used; using only pressure data.")
            data = pressure
            data["flight"] = data["label"] == "flight"

        # Sort by tag_id and datetime
        data = data.sort_values(["tag_id", "datetime"]).reset_index(drop=True)

        return data

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if hasattr(self, "data") and self.data is not None:
            return  # Already loaded, skip reloading

        data = self.read_data()

        # Assign train/val/test splits by tag_id, stratified by conceptdoi (like year_group)
        data["tvt"] = None
        concept_grp = {
            concept: np.unique(data.loc[data["conceptdoi"] == concept, "tag_id"])
            for concept in data["conceptdoi"].unique()
        }
        for tag_list in concept_grp.values():
            np.random.shuffle(tag_list)

        ratios = np.array(
            [self.split_samples[k] for k in ["train", "val", "test"]], dtype=float
        )
        ratios /= ratios.sum()

        for concept, tag_ids in concept_grp.items():
            splits = (ratios.cumsum()[:-1] * len(tag_ids)).astype(int)
            train_ids, val_ids, test_ids = np.split(tag_ids, splits)
            for ids, split in zip(
                [train_ids, val_ids, test_ids], ["train", "val", "test"]
            ):
                data.loc[
                    (data["conceptdoi"] == concept) & (data["tag_id"].isin(ids)), "tvt"
                ] = split

        # Subsample each split
        data["select"] = False

        for split, n_samples in self.split_samples.items():
            df_split = data[data["tvt"] == split]
            n_flight = int(self.flight_ratio * n_samples)
            n_non_flight = n_samples - n_flight

            idx_flight = (
                df_split[df_split["flight"] == True]
                .sample(
                    n=min(n_flight, (df_split["flight"] == True).sum()), random_state=42
                )
                .index
            )
            idx_non_flight = (
                df_split[df_split["flight"] == False]
                .sample(
                    n=min(n_non_flight, (df_split["flight"] == False).sum()),
                    random_state=42,
                )
                .index
            )

            data.loc[idx_flight, "select"] = True
            data.loc[idx_non_flight, "select"] = True

        # Log number of selected samples
        for split in ["train", "val", "test"]:
            n_selected = ((data["tvt"] == split) & (data["select"])).sum()
            n_flight = (
                (data["tvt"] == split) & (data["select"]) & (data["flight"])
            ).sum()
            log.info(
                f"{split.capitalize()} selected: {n_selected} (flight: {n_flight})"
            )

        self.data = data

    def train_dataloader(self):
        data_train = MeasurementDataset(
            self.data[self.data["tvt"] == "train"], window=self.window
        )
        return DataLoader(
            dataset=data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=False,
        )

    def val_dataloader(self):
        data_val = MeasurementDataset(
            self.data[self.data["tvt"] == "val"], window=self.window
        )
        return DataLoader(
            dataset=data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=False,
        )

    def test_dataloader(self):
        data_test = MeasurementDataset(
            self.data[self.data["tvt"] == "test"], window=self.window
        )
        return DataLoader(
            dataset=data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage=None):
        pass


if __name__ == "__main__":
    _ = LabelDataModule()
