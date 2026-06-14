import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import interp1d

from .config_manager import ConfigManager


class AgeInterpolator:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.global_settings = config_manager.get_global_settings()
        self.binning_rules = config_manager.get_binning_rules()
        self.calibration_table = config_manager.get_age_calibration_table()
        self.stratigraphy_epochs = config_manager.get_stratigraphy_epochs()

    def build_common_age_grid(
        self,
        data_dict: Dict[str, pd.DataFrame],
        resolution: Optional[int] = None,
        age_min: Optional[float] = None,
        age_max: Optional[float] = None,
    ) -> np.ndarray:
        if resolution is None:
            resolution = self.global_settings.age_grid_resolution

        ages_all = []
        for df in data_dict.values():
            if "age" in df.columns:
                ages_all.extend(df["age"].dropna().tolist())

        if age_min is None:
            age_min = int(np.floor(min(ages_all) / resolution) * resolution) if ages_all else 0
        if age_max is None:
            age_max = int(np.ceil(max(ages_all) / resolution) * resolution) if ages_all else 500000

        return np.arange(age_min, age_max + resolution, resolution, dtype=float)

    def interpolate_to_age_grid(
        self,
        df: pd.DataFrame,
        age_grid: np.ndarray,
        method: Optional[str] = None,
    ) -> pd.DataFrame:
        if method is None:
            method = self.global_settings.interpolation_method

        if "age" not in df.columns:
            raise ValueError("数据帧缺少 'age' 列")

        df_sorted = df.sort_values("age").dropna(subset=["age"]).copy()

        resolution = age_grid[1] - age_grid[0] if len(age_grid) > 1 else self.global_settings.age_grid_resolution
        df_sorted["_grid_key"] = np.round(df_sorted["age"].values / resolution) * resolution

        agg_dict = {}
        for col in df_sorted.columns:
            if col == "_grid_key":
                continue
            if col == "age":
                agg_dict[col] = "mean"
            elif pd.api.types.is_numeric_dtype(df_sorted[col]):
                agg_dict[col] = "mean"
            else:
                agg_dict[col] = "first"

        df_sorted = df_sorted.groupby("_grid_key", as_index=False).agg(agg_dict)
        df_sorted = df_sorted.drop(columns=["_grid_key"])
        df_sorted = df_sorted.sort_values("age").reset_index(drop=True)

        if len(df_sorted) < 4 and method in ["cubic", "quadratic"]:
            method = "linear"

        proxy_cols = [
            c for c in df_sorted.columns
            if c in ["d18O", "dD", "dust", "d13C", "mg_ca", "sr_ca", "temperature", "precipitation"]
            and pd.api.types.is_numeric_dtype(df_sorted[c])
        ]

        result = pd.DataFrame({"age": age_grid})

        if len(df_sorted) == 0:
            for col in proxy_cols:
                result[col] = np.nan
            return result

        age_original = df_sorted["age"].values

        for col in proxy_cols:
            values = df_sorted[col].values
            mask_valid = ~np.isnan(values)

            if mask_valid.sum() < 2:
                result[col] = np.nan
                continue

            try:
                age_valid = age_original[mask_valid]
                val_valid = values[mask_valid]

                if method == "linear":
                    f = interp1d(
                        age_valid, val_valid, kind="linear",
                        bounds_error=False, fill_value=np.nan,
                    )
                elif method == "cubic":
                    if len(age_valid) >= 4:
                        f = interp1d(
                            age_valid, val_valid, kind="cubic",
                            bounds_error=False, fill_value=np.nan,
                        )
                    else:
                        f = interp1d(
                            age_valid, val_valid, kind="linear",
                            bounds_error=False, fill_value=np.nan,
                        )
                elif method == "quadratic":
                    if len(age_valid) >= 3:
                        f = interp1d(
                            age_valid, val_valid, kind="quadratic",
                            bounds_error=False, fill_value=np.nan,
                        )
                    else:
                        f = interp1d(
                            age_valid, val_valid, kind="linear",
                            bounds_error=False, fill_value=np.nan,
                        )
                elif method == "nearest":
                    f = interp1d(
                        age_valid, val_valid, kind="nearest",
                        bounds_error=False, fill_value=np.nan,
                    )
                else:
                    f = interp1d(
                        age_valid, val_valid, kind="linear",
                        bounds_error=False, fill_value=np.nan,
                    )

                result[col] = f(age_grid)
            except Exception:
                result[col] = np.nan

        result["source_id"] = df["source_id"].iloc[0] if "source_id" in df.columns else ""
        result["source_name"] = df["source_name"].iloc[0] if "source_name" in df.columns else ""
        result = self._assign_epoch(result)
        return result

    def bin_by_time(
        self,
        df: pd.DataFrame,
        bin_size: Optional[int] = None,
    ) -> pd.DataFrame:
        if bin_size is None:
            bin_size = self.binning_rules.get("time_bin_size_yr", 100)

        if "age" not in df.columns:
            raise ValueError("数据帧缺少 'age' 列")

        df = df.copy()
        df["age_bin"] = (df["age"] // bin_size) * bin_size

        agg_methods = self.binning_rules.get("aggregation_methods", {})
        agg_dict = {}
        for col in df.columns:
            if col in ["age", "age_bin"]:
                continue
            if col in agg_methods:
                agg_dict[col] = agg_methods[col]
            elif pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = "mean"
            else:
                agg_dict[col] = "first"

        result = df.groupby("age_bin", as_index=False).agg(agg_dict)
        result = result.rename(columns={"age_bin": "age"})
        return result

    def calibrate_age_scale(
        self,
        df: pd.DataFrame,
        tie_points: Optional[List[Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        if tie_points is None:
            tie_points = self._get_default_tie_points()

        if len(tie_points) < 2:
            return df

        df = df.copy()
        tie_points = sorted(tie_points, key=lambda x: x["original_age"])

        orig_ages = np.array([tp["original_age"] for tp in tie_points])
        calib_ages = np.array([tp["calibrated_age"] for tp in tie_points])

        f = interp1d(
            orig_ages, calib_ages, kind="linear",
            bounds_error=False, fill_value="extrapolate",
        )

        original_age = df["age"].values
        df["age_uncalibrated"] = original_age
        df["age"] = f(original_age)
        df["age_calibrated"] = True
        return df

    def _get_default_tie_points(self) -> List[Dict[str, float]]:
        if self.calibration_table.empty:
            return []
        tie_points = []
        for _, row in self.calibration_table.iterrows():
            tie_points.append({
                "original_age": row["calibrated_age_yrBP"],
                "calibrated_age": row["calibrated_age_yrBP"],
                "event": row["event_name"],
            })
        return tie_points

    def _assign_epoch(self, df: pd.DataFrame) -> pd.DataFrame:
        df["epoch"] = "Unknown"
        df["epoch_code"] = "UNK"
        for epoch in self.stratigraphy_epochs:
            mask = (df["age"] >= epoch["age_start"]) & (df["age"] < epoch["age_end"])
            df.loc[mask, "epoch"] = epoch["name"]
            df.loc[mask, "epoch_code"] = epoch["code"]
        return df

    def merge_proxies_to_single_frame(
        self,
        data_dict: Dict[str, pd.DataFrame],
        age_grid: np.ndarray,
    ) -> pd.DataFrame:
        frames = []
        for source_id, df in data_dict.items():
            interp_df = self.interpolate_to_age_grid(df, age_grid)
            rename_map = {}
            for col in interp_df.columns:
                if col in ["age", "epoch", "epoch_code", "source_id", "source_name"]:
                    continue
                rename_map[col] = f"{source_id}_{col}"
            interp_df = interp_df.rename(columns=rename_map)
            keep_cols = ["age"] + [c for c in interp_df.columns if c.startswith(f"{source_id}_")]
            frames.append(interp_df[keep_cols])

        merged = frames[0]
        for f in frames[1:]:
            merged = pd.merge(merged, f, on="age", how="outer")

        merged = merged.sort_values("age").reset_index(drop=True)
        merged = self._assign_epoch(merged)
        return merged

    def save_interpolated_data(self, df: pd.DataFrame, filename: str) -> str:
        out_dir = self.config.get_processed_data_dir()
        if not filename.endswith(".csv"):
            filename += ".csv"
        filepath = os.path.join(out_dir, filename)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        return filepath

    def process_all_sources(
        self,
        cleaned_data: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray]:
        age_grid = self.build_common_age_grid(cleaned_data)

        per_source_interp = {}
        for source_id, df in cleaned_data.items():
            interp = self.interpolate_to_age_grid(df, age_grid)
            binned = self.bin_by_time(interp)
            per_source_interp[source_id] = binned

        merged = self.merge_proxies_to_single_frame(cleaned_data, age_grid)
        merged_binned = self.bin_by_time(merged)
        self.save_interpolated_data(merged_binned, "merged_interpolated_timeseries")

        for source_id, df in per_source_interp.items():
            self.save_interpolated_data(df, f"interpolated_{source_id.lower()}")

        return merged_binned, per_source_interp, age_grid
