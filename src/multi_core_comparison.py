import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats

from .config_manager import ConfigManager, DataSource


class MultiCoreComparator:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.global_settings = config_manager.get_global_settings()
        self.epochs = config_manager.get_stratigraphy_epochs()
        self.ice_core_sources = config_manager.get_ice_core_sources(enabled_only=True)

    def build_comparison_dataset(
        self,
        per_source_interp: Dict[str, pd.DataFrame],
        target_proxy: str = "d18O",
    ) -> pd.DataFrame:
        core_data = {}
        for source_id, df in per_source_interp.items():
            if target_proxy in df.columns:
                core_data[source_id] = df[["age", target_proxy]].copy()
                core_data[source_id] = core_data[source_id].rename(columns={target_proxy: source_id})

        if not core_data:
            return pd.DataFrame()

        merged = None
        for source_id, df in core_data.items():
            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on="age", how="outer")

        merged = merged.sort_values("age").reset_index(drop=True)
        merged = self._assign_epoch(merged)
        return merged

    def compute_pairwise_differences(
        self,
        comparison_df: pd.DataFrame,
    ) -> pd.DataFrame:
        core_cols = [c for c in comparison_df.columns if c not in ["age", "epoch", "epoch_code"]]
        if len(core_cols) < 2:
            return pd.DataFrame()

        diff_df = pd.DataFrame({"age": comparison_df["age"].values})

        for i in range(len(core_cols)):
            for j in range(i + 1, len(core_cols)):
                col_a = core_cols[i]
                col_b = core_cols[j]
                diff_name = f"delta_{col_a}_minus_{col_b}"
                diff_df[diff_name] = comparison_df[col_a] - comparison_df[col_b]

        diff_df = self._assign_epoch(diff_df)
        return diff_df

    def compute_cross_core_correlation(
        self,
        comparison_df: pd.DataFrame,
        method: str = "pearson",
    ) -> pd.DataFrame:
        core_cols = [c for c in comparison_df.columns if c not in ["age", "epoch", "epoch_code"]]
        if len(core_cols) < 2:
            return pd.DataFrame(index=core_cols, columns=core_cols)

        n = len(core_cols)
        corr_mat = pd.DataFrame(np.nan, index=core_cols, columns=core_cols)
        p_mat = pd.DataFrame(np.nan, index=core_cols, columns=core_cols)

        for i in range(n):
            for j in range(i, n):
                col_a = core_cols[i]
                col_b = core_cols[j]
                clean = comparison_df[[col_a, col_b]].dropna()
                if len(clean) < 2:
                    continue
                if method == "pearson":
                    corr, p_val = stats.pearsonr(clean[col_a], clean[col_b])
                elif method == "spearman":
                    corr, p_val = stats.spearmanr(clean[col_a], clean[col_b])
                else:
                    corr, p_val = stats.pearsonr(clean[col_a], clean[col_b])
                corr_mat.loc[col_a, col_b] = corr
                corr_mat.loc[col_b, col_a] = corr
                p_mat.loc[col_a, col_b] = p_val
                p_mat.loc[col_b, col_a] = p_val

        return corr_mat

    def epoch_wise_comparison(
        self,
        comparison_df: pd.DataFrame,
        target_proxy: str = "d18O",
    ) -> pd.DataFrame:
        core_cols = [c for c in comparison_df.columns if c not in ["age", "epoch", "epoch_code"]]
        if "epoch" not in comparison_df.columns or len(core_cols) < 1:
            return pd.DataFrame()

        rows = []
        for epoch_name, group in comparison_df.groupby("epoch", dropna=True):
            for col in core_cols:
                series = group[col].dropna()
                if len(series) == 0:
                    continue
                rows.append({
                    "epoch": epoch_name,
                    "core_id": col,
                    "n_points": len(series),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                })

        return pd.DataFrame(rows)

    def compute_spatial_gradient(
        self,
        comparison_df: pd.DataFrame,
        source_meta: Dict[str, Dict[str, float]],
        target_proxy: str = "temperature",
    ) -> pd.DataFrame:
        core_cols = [c for c in comparison_df.columns if c not in ["age", "epoch", "epoch_code"]]
        if len(core_cols) < 2:
            return pd.DataFrame()

        gradient_df = pd.DataFrame({"age": comparison_df["age"].values})
        valid_cores = [c for c in core_cols if c in source_meta]

        if len(valid_cores) < 2:
            return gradient_df

        latitudes = [source_meta[c]["latitude"] for c in valid_cores]
        sorted_idx = np.argsort(latitudes)
        sorted_cores = [valid_cores[i] for i in sorted_idx]
        sorted_lats = [latitudes[i] for i in sorted_idx]

        for i in range(len(sorted_cores) - 1):
            core_a = sorted_cores[i]
            core_b = sorted_cores[i + 1]
            lat_diff = sorted_lats[i + 1] - sorted_lats[i]
            if lat_diff == 0:
                continue
            col_name = f"gradient_{core_a}_to_{core_b}_per_degree"
            gradient_df[col_name] = (
                (comparison_df[core_b] - comparison_df[core_a]) / lat_diff
            )

        gradient_df = self._assign_epoch(gradient_df)
        return gradient_df

    def compute_rolling_correlation(
        self,
        comparison_df: pd.DataFrame,
        col_a: str,
        col_b: str,
        window_size: int = 2000,
        step_size: int = 500,
        method: str = "pearson",
    ) -> pd.DataFrame:
        df_clean = comparison_df[["age", col_a, col_b]].dropna().sort_values("age")
        if len(df_clean) < 10:
            return pd.DataFrame()

        ages = df_clean["age"].values
        vals_a = df_clean[col_a].values
        vals_b = df_clean[col_b].values

        results = []
        min_age = ages.min()
        max_age = ages.max()
        current_age = min_age

        while current_age + window_size <= max_age:
            mask = (ages >= current_age) & (ages < current_age + window_size)
            window_a = vals_a[mask]
            window_b = vals_b[mask]

            if len(window_a) >= 10:
                if method == "pearson":
                    corr, p_val = stats.pearsonr(window_a, window_b)
                elif method == "spearman":
                    corr, p_val = stats.spearmanr(window_a, window_b)
                else:
                    corr, p_val = stats.pearsonr(window_a, window_b)

                results.append({
                    "age_center": current_age + window_size / 2,
                    "age_start": current_age,
                    "age_end": current_age + window_size,
                    "correlation": float(corr),
                    "p_value": float(p_val),
                    "n_points": int(mask.sum()),
                })

            current_age += step_size

        return pd.DataFrame(results)

    def anomaly_analysis(
        self,
        comparison_df: pd.DataFrame,
        reference_epoch: str = "全新世",
        reference_age_min: float = 0,
        reference_age_max: float = 1000,
    ) -> pd.DataFrame:
        core_cols = [c for c in comparison_df.columns if c not in ["age", "epoch", "epoch_code"]]
        if len(core_cols) == 0:
            return pd.DataFrame()

        ref_mask = (comparison_df["age"] >= reference_age_min) & (comparison_df["age"] < reference_age_max)
        ref_data = comparison_df[ref_mask]

        anomaly_df = pd.DataFrame({"age": comparison_df["age"].values})

        for col in core_cols:
            ref_mean = ref_data[col].mean()
            if pd.isna(ref_mean):
                continue
            anomaly_df[f"{col}_anomaly"] = comparison_df[col] - ref_mean

        anomaly_df = self._assign_epoch(anomaly_df)
        return anomaly_df

    def _assign_epoch(self, df: pd.DataFrame) -> pd.DataFrame:
        df["epoch"] = "Unknown"
        df["epoch_code"] = "UNK"
        for epoch in self.epochs:
            mask = (df["age"] >= epoch["age_start"]) & (df["age"] < epoch["age_end"])
            df.loc[mask, "epoch"] = epoch["name"]
            df.loc[mask, "epoch_code"] = epoch["code"]
        return df

    def save_comparison_data(
        self,
        df: pd.DataFrame,
        filename: str,
    ) -> str:
        out_dir = self.config.get_processed_data_dir()
        if not filename.endswith(".csv"):
            filename += ".csv"
        filepath = os.path.join(out_dir, filename)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        return filepath

    def run_full_comparison(
        self,
        per_source_interp: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        results = {}

        comparison_d18o = self.build_comparison_dataset(per_source_interp, target_proxy="d18O")
        if not comparison_d18o.empty:
            results["comparison_d18O"] = comparison_d18o
            self.save_comparison_data(comparison_d18o, "multicore_comparison_d18O")

            diff_df = self.compute_pairwise_differences(comparison_d18o)
            if not diff_df.empty:
                results["pairwise_differences"] = diff_df
                self.save_comparison_data(diff_df, "multicore_pairwise_differences")

            corr_pearson = self.compute_cross_core_correlation(comparison_d18o, method="pearson")
            if not corr_pearson.empty:
                results["correlation_matrix_pearson"] = corr_pearson
                self.save_comparison_data(corr_pearson, "multicore_correlation_pearson")

            corr_spearman = self.compute_cross_core_correlation(comparison_d18o, method="spearman")
            if not corr_spearman.empty:
                results["correlation_matrix_spearman"] = corr_spearman
                self.save_comparison_data(corr_spearman, "multicore_correlation_spearman")

            epoch_stats = self.epoch_wise_comparison(comparison_d18o, target_proxy="d18O")
            if not epoch_stats.empty:
                results["epoch_wise_comparison"] = epoch_stats
                self.save_comparison_data(epoch_stats, "multicore_epoch_wise_stats")

        comparison_temp = self.build_comparison_dataset(per_source_interp, target_proxy="temperature")
        if not comparison_temp.empty:
            results["comparison_temperature"] = comparison_temp
            self.save_comparison_data(comparison_temp, "multicore_comparison_temperature")

            anomaly_df = self.anomaly_analysis(comparison_temp)
            if not anomaly_df.empty:
                results["temperature_anomalies"] = anomaly_df
                self.save_comparison_data(anomaly_df, "multicore_temperature_anomalies")

        return results
