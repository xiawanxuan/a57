import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats

from .config_manager import ConfigManager, DataSource


class StratigraphyDataCleaner:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.global_settings = config_manager.get_global_settings()
        self.outlier_rules = config_manager.get_outlier_detection_rules()
        self.quality_flags = config_manager.get_quality_flags()
        self.stratigraphy_epochs = config_manager.get_stratigraphy_epochs()

    def load_raw_data(self, source: DataSource) -> pd.DataFrame:
        if not os.path.exists(source.file_path):
            raise FileNotFoundError(f"数据文件不存在: {source.file_path}")

        if source.file_type.lower() == "csv":
            df = pd.read_csv(source.file_path, encoding="utf-8")
        elif source.file_type.lower() == "xlsx":
            df = pd.read_excel(source.file_path)
        else:
            raise ValueError(f"不支持的文件类型: {source.file_type}")

        df = self._rename_columns(df, source)
        df["source_id"] = source.id
        df["source_name"] = source.name
        return df

    def _rename_columns(self, df: pd.DataFrame, source: DataSource) -> pd.DataFrame:
        rename_map = {}
        for std_name, raw_name in source.columns.items():
            if raw_name in df.columns:
                rename_map[raw_name] = std_name
        df = df.rename(columns=rename_map)

        required = ["depth", "age"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"数据源 {source.id} 缺少必需列: {col}")
        return df

    def clean_dataset(
        self,
        df: pd.DataFrame,
        source: DataSource,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        report = {}
        original_count = len(df)
        report["original_rows"] = original_count

        df = self._handle_missing_values(df)
        report["rows_after_missing"] = len(df)

        df = self._ensure_numeric(df)
        df = self._sort_by_depth_age(df)

        df = self._assign_stratigraphic_unit(df)

        proxy_cols = [c for c in df.columns if c in ["d18O", "dD", "dust", "d13C", "mg_ca", "sr_ca"]]

        df["quality_flag"] = self.quality_flags.get("GOOD", 0)
        outlier_mask = pd.Series(False, index=df.index)
        for col in proxy_cols:
            if col in df.columns:
                col_outliers = self._detect_outliers(df, col)
                outlier_mask |= col_outliers
                report[f"outliers_{col}"] = int(col_outliers.sum())

        df.loc[outlier_mask, "quality_flag"] = self.quality_flags.get("OUTLIER", 2)
        df_clean = df[~outlier_mask].copy()
        report["rows_after_outlier"] = len(df_clean)
        report["outliers_total"] = int(outlier_mask.sum())

        df_clean = self._remove_duplicate_depth_age(df_clean)
        report["rows_after_dedup"] = len(df_clean)
        report["total_removed"] = original_count - len(df_clean)
        report["retention_rate"] = len(df_clean) / original_count if original_count > 0 else 0.0

        if source.d18o_to_temp_coef is not None and "d18O" in df_clean.columns:
            df_clean["temperature"] = (
                source.d18o_to_temp_coef * df_clean["d18O"] + source.d18o_to_temp_intercept
            )
            report["temperature_derived"] = True

        if source.d18o_to_precip_coef is not None and "d18O" in df_clean.columns:
            df_clean["precipitation"] = (
                source.d18o_to_precip_coef * df_clean["d18O"] + source.d18o_to_precip_intercept
            )
            report["precipitation_derived"] = True

        return df_clean, report

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df.dropna(subset=["depth", "age"])
        for col in numeric_cols:
            if col not in ["depth", "age"]:
                median_val = df[col].median()
                if not pd.isna(median_val):
                    df[col] = df[col].fillna(median_val)
        return df

    def _ensure_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [c for c in df.columns if c in ["depth", "age", "d18O", "dD", "dust", "d13C", "mg_ca", "sr_ca"]]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _sort_by_depth_age(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=["depth", "age"], ascending=[True, True]).reset_index(drop=True)
        return df

    def _assign_stratigraphic_unit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["epoch"] = "Unknown"
        df["epoch_code"] = "UNK"
        for epoch in self.stratigraphy_epochs:
            mask = (df["age"] >= epoch["age_start"]) & (df["age"] < epoch["age_end"])
            df.loc[mask, "epoch"] = epoch["name"]
            df.loc[mask, "epoch_code"] = epoch["code"]
        return df

    def _detect_outliers(self, df: pd.DataFrame, column: str) -> pd.Series:
        method = self.outlier_rules.get("method", "zscore")
        mask = pd.Series(False, index=df.index)

        if self.outlier_rules.get("by_stratigraphic_unit", True):
            for _, group in df.groupby("epoch", dropna=True):
                if len(group) < 10:
                    continue
                group_outliers = self._detect_outliers_in_series(group[column], method)
                mask.loc[group_outliers[group_outliers].index] = True
        else:
            mask = self._detect_outliers_in_series(df[column], method)

        return mask

    def _detect_outliers_in_series(self, series: pd.Series, method: str) -> pd.Series:
        series = series.dropna()
        if len(series) < 5:
            return pd.Series(False, index=series.index)

        if method == "zscore":
            threshold = self.outlier_rules.get("zscore_threshold", 3.0)
            z = np.abs(stats.zscore(series))
            return pd.Series(z > threshold, index=series.index)
        elif method == "iqr":
            multiplier = self.outlier_rules.get("iqr_multiplier", 1.5)
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            return (series < lower) | (series > upper)
        elif method == "rolling":
            window = self.outlier_rules.get("rolling_window", 50)
            rolling_mean = series.rolling(window=window, center=True, min_periods=5).mean()
            rolling_std = series.rolling(window=window, center=True, min_periods=5).std()
            threshold = self.outlier_rules.get("zscore_threshold", 3.0)
            z = np.abs((series - rolling_mean) / rolling_std)
            return z > threshold
        else:
            raise ValueError(f"未知的异常检测方法: {method}")

    def _remove_duplicate_depth_age(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        agg_dict = {c: "mean" for c in numeric_cols if c not in ["depth", "age"]}
        df = df.groupby(["depth", "age"], as_index=False).agg(agg_dict)
        return df

    def save_cleaned_data(self, df: pd.DataFrame, source_id: str) -> str:
        out_dir = self.config.get_processed_data_dir()
        filename = f"cleaned_{source_id.lower()}.csv"
        filepath = os.path.join(out_dir, filename)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        return filepath

    def process_all_sources(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
        results = {}
        reports = {}

        for source in self.config.get_all_sources(enabled_only=True):
            try:
                raw_df = self.load_raw_data(source)
                cleaned_df, report = self.clean_dataset(raw_df, source)
                filepath = self.save_cleaned_data(cleaned_df, source.id)
                report["output_file"] = filepath
                results[source.id] = cleaned_df
                reports[source.id] = report
            except Exception as e:
                reports[source.id] = {"error": str(e)}

        return results, reports
