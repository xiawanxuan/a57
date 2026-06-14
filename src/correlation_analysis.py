import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats

from .config_manager import ConfigManager


class CorrelationAnalyzer:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.settings = config_manager.get_correlation_settings()
        self.global_settings = config_manager.get_global_settings()

    def pairwise_correlation(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        if methods is None:
            methods = self.settings.methods

        results = {}
        df_clean = df[[col_a, col_b]].dropna()
        n = len(df_clean)

        if n < self.settings.min_overlap_points:
            return {
                method: {
                    "correlation": np.nan,
                    "p_value": np.nan,
                    "n_points": n,
                    "significant": False,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
                for method in methods
            }

        x = df_clean[col_a].values
        y = df_clean[col_b].values
        ci_level = self.settings.confidence_level

        for method in methods:
            if method == "pearson":
                corr, p_val = stats.pearsonr(x, y)
                se = 1.0 / np.sqrt(n - 3)
                z = np.arctanh(corr)
                z_crit = stats.norm.ppf((1 + ci_level) / 2)
                ci_lower = np.tanh(z - z_crit * se)
                ci_upper = np.tanh(z + z_crit * se)
            elif method == "spearman":
                corr, p_val = stats.spearmanr(x, y)
                se = 1.0 / np.sqrt(n - 3)
                z = np.arctanh(corr)
                z_crit = stats.norm.ppf((1 + ci_level) / 2)
                ci_lower = np.tanh(z - z_crit * se)
                ci_upper = np.tanh(z + z_crit * se)
            elif method == "kendall":
                corr, p_val = stats.kendalltau(x, y)
                ci_lower = np.nan
                ci_upper = np.nan
            else:
                raise ValueError(f"不支持的相关性方法: {method}")

            results[method] = {
                "correlation": float(corr),
                "p_value": float(p_val),
                "n_points": int(n),
                "significant": bool(p_val < (1 - ci_level)),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
            }

        return results

    def cross_correlation_with_lag(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        age_col: str = "age",
        max_lag: Optional[int] = None,
    ) -> pd.DataFrame:
        if max_lag is None:
            max_lag = self.settings.lag_max

        df_clean = df[[age_col, col_a, col_b]].dropna().sort_values(age_col)
        if len(df_clean) < self.settings.min_overlap_points:
            return pd.DataFrame()

        age_step = np.median(np.diff(df_clean[age_col].values))
        if age_step <= 0:
            age_step = self.global_settings.age_grid_resolution

        lag_steps = int(max_lag / age_step)

        results = []
        x = df_clean[col_a].values
        y = df_clean[col_b].values
        x_norm = (x - np.mean(x)) / (np.std(x) * len(x))
        y_norm = (y - np.mean(y)) / np.std(y)

        for lag in range(-lag_steps, lag_steps + 1):
            if lag == 0:
                corr = np.sum(x_norm * y_norm) * len(x)
            elif lag > 0:
                corr = np.sum(x_norm[lag:] * y_norm[:-lag]) * len(x)
            else:
                corr = np.sum(x_norm[:lag] * y_norm[-lag:]) * len(x)

            results.append({
                "lag_yr": lag * age_step,
                "correlation": float(corr),
            })

        return pd.DataFrame(results)

    def correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
    ) -> pd.DataFrame:
        if columns is None:
            columns = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and c != "age"
            ]

        n_cols = len(columns)
        corr_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)
        p_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)

        for i in range(n_cols):
            for j in range(i, n_cols):
                col_a = columns[i]
                col_b = columns[j]
                df_clean = df[[col_a, col_b]].dropna()
                if len(df_clean) >= self.settings.min_overlap_points:
                    if method == "pearson":
                        corr, p_val = stats.pearsonr(df_clean[col_a], df_clean[col_b])
                    elif method == "spearman":
                        corr, p_val = stats.spearmanr(df_clean[col_a], df_clean[col_b])
                    else:
                        corr, p_val = stats.pearsonr(df_clean[col_a], df_clean[col_b])
                    corr_matrix.loc[col_a, col_b] = corr
                    corr_matrix.loc[col_b, col_a] = corr
                    p_matrix.loc[col_a, col_b] = p_val
                    p_matrix.loc[col_b, col_a] = p_val

        return corr_matrix

    def correlation_by_epoch(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        epoch_col: str = "epoch",
    ) -> pd.DataFrame:
        if epoch_col not in df.columns:
            raise ValueError(f"数据缺少地层列: {epoch_col}")

        results = []
        for epoch_name, group in df.groupby(epoch_col, dropna=True):
            corr_res = self.pairwise_correlation(group, col_a, col_b)
            for method, values in corr_res.items():
                results.append({
                    "epoch": epoch_name,
                    "method": method,
                    "variable_a": col_a,
                    "variable_b": col_b,
                    **values,
                })

        return pd.DataFrame(results)

    def summary_statistics(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if columns is None:
            columns = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
            ]

        stats_rows = []
        for col in columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            stats_rows.append({
                "variable": col,
                "n_points": len(series),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "p75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
            })

        return pd.DataFrame(stats_rows)

    def save_correlation_results(
        self,
        results_df: pd.DataFrame,
        filename: str,
    ) -> str:
        out_dir = self.config.get_output_dir()
        if not filename.endswith(".csv"):
            filename += ".csv"
        filepath = os.path.join(out_dir, filename)
        results_df.to_csv(filepath, index=False, encoding="utf-8-sig")
        return filepath

    def run_full_analysis(
        self,
        merged_df: pd.DataFrame,
        per_source: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        output = {}

        numeric_cols = [
            c for c in merged_df.columns
            if pd.api.types.is_numeric_dtype(merged_df[c]) and c != "age"
        ]

        summary = self.summary_statistics(merged_df, numeric_cols)
        self.save_correlation_results(summary, "summary_statistics")
        output["summary_statistics"] = summary

        for method in self.settings.methods:
            corr_mat = self.correlation_matrix(merged_df, numeric_cols, method=method)
            self.save_correlation_results(corr_mat, f"correlation_matrix_{method}")
            output[f"correlation_matrix_{method}"] = corr_mat

        pairwise_results = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_a = numeric_cols[i]
                col_b = numeric_cols[j]
                pw = self.pairwise_correlation(merged_df, col_a, col_b)
                for method, vals in pw.items():
                    pairwise_results.append({
                        "variable_a": col_a,
                        "variable_b": col_b,
                        "method": method,
                        **vals,
                    })

        pairwise_df = pd.DataFrame(pairwise_results)
        self.save_correlation_results(pairwise_df, "pairwise_correlations")
        output["pairwise_correlations"] = pairwise_df

        return output
