import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .config_manager import ConfigManager


class StratigraphyVisualizer:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.settings = config_manager.get_visualization_settings()
        self.global_settings = config_manager.get_global_settings()
        self.epochs = config_manager.get_stratigraphy_epochs()
        self.colors = self.settings.color_palette

    def _get_color(self, name: str) -> str:
        return self.colors.get(name, "#2C3E50")

    def _add_epoch_shading(self, fig: go.Figure, age_min: float, age_max: float, row: int = 1) -> None:
        shapes = []
        annotations = []
        for epoch in self.epochs:
            e_start = epoch["age_start"]
            e_end = epoch["age_end"]
            overlap_start = max(age_min, e_start)
            overlap_end = min(age_max, e_end)
            if overlap_end <= overlap_start:
                continue
            shapes.append({
                "type": "rect",
                "x0": overlap_start,
                "x1": overlap_end,
                "y0": 0,
                "y1": 1,
                "xref": "x",
                "yref": f"y{row} domain",
                "fillcolor": epoch.get("color", "#D5D8DC"),
                "opacity": 0.12,
                "layer": "below",
                "line_width": 0,
            })
            annotations.append({
                "x": (overlap_start + overlap_end) / 2,
                "y": 1.03,
                "xref": "x",
                "yref": f"y{row} domain",
                "text": epoch["name"],
                "showarrow": False,
                "font": {"size": 10, "color": "#2C3E50"},
                "xanchor": "center",
                "yanchor": "bottom",
            })
        fig.update_layout(shapes=shapes + list(fig.layout.shapes or []),
                          annotations=annotations + list(fig.layout.annotations or []))

    def plot_multi_axis_timeseries(
        self,
        merged_df: pd.DataFrame,
        output_name: str = "multi_axis_timeseries",
    ) -> Dict[str, str]:
        df = merged_df.copy()
        df = df.sort_values("age")
        if len(df) == 0:
            return {}

        age_min = float(df["age"].min())
        age_max = float(df["age"].max())

        temp_cols = [c for c in df.columns if "temperature" in c.lower()]
        precip_cols = [c for c in df.columns if "precipitation" in c.lower()]
        d18o_cols = [c for c in df.columns if "d18o" in c.lower()]
        other_cols = [c for c in df.columns
                      if c not in ["age", "epoch", "epoch_code", "source_id", "source_name"]
                      and c not in temp_cols + precip_cols + d18o_cols
                      and pd.api.types.is_numeric_dtype(df[c])]

        y_axes = []
        if temp_cols:
            y_axes.append({"title": f"温度 ({self.global_settings.temperature_unit})", "color": self._get_color("temperature")})
        if precip_cols:
            y_axes.append({"title": f"降水 ({self.global_settings.precipitation_unit})", "color": self._get_color("precipitation")})
        if d18o_cols:
            y_axes.append({"title": "δ¹⁸O (‰)", "color": self._get_color("d18O_ice")})
        if other_cols:
            y_axes.append({"title": "其他代用指标", "color": self._get_color("dust")})

        if len(y_axes) == 0:
            return {}

        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": len(y_axes) > 1}]],
        )

        idx = 0
        for col in temp_cols:
            fig.add_trace(go.Scatter(
                x=df["age"], y=df[col],
                mode="lines", name=col,
                line=dict(color=self._get_color("temperature"), width=1.5),
                yaxis="y",
            ))
            idx += 1

        for col in precip_cols:
            fig.add_trace(go.Scatter(
                x=df["age"], y=df[col],
                mode="lines", name=col,
                line=dict(color=self._get_color("precipitation"), width=1.5),
                yaxis="y2",
            ))
            idx += 1

        for i, col in enumerate(d18o_cols):
            color = self._get_color("d18O_ice") if i == 0 else self._get_color("d18O_stalagmite")
            yaxis_key = "y" if idx == 0 else ("y2" if idx == 1 else ("y3" if idx == 2 else "y4"))
            fig.add_trace(go.Scatter(
                x=df["age"], y=df[col],
                mode="lines", name=col,
                line=dict(color=color, width=1.5, dash="dash"),
                yaxis=yaxis_key,
            ))
            idx += 1

        for col in other_cols:
            yaxis_key = "y" if idx == 0 else ("y2" if idx == 1 else ("y3" if idx == 2 else "y4"))
            fig.add_trace(go.Scatter(
                x=df["age"], y=df[col],
                mode="lines", name=col,
                line=dict(color=self._get_color("dust"), width=1.2, dash="dot"),
                yaxis=yaxis_key,
            ))
            idx += 1

        self._add_epoch_shading(fig, age_min, age_max)

        layout_updates = {
            "xaxis": {
                "title": f"年代 ({self.global_settings.age_unit})",
                "autorange": "reversed",
                "showgrid": True,
                "gridcolor": "#E5E8E8",
            },
            "height": self.settings.figure_height,
            "width": self.settings.figure_width,
            "template": self.settings.theme,
            "title": "多代用指标万年尺度温度-降水时序曲线",
            "legend": {"orientation": "h", "y": -0.15},
            "hovermode": "x unified",
            "margin": {"t": 100, "b": 100},
        }

        for i, axis_info in enumerate(y_axes):
            key = "yaxis" if i == 0 else f"yaxis{i+1}"
            layout_updates[key] = {
                "title": axis_info["title"],
                "titlefont": {"color": axis_info["color"]},
                "tickfont": {"color": axis_info["color"]},
                "showgrid": i == 0,
                "gridcolor": "#E5E8E8",
            }
            if i > 0:
                layout_updates[key]["overlaying"] = "y"
                side = "right" if i % 2 == 1 else "left"
                if i >= 2:
                    layout_updates[key]["side"] = "right"
                    layout_updates[key]["position"] = 1.0 + (i - 1) * 0.08
                else:
                    layout_updates[key]["side"] = side

        fig.update_layout(**layout_updates)

        return self._save_figure(fig, output_name)

    def plot_temperature_precipitation(
        self,
        merged_df: pd.DataFrame,
        output_name: str = "temperature_precipitation",
    ) -> Dict[str, str]:
        df = merged_df.copy().sort_values("age")
        if len(df) == 0:
            return {}

        temp_cols = [c for c in df.columns if "temperature" in c.lower()]
        precip_cols = [c for c in df.columns if "precipitation" in c.lower()]

        if not temp_cols and not precip_cols:
            return {}

        age_min = float(df["age"].min())
        age_max = float(df["age"].max())

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for col in temp_cols:
            fig.add_trace(go.Scatter(
                x=df["age"], y=df[col],
                mode="lines", name=col,
                line=dict(color=self._get_color("temperature"), width=2),
            ), secondary_y=False)

        for col in precip_cols:
            fig.add_trace(go.Scatter(
                x=df["age"], y=df[col],
                mode="lines", name=col,
                line=dict(color=self._get_color("precipitation"), width=2),
            ), secondary_y=True)

        self._add_epoch_shading(fig, age_min, age_max)

        fig.update_layout(
            title="万年尺度温度与降水波动时序",
            xaxis_title=f"年代 ({self.global_settings.age_unit})",
            xaxis=dict(autorange="reversed", showgrid=True, gridcolor="#E5E8E8"),
            yaxis_title=f"温度 ({self.global_settings.temperature_unit})",
            yaxis=dict(titlefont=dict(color=self._get_color("temperature")),
                       tickfont=dict(color=self._get_color("temperature")),
                       showgrid=True, gridcolor="#E5E8E8"),
            yaxis2_title=f"降水 ({self.global_settings.precipitation_unit})",
            yaxis2=dict(titlefont=dict(color=self._get_color("precipitation")),
                        tickfont=dict(color=self._get_color("precipitation")),
                        overlaying="y", side="right"),
            height=self.settings.figure_height,
            width=self.settings.figure_width,
            template=self.settings.theme,
            legend=dict(orientation="h", y=-0.15),
            hovermode="x unified",
            margin=dict(t=100, b=100),
        )

        return self._save_figure(fig, output_name)

    def plot_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        output_name: str = "correlation_heatmap",
    ) -> Dict[str, str]:
        if corr_matrix.empty:
            return {}

        fig = px.imshow(
            corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto",
            text_auto=".2f",
        )

        fig.update_layout(
            title="代用指标相关性热力图",
            height=max(600, len(corr_matrix.columns) * 40),
            width=max(800, len(corr_matrix.columns) * 80),
            template=self.settings.theme,
            xaxis_tickangle=-45,
        )

        return self._save_figure(fig, output_name)

    def plot_epoch_boxplot(
        self,
        df: pd.DataFrame,
        value_col: str,
        output_name: Optional[str] = None,
    ) -> Dict[str, str]:
        if "epoch" not in df.columns or value_col not in df.columns:
            return {}

        fig = px.box(
            df, x="epoch", y=value_col, color="epoch",
            title=f"各地层单位 {value_col} 分布对比",
        )
        fig.update_layout(
            height=self.settings.figure_height,
            width=self.settings.figure_width,
            template=self.settings.theme,
            xaxis_title="地层单位",
            yaxis_title=value_col,
            showlegend=False,
        )
        name = output_name or f"epoch_boxplot_{value_col}"
        return self._save_figure(fig, name)

    def plot_proxy_scatter(
        self,
        df: pd.DataFrame,
        col_x: str,
        col_y: str,
        output_name: Optional[str] = None,
    ) -> Dict[str, str]:
        if col_x not in df.columns or col_y not in df.columns:
            return {}

        df_clean = df[[col_x, col_y, "epoch"]].dropna() if "epoch" in df.columns else df[[col_x, col_y]].dropna()
        color_col = "epoch" if "epoch" in df_clean.columns else None

        fig = px.scatter(
            df_clean, x=col_x, y=col_y, color=color_col,
            trendline="ols", trendline_scope="overall",
            title=f"{col_x} vs {col_y} 散点图",
        )
        fig.update_layout(
            height=self.settings.figure_height,
            width=self.settings.figure_width,
            template=self.settings.theme,
        )
        name = output_name or f"scatter_{col_x}_{col_y}"
        return self._save_figure(fig, name)

    def plot_depth_profile(
        self,
        df: pd.DataFrame,
        value_col: str,
        output_name: Optional[str] = None,
    ) -> Dict[str, str]:
        if "depth" not in df.columns or value_col not in df.columns:
            return {}
        df_sorted = df.sort_values("depth")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sorted[value_col], y=df_sorted["depth"],
            mode="lines", name=value_col,
            line=dict(color="#2C3E50", width=1.5),
        ))
        fig.update_layout(
            title=f"地层深度剖面 - {value_col}",
            xaxis_title=value_col,
            yaxis_title=f"深度 ({self.global_settings.depth_unit})",
            yaxis=dict(autorange="reversed", showgrid=True, gridcolor="#E5E8E8"),
            height=self.settings.figure_height,
            width=self.settings.figure_width,
            template=self.settings.theme,
        )
        name = output_name or f"depth_profile_{value_col}"
        return self._save_figure(fig, name)

    def _save_figure(self, fig: go.Figure, output_name: str) -> Dict[str, str]:
        out_dir = self.config.get_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        paths = {}

        for fmt in self.settings.save_formats:
            filepath = os.path.join(out_dir, f"{output_name}.{fmt}")
            if fmt == "html":
                fig.write_html(filepath, include_plotlyjs="cdn")
            elif fmt == "png":
                try:
                    fig.write_image(filepath, scale=self.settings.dpi / 100)
                except Exception:
                    continue
            paths[fmt] = filepath

        return paths

    def generate_all_plots(
        self,
        merged_df: pd.DataFrame,
        per_source: Dict[str, pd.DataFrame],
        analysis_results: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[str, str]]:
        output_paths = {}

        output_paths["multi_axis_timeseries"] = self.plot_multi_axis_timeseries(merged_df)
        output_paths["temperature_precipitation"] = self.plot_temperature_precipitation(merged_df)

        for method in ["pearson", "spearman"]:
            key = f"correlation_matrix_{method}"
            if key in analysis_results:
                output_paths[f"heatmap_{method}"] = self.plot_correlation_heatmap(
                    analysis_results[key], f"correlation_heatmap_{method}"
                )

        numeric_cols = [
            c for c in merged_df.columns
            if pd.api.types.is_numeric_dtype(merged_df[c]) and c != "age"
        ]
        for col in numeric_cols[:4]:
            output_paths[f"boxplot_{col}"] = self.plot_epoch_boxplot(merged_df, col)

        return output_paths
