import os
import yaml
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class DataSource:
    id: str
    name: str
    location: str
    file_path: str
    file_type: str
    columns: Dict[str, str]
    enabled: bool = True
    d18o_to_temp_coef: Optional[float] = None
    d18o_to_temp_intercept: Optional[float] = None
    d18o_to_precip_coef: Optional[float] = None
    d18o_to_precip_intercept: Optional[float] = None


@dataclass
class GlobalSettings:
    age_unit: str = "yr BP"
    depth_unit: str = "m"
    temperature_unit: str = "°C"
    precipitation_unit: str = "mm/yr"
    random_seed: int = 42
    outlier_zscore_threshold: float = 3.0
    interpolation_method: str = "cubic"
    age_grid_resolution: int = 100


@dataclass
class CorrelationSettings:
    methods: List[str] = field(default_factory=lambda: ["pearson", "spearman"])
    min_overlap_points: int = 30
    confidence_level: float = 0.95
    lag_max: int = 500


@dataclass
class VisualizationSettings:
    theme: str = "plotly_white"
    figure_width: int = 1200
    figure_height: int = 700
    color_palette: Dict[str, str] = field(default_factory=dict)
    dpi: int = 150
    save_formats: List[str] = field(default_factory=lambda: ["html", "png"])


@dataclass
class ReportSettings:
    title: str = "第四纪古气候代用指标分析报告"
    author: str = "第四纪地质研究所"
    include_correlation_table: bool = True
    include_summary_statistics: bool = True
    font_family: str = "SimSun"
    font_size: int = 12


class ConfigManager:
    def __init__(
        self,
        data_sources_path: str = "./config/data_sources.yaml",
        stratigraphy_rules_path: str = "./config/stratigraphy_rules.yaml",
        age_calibration_path: str = "./config/age_calibration.csv",
    ):
        self.data_sources_path = self._resolve_path(data_sources_path)
        self.stratigraphy_rules_path = self._resolve_path(stratigraphy_rules_path)
        self.age_calibration_path = self._resolve_path(age_calibration_path)

        self._data_sources_config: Optional[Dict[str, Any]] = None
        self._stratigraphy_config: Optional[Dict[str, Any]] = None
        self._age_calibration_table: Optional[pd.DataFrame] = None

    @staticmethod
    def _resolve_path(rel_path: str) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_path = os.path.normpath(os.path.join(project_root, rel_path))
        return abs_path

    def load_all_configs(self) -> None:
        self._load_data_sources()
        self._load_stratigraphy_rules()
        self._load_age_calibration()

    def _load_data_sources(self) -> None:
        with open(self.data_sources_path, "r", encoding="utf-8") as f:
            self._data_sources_config = yaml.safe_load(f)

    def _load_stratigraphy_rules(self) -> None:
        with open(self.stratigraphy_rules_path, "r", encoding="utf-8") as f:
            self._stratigraphy_config = yaml.safe_load(f)

    def _load_age_calibration(self) -> None:
        if os.path.exists(self.age_calibration_path):
            self._age_calibration_table = pd.read_csv(self.age_calibration_path, encoding="utf-8")
        else:
            self._age_calibration_table = pd.DataFrame()

    def get_global_settings(self) -> GlobalSettings:
        if self._data_sources_config is None:
            self._load_data_sources()
        gs = self._data_sources_config.get("global_settings", {})
        return GlobalSettings(**gs)

    def get_correlation_settings(self) -> CorrelationSettings:
        if self._data_sources_config is None:
            self._load_data_sources()
        cs = self._data_sources_config.get("correlation_settings", {})
        return CorrelationSettings(**cs)

    def get_visualization_settings(self) -> VisualizationSettings:
        if self._data_sources_config is None:
            self._load_data_sources()
        vs = self._data_sources_config.get("visualization", {})
        return VisualizationSettings(**vs)

    def get_report_settings(self) -> ReportSettings:
        if self._data_sources_config is None:
            self._load_data_sources()
        rs = self._data_sources_config.get("report", {})
        return ReportSettings(**rs)

    def get_ice_core_sources(self, enabled_only: bool = True) -> List[DataSource]:
        if self._data_sources_config is None:
            self._load_data_sources()
        sources = []
        for item in self._data_sources_config.get("ice_core_sources", []):
            src = DataSource(
                id=item["id"],
                name=item["name"],
                location=item["location"],
                file_path=self._resolve_path(item["file_path"].replace("./", "./")),
                file_type=item["file_type"],
                columns=item["columns"],
                enabled=item.get("enabled", True),
                d18o_to_temp_coef=item.get("d18o_to_temp_coef"),
                d18o_to_temp_intercept=item.get("d18o_to_temp_intercept"),
            )
            if enabled_only and not src.enabled:
                continue
            sources.append(src)
        return sources

    def get_stalagmite_sources(self, enabled_only: bool = True) -> List[DataSource]:
        if self._data_sources_config is None:
            self._load_data_sources()
        sources = []
        for item in self._data_sources_config.get("stalagmite_sources", []):
            src = DataSource(
                id=item["id"],
                name=item["name"],
                location=item["location"],
                file_path=self._resolve_path(item["file_path"].replace("./", "./")),
                file_type=item["file_type"],
                columns=item["columns"],
                enabled=item.get("enabled", True),
                d18o_to_precip_coef=item.get("d18o_to_precip_coef"),
                d18o_to_precip_intercept=item.get("d18o_to_precip_intercept"),
            )
            if enabled_only and not src.enabled:
                continue
            sources.append(src)
        return sources

    def get_all_sources(self, enabled_only: bool = True) -> List[DataSource]:
        return self.get_ice_core_sources(enabled_only) + self.get_stalagmite_sources(enabled_only)

    def get_stratigraphy_epochs(self) -> List[Dict[str, Any]]:
        if self._stratigraphy_config is None:
            self._load_stratigraphy_rules()
        qua = self._stratigraphy_config.get("stratigraphy_definitions", {}).get("quaternary", {})
        return qua.get("epochs", [])

    def get_binning_rules(self) -> Dict[str, Any]:
        if self._stratigraphy_config is None:
            self._load_stratigraphy_rules()
        return self._stratigraphy_config.get("binning_rules", {})

    def get_outlier_detection_rules(self) -> Dict[str, Any]:
        if self._stratigraphy_config is None:
            self._load_stratigraphy_rules()
        return self._stratigraphy_config.get("outlier_detection", {})

    def get_quality_flags(self) -> Dict[str, int]:
        if self._stratigraphy_config is None:
            self._load_stratigraphy_rules()
        return self._stratigraphy_config.get("quality_flags", {})

    def get_age_calibration_table(self) -> pd.DataFrame:
        if self._age_calibration_table is None:
            self._load_age_calibration()
        return self._age_calibration_table.copy()

    def get_output_dir(self) -> str:
        if self._data_sources_config is None:
            self._load_data_sources()
        rel = self._data_sources_config.get("project", {}).get("output_dir", "./output")
        path = self._resolve_path(rel)
        os.makedirs(path, exist_ok=True)
        return path

    def get_processed_data_dir(self) -> str:
        if self._data_sources_config is None:
            self._load_data_sources()
        rel = self._data_sources_config.get("project", {}).get("processed_data_dir", "./data/processed")
        path = self._resolve_path(rel)
        os.makedirs(path, exist_ok=True)
        return path

    def get_raw_data_dir(self) -> str:
        if self._data_sources_config is None:
            self._load_data_sources()
        rel = self._data_sources_config.get("project", {}).get("raw_data_dir", "./data/raw")
        path = self._resolve_path(rel)
        os.makedirs(path, exist_ok=True)
        return path

    def add_ice_core_source(self, source_dict: Dict[str, Any]) -> None:
        if self._data_sources_config is None:
            self._load_data_sources()
        self._data_sources_config["ice_core_sources"].append(source_dict)
        self._save_data_sources()

    def add_stalagmite_source(self, source_dict: Dict[str, Any]) -> None:
        if self._data_sources_config is None:
            self._load_data_sources()
        self._data_sources_config["stalagmite_sources"].append(source_dict)
        self._save_data_sources()

    def toggle_source(self, source_id: str, enabled: bool) -> None:
        if self._data_sources_config is None:
            self._load_data_sources()
        for group in ["ice_core_sources", "stalagmite_sources"]:
            for src in self._data_sources_config.get(group, []):
                if src["id"] == source_id:
                    src["enabled"] = enabled
                    self._save_data_sources()
                    return

    def _save_data_sources(self) -> None:
        with open(self.data_sources_path, "w", encoding="utf-8") as f:
            yaml.dump(self._data_sources_config, f, allow_unicode=True, sort_keys=False)
