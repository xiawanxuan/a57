from .config_manager import ConfigManager, DataSource, GlobalSettings
from .data_cleaning import StratigraphyDataCleaner
from .age_interpolation import AgeInterpolator
from .correlation_analysis import CorrelationAnalyzer
from .visualization import StratigraphyVisualizer
from .report_generator import ReportGenerator
from .multi_core_comparison import MultiCoreComparator

__all__ = [
    "ConfigManager",
    "DataSource",
    "GlobalSettings",
    "StratigraphyDataCleaner",
    "AgeInterpolator",
    "CorrelationAnalyzer",
    "StratigraphyVisualizer",
    "ReportGenerator",
    "MultiCoreComparator",
]

__version__ = "1.1.0"
