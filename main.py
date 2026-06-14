import os
import sys
import json
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config_manager import ConfigManager
from src.data_cleaning import StratigraphyDataCleaner
from src.age_interpolation import AgeInterpolator
from src.correlation_analysis import CorrelationAnalyzer
from src.visualization import StratigraphyVisualizer
from src.report_generator import ReportGenerator


def print_banner():
    print("=" * 70)
    print("  第四纪古气候数据分析工程 v1.0")
    print("  Quaternary Paleoclimate Data Analysis Pipeline")
    print("  第四纪地质研究所 | 冰芯-石笋多指标古气候重建")
    print("=" * 70)
    print()


def stage_header(stage_name: str):
    print()
    print("─" * 70)
    print(f"  ▶ {stage_name}")
    print("─" * 70)


def stage_ok(message: str):
    print(f"  ✓ {message}")


def stage_info(message: str):
    print(f"    {message}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="第四纪古气候数据分析工程 - 冰芯/石笋代用指标数据处理流水线"
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="先生成示例数据（如不存在）再运行分析",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="跳过 Word 报告生成",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="跳过可视化绘图",
    )
    parser.add_argument(
        "--only-source",
        type=str,
        default=None,
        help="仅处理指定数据源 ID（如 ICE_DOME_A）",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="列出所有已配置的数据源并退出",
    )
    return parser.parse_args()


def generate_sample_data_if_needed(force: bool = False):
    from scripts.generate_sample_data import generate_ice_core_data, generate_stalagmite_data

    cm = ConfigManager()
    cm.load_all_configs()
    raw_dir = cm.get_raw_data_dir()

    ice_path = os.path.join(raw_dir, "ice_core_dome_a.csv")
    stal_path = os.path.join(raw_dir, "stalagmite_hulu.csv")

    need_ice = force or not os.path.exists(ice_path)
    need_stal = force or not os.path.exists(stal_path)

    if need_ice or need_stal:
        stage_header("生成示例数据")
        if need_ice:
            ice_df = generate_ice_core_data()
            ice_df.to_csv(ice_path, index=False, encoding="utf-8-sig")
            stage_ok(f"Dome A 冰芯示例数据: {len(ice_df)} 条 → {os.path.basename(ice_path)}")
        if need_stal:
            stal_df = generate_stalagmite_data()
            stal_df.to_csv(stal_path, index=False, encoding="utf-8-sig")
            stage_ok(f"葫芦洞石笋示例数据: {len(stal_df)} 条 → {os.path.basename(stal_path)}")


def list_configured_sources():
    cm = ConfigManager()
    cm.load_all_configs()
    stage_header("已配置数据源")
    sources = cm.get_all_sources(enabled_only=False)
    for src in sources:
        status = "✓启用" if src.enabled else "  禁用"
        stype = "冰芯  " if src.id.startswith("ICE") else "石笋"
        print(f"  [{status}] [{stype}] {src.id:<15} {src.name:<16} @ {src.location}")
    print()


def run_pipeline(args):
    cm = ConfigManager()
    cm.load_all_configs()

    cleaner = StratigraphyDataCleaner(cm)
    interpolator = AgeInterpolator(cm)
    analyzer = CorrelationAnalyzer(cm)
    visualizer = StratigraphyVisualizer(cm)
    reporter = ReportGenerator(cm)

    pipeline_summary = {
        "started_at": datetime.now().isoformat(),
        "stages": {},
    }

    stage_header("阶段 1/5 - 地层深度数据清洗")
    cleaned_data = {}
    cleaning_reports = {}
    for source in cm.get_all_sources(enabled_only=True):
        if args.only_source and source.id != args.only_source:
            continue
        try:
            raw_df = cleaner.load_raw_data(source)
            stage_info(f"读取 {source.name} ({source.id}): {len(raw_df)} 条原始记录")
            cleaned_df, report = cleaner.clean_dataset(raw_df, source)
            out_path = cleaner.save_cleaned_data(cleaned_df, source.id)
            cleaned_data[source.id] = cleaned_df
            cleaning_reports[source.id] = report
            stage_ok(
                f"{source.id}: 清洗后 {len(cleaned_df)} 条, "
                f"保留率 {report.get('retention_rate', 0):.2%}, "
                f"剔除异常值 {report.get('outliers_total', 0)}"
            )
        except Exception as e:
            stage_info(f"✗ {source.id} 处理失败: {e}")
            cleaning_reports[source.id] = {"error": str(e)}

    pipeline_summary["stages"]["cleaning"] = cleaning_reports

    if not cleaned_data:
        print("\n  ✗ 没有可处理的数据源，流水线终止。")
        return

    stage_header("阶段 2/5 - 年代插值与时间轴对齐")
    merged_df, per_source_interp, age_grid = interpolator.process_all_sources(cleaned_data)
    stage_ok(f"公共年代栅格: {age_grid[0]:.0f} ~ {age_grid[-1]:.0f} yr BP, 步长 {age_grid[1]-age_grid[0]:.0f} yr")
    stage_ok(f"合并数据集: {len(merged_df)} 个时间点, {len(merged_df.columns)-1} 个代用指标列")
    for sid, df in per_source_interp.items():
        stage_info(f"  {sid}: 插值后 {len(df)} 行")
    pipeline_summary["stages"]["interpolation"] = {
        "age_grid_min": float(age_grid[0]),
        "age_grid_max": float(age_grid[-1]),
        "n_points": int(len(age_grid)),
        "merged_rows": int(len(merged_df)),
    }

    stage_header("阶段 3/5 - 多指标相关性计算")
    analysis_results = analyzer.run_full_analysis(merged_df, per_source_interp)
    for key, df in analysis_results.items():
        stage_ok(f"{key}: {df.shape[0]} 行 × {df.shape[1]} 列")
    pipeline_summary["stages"]["correlation"] = {
        k: list(df.columns) if isinstance(df, pd.DataFrame) else df
        for k, df in analysis_results.items()
    }

    figure_paths = {}
    if not args.skip_plots:
        stage_header("阶段 4/5 - 分层时序可视化")
        figure_paths = visualizer.generate_all_plots(merged_df, per_source_interp, analysis_results)
        for plot_name, paths in figure_paths.items():
            for fmt, fp in paths.items():
                stage_ok(f"{plot_name}.{fmt}: {os.path.basename(fp)}")
    else:
        stage_header("阶段 4/5 - 分层时序可视化 (已跳过)")

    pipeline_summary["stages"]["visualization"] = {
        k: list(v.keys()) for k, v in figure_paths.items()
    }

    report_path = None
    if not args.skip_report:
        stage_header("阶段 5/5 - Word 研究报告生成")
        try:
            report_path = reporter.generate_report(
                cleaned_data=cleaned_data,
                cleaning_reports=cleaning_reports,
                merged_df=merged_df,
                analysis_results=analysis_results,
                figure_paths=figure_paths,
            )
            stage_ok(f"报告已保存: {report_path}")
        except Exception as e:
            stage_info(f"✗ 报告生成失败: {e}")
    else:
        stage_header("阶段 5/5 - Word 研究报告生成 (已跳过)")

    pipeline_summary["stages"]["report"] = {"path": report_path}
    pipeline_summary["finished_at"] = datetime.now().isoformat()

    out_dir = cm.get_output_dir()
    summary_path = os.path.join(out_dir, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, ensure_ascii=False, indent=2, default=str)

    print()
    print("=" * 70)
    print("  流水线执行完成！")
    print(f"  输出目录: {out_dir}")
    print(f"  执行摘要: {os.path.basename(summary_path)}")
    if report_path:
        print(f"  研究报告: {os.path.basename(report_path)}")
    print("=" * 70)


def main():
    print_banner()
    args = parse_args()

    if args.list_sources:
        list_configured_sources()
        return

    if args.generate_data:
        generate_sample_data_if_needed(force=True)
    else:
        generate_sample_data_if_needed(force=False)

    run_pipeline(args)


if __name__ == "__main__":
    main()
