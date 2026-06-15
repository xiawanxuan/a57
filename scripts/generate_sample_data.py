import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config_manager import ConfigManager


def generate_ice_core_data(n_samples: int = 1200, seed: int = 42, source_id: str = "ICE_DOME_A") -> pd.DataFrame:
    np.random.seed(seed)
    age_min, age_max = 0, 130000
    ages = np.sort(np.random.uniform(age_min, age_max, n_samples))
    ages = ages + np.random.normal(0, 10, n_samples)
    ages = np.clip(np.sort(ages), age_min, age_max)

    depths = ages * 0.0008 + np.random.normal(0, 0.05, n_samples)
    depths = np.clip(depths, 0, 200)

    source_params = {
        "ICE_DOME_A": {"base_d18O": -60.0, "amp_100k": 4.0, "amp_40k": 2.0, "amp_23k": 1.2, "phase_100k": 0.0, "noise": 0.3},
        "ICE_GRIP": {"base_d18O": -35.0, "amp_100k": 5.0, "amp_40k": 2.5, "amp_23k": 1.5, "phase_100k": 0.1, "noise": 0.4},
        "ICE_VOSTOK": {"base_d18O": -55.0, "amp_100k": 3.5, "amp_40k": 1.8, "amp_23k": 1.0, "phase_100k": -0.1, "noise": 0.25},
        "ICE_EPICA": {"base_d18O": -52.0, "amp_100k": 3.8, "amp_40k": 1.9, "amp_23k": 1.1, "phase_100k": -0.05, "noise": 0.28},
    }
    p = source_params.get(source_id, source_params["ICE_DOME_A"])

    orbital_100k = p["amp_100k"] * np.sin(2 * np.pi * ages / 100000 + p["phase_100k"] * np.pi)
    orbital_40k = p["amp_40k"] * np.sin(2 * np.pi * ages / 40000)
    orbital_23k = p["amp_23k"] * np.sin(2 * np.pi * ages / 23000)
    millennial = 0.8 * np.sin(2 * np.pi * ages / 1500)
    trend = -0.00001 * ages
    noise = np.random.normal(0, p["noise"], n_samples)

    d18o = p["base_d18O"] + orbital_100k + orbital_40k + orbital_23k + millennial + trend + noise
    dD = 8 * (d18o - p["base_d18O"]) + p["base_d18O"] * 0.9 + np.random.normal(0, 2, n_samples)
    dust = 100 + 50 * orbital_100k + np.random.normal(0, 15, n_samples)
    dust = np.clip(dust, 5, 500)

    outlier_idx = np.random.choice(n_samples, size=max(5, n_samples // 50), replace=False)
    d18o[outlier_idx] += np.random.choice([-10, 10], size=len(outlier_idx))

    if source_id == "ICE_DOME_A":
        df = pd.DataFrame({
            "depth_m": np.round(depths, 3),
            "age_yrBP": np.round(ages, 1),
            "d18O_permil": np.round(d18o, 3),
            "dD_permil": np.round(dD, 2),
            "dust_ppm": np.round(dust, 2),
        })
    elif source_id == "ICE_GRIP":
        df = pd.DataFrame({
            "depth_m": np.round(depths, 3),
            "age_yrBP": np.round(ages, 1),
            "delta18O": np.round(d18o, 3),
            "deltaD": np.round(dD, 2),
            "dust_ppb": np.round(dust * 1000, 1),
        })
    elif source_id == "ICE_VOSTOK":
        df = pd.DataFrame({
            "depth_m": np.round(depths, 3),
            "age_yrBP": np.round(ages, 1),
            "d18O_permil": np.round(d18o, 3),
            "dust_concentration": np.round(dust, 2),
        })
    elif source_id == "ICE_EPICA":
        df = pd.DataFrame({
            "depth_m": np.round(depths, 3),
            "age_yrBP": np.round(ages, 1),
            "delta_18O": np.round(d18o, 3),
            "delta_D": np.round(dD, 2),
        })
    else:
        df = pd.DataFrame({
            "depth_m": np.round(depths, 3),
            "age_yrBP": np.round(ages, 1),
            "d18O_permil": np.round(d18o, 3),
        })

    return df


def generate_stalagmite_data(n_samples: int = 800, seed: int = 43, source_id: str = "STAL_HULU") -> pd.DataFrame:
    np.random.seed(seed)
    age_min, age_max = 0, 100000
    ages = np.sort(np.random.uniform(age_min, age_max, n_samples))
    ages = ages + np.random.normal(0, 20, n_samples)
    ages = np.clip(np.sort(ages), age_min, age_max)

    depths_mm = ages * 0.05 + np.random.normal(0, 0.5, n_samples)
    depths_mm = np.clip(depths_mm, 0, 5000)

    source_params = {
        "STAL_HULU": {"base_d18O": -6.0, "base_d13C": -8.0, "amp_insolation": 3.0, "amp_millennial": 1.2, "phase_shift": 0.0},
        "STAL_SANBAO": {"base_d18O": -8.5, "base_d13C": -7.5, "amp_insolation": 2.8, "amp_millennial": 1.0, "phase_shift": 0.15},
    }
    p = source_params.get(source_id, source_params["STAL_HULU"])

    insolation = p["amp_insolation"] * np.sin(2 * np.pi * ages / 23000 + p["phase_shift"] * np.pi)
    orbital_100k = 2.5 * np.sin(2 * np.pi * ages / 100000)
    millennial = p["amp_millennial"] * np.sin(2 * np.pi * ages / 2000)
    trend = 0.000005 * ages
    noise = np.random.normal(0, 0.4, n_samples)

    d18o_calcite = p["base_d18O"] - insolation - orbital_100k * 0.5 - millennial + trend + noise
    d13c_calcite = p["base_d13C"] + 0.5 * orbital_100k + np.random.normal(0, 0.5, n_samples)
    mg_ca = 0.05 + 0.02 * orbital_100k + 0.01 * np.sin(2 * np.pi * ages / 10000) + np.random.normal(0, 0.005, n_samples)
    sr_ca = 0.001 + 0.0005 * millennial + np.random.normal(0, 0.0002, n_samples)

    mg_ca = np.clip(mg_ca, 0.001, 0.5)
    sr_ca = np.clip(sr_ca, 0.0001, 0.01)

    outlier_idx = np.random.choice(n_samples, size=max(4, n_samples // 60), replace=False)
    d18o_calcite[outlier_idx] += np.random.choice([-3, 3], size=len(outlier_idx))

    if source_id == "STAL_HULU":
        df = pd.DataFrame({
            "depth_mm": np.round(depths_mm, 2),
            "age_yrBP": np.round(ages, 1),
            "d18O_calcite": np.round(d18o_calcite, 3),
            "d13C_calcite": np.round(d13c_calcite, 3),
            "Mg_Ca_ratio": np.round(mg_ca, 5),
            "Sr_Ca_ratio": np.round(sr_ca, 6),
        })
    elif source_id == "STAL_SANBAO":
        df = pd.DataFrame({
            "depth_mm": np.round(depths_mm, 2),
            "age_yrBP": np.round(ages, 1),
            "d18O": np.round(d18o_calcite, 3),
            "Mg/Ca": np.round(mg_ca, 5),
        })
    else:
        df = pd.DataFrame({
            "depth_mm": np.round(depths_mm, 2),
            "age_yrBP": np.round(ages, 1),
            "d18O_calcite": np.round(d18o_calcite, 3),
        })

    return df


def main():
    cm = ConfigManager()
    cm.load_all_configs()
    raw_dir = cm.get_raw_data_dir()
    os.makedirs(raw_dir, exist_ok=True)

    print("=" * 60)
    print("第四纪古气候数据分析工程 - 示例数据生成")
    print("=" * 60)
    print()

    all_sources = cm.get_all_sources(enabled_only=True)
    for source in all_sources:
        seed = sum(ord(c) for c in source.id)
        if source.id.startswith("ICE"):
            df = generate_ice_core_data(seed=seed, source_id=source.id)
        else:
            df = generate_stalagmite_data(seed=seed * 2, source_id=source.id)

        df.to_csv(source.file_path, index=False, encoding="utf-8-sig")
        print(f"[✓] {source.name} ({source.id}): {len(df)} 条样本")
        print(f"    保存路径: {source.file_path}")
        age_col = source.columns.get("age", "age_yrBP")
        if age_col in df.columns:
            print(f"    年龄范围: {df[age_col].min():.0f} ~ {df[age_col].max():.0f} yr BP")
        print()

    print("示例数据生成完成！")
    print()
    print("共生成", len(all_sources), "个数据源：")
    ice_sources = [s for s in all_sources if s.id.startswith("ICE")]
    stal_sources = [s for s in all_sources if not s.id.startswith("ICE")]
    if ice_sources:
        print(f"  冰芯 ({len(ice_sources)} 个):")
        for s in ice_sources:
            print(f"    - {s.id}: {s.name} @ {s.location}")
    if stal_sources:
        print(f"  石笋 ({len(stal_sources)} 个):")
        for s in stal_sources:
            print(f"    - {s.id}: {s.name} @ {s.location}")


if __name__ == "__main__":
    main()
