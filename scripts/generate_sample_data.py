import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config_manager import ConfigManager


def generate_ice_core_data(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    age_min, age_max = 0, 130000
    ages = np.sort(np.random.uniform(age_min, age_max, n_samples))
    ages = ages + np.random.normal(0, 10, n_samples)
    ages = np.clip(np.sort(ages), age_min, age_max)

    depths = ages * 0.0008 + np.random.normal(0, 0.05, n_samples)
    depths = np.clip(depths, 0, 200)

    orbital_100k = 3.0 * np.sin(2 * np.pi * ages / 100000)
    orbital_40k = 1.5 * np.sin(2 * np.pi * ages / 40000)
    orbital_23k = 1.0 * np.sin(2 * np.pi * ages / 23000)
    millennial = 0.8 * np.sin(2 * np.pi * ages / 1500)
    trend = -0.00001 * ages
    noise = np.random.normal(0, 0.3, n_samples)

    d18o = -40 + orbital_100k + orbital_40k + orbital_23k + millennial + trend + noise
    dD = 8 * d18o + 10 + np.random.normal(0, 2, n_samples)
    dust = 100 + 50 * orbital_100k + np.random.normal(0, 15, n_samples)
    dust = np.clip(dust, 5, 500)

    outlier_idx = np.random.choice(n_samples, size=max(5, n_samples // 50), replace=False)
    d18o[outlier_idx] += np.random.choice([-10, 10], size=len(outlier_idx))

    df = pd.DataFrame({
        "depth_m": np.round(depths, 3),
        "age_yrBP": np.round(ages, 1),
        "d18O_permil": np.round(d18o, 3),
        "dD_permil": np.round(dD, 2),
        "dust_ppm": np.round(dust, 2),
    })
    return df


def generate_stalagmite_data(n_samples: int = 800, seed: int = 43) -> pd.DataFrame:
    np.random.seed(seed)
    age_min, age_max = 0, 100000
    ages = np.sort(np.random.uniform(age_min, age_max, n_samples))
    ages = ages + np.random.normal(0, 20, n_samples)
    ages = np.clip(np.sort(ages), age_min, age_max)

    depths_mm = ages * 0.05 + np.random.normal(0, 0.5, n_samples)
    depths_mm = np.clip(depths_mm, 0, 5000)

    insolation = 3.0 * np.sin(2 * np.pi * ages / 23000)
    orbital_100k = 2.5 * np.sin(2 * np.pi * ages / 100000)
    millennial = 1.2 * np.sin(2 * np.pi * ages / 2000)
    trend = 0.000005 * ages
    noise = np.random.normal(0, 0.4, n_samples)

    d18o_calcite = -6.0 - insolation - orbital_100k * 0.5 - millennial + trend + noise
    d13c_calcite = -8.0 + 0.5 * orbital_100k + np.random.normal(0, 0.5, n_samples)
    mg_ca = 0.05 + 0.02 * orbital_100k + 0.01 * np.sin(2 * np.pi * ages / 10000) + np.random.normal(0, 0.005, n_samples)
    sr_ca = 0.001 + 0.0005 * millennial + np.random.normal(0, 0.0002, n_samples)

    mg_ca = np.clip(mg_ca, 0.001, 0.5)
    sr_ca = np.clip(sr_ca, 0.0001, 0.01)

    outlier_idx = np.random.choice(n_samples, size=max(4, n_samples // 60), replace=False)
    d18o_calcite[outlier_idx] += np.random.choice([-3, 3], size=len(outlier_idx))

    df = pd.DataFrame({
        "depth_mm": np.round(depths_mm, 2),
        "age_yrBP": np.round(ages, 1),
        "d18O_calcite": np.round(d18o_calcite, 3),
        "d13C_calcite": np.round(d13c_calcite, 3),
        "Mg_Ca_ratio": np.round(mg_ca, 5),
        "Sr_Ca_ratio": np.round(sr_ca, 6),
    })
    return df


def main():
    cm = ConfigManager()
    cm.load_all_configs()
    raw_dir = cm.get_raw_data_dir()

    print("=" * 60)
    print("第四纪古气候数据分析工程 - 示例数据生成")
    print("=" * 60)

    ice_df = generate_ice_core_data()
    ice_path = os.path.join(raw_dir, "ice_core_dome_a.csv")
    ice_df.to_csv(ice_path, index=False, encoding="utf-8-sig")
    print(f"[✓] Dome A 冰芯数据已生成: {len(ice_df)} 条样本")
    print(f"    保存路径: {ice_path}")
    print(f"    年龄范围: {ice_df['age_yrBP'].min():.0f} ~ {ice_df['age_yrBP'].max():.0f} yr BP")
    print()

    stal_df = generate_stalagmite_data()
    stal_path = os.path.join(raw_dir, "stalagmite_hulu.csv")
    stal_df.to_csv(stal_path, index=False, encoding="utf-8-sig")
    print(f"[✓] 葫芦洞石笋数据已生成: {len(stal_df)} 条样本")
    print(f"    保存路径: {stal_path}")
    print(f"    年龄范围: {stal_df['age_yrBP'].min():.0f} ~ {stal_df['age_yrBP'].max():.0f} yr BP")
    print()
    print("示例数据生成完成！")
    print()
    print("数据字段说明：")
    print("  冰芯 (ice_core_dome_a.csv):")
    print("    - depth_m      : 地层深度 (米)")
    print("    - age_yrBP     : 年代 (距今年)")
    print("    - d18O_permil  : 氧同位素 δ¹⁸O (‰)")
    print("    - dD_permil    : 氘同位素 δD (‰)")
    print("    - dust_ppm     : 粉尘浓度 (ppm)")
    print()
    print("  石笋 (stalagmite_hulu.csv):")
    print("    - depth_mm     : 地层深度 (毫米)")
    print("    - age_yrBP     : 年代 (距今年)")
    print("    - d18O_calcite : 方解石氧同位素 δ¹⁸O (‰)")
    print("    - d13C_calcite : 方解石碳同位素 δ¹³C (‰)")
    print("    - Mg_Ca_ratio  : Mg/Ca 比值")
    print("    - Sr_Ca_ratio  : Sr/Ca 比值")


if __name__ == "__main__":
    main()
