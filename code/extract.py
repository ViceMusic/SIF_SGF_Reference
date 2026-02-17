# 环境检查
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent

# 核心库导入
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# 项目模块导入
from feature_extraction import PeptideFeaturizer
from feature_extraction.utils import (
    get_csv_files, load_csv_safely, extract_molecular_features,
    convert_label_to_minutes, save_features_to_npz
)

# 设置显示选项
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')

# ============== 参数配置区 ==============
# 用户可根据需要修改以下参数

CONFIG = {
    # csv输出路径
    'processed_dir': project_root/'data'/'csv',
    'features_dir': project_root/'data'/'npz',
    'avalon_bits': 512,            # Avalon指纹位数512
    'molformer_bits':768,   # 新增：MolFormer指纹位数768
    'molt5_bits':768,     # 新增：MolT5 768
    # 可视化参数
    'dpi': 300,              # 图像分辨率
    'format': 'png',         # 图像格式 (png/pdf/svg)
    'display_plots': True,   # 是否在notebook中显示关键图表
    'max_display_plots': 3,  # 最多显示几个图表
    'split_random':42
}

# 这里分成SIF任务和SGF任务
# 读取两个任务并且保存到对应features_dir下面
tasks = [
    {
        "name": "SIF",
        "featurizer": PeptideFeaturizer(
            avalon_bits=CONFIG['avalon_bits'],
            molformer_bits=CONFIG['molformer_bits']
        ),
    },
    {
        "name": "SGF",
        "featurizer": PeptideFeaturizer(
            avalon_bits=CONFIG['avalon_bits'],
            molt5_bits=CONFIG['molt5_bits']
        ),
    }
]



# 创建输出目录
CONFIG['processed_dir'].mkdir(parents=True, exist_ok=True)
CONFIG['features_dir'].mkdir(parents=True, exist_ok=True)



def process_and_merge_datasets(raw_file_path):
    # 不进行拆分了，仅仅根据数据内容把数据拆分成SIF任务和SGF两种任务格式，并且生成csv文件
    # 再填入到对应的文件中
    
    # =========================
    # 为两个任务分别准备容器
    # =========================
    test_list_sif = []
    test_list_sgf = []
    #print(f"正在处理: {raw_file_path}")
    df = pd.read_csv(raw_file_path)
    file_name=raw_file_path

    # ==================================================
    # 【清理废数据】全局生效
    # 1. SMILES 不能为空
    # ==================================================
    #print(f"{file_name}原本数据长度{len(df)}")
    df = df[df["SMILES"].notna()].copy()
    #print(f"{file_name}清理空SMILE后：{len(df)}")

    # ------------------ 原有预处理逻辑开始 ------------------
    # 提取分子特征
    feature_records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {file_name}", leave=False):
        features = extract_molecular_features(row["SMILES"])
        feature_records.append(features)

    df = pd.concat([df, pd.DataFrame(feature_records)], axis=1)

    # 转换标签到分钟
    if "SIF_class" in df.columns:
        df["SIF_minutes"] = df["SIF_class"].apply(convert_label_to_minutes)
    else:
        df["SIF_minutes"] = -1

    if "SGF_class" in df.columns:
        df["SGF_minutes"] = df["SGF_class"].apply(convert_label_to_minutes)
    else:
        df["SGF_minutes"] = -1

    # ==================================================
    # 【清理废数据】
    # SIF 和 SGF 同时为 -1 的样本直接删除
    # ==================================================
    mask_both_missing = (df["SIF_minutes"] == -1) & (df["SGF_minutes"] == -1)
    df_processed = df[~mask_both_missing].copy()
    #print(f"{file_name}去掉均为空的脏数据：{len(df_processed)}")
    # 提前筛选：只保留 monomer
    if "is_monomer" in df_processed.columns:
        df_processed = df_processed[df_processed["is_monomer"] == True].copy()
    else:
        print(f"--- 警告: {file_name} 中不存在 is_monomer 字段")
    #print(f"{file_name}清理空非单体后：{len(df_processed)}")
    # 记录数据来源（用于后续分析 / 泄露检查）
    df_processed["source_name"] = file_name

    # ------------------ 原有预处理逻辑结束 ------------------
    # SIF
    sif_df = df_processed[(df_processed["SIF_minutes"] != -1) & (df_processed["SIF_minutes"] <= 700)].copy()
    sif_df["label"] = (sif_df["SIF_minutes"] >= 270).astype(int)
    test_list_sif.append(sif_df)

    # SGF
    sgf_df = df_processed[(df_processed["SGF_minutes"] != -1) & (df_processed["SGF_minutes"] <= 700)].copy()
    sgf_df["label"] = (sgf_df["SGF_minutes"] >= 250).astype(int)
    test_list_sgf.append(sgf_df)


    # ==================================================
    # 合并 & 保存
    # ==================================================
    final_test_sif  = pd.concat(test_list_sif, ignore_index=True) if test_list_sif else pd.DataFrame()
    final_test_sgf  = pd.concat(test_list_sgf, ignore_index=True) if test_list_sgf else pd.DataFrame()


    final_test_sif.to_csv(CONFIG["processed_dir"]/f"sif.csv", index=False)
    final_test_sgf.to_csv(CONFIG["processed_dir"]/f"sgf.csv", index=False)

    #print("\n" + "="*30)
    #print(f"SIF   Test: {len(final_test_sif)}")
    #print(f"SGF   Test: {len(final_test_sgf)}")
    #print("="*30)

from pathlib import Path
# 将csv文件统计到新的内容中
def extract_rdkit_features(csv_path: Path, output_dir: Path, featurizer):
    """
    从CSV提取RDKit特征并保存为NPZ
    
    Args:
        csv_path: 输入CSV文件路径
        output_dir: 输出目录
        featurizer: PeptideFeaturizer实例
    
    Returns:
        dict: 统计信息
    """
    # 加载CSV
    df, _ = load_csv_safely(csv_path, required_columns=["id", "SMILES", "SIF_minutes", "SGF_minutes"])
    if df is None:
        return {"error": "Failed to load CSV"}
    
    X = []
    y_sif = []
    y_sgf = []
    ids = []
    valid_count = 0
    
    # 提取特征
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"提取特征 {csv_path.name}", leave=False):
        smiles = str(row["SMILES"])
        features, success = featurizer.featurize(smiles)
        
        if success and features is not None:
            X.append(features)
            y_sif.append(int(row["SIF_minutes"]) if not pd.isna(row["SIF_minutes"]) else -1)
            y_sgf.append(int(row["SGF_minutes"]) if not pd.isna(row["SGF_minutes"]) else -1)
            ids.append(str(row["id"]))
            valid_count += 1
    
    # 转换为NumPy数组
    X = np.array(X, dtype=np.float32)
    y_sif = np.array(y_sif, dtype=np.int32)
    y_sgf = np.array(y_sgf, dtype=np.int32)
    ids = np.array(ids, dtype=object)
    feature_names = featurizer.get_feature_names()
    
    # 保存NPZ
    output_path = output_dir / csv_path.name.replace('.csv', '.npz')
    np.savez_compressed(
        output_path,
        X=X,
        y_sif=y_sif,
        y_sgf=y_sgf,
        ids=ids,
        feature_names=feature_names,
    )
    
    return {
        "file": csv_path.name,
        "total_samples": len(df),
        "valid_samples": valid_count,
        "feature_dim": X.shape[1],
        "output_path": output_path,
    }






# ============测试区域============这样应该是没问题的了
def test():
    print('测试')


def process_by_raw_csv(raw_path):
    # 根据输入数据整理内容
    process_and_merge_datasets(raw_path)
    feature_stats = []
    for task in tasks:
        task_name = task["name"]
        featurizer = task["featurizer"]
        csv_path = CONFIG['processed_dir'] / f"{task_name.lower()}.csv"
        stats = extract_rdkit_features(csv_path, CONFIG['features_dir'], featurizer)
        if "error" not in stats:
            feature_stats.append(stats)
            # print(f"✓ {stats['file']}: {stats['valid_samples']} samples, {stats['feature_dim']} features")
        else:
            print(f"✗ {task_name} 处理失败：{stats.get('error')}")

    # 汇总
    feat_summary_df = pd.DataFrame(feature_stats)
    #print(f'========================数据{raw_path}处理成功=================================')