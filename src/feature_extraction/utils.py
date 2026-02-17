"""
辅助函数模块

提供特征提取过程中的辅助工具函数。
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional, Union

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)


def validate_csv_columns(
    df: pd.DataFrame, required_columns: List[str]
) -> Tuple[bool, List[str]]:
    """
    验证数据框是否包含所有必需列。

    Args:
        df (pd.DataFrame): 待验证的数据框。
        required_columns (List[str]): 必需的列名列表。

    Returns:
        Tuple[bool, List[str]]: 
            - 第一个元素：所有必需列是否都存在（bool）
            - 第二个元素：缺失的列名列表（若所有列都存在，则为空列表）
    """
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


def get_csv_files(directory: Path) -> List[Path]:
    """
    获取目录下所有 CSV 文件。

    Args:
        directory (Path): 输入目录路径。

    Returns:
        List[Path]: CSV 文件路径列表（按修改时间排序）。
    """
    if not directory.exists():
        logger.error(f"目录不存在: {directory}")
        return []
    
    if not directory.is_dir():
        logger.error(f"路径不是目录: {directory}")
        return []
    
    csv_files = sorted(directory.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    logger.info(f"在 {directory} 中找到 {len(csv_files)} 个 CSV 文件")
    
    return csv_files


def load_csv_safely(
    csv_path: Path, required_columns: Optional[List[str]] = None
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    安全地加载 CSV 文件。

    Args:
        csv_path (Path): CSV 文件路径。
        required_columns (Optional[List[str]]): 必需的列名列表。
            若为 None，则不进行列验证。

    Returns:
        Tuple[Optional[pd.DataFrame], str]:
            - 第一个元素：加载的数据框，若加载失败则为 None
            - 第二个元素：状态信息字符串
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"成功加载 CSV 文件: {csv_path} ({len(df)} 行)")
        
        if required_columns is not None:
            valid, missing = validate_csv_columns(df, required_columns)
            if not valid:
                msg = f"缺失必需列: {', '.join(missing)}"
                logger.error(msg)
                return None, msg
        
        return df, "ok"
    
    except FileNotFoundError:
        msg = f"文件不存在: {csv_path}"
        logger.error(msg)
        return None, msg
    
    except pd.errors.EmptyDataError:
        msg = f"CSV 文件为空: {csv_path}"
        logger.error(msg)
        return None, msg
    
    except Exception as e:
        msg = f"加载 CSV 文件时出错 ({csv_path}): {e}"
        logger.error(msg)
        return None, msg


def build_output_path(
    input_path: Path, output_dir: Path, suffix: str = ".npz"
) -> Path:
    """
    根据输入文件路径和输出目录构建输出文件路径。

    Args:
        input_path (Path): 输入文件路径。
        output_dir (Path): 输出目录。
        suffix (str): 输出文件后缀。默认 ".npz"。

    Returns:
        Path: 输出文件路径。
    """
    stem = input_path.stem  # 不包含后缀的文件名
    output_path = output_dir / f"{stem}{suffix}"
    return output_path


def save_features_to_npz(
    output_path: Path,
    X: np.ndarray,
    y_sif: np.ndarray,
    y_sgf: np.ndarray,
    ids: np.ndarray,
    feature_names: np.ndarray,
    mask_valid: np.ndarray,
    metadata: dict,
) -> Tuple[bool, str]:
    """
    将特征和标签保存为 NPZ 文件。

    Args:
        output_path (Path): 输出文件路径。
        X (np.ndarray): 特征矩阵，shape: (n_samples, n_features)。
        y_sif (np.ndarray): SIF 类标签，shape: (n_samples,)。
        y_sgf (np.ndarray): SGF 类标签，shape: (n_samples,)。
        ids (np.ndarray): 样本 ID，shape: (n_samples,)。
        feature_names (np.ndarray): 特征名称列表，shape: (n_features,)。
        mask_valid (np.ndarray): 有效样本掩码，shape: (n_rows,)。
        metadata (dict): 元数据字典。

    Returns:
        Tuple[bool, str]:
            - 第一个元素：是否保存成功（bool）
            - 第二个元素：状态信息字符串
    """
    try:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_path,
            X=X,
            y_sif=y_sif,
            y_sgf=y_sgf,
            ids=ids,
            feature_names=feature_names,
            mask_valid=mask_valid,
            metadata=np.asarray(metadata, dtype=object),
        )
        
        msg = f"成功保存特征文件: {output_path} ({X.shape[0]} 个有效样本, {X.shape[1]} 个特征)"
        logger.info(msg)
        return True, msg
    
    except Exception as e:
        msg = f"保存特征文件时出错 ({output_path}): {e}"
        logger.error(msg)
        return False, msg


def format_batch_summary(
    total_files: int,
    successful_files: int,
    total_samples: int,
    valid_samples: int,
    invalid_samples: int,
) -> str:
    """
    格式化批量处理摘要信息。

    Args:
        total_files (int): 处理的文件总数。
        successful_files (int): 成功处理的文件数。
        total_samples (int): 总样本数。
        valid_samples (int): 有效样本数。
        invalid_samples (int): 无效样本数。

    Returns:
        str: 格式化的摘要信息。
    """
    summary = (
        f"\n{'='*60}\n"
        f"批量特征提取摘要\n"
        f"{'='*60}\n"
        f"处理文件数: {successful_files}/{total_files}\n"
        f"总样本数: {total_samples}\n"
        f"  - 有效样本: {valid_samples}\n"
        f"  - 无效样本: {invalid_samples}\n"
        f"{'='*60}\n"
    )
    return summary


# ============================================================================
# 分子特征检测函数
# ============================================================================

def detect_dimer(smiles: str) -> bool:
    """
    检测肽链是否为二聚体。

    检测策略：
    1. 分子量：二聚体通常 > 2000 Da
    2. SMILES 长度：二聚体通常很长 (> 400 字符)
    3. 连接基团：检测 PEG 连接子 (CCOCCOCC) 或特定连接模式

    Args:
        smiles (str): SMILES 字符串

    Returns:
        bool: True 表示二聚体，False 表示单体
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug(f"无法解析 SMILES (dimer detection): {smiles[:50]}...")
            return False

        # 策略 1: 分子量判断
        mw = Descriptors.MolWt(mol)
        if mw > 2500:  # 二聚体通常 > 2500 Da
            return True

        # 策略 2: SMILES 长度判断（辅助）
        if len(smiles) > 500:  # 长 SMILES 通常是二聚体或 PEG 化合物
            # 进一步检查 PEG 连接子
            if "CCOCCOCC" in smiles or "CCOCCOCCOC" in smiles:
                return True

        # 策略 3: 检测多个肽链特征（保守判断）
        # 计算肽键数量 (C(=O)N 模式)
        peptide_bonds = len(re.findall(r'C\(=O\)N', smiles))
        if peptide_bonds > 20 and mw > 2000:  # 大量肽键 + 高分子量
            return True

        return False

    except Exception as e:
        logger.debug(f"检测二聚体时出错: {e}")
        return False


def detect_cyclic(smiles: str) -> bool:
    """
    检测肽链是否含有环状结构。

    检测策略：
    1. RDKit 环计数
    2. SMILES 中的环闭合标记（数字对）

    Args:
        smiles (str): SMILES 字符串

    Returns:
        bool: True 表示含环，False 表示无环
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug(f"无法解析 SMILES (cyclic detection): {smiles[:50]}...")
            return False

        # 策略 1: RDKit 环计数
        num_rings = Descriptors.RingCount(mol)
        if num_rings > 0:
            return True

        # 策略 2: SMILES 环闭合标记检测（备用）
        # 环闭合用数字表示，如 C1CCCCC1 表示环己烷
        ring_closures = re.findall(r'\d+', smiles)
        if len(ring_closures) >= 2:  # 至少一对环闭合标记
            return True

        return False

    except Exception as e:
        logger.debug(f"检测环状结构时出错: {e}")
        return False


def detect_disulfide(smiles: str) -> bool:
    """
    检测肽链是否含有二硫键。

    检测策略：
    1. SMILES 中查找 "SS" 模式
    2. 使用 RDKit 子结构匹配

    Args:
        smiles (str): SMILES 字符串

    Returns:
        bool: True 表示含二硫键，False 表示无二硫键
    """
    try:
        # 策略 1: 简单模式匹配
        if "SS" in smiles:
            return True

        # 策略 2: RDKit 子结构匹配（更准确）
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug(f"无法解析 SMILES (disulfide detection): {smiles[:50]}...")
            return False

        # 二硫键 SMARTS 模式: [S]-[S]
        disulfide_pattern = Chem.MolFromSmarts("[S][S]")
        if disulfide_pattern is not None:
            if mol.HasSubstructMatch(disulfide_pattern):
                return True

        return False

    except Exception as e:
        logger.debug(f"检测二硫键时出错: {e}")
        return False


# ============================================================================
# 标签转换函数
# ============================================================================

def convert_label_to_minutes(label: Union[str, int, float]) -> int:
    """
    将各种标签编码统一转换为分钟（使用区间中点值）。

    转换规则：
    1. 星号系统 (*****～*):
       - ***** (< 60 min) → 30
       - **** (60-120 min) → 90
       - *** (120-180 min) → 150
       - ** (180-360 min) → 270
       - * (> 360 min) → 420

    2. 数字分类 (1-4):
       - 1 (< 1 hour / < 60 min) → 30
       - 2 (1-1.5 hours / 60-90 min) → 75
       - 3 (1.5-2 hours / 90-120 min) → 105
       - 4 (> 2 hours / > 120 min) → 150

    3. 直接分钟值: 保持不变

    4. 缺失值 (空值、"---"、NaN): → -1

    Args:
        label: 原始标签值（可以是字符串、整数或浮点数）

    Returns:
        int: 转换后的分钟值，缺失值返回 -1
    """
    # 处理缺失值
    if pd.isna(label) or label == "" or label == "---" or label is None:
        return -1

    # 转换为字符串处理
    label_str = str(label).strip()

    # 空字符串
    if not label_str:
        return -1

    # 处理星号系统（去掉可能的 "S" 后缀）
    label_clean = label_str.rstrip("S").strip()

    if label_clean == "*****":
        return 30
    elif label_clean == "****":
        return 90
    elif label_clean == "***":
        return 150
    elif label_clean == "**":
        return 270
    elif label_clean == "*":
        return 420

    # 处理数字分类 (1-4)
    try:
        num_val = int(float(label_clean))
        if num_val == 1:
            return 30
        elif num_val == 2:
            return 75
        elif num_val == 3:
            return 105
        elif num_val == 4:
            return 150
        # 如果是其他整数，尝试作为直接分钟值
        elif num_val >= 0:
            return num_val
        else:
            return -1
    except ValueError:
        # 无法转换为数字，标记为缺失
        logger.debug(f"无法转换标签为分钟值: '{label}'")
        return -1


def extract_molecular_features(smiles: str) -> dict:
    """
    提取单个 SMILES 的分子特征。

    Args:
        smiles (str): SMILES 字符串

    Returns:
        dict: 包含以下键的字典：
            - is_monomer (bool): 是否为单体
            - is_dimer (bool): 是否为二聚体
            - is_cyclic (bool): 是否含环
            - has_disulfide_bond (bool): 是否含二硫键
    """
    is_dimer_flag = detect_dimer(smiles)

    return {
        "is_monomer": not is_dimer_flag,
        "is_dimer": is_dimer_flag,
        "is_cyclic": detect_cyclic(smiles),
        "has_disulfide_bond": detect_disulfide(smiles),
    }

