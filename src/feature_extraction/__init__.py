"""
feature_extraction 模块

用于肽类分子 SIF/SGF 稳定性的特征提取和预测。

主要功能：
- 分子特征化（Morgan 指纹、Avalon 指纹、QED 属性、物理化学描述符）
- 批量特征提取
- 结果保存为 NumPy 格式

主要导出：
- PeptideFeaturizer: 肽类特征提取器
"""

import logging

from .featurizer import PeptideFeaturizer

__version__ = "0.1.0"

# 配置日志
logger = logging.getLogger(__name__)

__all__ = [
    "PeptideFeaturizer",
]

