"""简化版推理脚本

功能：直接读取包含 `SMILES` 列的 CSV 文件，对每个 SMILES 逐条做特征提取，
并使用已保存的模型对 SIF/SGF 两个任务分别输出 0/1 预测。

使用示例：
    from pathlib import Path
    results = predict_from_csv(Path('data/raw/myfile.csv'))

返回：list，每个元素为 dict，包含 'SMILES','SIF','SGF' 三个字段
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import joblib
import re
from tqdm import tqdm

from feature_extraction import PeptideFeaturizer


# 配置（仅推理相关）
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = (PROJECT_ROOT / 'data' / 'weights').resolve()

# 默认位宽设置
DEFAULT_BITS = {
    'avalon_bits': 512,
    'molformer_bits': 768,
    'molt5_bits': 768,
}

# 推理时使用的任务和默认模型/表征配置（可按需修改）
INFERENCE_CONFIG = {
    'TASKS': ['SIF', 'SGF'],
    'EXECUTE': {
        'SIF': {'model': 'lr', 'para': {'PRINTS': 'Avalon(512)_molFormer(768)', 'lr_max_iter': 120}},
        'SGF': {'model': 'lr', 'para': {'PRINTS': 'Avalon(512)_MolT5(768)', 'lr_max_iter': 120}},
    }
}


def _select_bits_from_prints(print_name: str):
    """从 PRINTS 名称判断应启用哪些位（Avalon/molFormer/MolT5）。"""
    flags = {'avalon_bits': 0, 'molformer_bits': 0, 'molt5_bits': 0}
    if 'Avalon' in print_name:
        flags['avalon_bits'] = DEFAULT_BITS['avalon_bits']
    if re.search(r'molFormer', print_name, re.IGNORECASE):
        flags['molformer_bits'] = DEFAULT_BITS['molformer_bits']
    if re.search(r'MolT5', print_name, re.IGNORECASE):
        flags['molt5_bits'] = DEFAULT_BITS['molt5_bits']
    return flags


def _param_str_from_para(model_name: str, para: dict) -> str:
    if model_name == 'lr':
        return f"maxiter{para.get('lr_max_iter', 0)}"
    if model_name == 'rf':
        return f"nest{para.get('rf_n_estimators', 0)}_jobs{para.get('rf_n_jobs', -1)}"
    if model_name == 'xgb':
        return (
            f"depth{para.get('xgb_max_depth', 0)}_"
            f"lr{para.get('xgb_learning_rate', 0)}_"
            f"nest{para.get('xgb_n_estimators', 0)}"
        )
    return 'default'


def _load_model_for_task(task: str):
    cfg = INFERENCE_CONFIG['EXECUTE'].get(task)
    if cfg is None:
        return None, None
    model_name = cfg['model']
    para = cfg['para']
    feature_name = para.get('PRINTS', 'unknown')
    param_str = _param_str_from_para(model_name, para)
    filename = f"{model_name}_{task.lower()}_{feature_name}_{param_str}.pkl"
    model_path = MODEL_DIR / filename
    if not model_path.exists():
        return None, str(model_path)
    model = joblib.load(model_path)
    print(f"模型加载成功{model_path}")
    return model, str(model_path)


def _featurize_smiles_list(smiles_list, featurizer: PeptideFeaturizer, desc: str = None):
    """将 SMILES 列表批量 featurize，返回特征矩阵和失败索引列表。

    参数 `desc` 用于 tqdm 显示进度描述。
    """
    X = []
    fail_idx = []
    # 预取特征名长度以便失败占位
    try:
        dim = len(featurizer.get_feature_names())
    except Exception:
        dim = 0

    iterator = enumerate(smiles_list)
    if desc is not None:
        iterator = tqdm(iterator, total=len(smiles_list), desc=desc)

    for i, s in iterator:
        try:
            vec, ok = featurizer.featurize(s)
            if not ok or vec is None:
                fail_idx.append(i)
                X.append(np.zeros(dim, dtype=np.float32))
            else:
                X.append(np.array(vec, dtype=np.float32))
        except Exception:
            fail_idx.append(i)
            X.append(np.zeros(dim, dtype=np.float32))

    if len(X) == 0:
        return np.zeros((0, 0), dtype=np.float32), fail_idx
    X = np.vstack(X)
    return X, fail_idx


def predict_from_csv(csv_path):
    """主入口：读取 CSV（必须含 `SMILES` 列），对每条 SMILES 输出 SIF/SGF 0/1 预测。

    返回: list of dict: {'SMILES': str, 'SIF': 0/1/None, 'SGF': 0/1/None}
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'SMILES' not in df.columns:
        raise ValueError('Input CSV must contain a SMILES column named exactly "SMILES"')

    smiles_list = df['SMILES'].astype(str).tolist()

    # 按 PRINTS 分组，避免对同一表征重复 featurize
    prints_to_tasks = {}
    for task in INFERENCE_CONFIG['TASKS']:
        cfg = INFERENCE_CONFIG['EXECUTE'].get(task)
        para = cfg['para']
        prints = para.get('PRINTS', '')
        prints_to_tasks.setdefault(prints, []).append(task)

    # 为每个唯一 PRINTS 执行一次 featurize（带进度条），并缓存结果
    features_cache = {}
    for prints, tasks_for_prints in prints_to_tasks.items():
        bits = _select_bits_from_prints(prints)
        featurizer = PeptideFeaturizer(**bits)
        desc = f"Featurize {prints}"
        X, fails = _featurize_smiles_list(smiles_list, featurizer, desc=desc)
        features_cache[prints] = X

    # 加载所有 task 的模型（并报告缺失）
    models = {}
    for task in INFERENCE_CONFIG['TASKS']:
        model, path_or_err = _load_model_for_task(task)
        if model is None:
            print(f"⚠ 模型未找到 for {task}: {path_or_err}")
        else:
            print(f"✓ loaded model for {task}: {path_or_err}")
        models[task] = model

    results = [ {'SMILES': s, 'SIF': None, 'SGF': None} for s in smiles_list ]

    # 对每个 task 使用对应的缓存特征进行预测，并展示分配进度条
    for prints, tasks_for_prints in prints_to_tasks.items():
        X = features_cache.get(prints)
        for task in tasks_for_prints:
            model = models.get(task)
            if model is None:
                for i in range(len(results)):
                    results[i][task] = None
                continue

            try:
                preds = model.predict(X)
            except Exception as e:
                print(f"⚠ 预测失败 ({task}): {e}")
                preds = [None] * len(results)

            if len(preds) != len(results):
                preds = list(preds) + [None] * (len(results) - len(preds))

            # 分配预测结果时显示进度条
            for i, p in tqdm(enumerate(preds), total=len(preds), desc=f"Assign {task}"):
                try:
                    results[i][task] = int(p) if p is not None else None
                except Exception:
                    results[i][task] = None

    return results


'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict SIF/SGF from CSV with SMILES column')
    parser.add_argument('csv', help='input CSV path (must contain SMILES column)')
    args = parser.parse_args()
    out = predict_from_csv(args.csv)
    print(out)

'''