# 推理逻辑，对应原本的phase3

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
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# 机器学习
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score,log_loss
)
from xgboost import XGBClassifier

# 检查GPU可用性
try:
    import torch
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ GPU不可用，将使用CPU训练")
except ImportError:
    gpu_available = False
    print("⚠ PyTorch未安装，将使用CPU训练")

# 设置显示选项
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (12, 6)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# 对象转换的方法，也放在这里
def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# 配置参数区域
CONFIG = {

    # -------------------------------------文件读取路径--------------------------------------
    # 文件读取
    'processed_dir': project_root/'data'/'csv',
    'features_dir': project_root/'data'/'npz',
    # 权重获取地点
    'Train_weight_result':project_root/'data'/'weights',
    



    # -------------------------------------手动设置区域-----------------------------------------------------

    # 需要执行的任务包括
    'TASKS':["SIF","SGF"],
    'FEATURES': [ # 该部分作为一个备份，旨在找不到东西的时候可以来这里面看看
        {   # Morgan fingerprint（ECFP-like，强基线，1024 bits）
            'PRINTS': "Morgan(1024)",
            'lr_max_iter': 160,
            'rf_n_estimators': 37, 'rf_n_jobs': -1,
            'xgb_max_depth': 5, 'xgb_learning_rate': 0.1, 'xgb_n_estimators': 200,
        },
        {   # Avalon fingerprint（substructure-based，512 bits）
            'PRINTS': "Avalon(512)",
            'lr_max_iter': 160,
            'rf_n_estimators': 9, 'rf_n_jobs': -1,
            'xgb_max_depth': 6, 'xgb_learning_rate': 0.1, 'xgb_n_estimators': 100,
        },
        {   # molFormer（SMILES Transformer 预训练表示，768 dim）
            'PRINTS': "molFormer(768)",
            'lr_max_iter': 500,
            'rf_n_estimators': 21, 'rf_n_jobs': -1,
            'xgb_max_depth': 4, 'xgb_learning_rate': 0.05, 'xgb_n_estimators': 300,
        },
        {   # MolT5（基于 T5 的 SMILES 预训练模型，768 dim）
            'PRINTS': "MolT5(768)",
            'lr_max_iter': 1000,
            'rf_n_estimators': 113, 'rf_n_jobs': -1,
            'xgb_max_depth': 4, 'xgb_learning_rate': 0.05, 'xgb_n_estimators': 300,
        },
        {   # HashMorgan（Hash 版本的 Morgan 指纹，1024 bits）
            'PRINTS': "HashMorgan(1024)",
            'lr_max_iter': 80,
            'rf_n_estimators': 253, 'rf_n_jobs': -1,
            'xgb_max_depth': 6, 'xgb_learning_rate': 0.03, 'xgb_n_estimators': 600,
        },
        {   # 分子描述符（物化性质，220 dim，连续特征）
            'PRINTS': "MolDescriptor(220)",
            'lr_max_iter': 0,
            'rf_n_estimators': 9, 'rf_n_jobs': -1,
            'xgb_max_depth': 4, 'xgb_learning_rate': 0.05, 'xgb_n_estimators': 300,
        },
        # ===================== 组合表征 =====================
        {   # Avalon + molFormer（结构指纹 + 预训练语义）
            'PRINTS': "Avalon(512)_molFormer(768)",
            'lr_max_iter': 120,
            'rf_n_estimators': 33, 'rf_n_jobs': -1,
            'xgb_max_depth': 6, 'xgb_learning_rate': 0.1, 'xgb_n_estimators': 100,
        },
        {   # Avalon + MolT5（结构指纹 + T5 语义表示）
            'PRINTS': "Avalon(512)_MolT5(768)",
            'lr_max_iter': 120,
            'rf_n_estimators': 57, 'rf_n_jobs': -1,
            'xgb_max_depth': 4, 'xgb_learning_rate': 0.05, 'xgb_n_estimators': 300,
        },
        {   # Avalon + molFormer + MolT5（三种信息融合）
            'PRINTS': "Avalon(512)_molFormer(768)_MolT5(768)",
            'lr_max_iter': 500,
            'rf_n_estimators': 61, 'rf_n_jobs': -1,
            'xgb_max_depth': 5, 'xgb_learning_rate': 0.1, 'xgb_n_estimators': 200,
        },
        {   # Avalon + HashMorgan（两种结构型指纹组合）
            'PRINTS': "Avalon(512)_HashMorgan(1024)",
            'lr_max_iter': 80,
            'rf_n_estimators': 9, 'rf_n_jobs': -1,
            'xgb_max_depth': 6, 'xgb_learning_rate': 0.03, 'xgb_n_estimators': 600,
        },
        {   # Avalon + 分子描述符（离散结构 + 连续物化性质）
            'PRINTS': "Avalon(512)_MolDescriptor(220)",
            'lr_max_iter': 0,
            'rf_n_estimators': 9, 'rf_n_jobs': -1,
            'xgb_max_depth': 4, 'xgb_learning_rate': 0.05, 'xgb_n_estimators': 300,
        },
        {   # molFormer + MolT5（两种预训练语义表示）
            'PRINTS': "molFormer(768)_MolT5(768)",
            'lr_max_iter': 300,
            'rf_n_estimators': 261, 'rf_n_jobs': -1,
            'xgb_max_depth': 6, 'xgb_learning_rate': 0.1, 'xgb_n_estimators': 100,
        },
        {   # molFormer + HashMorgan（语义表示 + 结构指纹）
            'PRINTS': "molFromer(768)_HashMorgan(1024)",
            'lr_max_iter': 80,
            'rf_n_estimators': 253, 'rf_n_jobs': -1,
            'xgb_max_depth': 6, 'xgb_learning_rate': 0.03, 'xgb_n_estimators': 600,
        },
        {   # molFormer + 分子描述符（语义表示 + 连续特征）
            'PRINTS': "molFormer(768)_MolDescriptor(220)",
            'lr_max_iter': 0,
            'rf_n_estimators': 293, 'rf_n_jobs': -1,
            'xgb_max_depth': 4, 'xgb_learning_rate': 0.05, 'xgb_n_estimators': 300,
        },
        {   # HashMorgan + 分子描述符（结构指纹 + 连续物化）
            'PRINTS': "HashMorgan(1024)_MolDescriptor(220)",
            'lr_max_iter': 0,
            'rf_n_estimators': 9, 'rf_n_jobs': -1,
            'xgb_max_depth': 4, 'xgb_learning_rate': 0.05, 'xgb_n_estimators': 300,
        },
    ],
    # 测试执行：
    'EXECUTE':{
        'SIF':{
            'model':'lr',
            'para': {   # Avalon + molFormer（结构指纹 + 预训练语义）
                'PRINTS': "Avalon(512)_molFormer(768)",
                'lr_max_iter': 120,
                'rf_n_estimators': 33, 'rf_n_jobs': -1,
                'xgb_max_depth': 6, 'xgb_learning_rate': 0.1, 'xgb_n_estimators': 100,
            },

        },
        'SGF':{
            'model':'lr',
            'para': {   # Avalon + MolT5（结构指纹 + T5 语义表示）
                'PRINTS': "Avalon(512)_MolT5(768)",
                'lr_max_iter': 120,
                'rf_n_estimators': 57, 'rf_n_jobs': -1,
                'xgb_max_depth': 4, 'xgb_learning_rate': 0.05, 'xgb_n_estimators': 300,
            },

        }
    },

    # 掩码文件的所在位置（要求，特定任务上，无需使用的特征为0）
    'mask':"_error__importance_mask.json",




    # -------------------------------------超参数验证区域--------------------------------------------------

    # 模型选择（可选: 'lr', 'rf', 'xgb'）
    'models_to_train': ['lr', 'rf', 'xgb'],
    
    # 交叉验证参数
    'n_folds': 5,
    'random_state': 42,

    # 另外三个参数将会在后面想办法设置
    # Logistic Regression参数
    # Random Forest参数
    # XGBoost参数
    # 阈值
    'threshold':{
        'SIF':270,
        'SGF':250
    },
    # 是否只提取单体分子
    'is_monomer': True,


}

# 选择模型：这里已经强制了随机种子1-42
def get_model(model_name: str, config):
    
    # CONFIG['random_state']=random.randint(1, 1000000)
    # print("选择模型为", model_name, "随机种子", CONFIG['random_state'])
    """
    创建模型实例
    """
    if model_name == 'lr':
        return LogisticRegression(
            max_iter=config['lr_max_iter'],
            class_weight='balanced',
            random_state=CONFIG['random_state'] #random.randint(1, 1000000)
        )
    elif model_name == 'rf':
        return RandomForestClassifier(
            n_estimators=config['rf_n_estimators'],
            class_weight='balanced',
            n_jobs=config['rf_n_jobs'],
            random_state=CONFIG['random_state'] #random.randint(1, 1000000)
        )
    elif model_name == 'xgb':
        params = {
            'max_depth': config['xgb_max_depth'],
            'learning_rate': config['xgb_learning_rate'],
            'n_estimators': config['xgb_n_estimators'],
            'random_state': CONFIG['random_state'], #random.randint(1, 1000000),
            'tree_method': 'hist',
        }
        if gpu_available:
            params['device'] = 'cuda:0'
        return XGBClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# 进行独立验证的函数，只验证测试内容
def independent_validate(X_test, y_test, model_name: str, config, task):
    """
    在训练集上进行全量训练，在独立的测试集上进行单次验证
    """
    # 检查类别情况，防止只有单类别无法计算指标
    unique_classes = np.unique(y_test)
    if len(unique_classes) < 2:
        print(f"    ⚠ 测试集只有类别 {unique_classes}，无法进行二分类评估，跳过该任务")
        return None

    # --- 核心训练过程 ---
    model = get_model(model_name, config)

    # 获取表征的名称
    featureName=config['PRINTS']

    # 训练以后保存模型的内容如下
    if model_name == 'lr':
        param_str = f"maxiter{config['lr_max_iter']}"
    elif model_name == 'rf':
        param_str = f"nest{config['rf_n_estimators']}_jobs{config['rf_n_jobs']}"
    elif model_name == 'xgb':
        param_str = (
            f"depth{config['xgb_max_depth']}_"
            f"lr{config['xgb_learning_rate']}_"
            f"nest{config['xgb_n_estimators']}"
        )
    # 2. 尝试不要训练而是直接加载模型信息
    weight_path = Path(CONFIG['Train_weight_result']) / \
        f"{model_name}_{task}_{featureName}_{param_str}.pkl"

    if not weight_path.exists():
        raise FileNotFoundError(f"Model weight not found: {weight_path}")

    # 直接加载已训练好的模型
    model = joblib.load(weight_path)

    print(f"Loaded model from: {weight_path}")

    
    
    # --- 预测过程 ---
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    
    if proba.shape[1] == 2:
        y_proba = proba[:, 1]
    else:
        y_proba = None

    # --- 计算指标 ---
    # 为了保持输出结构一致，我们将结果存入列表（虽然只有一个值）
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'AUPRC':0,
        'auc': 0
    }
    
    if y_proba is not None and len(unique_classes) > 1:
        metrics['auc']=roc_auc_score(y_test, y_proba)
    else:
        metrics['auc']=np.nan
    
    if y_proba is not None and len(np.unique(y_test)) > 1:
        metrics['AUPRC']=average_precision_score(y_test, y_proba)
    else:
        metrics['AUPRC']=np.nan

    # 补充预测，概率是否为贴地飞行
    proba = model.predict_proba(X_test)
    y_proba = proba[:, 1]

    metrics['proba_stats'] = {
        "min": float(np.min(y_proba)),
        "mean": float(np.mean(y_proba)),
        "max": float(np.max(y_proba)),
        'pos_ratio':y_test.mean(),
        'sanity':abs(float(np.mean(y_proba))- y_test.mean()),
        'ce_loss':float(log_loss(y_test, y_proba)) if len(np.unique(y_test))>1 else float(-np.mean(np.log(1 - y_proba)))
    }
    
    return metrics


# 进行数据二值化
def load_and_binarize_dataset(npz_path: Path, csv_path: Path, target: str, is_monomer: bool = None):
    """
    加载数据并将标签二值化，采用索引顺序匹配而非ID匹配
    (2026-01-16 更新: 解决ID重复问题，直接按行顺序对应)
    
    Args:
        npz_path: NPZ特征文件 (包含 X, y_sif, y_sgf 等)
        csv_path: 处理后的CSV文件 (包含标签和筛选列)
        target: 'SIF' or 'SGF'
        is_monomer: 如果为True，只保留monomer；False，只保留非monomer；None不筛选
    
    Returns:
        X_valid, y_binary, median, feature_names
    """
    # 1. 加载NPZ特征
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    feature_names = data['feature_names']
    # 注意：这里不再提取 ids_npz，因为我们直接按物理顺序读取
    
    # 2. 加载CSV获取标签及筛选条件
    df = pd.read_csv(csv_path)
    
    # 3. 顺序匹配逻辑
    # 既然 npz 的 X 属性与 csv 默认顺序对应，我们直接根据 df 的索引进行筛选
    valid_indices = []
    valid_labels = []
    
    label_col = f"{target}_minutes"
    
    # 使用 zip 或直接遍历索引，确保 X 的第 i 行对应 df 的第 i 行
    for i, row in df.iterrows():
        # 获取当前行的标签
        label = row[label_col]
        
        # =================================== 过滤操作, 虽然已经操作过了 ===================================
        # 1. 标签有效性判断
        if label == -1 or pd.isna(label):
            continue
            
        # 2. monomer 条件筛选
        if is_monomer is not None and row['is_monomer'] != is_monomer:
            continue
            
        # 3. 特殊阈值处理 (SIF/SGF_minutes > 700 排除)
        if label > 700:
            continue
        # ===============================================================================
        
        # 记录通过筛选的行索引和对应的标签
        valid_indices.append(i) # 选择合理的X
        valid_labels.append(label) # 选择合理的y
    
    # 4. 筛选有效样本
    # 利用 numpy 的高级索引，根据保存的行索引一次性提取对应的特征行
    X_valid = X[valid_indices]
    y_minutes = np.array(valid_labels)

    
    
    # 5. 设定阈值并二值化
    # 假设 CONFIG 已经在全局定义，如果没有，请确保在此处能访问到
    median = None
    if target == 'SIF':
        median = CONFIG['threshold']['SIF']
    elif target == 'SGF':
        median = CONFIG['threshold']['SGF']
    else:
        raise ValueError(f"Unknown target: {target}")

    # 1 = 稳定, 0 = 不稳定
    y_binary = (y_minutes >= median).astype(int)
    
    print(f"--- 数据加载完成 ---")
    print(f"  有效样本数: {len(X_valid)}")
    print(f"  任务目标: {target} (中位数阈值: {median:.1f})")
    print(f"  类别分布 (1/0): {np.sum(y_binary == 1)} / {np.sum(y_binary == 0)}")
    
    return X_valid, y_binary, median, feature_names


    # 数据集来源要求有限制，具体详见config.json

def predict_by_processed_data():
    # 准备数据集
    datasets_data = {}
    # 进行数据获取
    # ===================== Batch loading =====================
    for task in CONFIG['TASKS']:
        task_lower = task.lower()

        test_npz  = CONFIG['features_dir'] / f"{task_lower}.npz"
        test_csv  = CONFIG['processed_dir'] / f"{task_lower}.csv"

        try:

            # -------- Test --------
            X_te, y_te, median_te, _ = load_and_binarize_dataset(
                test_npz,
                test_csv,
                task,
                CONFIG['is_monomer']
            )

            dataset_key_test  = test_npz.stem.replace('', '')


            datasets_data[dataset_key_test] = {
                f'X_{task_lower}': X_te,
                f'y_{task_lower}': y_te,
                f'median_{task_lower}': median_te,
            }

            print(f"✓ [{task}] 加载完成")

        except FileNotFoundError as e:
            print(f"⚠ 缺失文件，跳过: [{task}] ")
        except Exception as e:
            print(f"⚠ 加载失败 [{task}] : {e}")

    print(f"\n✅ 数据加载完成！")
    import matplotlib.pyplot as plt
    import numpy as np


    # 将任务拆分成两种
    for task in CONFIG["TASKS"]:
        config=CONFIG['EXECUTE'][task]
        test_name=f"{task.lower()}" # 获取任务名称
        test_data = datasets_data.get(test_name)

        if test_data:
            target=task
            X_te, y_te = test_data[f'X_{target.lower()}'], test_data[f'y_{target.lower()}']
            X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
            # 注意：这里填入的是修改后的
            metric= independent_validate( X_te, y_te, config['model'], config['para'], task.lower())
            print(metric)