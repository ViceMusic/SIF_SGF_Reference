# 检测脚本以及说明，用于检测该文件夹中所有的npy_data数据全都符合理想情况
# Morgan分子指纹数目为1024，Avalon分子指纹数目为512，如果啥都没有应该是20多个


#---------------该cell是一些测试的东西，可以忽略掉--------------------------------- 
# 库函数
from pathlib import Path
import numpy as np

CONFIG = {
    'npy_datas_dir': Path().resolve().parents[0] / 'src' / 'npy_datas',
    'dataset_names': ['sif_sgf_second', 'US9624268', 'US9809623B2','US20140294902A1', 'WO2017011820A2'],
    'dataset': {
        'sif_sgf_second': 'sif_sgf_second_processed',
        'US9624268': 'US9624268_processed',
        'US9809623B2': 'US9809623B2_processed',
        'US20140294902A1': 'US20140294902A1_processed',
        'WO2017011820A2': 'WO2017011820A2_processed',
    },
    'abnormal_point':700,
    'threshold'   : {
        'sif': 270,
        'sgf': 250,
    },
    'exclude':{  # 需要排除的类型
        'Avalon':"NoAvalon",
        'Morgan':"NoMorgan",
        'None':"Overall",
        'All':'NoMorganAndAvalon'
    }

}

#print("加载数据成功")
#print("判断所有数据均存在...")
#for dataset_name in CONFIG['dataset_names']:
#    dataset_dir = CONFIG['npy_datas_dir']/CONFIG['exclude']['Avalon']/ CONFIG['dataset'][dataset_name]
#    required_files = ['X.npy', 'y_sif.npy', 'y_sgf.npy', 'feature_names.npy']
#    for file_name in required_files:
#        file_path = dataset_dir / file_name
#        if not file_path.exists():
#            raise FileNotFoundError(f"缺少文件: {file_path}")
#    print(f"{dataset_name} 的npy数据文件存在。")
#print("所有数据文件均存在。")



# 该方法的返回值为筛选后的X（特定数据集，特定的任务，以及异常值筛选），二值化后的y（根据特定任务进行阈值划分），以及特征名称列表

def load_data_from_npy(
    dataset: str,      # 数据集的名称即可，_processed之类的后缀不需要,代码里面会加上的
    target: str,             # "SIF" or "SGF"，大写与否都无所谓
    exclude: str
):
    """
    从 NPY 文件中读取数据，根据任务类型和 is_monomer 进行筛选，
    并对标签进行二值化处理。

    Returns
    -------
    X_filtered : np.ndarray, shape (M, D)
    y_binary   : np.ndarray, shape (M,)
    feature_names : np.ndarray or list
    """
    dataset_name=CONFIG['dataset'][f'{dataset}']
    #print(dataset_name)

    # 1.读取数据
    dataset_dir = CONFIG['npy_datas_dir'] /CONFIG['exclude'][f"{exclude}"]/dataset_name
    X = np.load(dataset_dir / "X.npy")
    y_sif = np.load(dataset_dir / "y_sif.npy")
    y_sgf = np.load(dataset_dir / "y_sgf.npy")
    feature_names = np.load(dataset_dir / "feature_names.npy", allow_pickle=True)

    # 一致性检查，如果不通过就报错
    n = X.shape[0]
    assert y_sif.shape[0] == n
    assert y_sgf.shape[0] == n

    # 2. 找到 is_monomer 在 X 中对应的列号-----这个后面根据序号进行扩充吧,该方法已经被完全废弃，因为筛选已经移动到csv中
    #feature_names = list(feature_names)
    #if "is_monomer" not in feature_names:
    #    raise ValueError("feature_names 中未找到 'is_monomer' 特征")
    #monomer_col_idx = feature_names.index("is_monomer")

    y_raw = None
    threshold = None
    abnormal_point = CONFIG['abnormal_point']   # 异常值

    # 3. 选择任务标签
    if target.upper() == "SIF":
        y_raw = y_sif
        threshold = CONFIG['threshold']['sif']
    elif target.upper() == "SGF":
        y_raw = y_sgf
        threshold = CONFIG['threshold']['sgf']
    else:
        raise ValueError("target 必须是 'SIF' 或 'SGF'")

    
    # ------------------------------------------------------------------
    # 4和5用了三个很有意思的特性， csv转npy时候的自动转化操作，numpy的筛选操作，多重叠加的按位与操作
    # ------------------------------------------------------------------

    # 4. 构造样本筛选 mask
    valid_mask = np.ones(n, dtype=bool)  # 生成的原址数值全都为1

    # 4.1 根据 is_monomer 过滤（如果指定），该方法已经被完全废弃，因为筛选已经移动到csv中
    #if is_monomer is not None:
    #    valid_mask &= (X[:, monomer_col_idx] == int(is_monomer)) # 如果是单体，则筛选出单体样本，否则筛选出非单体样本

    # 4.2 过滤无效标签(mask是按位与操作)
    valid_mask &= (y_raw != -1)   # 添加筛选条件，若为-1则过滤
    valid_mask &= ~np.isnan(y_raw) # 添加筛选条件，如果是nan则过滤掉（~为numpy中的取反操作）
    valid_mask &= (y_raw <= abnormal_point)  # 添加筛选条件，若大于异常值则过滤掉

    # 5. 应用筛选（按位置删除）
    X_valid = X[valid_mask]
    y_valid = y_raw[valid_mask]

    #print(f"筛选后样本数: {X_valid.shape[0]} / {X.shape[0]}")

    # 6. 二值化标签，半衰期大于阈值则视为稳定，否则视为不稳定
    y_binary = (y_valid >= threshold).astype(np.int32)

    #print("============================================================================")
    #print(f"根据任务{target}筛选数据集 {dataset_name} 后的统计信息,去除分子指纹为{exclude}:")
    #print(f"  样本数: {X_valid.shape[0]}")
    #print(f"  本次筛选设定阈值为: {threshold} 分钟，异常值为{abnormal_point}分钟")
    #print(f"  稳定/不稳定: {(y_binary == 1).sum()} / {(y_binary == 0).sum()}")
    #print(f"    符合该条件的数据条数：{X_valid.shape[0]}，特征维度: {X_valid.shape[1]}")
    #print(f"    特征名称示例: {feature_names[:5]} ...")
    #print(f"    {target}半衰期范围: {y_valid.min()} - {y_valid.max()}")
    #print("============================================================================")

    result={
        "X":X_valid,
        "y":y_valid,
        "y_binary":y_binary,
        "feature_names":feature_names
    }

    return result #,X_valid,y_valid, y_binary, feature_names





