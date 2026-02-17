"""
肽类分子特征提取器模块

提供 PeptideFeaturizer 类，用于从 SMILES 字符串提取分子特征，
包括 Morgan 指纹、Avalon 指纹、QED 属性和物理化学描述符。
"""
# ============特征拼接方法==============
'''
大致工作流程如下图所示
            ┌───────────┐
            │  SMILES   │
            └─────┬─────┘
                  │
        ┌─────────▼─────────┐
        │ RDKit 解析与校验   │  ← MolFromSmiles
        └─────────┬─────────┘
                  │
   ┌──────────────┼────────────────┐
   │              │                │
┌──▼───┐     ┌────▼────┐      ┌────▼────┐
│QED   │     │PhysChem │      │Gasteiger│
│8维   │     │11维     │      │5维      │
└──┬───┘     └────┬────┘      └────┬────┘
   │              │                │
   └──────────────┼────────────────┘
                  │
           ┌──────▼──────┐
           │ Morgan FP   │  ← 1024 bit
           └──────┬──────┘
                  │
        ┌─────────▼─────────┐
        │ Avalon FP (可选)  │  ← 512 bit
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │ ChemBERTa (可选)  │  ← 384 dim
        │ SMILES → embedding│
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │ HELM-BERT (可选)  │
        │ SMILES→HELM→emb   │ ← 768 dim
        └─────────┬─────────┘
                  │
            ┌─────▼─────┐
            │ 拼接特征  │
            └───────────┘


'''
# =========== work check list  ==============
'''
[ ] self.xxx_bits
    self.use_xxxx

[ ] __init__:
    - self.use_xxx
    - self.xxx_tokenizer
    - self.xxx_model

[ ] def _xxx_fingerprint()

[ ] try/except + zero vector fallback

[ ] featurize() 中 features.extend()

[ ] get_feature_names() 中声明

[ ] n_features += hidden_size


'''

# =========== 补充更新说明 ==============
'''
后续更新的其他预训练模型，会尽可能使用面向对象思维进行封装
统一要求：
- 模型的开关和参数在大类中使用
- 独立的模型类需要返回以下内容，其中包括
    - 初始化方法 __init__()
    - 特征提取方法 ()
    - 维度属性 n_features
'''

import logging
from typing import Optional, Tuple, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    AllChem, QED, rdMolDescriptors, Crippen, Descriptors
)

import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, T5ForConditionalGeneration
from unimol_tools import UniMolRepr
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import Chem, rdBase, DataStructs
from rdkit.Chem import AllChem


# 顺手把 UniMol Tools 也一起关了
logging.getLogger("Uni-Mol Tools").setLevel(logging.ERROR)

# 尝试导入 Avalon 指纹支持（可选）
try:
    from rdkit.Avalon import pyAvalonTools
    _HAS_AVALON = True
except (ImportError, ModuleNotFoundError):
    pyAvalonTools = None
    _HAS_AVALON = False

# 检查是否支持transformer和ChemBERTa
# 尝试检查环境是否支持 Transformers
try:
    from transformers import AutoTokenizer, AutoModel
    _HAS_TRANSFORMERS = True
except (ImportError, ModuleNotFoundError):
    _HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


# ====================== 封装好的肽类分子特征提取器 =====================

# == SmoleBert 特征提取器 ==
class SmoleBertFeaturizer:
    def __init__(self, 
                 model_name="UdS-LSV/smole-bert-guacamol-66", 
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def get_features(self, smiles: str):
        inputs = self.tokenizer(smiles, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        emb = outputs.last_hidden_state.mean(dim=1)

        return emb.squeeze(0).cpu().tolist()

    def get_dim(self):
        return 512  # SmoleBert hidden size
    
# == MolT5 特征提取器 ==
class MolT5Featurizer:
    def __init__(
            self, 
            model_name="laituan245/molt5-base", 
            device="cuda" if torch.cuda.is_available() else "cpu"
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=512
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name
        ).to(device)

        self.model.eval()
        self.device = device

    @torch.no_grad()
    def get_features(self, smiles: str):
        """
        输入: 单条 SMILES (str)
        输出: List[float]，长度 = hidden_dim
        """
        # -------- tokenize（单条，不需要 padding 也可以） --------
        inputs = self.tokenizer(
            smiles,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        # -------- encoder forward --------
        encoder_outputs = self.model.encoder(**inputs)
        hidden = encoder_outputs.last_hidden_state  # (1, L, D)

        # -------- mask-aware mean pooling --------
        mask = inputs["attention_mask"].unsqueeze(-1)  # (1, L, 1)
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, D)

        # -------- 去掉 batch 维 + 转成 list --------
        emb = emb.squeeze(0).cpu().tolist()  # (D,)

        return emb

    def get_dim(self):
        return 768  # MolT5 base hidden size

# == UniMol 特征提取器 ==
class UniMolFeaturizer:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = UniMolRepr(
            data_type="molecule",
            remove_hs=False,
            device=device
        )

    def get_features(self, smile: str):
        """
        输入: SMILES 字符串
        输出: 分子级 embedding (list[float])
        """
        embedding = self.model.get_repr(
            [smile],
            return_atomic_reprs=False
        )
        return embedding[0]
    def get_dim(self):
        return 512  # UniMol 分子级别特征维度固定为 512

# == HashMorgan特征提取器(来自HelmBert) == 
class HashMorganFeaturizer:
    def __init__(self, radius=3, size=1024):
        self.radius = radius
        self.size = size

    def get_features(self, smile: str):
        """
        输入: SMILES 字符串
        输出: 分子级 HashMorgan embedding (list[float])
        """
        mol = Chem.MolFromSmiles(smile)
        
        # 如果 SMILES 无效，返回全零向量，确保后续模型不报错
        if mol is None:
            return [0.0] * self.size

        # 生成 Hashed Morgan 指纹
        # 这里使用 GetHashedMorganFingerprint 以获得更好的特征分布
        fp_bits = AllChem.GetHashedMorganFingerprint(mol, self.radius, nBits=self.size)
        
        # 转换为 NumPy 数组
        fp_np = np.zeros((self.size,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp_bits, fp_np)
        
        # 转回 list 以保持和 UniMolFeaturizer 的输出格式一致
        return fp_np.tolist()

    def get_dim(self):
        return self.size

# == MolDescriptor补充理化性质 ================
class MolDescriptorFeaturizer:
    def __init__(self, missing_val=0):
        self.missing_val = missing_val
        # 预先获取名称列表，确定维度
        _, self.names = self._calculate(Chem.MolFromSmiles("C")) 

    def _calculate(self, mol):
        """ 内部核心计算方法 """
        values, names = [], []
        if mol is None:
            return None, None
            
        # 1. RDKit 标准描述符 (约 200+ 个)
        for nm, fn in Descriptors._descList:
            try:
                val = fn(mol)
            except:
                val = self.missing_val
            values.append(float(val))
            names.append(nm)

        # 2. 自定义 Lipinski 参数
        custom_descriptors = {
            'hydrogen-bond donors': rdMolDescriptors.CalcNumLipinskiHBD,
            'hydrogen-bond acceptors': rdMolDescriptors.CalcNumLipinskiHBA,
            'rotatable bonds': rdMolDescriptors.CalcNumRotatableBonds,
        }
        
        for nm, fn in custom_descriptors.items():
            try:
                val = fn(mol)
            except:
                val = self.missing_val
            values.append(float(val))
            names.append(nm)
            
        return values, names

    def get_features(self, smile: str):
        """
        输入: SMILES 字符串
        输出: 理化性质 embedding (list[float])
        """
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            # 返回与维度匹配的全零列表
            return [float(self.missing_val)] * self.get_dim()
            
        values, _ = self._calculate(mol)
        return values

    def get_dim(self):
        return len(self.names)

    def get_names(self):
        """ 返回描述符的名称列表，方便后期做特征解释 """
        return self.names

# ======================================================================

class PeptideFeaturizer:
    """
    肽类分子特征提取器。

    用于从 SMILES 字符串提取分子特征，包括：
    - QED 属性（分子量、LogP、HBA、HBD、PSA、旋转键数、芳香性、警报）
    - 物理化学描述符（脂溶性、刚性、分子大小等）
    - Gasteiger 电荷统计
    - Morgan 指纹（位向量）
    - Avalon 指纹（可选）

    Attributes:
        morgan_bits (int): Morgan 指纹位数（默认 1024）
        avalon_bits (int): Avalon 指纹位数（默认 512）
        use_avalon (bool): 是否使用 Avalon 指纹（默认 True）
    """

    # ------- 其他配置 --------

    def __init__(
        self,
        morgan_bits: int =0,#= 1024,
        avalon_bits: int =0,# = 512,

        chemberta_bits: int =0,#=384,
        helmbert_bits: int =0,#= 768,
        molformer_bits:int =0,#=768,
        smilesbert_bits:int =0,#=768,
        smolebert_bits:int =0,#=768,
        molt5_bits:int =0,#=768,
        unimol_bits:int =0,#=512,
        # 注，后续新增部分开始，不再需要手动确认是否使用特征了，会自动根据是不是0来决定调用情况，后续也会随之更新这个
        hashrmorgan_bits:int =0,#=1024,
        moldescriptor_bits:int =0,#=220,

    
        use_helmbert: bool =False,# 这个玩意以后也用不上了========作为分割线使用吧

        chemberta_name: str = "deepchem/ChemBERTa-77M-MLM", # 新增：ChemBerta应该是最好的一个了，但是也没啥效果
        helmbert_name: str ="Flansma/helm-bert",         # ---  新增：helmbert 相关参数 ---
        molformer_name: str = "ibm-research/MoLFormer-XL-both-10pct",
        smilesbert_name: str = "JuIm/SMILES_BERT",
    ) -> None:
        """
        初始化肽类特征提取器。

        Args:
            morgan_bits (int): Morgan 指纹位数。默认 1024。
            avalon_bits (int): Avalon 指纹位数。默认 512。
            use_avalon (bool): 是否使用 Avalon 指纹。默认 True。
                如果系统不支持 Avalon，将自动设置为 False。
        """
        self.morgan_bits = morgan_bits
        self.use_morgan= True if morgan_bits==1024 else False


        self.avalon_bits = avalon_bits
        self.use_avalon = (True if avalon_bits==512 else False) and _HAS_AVALON
        if self.use_avalon and not _HAS_AVALON:
            logger.warning(
                "Avalon 指纹在此环境中不可用，已禁用。"
                "若需使用，请确保 RDKit 编译时启用了 Avalon 支持。"
            )

        # --- 2. 新增 ChemBERTa 初始化 ---
        self.chemberta_bits=chemberta_bits
        self.use_chemberta = (True if chemberta_bits==384 else False) and _HAS_TRANSFORMERS
        if self.use_chemberta:
            logger.info(f"正在加载 ChemBERTa 模型: {chemberta_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(chemberta_name)
            self.chemberta = AutoModel.from_pretrained(chemberta_name)
            self.chemberta.eval() # 设置为评估模式
            # 如果有 GPU，可以移动到 GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.chemberta.to(self.device)

        # --- 4. 新增helm-bert 初始化-------(暂时废弃不使用)-------
        self.helmbert_bits=helmbert_bits
        self.use_helmbert = use_helmbert and True
        if self.use_helmbert:
            try:
                self.helmbert_tokenizer = AutoTokenizer.from_pretrained(helmbert_name, trust_remote_code=True)
                self.helmbert = AutoModel.from_pretrained(helmbert_name, trust_remote_code=True)
                self.helmbert.eval()
                DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.helmbert.to(DEVICE)
            except Exception as e:
                logger.warning(f"加载 HELM-BERT 模型失败: {e}")
                self.helmbert_tokenizer = None
                self.helmbert = None
                DEVICE = None

        # --- 5. 新增 Molformer 初始化 ---
        self.molformer_bits=molformer_bits
        self.use_molformer = True if molformer_bits==768 else False
        if self.use_molformer:
            logger.info(f"正在加载 ChemBERTa 模型: {molformer_name}...")
            self.molformer_tokenizer = AutoTokenizer.from_pretrained(molformer_name, trust_remote_code=True)
            self.molformer = AutoModel.from_pretrained(molformer_name, trust_remote_code=True)
            self.molformer.eval() # 设置为评估模式
            # 如果有 GPU，可以移动到 GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.molformer.to(self.device)

        # --- 6. 新增 SMILES_BERT 初始化 ---
        self.smilesbert_bits=smilesbert_bits
        self.use_smilesbert = True if smilesbert_bits==768 else False
        if self.use_smilesbert:
            logger.info(f"正在加载 SMILES_BERT 模型: {smilesbert_name}...")
            self.smilesbert_tokenizer = AutoTokenizer.from_pretrained(smilesbert_name)
            self.smilesbert = AutoModel.from_pretrained(smilesbert_name)
            self.smilesbert.eval() # 设置为评估模式
            # 如果有 GPU，可以移动到 GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.smilesbert.to(self.device)

        # --- 7. 新增 SmoleBert 初始化 ---
        self.smolebert_bits=smolebert_bits
        self.use_smolebert = True if smolebert_bits==512 else False
        if self.use_smolebert:
            self.smolebert_featurizer = SmoleBertFeaturizer()

        # --- 8. 新增 MolT5 初始化 ---
        self.molt5_bits=molt5_bits
        self.use_molt5 = True if molt5_bits==768 else False
        if self.use_molt5:
            self.molt5_featurizer = MolT5Featurizer()

        # --- 9. 新增 UniMol 初始化 ---
        self.unimol_bits=unimol_bits
        self.use_unimol = True if unimol_bits==512 else False
        if self.use_unimol:
            self.unimol_featurizer = UniMolFeaturizer()
        
        # --- 10. 新增HashMorgan 初始化
        self.hashmorgan_bits=hashrmorgan_bits
        self.use_hashmorgan= True if hashrmorgan_bits==1024 else False
        if self.use_hashmorgan:
            self.hashmorgan_featurizer=HashMorganFeaturizer()
        
        # --- 11. 新增MolDescriptor 初始化
        self.moldescriptor_bits=moldescriptor_bits
        self.use_moldescriptor= True if moldescriptor_bits==220 else False
        if self.use_moldescriptor:
            self.moldescriptor_featurizer=MolDescriptorFeaturizer()       

    @staticmethod
    def _safe_float(value: any, default: float = 0.0) -> float:
        """
        安全地将值转换为浮点数，处理 NaN 和 Inf。

        Args:
            value: 待转换的值。
            default (float): 转换失败时的默认值。

        Returns:
            float: 转换后的浮点数。
        """
        try:
            val = float(value)
            if np.isnan(val) or np.isinf(val):
                return default
            return val
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _gasteiger_charge_stats(mol: Chem.Mol) -> List[float]:
        """
        计算分子的 Gasteiger 电荷统计。

        Args:
            mol (Chem.Mol): RDKit 分子对象。

        Returns:
            List[float]: 包含 5 个统计值的列表
                [均值, 最大值, 最小值, 标准差, 总和]。
                若计算失败，返回全零列表。
        """
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = []
            for i in range(mol.GetNumAtoms()):
                try:
                    charge = float(
                        mol.GetAtomWithIdx(i).GetProp("_GasteigerCharge")
                    )
                    charges.append(charge)
                except (TypeError, KeyError):
                    charges.append(0.0)
            
            if len(charges) == 0:
                charges = [0.0]
            
            charges_arr = np.asarray(charges, dtype=np.float64)
            # 替换非有限值为 0
            charges_arr = np.where(np.isfinite(charges_arr), charges_arr, 0.0)
            
            return [
                float(np.mean(charges_arr)),
                float(np.max(charges_arr)),
                float(np.min(charges_arr)),
                float(np.std(charges_arr)),
                float(np.sum(charges_arr)),
            ]
        except Exception as e:
            logger.debug(f"计算 Gasteiger 电荷时出错: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def _qed_properties(mol: Chem.Mol) -> List[float]:
        """
        提取 QED（药物相似性）属性。

        Args:
            mol (Chem.Mol): RDKit 分子对象。

        Returns:
            List[float]: 包含 8 个 QED 属性的列表
                [MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS]。
                若计算失败，返回全零列表。
        """
        try:
            props = QED.properties(mol)
            return [
                float(props.MW),
                float(props.ALOGP),
                float(props.HBA),
                float(props.HBD),
                float(props.PSA),
                float(props.ROTB),
                float(props.AROM),
                float(props.ALERTS),
            ]
        except Exception as e:
            logger.debug(f"计算 QED 属性时出错: {e}")
            return [0.0] * 8

    @staticmethod
    def _physchem_descriptors(mol: Chem.Mol) -> List[float]:
        """
        提取物理化学描述符（脂溶性、刚性、分子大小等）。

        Args:
            mol (Chem.Mol): RDKit 分子对象。

        Returns:
            List[float]: 包含 11 个物理化学描述符的列表
                [分子量, LogP, HBA, HBD, TPSA, 旋转键数, 环数, Fsp3, 
                 重原子数, 原子总数, 刚性指标]。
                若计算失败，返回全零列表。
        """
        try:
            mw = float(Descriptors.MolWt(mol))
            logp = float(Crippen.MolLogP(mol))
            hba = float(rdMolDescriptors.CalcNumHBA(mol))
            hbd = float(rdMolDescriptors.CalcNumHBD(mol))
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))
            rotb = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
            num_rings = float(rdMolDescriptors.CalcNumRings(mol))
            fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
            heavy_atoms = float(Descriptors.HeavyAtomCount(mol))
            total_atoms = float(mol.GetNumAtoms())
            
            # 简单的刚性指标：环数 / (1 + 旋转键数)
            rigidity_proxy = num_rings / (1.0 + rotb)
            
            return [
                mw, logp, hba, hbd, tpsa, rotb, num_rings, fsp3,
                heavy_atoms, total_atoms, rigidity_proxy
            ]
        except Exception as e:
            logger.debug(f"计算物理化学描述符时出错: {e}")
            return [0.0] * 11

    # ---将SMILE转化为HELM格式的函数（暂时用不上）----
    def smiles_to_helm(self, smiles: str, return_fingerprint: bool = False):
            # ======= helm-bert ======
        # 20 种天然氨基酸残基 SMARTS，用于SMILE转为HELM
        MONOMER_SMARTS = {
            'A':'NC(C)C(=O)',  'R':'NC(CCCNC(N)=N)C(=O)', 'N':'NC(CC(N)=O)C(=O)',
            'D':'NC(CC(O)=O)C(=O)','C':'NC(CS)C(=O)','Q':'NC(CCC(N)=O)C(=O)',
            'E':'NC(CCC(O)=O)C(=O)','G':'NCC(=O)','H':'NC(Cc1c[nH]cn1)C(=O)',
            'I':'NC(C(C)CC)C(=O)','L':'NC(CC(C)C)C(=O)','K':'NC(CCCCN)C(=O)',
            'M':'NC(CCSC)C(=O)','F':'NC(Cc1ccccc1)C(=O)','P':'N1CCCC1C(=O)',
            'S':'NC(CO)C(=O)','T':'NC(C(O)C)C(=O)','W':'NC(Cc1c[nH]c2ccccc12)C(=O)',
            'Y':'NC(Cc1ccc(O)cc1)C(=O)','V':'NC(C(C)C)C(=O)'
        }
        # 转成 RDKit 分子对象
        MONOMER_MOLS = {k: Chem.MolFromSmarts(v) for k,v in MONOMER_SMARTS.items()}

        """
        输入 SMILES，返回 HELM 字符串。
        如果 return_fingerprint=True，同时返回 HELM-BERT embedding。
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            logger.warning(f"无效 SMILES: {smiles}")
            return (None, None) if return_fingerprint else None

        # 1️⃣ 提取氨基酸序列
        matches = []
        for aa, pattern in MONOMER_MOLS.items():
            for match in mol.GetSubstructMatches(pattern):
                matches.append((match[0], aa))
        matches.sort(key=lambda x: x[0])
        sequence = "".join([aa for _, aa in matches]) if matches else None
        if not sequence:
            logger.warning(f"未识别到已知氨基酸: {smiles}")
            return (None, None) if return_fingerprint else None

        # 2️⃣ 转 HELM
        helm = f"PEPTIDE1{{{'.'.join(sequence)}}}$$$$V2.0"

        # 3️⃣ 可选：计算 HELM-BERT 指纹
        fingerprint = None
        if return_fingerprint and self.helmbert and self.helmbert_tokenizer:
            inputs = self.helmbert_tokenizer(helm, return_tensors="pt", padding=True, truncation=True)# .to(DEVICE)
            with torch.no_grad():
                outputs = self.helmbert(**inputs)
                fingerprint = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()

        return (helm, fingerprint) if return_fingerprint else helm

    def _morgan_fingerprint(
        self, mol: Chem.Mol, radius: int = 2, use_chirality: bool = True
    ) -> List[float]:
        """
        提取 Morgan 指纹（位向量）。

        Args:
            mol (Chem.Mol): RDKit 分子对象。
            radius (int): Morgan 指纹半径。默认 2。
            use_chirality (bool): 是否考虑手性。默认 True。

        Returns:
            List[float]: Morgan 指纹位向量，长度为 morgan_bits。
                若计算失败，返回全零列表。
        """
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=self.morgan_bits, useChirality=use_chirality
            )
            arr = np.zeros(self.morgan_bits, dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr.astype(np.float32).tolist()
        except Exception as e:
            logger.debug(f"计算 Morgan 指纹时出错: {e}")
            return [0.0] * self.morgan_bits

    def _avalon_fingerprint(self, mol: Chem.Mol) -> Optional[List[float]]:
        """
        提取 Avalon 指纹（位向量）。

        Args:
            mol (Chem.Mol): RDKit 分子对象。

        Returns:
            Optional[List[float]]: Avalon 指纹位向量，长度为 avalon_bits。
                若不支持或计算失败，返回 None。
        """
        if not self.use_avalon:
            return None
        
        try:
            fp = pyAvalonTools.GetAvalonFP(mol, self.avalon_bits)
            arr = np.zeros(self.avalon_bits, dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr.astype(np.float32).tolist()
        except Exception as e:
            logger.debug(f"计算 Avalon 指纹时出错: {e}")
            return [0.0] * self.avalon_bits

    # --- 新增：ChemBERTa 特征提取私有方法 ---
    def _chemberta_fingerprint(self, smiles: str) -> Optional[List[float]]:
        if not self.use_chemberta:
            return None
        try:
            inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.chemberta(**inputs)
                # 取最后一层隐藏层所有 token 的平均值 (Mean Pooling)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
                
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.debug(f"计算 ChemBERTa 特征时出错: {e}")
            # 注意：ChemBERTa-77M 的维度通常是 384
            return [0.0] * self.chemberta.config.hidden_size
    

    # --- 5. 新增helm-bert 特征提取方法（方法组）
    def _helmbert_fingerprint(self, smiles: str) -> Optional[List[float]]:
        """
        使用 HELM-BERT 提取分子指纹（embedding）。
        输入 SMILES，会先转 HELM，然后直接返回 HELM-BERT 的 embedding。
        """
        
        if not self.use_helmbert:
            logger.debug("没用上helmbert")
            return None

        try:
            # 1️⃣ 将 SMILES 转 HELM + 直接获取 embedding
            helm_result, fingerprint = self.smiles_to_helm(smiles, return_fingerprint=True)
            if fingerprint is None:
                return None

            # 2️⃣ 转成 list，保证类型一致
            return list(fingerprint)

        except Exception as e:
            logger.debug(f"计算 HELM-BERT 特征时出错: {e}")
            # 否则返回空列表
            return []
    
    # --- 6. 新增MolFormer 特征提取方法
    def _molformer_fingerprint(self,smiles:str) ->Optional[List[float]]:

        """
        使用预训练模型molformer进行分子指纹的提取操作
        """
        
        if not self.use_molformer:
            logger.debug("此模型无法使用")
            return None

        try:
            tokens = self.molformer_tokenizer(smiles, padding=True, return_tensors="pt")# , truncation=True
            with torch.no_grad():
                outputs = self.molformer(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()  # 如果这个属性可用（否则自己做 pooling）
            embeddings_normalized = F.normalize(embeddings, p=2, dim=-1)
            fingerprint = embeddings_normalized.cpu().numpy().tolist()
            return fingerprint

        except Exception as e:
            logger.debug(f"计算 HELM-BERT 特征时出错: {e}")
            # 否则返回空列表
            return []

    # --- 7. 新增 SMILES_BERT 特征提取方法
    def _smilesbert_fingerprint(self, smiles: str, pooling: str = "mean") -> Optional[List[float]]:

        """
        使用预训练模型 SMILES_BERT 进行分子指纹的提取操作
        输入：
            smiles: 单条 SMILES 字符串
            pooling: 'mean' 或 'cls'，控制输出方式
        输出：
            embedding: 一维 list
        
        """
        if not self.use_smilesbert:
            logger.debug("此模型无法使用")
            return None
        try:
            inputs = self.smilesbert_tokenizer(smiles, return_tensors="pt").to(self.device)
            outputs = self.smilesbert(**inputs)

            with torch.no_grad():
                if pooling == "cls":
                    # 直接取 [CLS] token
                    emb = outputs.last_hidden_state[0, 0]
                elif pooling == "mean":
                    # mean pooling 所有 token
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
                    emb = emb[0]  # batch size = 1
                else:
                    raise ValueError("pooling must be 'cls' or 'mean'")

            return emb.cpu().tolist()  # 一维 list

        except Exception as e:
            logger.debug(f"计算 SMILES_BERT 特征时出错: {e}")
            # 否则返回空列表
            return []


    def featurize(
        self, smiles: str
    ) -> Tuple[Optional[List[float]], bool]:
        """
        从 SMILES 字符串提取分子特征。

        Args:
            smiles (str): 分子的 SMILES 字符串。

        Returns:
            Tuple[Optional[List[float]], bool]: 
                - 第一个元素：特征列表，若提取失败则为 None
                - 第二个元素：是否成功提取（bool）
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.debug(f"无效的 SMILES 字符串: {smiles}")
                return None, False
            
            # 依次提取各部分特征
            qed_feats = self._qed_properties(mol)
            physchem_feats = self._physchem_descriptors(mol)
            charge_feats = self._gasteiger_charge_stats(mol)
            
            logger.info("qed_feats的长度为："+str(len(qed_feats))) # 8
            logger.info("physchem_feats的长度为："+str(len(physchem_feats))) # 11
            logger.info("charge_feats的长度为："+str(len(charge_feats))) # 5
            
            
            
            # 组合所有理化特征
            all_features = (
                qed_feats + physchem_feats + charge_feats
            )

            # 如果启用了morgan（Morgan是唯一一个可以调整的内容）
            if self.use_morgan:
                morgan_feats = self._morgan_fingerprint(mol)
                if morgan_feats is not None:
                    logger.info("morgan_feats的长度为："+str(len(morgan_feats))) # 1024
                    all_features.extend(morgan_feats)

            
            # 如果启用了 Avalon，添加 Avalon 指纹
            if self.use_avalon:
                avalon_feats = self._avalon_fingerprint(mol)
                if avalon_feats is not None:
                    logger.info("avalon_feats的长度为："+str(len(avalon_feats))) # 512
                    all_features.extend(avalon_feats) # 512
            
            # --- 新增：整合 ChemBERTa ---
            if self.use_chemberta:
                berta_feats = self._chemberta_fingerprint(smiles)
                if berta_feats is not None: 
                    logger.info("chemberta_feats的长度为："+str(len(berta_feats))) # 512
                    all_features.extend(berta_feats) # 384
            # --- 6. 新增： 整合helmbert ---
            if self.use_helmbert:
                helmbert_feats = self._helmbert_fingerprint(smiles)

                if helmbert_feats is not None:
                    logger.info("helmbert_feats的长度为："+str(len(helmbert_feats))) # 768
                    all_features.extend(helmbert_feats) # 768
            # --- 7. 新增： 整合molformer ---
            if self.use_molformer:
                molformer_feats = self._molformer_fingerprint(smiles)

                if molformer_feats is not None:
                    logger.info("molformer_feats的长度为："+str(len(molformer_feats))) # 768
                    all_features.extend(molformer_feats) # 768
            # --- 8. 新增： 整合smilesbert ---
            if self.use_smilesbert:
                smilesbert_feats = self._smilesbert_fingerprint(smiles)
                if smilesbert_feats is not None:
                    logger.info("smilesbert_feats的长度为："+str(len(smilesbert_feats))) # 768
                    all_features.extend(smilesbert_feats) # 768
            # --- 9. 新增： 整合smolebert ---
            if self.use_smolebert:
                smolebert_feats = self.smolebert_featurizer.get_features(smiles)
                if smolebert_feats is not None:
                    logger.info("smolebert_feats的长度为："+str(len(smolebert_feats))) # 768
                    all_features.extend(smolebert_feats) # 768
            # --- 10. 新增： 整合molt5 ---
            if self.use_molt5:
                molt5_feats = self.molt5_featurizer.get_features(smiles)
                if molt5_feats is not None:
                    logger.info("molt5_feats的长度为："+str(len(molt5_feats))) # 768
                    all_features.extend(molt5_feats) # 768
            # --- 11. 新增： 整合unimol ---
            if self.use_unimol:
                unimol_feats = self.unimol_featurizer.get_features(smiles)
                if unimol_feats is not None:
                    logger.info("unimol_feats的长度为："+str(len(unimol_feats))) # 512
                    all_features.extend(unimol_feats) # 512
            # ---12. 整合HashMorgan 
            if self.use_hashmorgan:
                hashmorgan_feats=self.hashmorgan_featurizer.get_features(smiles)
                if hashmorgan_feats is not None:
                    logger.info("hashmorgan_feats的长度为："+str(len(hashmorgan_feats))) # 
                    all_features.extend(hashmorgan_feats) # 
            # ---13. 整合一些别的理化性质
            if self.use_moldescriptor:
                moldescriptor_feats=self.moldescriptor_featurizer.get_features(smiles)
                if moldescriptor_feats is not None:
                    logger.info("moldescriptor_feats的长度为："+str(len(moldescriptor_feats))) # 
                    all_features.extend(moldescriptor_feats) # 

            logger.info("汇总的长度为："+str(len(all_features))) # 3480
            return all_features, True
        
        except Exception as e:
            logger.error(f"提取 SMILES '{smiles}' 的特征时出错: {e}")
            return None, False

    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表。

        返回的特征顺序与 featurize() 方法返回的特征列表一致。

        Returns:
            List[str]: 特征名称列表。
        """
        names = []
        
        # QED 属性名
        names.extend([
            "QED_MW", "QED_ALOGP", "QED_HBA", "QED_HBD",
            "QED_PSA", "QED_ROTB", "QED_AROM", "QED_ALERTS"
        ])
        
        # 物理化学描述符名
        names.extend([
            "PC_MolWt", "PC_LogP", "PC_HBA", "PC_HBD", "PC_TPSA",
            "PC_RotB", "PC_Rings", "PC_FractionCSP3", "PC_HeavyAtomCount",
            "PC_NumAtoms", "PC_RigidityProxy"
        ])
        
        # Gasteiger 电荷统计名
        names.extend([
            "GC_Mean", "GC_Max", "GC_Min", "GC_Std", "GC_Sum"
        ])
        
        # Morgan 指纹名
        names.extend([f"Morgan_{i}" for i in range(self.morgan_bits)])
        
        # Avalon 指纹名（如果启用）
        if self.use_avalon:
            names.extend([f"Avalon_{i}" for i in range(self.avalon_bits)])

        # --- 新增：ChemBERTa 维度名称 ---
        if self.use_chemberta:
            hidden_size = self.chemberta.config.hidden_size
            names.extend([f"ChemBERTa_{i}" for i in range(hidden_size)])

        # --- 7. 新增： helmbert 维度名称 ---
        if self.use_helmbert:
            names.extend([f"HelmBERT_{i}" for i in range(self.helmbert_bits)])

        # --- 8. 新增： MolFormer 维度名称 ---
        if self.use_molformer:
            names.extend([f"MolFormer_{i}" for i in range(self.molformer_bits)])
        # --- 9. 新增： Smile 维度名称 ---
        if self.use_smilesbert:
            names.extend([f"SmilesBERT_{i}" for i in range(self.smilesbert_bits)])
        # --- 10. 新增： SmoleBert 维度名称 ---
        if self.use_smolebert:
            names.extend([f"SmoleBert_{i}" for i in range(self.smolebert_bits)])
        # --- 11. 新增： MolT5 维度名称 ---
        if self.use_molt5:
            names.extend([f"MolT5_{i}" for i in range(self.molt5_bits)])
        # --- 12. 新增： UniMol 维度名称 ---
        if self.use_unimol:
            names.extend([f"UniMol_{i}" for i in range(self.unimol_bits)])
        # --- 13 HashMorgan
        if self.use_hashmorgan:
            names.extend([f"HashMorgan_{i}" for i in range(self.hashmorgan_bits)])
        # --- 14 补充理化性质
        if self.use_moldescriptor:
            names.extend([f"MD_{i}" for i in self.moldescriptor_featurizer.get_names()])
        return names
    # --- 新增：总特征数属性, 不影响输出的话最好在这里做个修改 ---
    @property
    def n_features(self) -> int:
        n = 8 + 11 + 5 + self.morgan_bits
        if self.use_avalon: n += self.avalon_bits
        if self.use_chemberta: n += self.chemberta_bits
        # --- 8. 新增：总特征数更新 ---
        if self.use_helmbert: n += self.helmbert_bits
        if self.use_molformer: n += self.molformer_bits
        if self.use_smilesbert: n += self.smilesbert_bits
        if self.use_smolebert: n += self.smolebert_bits
        if self.use_molt5: n += self.molt5_bits
        if self.use_unimol: n += self.unimol_bits
        if self.use_hashmorgan: n+=self.hashmorgan_bits
        if self.use_moldescriptor: n+=self.moldescriptor_bits
        return n
