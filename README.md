# Pipeline of inference



## 🚀 快速开始（三分钟跑通版）

前提： 确保你电脑装了 Python 3.11（没装的自己去官网下）。

另外请自备包含字段为**SMILES**的csv文件

## ① 使用方法

### 0. 【前置要求】安装python
当前项目在开发阶段使用的python版本为3.11.*
通过初步测试发现，当前支持python3.11\*以及3.12\*
可能支持您的版本，也可能不支持


### 1. 创建虚拟环境 （强烈推荐）
```python -m venv venv```

激活虚拟环境后，命令行上会出现（venv）字样

### 2. 激活环境

**Windows环境：**
```.\venv\Scripts\activate```

**Linux环境**
```source venv/bin/activate```

### 3. 安装依赖
```pip install -r requirements.txt --no-deps```

另外，由于开发过程中使用依赖较多，安装虚拟环境耗时可能更长，请耐心等待

### 4. 准备数据【为举例子将其命名为your_data】
csv文件中至少应包含SMILES字段，在项目根目录中存在名为raw.csv的样例文件

### 5. 运行指令（推理）
```python start.py --input .\your_data.csv```

### 6. 结果查验

- **SIF**: 

    ```./SIF_result/sif_results.csv ```
    ```./SIF_result/sif_results.html```

- **SGF**: 

    ```./SGF_result/sgf_results.csv ```
    ```./SGF_result/sgf_results.html```


## ② 关于环境补充：
开发阶段使用的包管理工具为uv，但是涉及到以下两个问题

* 对于没使用过uv的用户，可能初步安装和配置需要时间
* 该项目中的部分库存在“可接受”的依赖冲突，即功能本身并不冲突，但某些库的依赖要求有版本上的冲突，不影响使用。但uv根据pyproject.toml生成uv.lock文件时会自动规划依赖，导致项目不可用。因此在pipeline的整合阶段，使用的是uv pip直接管理.venv环境。不经过标准的pyproject和lock文件。

因此，推荐直接使用requirements.txt部署环境。

由于项目本身存在可接受的冲突，可能会影响您的全局python环境（如果您不了解这个请务必注意！），推荐按照上述方法，配置当前文件夹下的虚拟环境。

