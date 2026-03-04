# Pipeline of inference

该项目提供对于SMILES分子在SIF，SGF两种任务下的稳定性预测

- SIF使用LR模型，表征为Avalon+MolFormer
- SGF使用LR模型，表征为Avalon+MolT5
- 另外提供Baseline校验（LR模型，Morgan表征），会生成对应的文件夹

（为了方便使用，我们提供了这种默认配置，如需调整请联系我们新增功能）

- 为了确保数据隐私，我们仅上传了模型权重，测试数据仅包含五条经过修改的数据

## 〇 Python版本要求

前提： 确保工作环境装了 Python 3.11。

检查方法为输入当前指令：
```
python --version
```
如果输出对应的版本字样，则为安装成功
```
3.11.*
```

另外请自备包含字段为**SMILES**的csv文件

## ① 使用方法

### 0. 【前置要求】安装python
当前项目在开发阶段使用的python版本为3.11.*

通过初步测试发现，当前支持python3.11\*以及3.12\*

可能支持您自己的版本，也可能不支持，因此请您尽量把python环境调整为3.11.*


### 1. 创建虚拟环境 （强烈推荐）
```python -m venv venv```

创建虚拟环境后，文件夹中会出现一个名为（venv）的文件夹，该文件夹即为您的虚拟环境。

拥有虚拟环境后，可以在您当前的文件夹目录下，创建一个和您本地电脑隔离的环境，按照下方的操作指引完成激活后，配置环境就不会影响您的电脑了。

> PS: 如果您的电脑中包含常用化学库，计算库等，恰好支持运行该内容的话，或许可以尝试直接进行推理，推理方法在后续内容中

### 2. 激活环境

根据您电脑的环境尝试激活虚拟环境，在项目根目录下执行：

**Windows环境：**
```.\venv\Scripts\activate```

**Linux环境**
```source venv/bin/activate```

若执行成功后，您的终端命令行从

``` D:\RA\pipeline> ```

变为

```(venv) D:\RA\pipeline> ```

出现前面的括号后，代表您从此开始执行的所有安装，运行python脚本的操作都是在一个隔离的虚拟环境下， 不会干扰到您的电脑。

如果您结束了该项目的工作，需要重新使用您的电脑环境，您可以执行

```deactivate```

即可退出您当前的虚拟环境

### 3. 安装依赖

激活虚拟环境后，请您在该目录下的命令行窗口执行

```pip install -r requirements.txt --no-deps```

另外，由于开发过程中使用依赖较多，安装虚拟环境耗时可能更长，请耐心等待

### 4. 准备数据【为举例子将其命名为your_data】

csv文件中至少应包含一个名为SMILES的字段，该字段为SMILES格式的化学式

并且仅需要该字段即可，其他字段并不影响

在项目根目录中存在名为raw.csv的样例文件，可以作为您的格式参考

### 5. 运行指令（推理）

在虚拟环境下安装好依赖后，请保持虚拟环境激活，在命令行窗口执行

```python start.py --input .\your_data.csv```

其中```.\your_data.csv```请替换为您准备好的csv文件

结果执行后，可以检查您的执行结果

### 6. 结果查验

当前版本中，我们会将预测结果输入到如下文件，并且使用html进行可视化。

**结果查看方法**：

- csv文件可以使用文本编辑器(如vscode)打开
- html文件可以使用浏览器打开

**结果存储路径**：

- **SIF**: 

    ```./SIF_result/sif_results.csv ```
    ```./SIF_result/sif_results.html```

- **SGF**: 

    ```./SGF_result/sgf_results.csv ```
    ```./SGF_result/sgf_results.html```

- **Morgan-Baseline结果**: 

    ```./SIF_result_baseline/sif_results.csv ```
    ```./SIF_result_baseline/sif_results.html```
    ```./SGF_result_baseline/sgf_results.csv ```
    ```./SGF_result_baseline/sgf_results.html```

## ② 关于环境补充（为什么要使用原生的虚拟环境）：
开发阶段使用的包管理工具为uv，但是涉及到以下两个问题

* 对于没使用过uv的用户，可能初步安装和配置需要时间
* 该项目中的部分库存在“可接受”的依赖冲突，即功能本身并不冲突，但某些库的依赖要求有版本上的冲突，不影响使用。但uv的标准用法会导致检查失败

因此，推荐直接使用虚拟环境 + requirements.txt部署环境。

没有使用conda也是一样的原因，conda会对项目依赖进行检查。

## ③ 关于可能产生的报错

经过测试应该不会有太多报错，但是基于其他项目中常出现的问题，给出可能的解决方法。

1. 如果提示某个库无法下载

    可以自行在激活虚拟环境后使用```pip install <包名称>```进行下载

    但是如果您下载的包涉及到numpy和transformer库的变动，这可能会造成依赖冲突，解决方法是在进安装完成以后，按照顺序执行如下两个安装命令, 会解决这个已知冲突

      pip install transformers==4.36.2 --force-reinstall --no-cache-dir

      pip install "numpy<2.0" --force-reinstall --no-cache-dir 

2. 关于报错信息

    报错信息没有做特别的拦截，会直接显示在命令行，如果出现了您解决不了的问题，请您联系我

    ```xarnudvilas@gmail.com```


