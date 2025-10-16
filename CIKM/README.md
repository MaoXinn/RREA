# RREA：关系反射实体对齐

[![arXiv](https://img.shields.io/badge/arXiv-2008.07962-b31b1b.svg)](https://arxiv.org/abs/2008.07962)

本代码库是论文 [Relational Reflection Entity Alignment](https://arxiv.org/pdf/2008.07962.pdf) 的 Keras 实现。

## 项目简介

实体对齐（Entity Alignment, EA）旨在发现并链接不同知识图谱（KG）中指向同一现实世界对象的实体。本项目实现了一个强大的、基于图的实体对齐模型——RREA。该模型通过一种新颖的“关系反射”机制，同时利用了知识图谱中的实体和关系信息，从而能够更高效地学习用于对齐的实体嵌入向量。

## 主要特性

-   **RREA 模型**: 忠实复现了关系反射实体对齐模型。
-   **图神经网络**: 采用图注意力网络（GAT）对知识图谱的结构信息进行编码。
-   **半监督学习**: 采用半监督策略，在训练过程中迭代地将高置信度的预测结果加入训练集，以提升模型性能。
-   **CSLS 评估**: 使用跨域自监督学习（CSLS）算法进行评估，该算法能更精确地匹配近邻，提高评估准确性。
-   **标准数据集**: 项目已预先配置了广泛使用的 DBP15K 数据集（`zh_en`, `ja_en`, `fr_en`）。

## 环境配置

请确保您已安装 Python 3.5+ 和 Anaconda。您可以按照以下步骤创建环境并安装所需依赖：

```bash
# 推荐创建一个新的 Conda 环境
conda create -n rrea python=3.6
conda activate rrea

# 安装依赖
pip install tensorflow==1.13.1
pip install keras==2.2.4
pip install scipy numpy tqdm
```

## 如何运行

项目的核心训练和评估逻辑都包含在 `RREA.ipynb` 这个 Jupyter Notebook 文件中。

1.  **启动 Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

2.  **打开 `RREA.ipynb` 文件**。

3.  **选择数据集**: 在第二个代码单元格中，修改 `lang` 变量的值来选择您想使用的数据集：
    ```python
    # 可选项: 'zh', 'ja', 'fr'
    lang = 'zh'
    ```

4.  **运行 Notebook**: 在菜单栏中点击 `Cell -> Run All` 来按顺序执行所有单元格。训练过程将自动开始，评估结果（Hits@1, 5, 10）会周期性地打印在输出中。

## 数据集

本项目使用的数据集来自 [GCN-Align](https://github.com/1049451037/GCN-Align)，并已包含在 `data/` 目录下。每个语言对（例如 `zh_en`）的数据结构如下：

-   `ent_ids_1`: 源知识图谱中的实体 ID。
-   `ent_ids_2`: 目标知识图谱中的实体 ID。
-   `ref_ent_ids`: 用于训练和测试的、预先链接好的实体对。
-   `triples_1`: 源知识图谱中的关系三元组。
-   `triples_2`: 目标知识图谱中的关系三元组。

## 引用

如果您在研究中使用了本项目的代码或模型，请引用以下原始论文：

```bibtex
@article{zhang2020relational,
  title={Relational Reflection Entity Alignment},
  author={Zhang, Wentao and Feng, Zhaodong and Yang, Yafang and Wang, Zirui and Chen, Xiang},
  journal={arXiv preprint arXiv:2008.07962},
  year={2020}
}
```

## 致谢

本项目的代码参考了以下开源实现：[keras-gat](https://github.com/danielegrattarola/keras-gat), [GCN-Align](https://github.com/1049451037/GCN-Align), and [TransEdge](https://github.com/nju-websoft/TransEdge)。感谢这些优秀的工作！