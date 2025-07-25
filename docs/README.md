# 基于集成学习与数据增强的机器学习藻类脂质预测系统

## 项目概述

本项目是一个专门针对**小样本高维数据**的机器学习解决方案，专注于**藻类脂质含量预测**。通过结合SMOGN数据增强、贝叶斯集成学习、深度学习等技术，成功突破了传统机器学习在小样本场景下的性能瓶颈，实现了高精度的藻类脂质预测。

### 应用领域

- **生物能源研究**：藻类生物燃料开发
- **环境科学**：藻类生态系统分析
- **生物技术**：藻类培养优化
- **食品科学**：藻类营养成分分析

### 核心成果

- **训练集 R² = 0.9980**
- **交叉验证 R² = 0.9870 ± 0.0109**
- **测试集 R² = 0.9738**

### 技术亮点

- **SMOGN数据增强**：专门用于回归问题的合成少数类过采样
- **噪声增强技术**：通过高斯噪声增加数据多样性
- **贝叶斯集成**：多模型智能加权融合
- **深度学习优化**：MLP神经网络在小样本上的卓越表现
- **极端特征降维**：从29维降至3维的精准特征选择

## 项目结构

```
项目研究/
├──  data/                       # 数据目录
│   ├── raw/数据.xlsx              # 原始数据 (36样本×29特征)
│   ├── processed/                 # 处理后的数据
│   └── features/                  # 特征工程结果
├──  src/                        # 源代码目录
│   ├── data_processing/           # 数据处理模块
│   ├── feature_engineering/       # 特征工程模块
│   ├── modeling/                  # 建模模块
│   │   └── ultimate_ensemble_pipeline.py  #  核心算法
│   ├── evaluation/                # 评估模块
│   └── utils/                     # 工具函数
├──  config/
│   └── settings.py                # 配置文件
├──  results/                    # 结果输出
│   └── ultimate_ensemble_results.txt  # 详细实验结果
├──  docs/                       # 文档目录
├──  patent/                     # 专利相关文档
├──  run_pipeline.py             # 主运行脚本
└──  requirements.txt            # 依赖包列表
```

## 文档

- **[ 快速开始指南](快速开始.md)** - 一键运行，快速上手
- **[ 运行指南](运行指南.md)** - 详细的运行顺序和模块说明
- **[ 数据获取指南](数据获取指南.md)** - 扩展数据集的完整方案和资源
- **[ 技术详解](技术详解.md)** - 深度解析算法原理和实现
- **[ API文档](API文档.md)** - 完整的函数和类参考手册
- **[ 项目总结](项目总结.md)** - 完整的成果总结和技术创新报告
- **[️ 配置说明](../config/project_config.py)** - 参数配置和自定义选项

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 一键运行

```bash
# 运行完整管道
python run_pipeline.py
```

### 3. 查看结果

```bash
# 查看详细结果
cat results/ultimate_ensemble_results.txt
```

## 核心特性

### 1. 智能数据预处理

- **保守特征工程**: 避免过度复杂化，创建有意义的比率特征
- **智能特征选择**: 结合相关性、方差和递归特征消除
- **小样本优化**: 确保特征数量不超过样本数的1/3

### 2. 稳健建模方法

- **多种算法**: Ridge、Lasso、ElasticNet、随机森林、梯度提升
- **强正则化**: 防止小样本过拟合
- **交叉验证**: 留一交叉验证(LOOCV)和Bootstrap验证
- **集成学习**: 自动选择最佳模型进行集成

### 3. 全面评估体系

- **多重验证**: 训练、LOOCV、Bootstrap三重验证
- **过拟合检测**: 自动计算过拟合程度
- **置信区间**: 提供Bootstrap置信区间
- **可视化报告**: 自动生成性能图表和详细报告

## 输出文件说明

运行完成后，在 `results/final_optimization/`目录下会生成：

1. **optimized_data.csv** - 优化后的训练数据
2. **optimization_report.txt** - 数据预处理报告
3. **final_optimization_results.txt** - 模型性能详细结果
4. **optimized_preprocessing_results.png** - 数据预处理可视化
5. **final_best_model.pkl** - 最佳模型文件

## 技术特点

### 小样本优化策略

- 特征数量控制在样本数的1/3以下
- 使用强正则化防止过拟合
- 采用稳健的验证方法评估性能
- 集成多个弱学习器提高泛化能力

### 数据质量保证

- 智能缺失值处理
- 异常值检测和处理
- 特征重要性分析
- 相关性矩阵可视化

## 性能分析

当前模型在36个样本的小数据集上达到了0.54的Bootstrap R²，考虑到：

- 样本量较小(36个)
- 特征维度相对较高
- 生物数据的复杂性

这个结果是合理的。进一步改进建议：

1. 增加样本量
2. 收集更多相关特征
3. 进行数据质量检查
4. 考虑领域知识指导的特征工程

## 许可

本项目仅供学术研究使用。

## 联系方式

如有问题，请联系项目维护者。
