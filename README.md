# 🧬 微藻脂质含量预测系统 (AlgaeV1)

基于机器学习的微藻脂质含量预测系统，通过环境参数和生物特征预测微藻脂质含量，为生物燃料生产提供技术支持。

## 🎯 项目概述

本项目实现了一个完整的机器学习流水线，用于预测微藻的脂质含量。系统包含数据预处理、特征选择、模型训练、性能评估和可视化预测应用等模块。

### 主要特性

- ✅ 完整的数据预处理流水线
- ✅ 多方法集成特征选择系统  
- ✅ SVM回归模型训练与测试
- ✅ Web可视化预测应用
- ✅ 详细的技术文档

## 🏗️ 项目结构

```
algaev1/
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   └── processed/                 # 预处理数据
├── src/                           # 源代码
│   ├── data_processing/           # 数据预处理
│   ├── feature_selection/         # 特征选择
│   ├── model_training/            # 模型训练
│   ├── model_testing/             # 模型测试
│   └── visualization/             # 可视化应用
├── results/                       # 结果输出
│   ├── data_preprocess/           # 预处理结果
│   ├── feature_selection/         # 特征选择结果
│   ├── model_training/            # 训练结果
│   └── model_testing/             # 测试结果
├── docs/                          # 技术文档
├── run_prediction_app.py          # 应用启动器
└── README.md                      # 项目说明
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- 主要依赖：pandas, numpy, scikit-learn, streamlit, plotly

### 安装依赖

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly joblib openpyxl
```

### 运行预测系统

```bash
# 启动Web预测应用
python run_prediction_app.py
```

系统将自动在浏览器中打开，默认地址：http://localhost:8501

## 📊 模型性能

### 最终特征

| 特征名称 | 描述 | 重要性排名 |
|---------|------|-----------|
| protein(%) | 蛋白质含量百分比 | 1 |
| H(%) | 氢元素含量百分比 | 2 |
| O(%) | 氧元素含量百分比 | 3 |
| pigment_per_cell | 单细胞色素含量 | 4 |

### 性能指标

- **训练集R²**: -0.183
- **测试集R²**: -0.194
- **测试集MAE**: 3.688
- **过拟合程度**: 低 (0.011)

*注：当前模型存在欠拟合问题，性能有待改进*

## 🔧 使用说明

### 完整流程执行

```bash
# 1. 数据预处理
cd src/data_processing
python data_preprocessing.py

# 2. 特征选择
cd ../feature_selection
python feature_selector.py

# 3. 模型训练
cd ../model_training
python model_train.py

# 4. 模型测试
cd ../model_testing
python model_test.py

# 5. 启动预测应用
cd ../../
python run_prediction_app.py
```

### Web预测系统使用

1. 启动系统后在浏览器中打开
2. 在左侧边栏输入4个特征参数
3. 点击"开始预测"按钮
4. 查看预测结果和等级评定

## 📚 技术文档

详细的技术文档位于 `docs/` 目录：

- [数据预处理详解](docs/data_preprocessing.md)
- [特征选择详解](docs/feature_selection.md)
- [模型训练详解](docs/model_training.md)
- [模型测试详解](docs/model_testing.md)
- [项目总览](docs/project_overview.md)

## 🎨 核心技术

- **数据预处理**: 缺失值处理、异常值处理、Robust标准化
- **特征选择**: 6种方法集成（F-regression、互信息、随机森林等）
- **机器学习**: SVM回归模型（多项式核）
- **Web应用**: Streamlit + Plotly交互式界面

## ⚠️ 已知限制

1. **数据量有限**: 仅35个样本，训练数据不足
2. **模型性能**: 当前存在欠拟合问题，R²为负值
3. **特征数量**: 仅使用4个特征，可能信息不足
4. **预测区分度**: 预测值变化范围较小

## 🔮 改进方向

- [ ] 增加训练样本数量
- [ ] 尝试其他机器学习算法（随机森林、神经网络）
- [ ] 改进特征工程和特征选择
- [ ] 使用集成学习方法
- [ ] 引入深度学习技术

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 📞 联系方式

如有问题或建议，请通过GitHub Issues联系。

---

**注**: 本项目为研究性质，当前模型性能有限，建议在实际应用前进行进一步优化和验证。
