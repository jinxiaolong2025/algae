# 小样本高维数据机器学习项目 - 完整技术流程图

## 项目整体架构流程图

```mermaid
flowchart TD
    %% 数据输入层
    A["📊 原始数据<br/>36样本 × 29特征<br/>data/raw/数据.xlsx"] --> B["🔍 数据质量诊断<br/>缺失值检测<br/>异常值识别<br/>数据类型验证"]
    
    %% 数据预处理层
    B --> C["🧹 智能数据预处理<br/>intelligent_data_processor.py<br/>• 缺失值智能填充<br/>• 数据标准化<br/>• 目标变量自动识别"]
    
    %% 特征工程层
    C --> D["⚙️ 高级特征选择<br/>selection.py<br/>• F统计量分析<br/>• 互信息计算<br/>• 方差分析<br/>• VIF多重共线性检测"]
    
    D --> E["📈 特征降维结果<br/>29维 → 3维 (90%降维)<br/>选择特征:<br/>• N(%)<br/>• electrical conductivity<br/>• Chroma"]
    
    %% 数据增强层
    E --> F["🔄 多策略数据增强<br/>ultimate_ensemble_pipeline.py"]
    
    F --> F1["📊 SMOGN增强<br/>• K近邻合成<br/>• 高斯噪声注入<br/>• 36→108样本"]
    F --> F2["🎲 噪声增强<br/>• 特征噪声添加<br/>• 目标值微调<br/>• 36→108样本"]
    F --> F3["📋 原始数据<br/>• 保持原始特征<br/>• 36样本"]
    
    %% 模型训练层
    F1 --> G["🤖 多算法模型训练"]
    F2 --> G
    F3 --> G
    
    G --> G1["🧠 深度学习<br/>MLP_Deep<br/>200-100-50层"]
    G --> G2["🌳 集成学习<br/>RandomForest<br/>GradientBoosting"]
    G --> G3["📏 线性模型<br/>Ridge, Lasso<br/>ElasticNet"]
    G --> G4["🎯 支持向量机<br/>SVR_Polynomial"]
    G --> G5["🔮 贝叶斯集成<br/>BayesianEnsemble"]
    
    %% 模型评估层
    G1 --> H["📊 综合模型评估<br/>comprehensive_evaluation.py"]
    G2 --> H
    G3 --> H
    G4 --> H
    G5 --> H
    
    H --> H1["✅ 留一交叉验证<br/>LOOCV"]
    H --> H2["🔄 Bootstrap验证<br/>500次重采样"]
    H --> H3["📈 多指标评估<br/>R², RMSE, MAE"]
    H --> H4["⚖️ 过拟合检测<br/>训练/验证差异"]
    
    %% 模型融合层
    H1 --> I["🎯 贝叶斯智能融合<br/>基于似然函数的权重学习"]
    H2 --> I
    H3 --> I
    H4 --> I
    
    %% 结果输出层
    I --> J["🏆 最优模型选择<br/>MLP_Deep + noise增强<br/>训练R²: 0.9980<br/>CV R²: 0.9870±0.0109"]
    
    J --> K["📋 结果输出"]
    
    K --> K1["📄 详细报告<br/>ultimate_ensemble_results.txt"]
    K --> K2["📊 可视化图表<br/>feature_visual/"]
    K --> K3["💾 模型保存<br/>results/modeling/"]
    K --> K4["📈 性能分析<br/>comprehensive_evaluation_results/"]
    
    %% 专利支持层
    K --> L["📜 专利技术支持<br/>patent/"]
    
    L --> L1["📋 技术文档<br/>patent_technical_document.md"]
    L --> L2["📊 技术流程图<br/>technical_flowchart.png"]
    L --> L3["🔧 专利系统<br/>system.py"]
    
    %% 样式定义
    classDef inputData fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef modeling fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef evaluation fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef output fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef patent fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    
    class A,B inputData
    class C,D,E,F,F1,F2,F3 processing
    class G,G1,G2,G3,G4,G5 modeling
    class H,H1,H2,H3,H4,I evaluation
    class J,K,K1,K2,K3,K4 output
    class L,L1,L2,L3 patent
```

## 核心技术创新点流程图

```mermaid
flowchart LR
    %% 创新点1：小样本优化
    A1["🎯 小样本优化策略"] --> A2["样本/特征比优化<br/>1.3 → 9.0"]
    A1 --> A3["模型复杂度控制<br/>防止过拟合"]
    A1 --> A4["正则化强度调节<br/>L1/L2平衡"]
    
    %% 创新点2：数据增强
    B1["🔄 双重数据增强"] --> B2["SMOGN回归增强<br/>结构化合成"]
    B1 --> B3["噪声随机增强<br/>多样性提升"]
    B1 --> B4["质量控制机制<br/>避免噪声污染"]
    
    %% 创新点3：特征工程
    C1["⚙️ 极端特征工程"] --> C2["多策略融合选择<br/>F统计+互信息+方差"]
    C1 --> C3["90%维度压缩<br/>29→3特征"]
    C1 --> C4["生物学意义保持<br/>可解释性强"]
    
    %% 创新点4：集成学习
    D1["🤖 贝叶斯集成"] --> D2["智能权重学习<br/>基于似然函数"]
    D1 --> D3["动态模型选择<br/>自适应融合"]
    D1 --> D4["多算法协同<br/>深度+传统"]
    
    %% 创新点5：评估体系
    E1["📊 全方位评估"] --> E2["LOOCV稳定验证<br/>小样本专用"]
    E1 --> E3["Bootstrap鲁棒性<br/>500次重采样"]
    E1 --> E4["综合评分机制<br/>防过拟合惩罚"]
    
    classDef innovation fill:#e3f2fd,stroke:#0277bd,stroke-width:3px
    classDef technique fill:#f9fbe7,stroke:#689f38,stroke-width:2px
    
    class A1,B1,C1,D1,E1 innovation
    class A2,A3,A4,B2,B3,B4,C2,C3,C4,D2,D3,D4,E2,E3,E4 technique
```

## 数据流转换过程图

```mermaid
flowchart TD
    %% 数据维度变化
    subgraph "数据维度演变"
        D1["原始数据<br/>36 × 29"] --> D2["预处理后<br/>36 × 29"]
        D2 --> D3["特征选择后<br/>36 × 3"]
        D3 --> D4["SMOGN增强后<br/>108 × 3"]
        D3 --> D5["噪声增强后<br/>108 × 3"]
    end
    
    %% 性能指标变化
    subgraph "性能指标演变"
        P1["原始数据性能<br/>CV R²: -2.86"]
        P2["SMOGN增强性能<br/>CV R²: 0.56"]
        P3["噪声增强性能<br/>CV R²: 0.77"]
        P4["最终集成性能<br/>CV R²: 0.987"]
        
        P1 --> P2 --> P3 --> P4
    end
    
    %% 模型复杂度控制
    subgraph "模型复杂度管理"
        M1["高复杂度模型<br/>过拟合风险"]
        M2["正则化控制<br/>复杂度平衡"]
        M3["集成融合<br/>稳定性提升"]
        M4["最优解<br/>泛化能力强"]
        
        M1 --> M2 --> M3 --> M4
    end
```

## 技术栈架构图

```mermaid
flowchart TB
    %% 应用层
    subgraph "应用层 Application Layer"
        APP1["run_pipeline.py<br/>主运行脚本"]
        APP2["配置管理<br/>config/"]
    end
    
    %% 业务逻辑层
    subgraph "业务逻辑层 Business Logic Layer"
        BL1["数据处理模块<br/>data_processing/"]
        BL2["特征工程模块<br/>feature_engineering/"]
        BL3["建模模块<br/>modeling/"]
        BL4["评估模块<br/>evaluation/"]
    end
    
    %% 算法层
    subgraph "算法层 Algorithm Layer"
        AL1["SMOGN增强算法"]
        AL2["贝叶斯集成算法"]
        AL3["深度学习算法"]
        AL4["传统ML算法"]
    end
    
    %% 数据层
    subgraph "数据层 Data Layer"
        DL1["原始数据<br/>data/raw/"]
        DL2["处理数据<br/>data/processed/"]
        DL3["特征数据<br/>data/features/"]
        DL4["结果数据<br/>results/"]
    end
    
    %% 工具层
    subgraph "工具层 Utility Layer"
        UL1["字体配置<br/>utils/font_config.py"]
        UL2["可视化工具<br/>matplotlib/seaborn"]
        UL3["科学计算<br/>numpy/pandas"]
        UL4["机器学习<br/>scikit-learn"]
    end
    
    %% 连接关系
    APP1 --> BL1
    APP1 --> BL2
    APP1 --> BL3
    APP1 --> BL4
    APP2 --> BL1
    
    BL1 --> AL1
    BL2 --> AL1
    BL3 --> AL2
    BL3 --> AL3
    BL3 --> AL4
    BL4 --> AL2
    
    BL1 --> DL1
    BL1 --> DL2
    BL2 --> DL3
    BL3 --> DL4
    BL4 --> DL4
    
    BL1 --> UL3
    BL2 --> UL4
    BL3 --> UL4
    BL4 --> UL2
    UL2 --> UL1
```

## 项目文件结构图

```mermaid
flowchart TD
    ROOT["项目根目录"] --> CONFIG["config/<br/>配置文件"]
    ROOT --> DATA["data/<br/>数据目录"]
    ROOT --> DOCS["docs/<br/>文档目录"]
    ROOT --> SRC["src/<br/>源代码"]
    ROOT --> RESULTS["results/<br/>结果输出"]
    ROOT --> PATENT["patent/<br/>专利支持"]
    
    CONFIG --> CONFIG1["project_config.py<br/>项目配置"]
    CONFIG --> CONFIG2["settings.py<br/>路径设置"]
    
    DATA --> DATA1["raw/<br/>原始数据"]
    DATA --> DATA2["processed/<br/>处理数据"]
    DATA --> DATA3["features/<br/>特征数据"]
    
    SRC --> SRC1["data_processing/<br/>数据处理"]
    SRC --> SRC2["feature_engineering/<br/>特征工程"]
    SRC --> SRC3["modeling/<br/>建模"]
    SRC --> SRC4["evaluation/<br/>评估"]
    SRC --> SRC5["utils/<br/>工具"]
    
    RESULTS --> RESULTS1["preprocessing/<br/>预处理结果"]
    RESULTS --> RESULTS2["feature_selection/<br/>特征选择结果"]
    RESULTS --> RESULTS3["modeling/<br/>建模结果"]
    RESULTS --> RESULTS4["evaluation/<br/>评估结果"]
    
    PATENT --> PATENT1["documents/<br/>技术文档"]
    PATENT --> PATENT2["figures/<br/>技术图表"]
    PATENT --> PATENT3["system.py<br/>专利系统"]
```

---

## 流程图说明

### 1. 项目整体架构流程图
展示了从原始数据到最终结果的完整技术路线，包括：
- **数据预处理**：智能清洗、标准化、目标识别
- **特征工程**：多策略选择、极端降维、生物学验证
- **数据增强**：SMOGN合成、噪声注入、质量控制
- **模型训练**：深度学习、集成学习、传统算法
- **综合评估**：LOOCV、Bootstrap、多指标评估
- **智能融合**：贝叶斯权重、动态选择、最优组合

### 2. 核心技术创新点流程图
突出展示了项目的5大技术创新：
- **小样本优化策略**：样本/特征比优化、复杂度控制
- **双重数据增强**：SMOGN+噪声的组合策略
- **极端特征工程**：90%降维的高效选择
- **贝叶斯集成**：智能权重学习和动态融合
- **全方位评估**：小样本专用的评估体系

### 3. 数据流转换过程图
展示了数据在整个流程中的维度变化和性能提升：
- 数据维度：36×29 → 36×3 → 108×3
- 性能指标：CV R² -2.86 → 0.56 → 0.77 → 0.987
- 模型复杂度：高风险 → 平衡控制 → 稳定融合 → 最优解

### 4. 技术栈架构图
展示了项目的分层架构设计：
- **应用层**：主运行脚本和配置管理
- **业务逻辑层**：四大核心模块
- **算法层**：核心算法实现
- **数据层**：数据存储和管理
- **工具层**：基础工具和库支持

### 5. 项目文件结构图
展示了项目的完整文件组织结构，便于理解各模块的职责和关系。

这些流程图全面展示了项目的技术深度、创新性和系统性，为专利申请提供了强有力的技术支撑。