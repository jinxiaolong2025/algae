"""
微藻脂质含量预测系统
基于训练好的SVM模型的可视化预测界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

# 设置页面配置
st.set_page_config(
    page_title="微藻脂质含量预测系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_info():
    """加载训练好的模型和相关信息"""
    try:
        import os

        # 获取当前脚本的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(current_dir))

        # 构建文件路径
        model_path = os.path.join(project_root, "results", "model_training", "trained_svm_model.pkl")
        feature_path = os.path.join(project_root, "results", "model_training", "model_features.csv")
        info_path = os.path.join(project_root, "results", "model_training", "model_info.csv")

        # 检查文件是否存在
        if not os.path.exists(model_path):
            st.error(f"模型文件不存在: {model_path}")
            return None, None, None

        # 加载模型
        model = joblib.load(model_path)

        # 加载特征信息
        feature_info = pd.read_csv(feature_path)
        feature_names = feature_info['feature_name'].tolist()

        # 加载模型信息
        model_info = pd.read_csv(info_path)

        return model, feature_names, model_info
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None, None, None

def get_feature_ranges():
    """获取特征的合理取值范围（基于原始数据）"""
    try:
        import os

        # 获取当前脚本的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(current_dir))

        # 构建原始数据路径
        raw_data_path = os.path.join(project_root, "data", "raw_analysis", "数据.xlsx")

        # 加载原始数据来获取特征范围
        raw_data = pd.read_excel(raw_data_path)

        feature_ranges = {
            'DO': (raw_data['DO'].min(), raw_data['DO'].max()),
            'electrical conductivity': (raw_data['electrical conductivity'].min(), raw_data['electrical conductivity'].max()),
            'nitrate nitrogen': (raw_data['nitrate nitrogen'].min(), raw_data['nitrate nitrogen'].max()),
            'TP': (raw_data['TP'].min(), raw_data['TP'].max()),
            'Dry cell weight': (raw_data['Dry cell weight'].min(), raw_data['Dry cell weight'].max()),
            'protein(%)': (raw_data['protein(%)'].min(), raw_data['protein(%)'].max()),
            'C(%)': (raw_data['C(%)'].min(), raw_data['C(%)'].max()),
            'H(%)': (raw_data['H(%)'].min(), raw_data['H(%)'].max()),
            'O(%)': (raw_data['O(%)'].min(), raw_data['O(%)'].max()),
            'N conversion rate(%)': (raw_data['N conversion rate(%)'].min(), raw_data['N conversion rate(%)'].max())
        }
        return feature_ranges
    except Exception as e:
        st.warning(f"无法加载原始数据，使用默认范围: {e}")
        # 基于实际数据的合理范围
        return {
            'DO': (7.0, 21.0),
            'electrical conductivity': (150.0, 900.0),
            'nitrate nitrogen': (0.0, 13.0),
            'TP': (0.0, 7.0),
            'Dry cell weight': (4.0, 112.0),
            'protein(%)': (0.0, 20.0),
            'C(%)': (20.0, 54.0),
            'H(%)': (3.0, 13.0),
            'O(%)': (24.0, 42.0),
            'N conversion rate(%)': (0.0, 1.0)
        }

def create_input_form(feature_names, feature_ranges):
    """创建输入表单"""
    st.sidebar.header("🔬 输入特征参数")
    
    # 特征单位和描述
    feature_descriptions = {
        'DO': ('mg/L', '溶解氧浓度'),
        'electrical conductivity': ('μS/cm', '电导率'),
        'nitrate nitrogen': ('mg/L', '硝态氮浓度'),
        'TP': ('mg/L', '总磷浓度'),
        'Dry cell weight': ('g/L', '干细胞重量'),
        'protein(%)': ('%', '蛋白质含量百分比'),
        'C(%)': ('%', '碳元素含量百分比'),
        'H(%)': ('%', '氢元素含量百分比'),
        'O(%)': ('%', '氧元素含量百分比'),
        'N conversion rate(%)': ('%', '氮转化率百分比')
    }
    
    input_values = {}
    
    for feature in feature_names:
        min_val, max_val = feature_ranges.get(feature, (0.0, 100.0))
        unit, description = feature_descriptions.get(feature, ('', ''))
        
        # 计算默认值（中位数）
        default_val = (min_val + max_val) / 2
        
        input_values[feature] = st.sidebar.number_input(
            f"{feature} ({unit})",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=0.01,
            help=description
        )
    
    return input_values

@st.cache_data
def get_standardization_params():
    """获取标准化参数（基于训练数据）"""
    try:
        import os

        # 获取当前脚本的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(current_dir))

        # 加载原始数据
        raw_data_path = os.path.join(project_root, "data", "raw_analysis", "数据.xlsx")
        raw_data = pd.read_excel(raw_data_path)

        # 获取特征列
        feature_columns = ['DO', 'electrical conductivity', 'nitrate nitrogen', 'TP',
                          'Dry cell weight', 'protein(%)', 'C(%)', 'H(%)', 'O(%)', 'N conversion rate(%)']

        # 计算均值和标准差
        means = raw_data[feature_columns].mean()
        stds = raw_data[feature_columns].std()

        return means, stds
    except Exception as e:
        st.error(f"无法获取标准化参数: {e}")
        return None, None

def standardize_input(input_values, feature_names):
    """标准化输入数据"""
    means, stds = get_standardization_params()

    if means is None or stds is None:
        st.error("无法进行数据标准化")
        return None

    # 创建DataFrame
    input_df = pd.DataFrame([input_values])

    # 标准化
    standardized_values = {}
    for feature in feature_names:
        if feature in means.index and feature in stds.index:
            standardized_values[feature] = (input_values[feature] - means[feature]) / stds[feature]
        else:
            standardized_values[feature] = input_values[feature]

    return standardized_values

def make_prediction(model, feature_names, input_values):
    """进行预测"""
    # 标准化输入数据
    standardized_values = standardize_input(input_values, feature_names)

    if standardized_values is None:
        return None

    # 准备输入数据
    input_data = pd.DataFrame([standardized_values])
    input_data = input_data[feature_names]  # 确保特征顺序正确

    # 进行预测
    prediction = model.predict(input_data)[0]

    return prediction

def display_prediction_result(prediction):
    """显示预测结果"""
    st.header("🎯 预测结果")
    
    # 创建三列布局
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="预测脂质含量",
            value=f"{prediction:.2f}%",
            delta=None
        )
    
    with col2:
        # 根据脂质含量给出评级
        if prediction >= 20:
            grade = "优秀"
            color = "green"
        elif prediction >= 15:
            grade = "良好"
            color = "blue"
        elif prediction >= 10:
            grade = "一般"
            color = "orange"
        else:
            grade = "较低"
            color = "red"
        
        st.metric(
            label="脂质含量等级",
            value=grade,
            delta=None
        )
    
    with col3:
        # 显示置信度（基于模型性能）
        confidence = "中等"  # 基于R²=0.59
        st.metric(
            label="预测置信度",
            value=confidence,
            delta=None
        )

def create_feature_importance_chart():
    """创建特征重要性图表"""
    # 这里可以基于RandomForest的特征重要性
    feature_importance = {
        'H(%)': 0.2891,
        'O(%)': 0.2016,
        'nitrate nitrogen': 0.1047,
        'DO': 0.0906,
        'Dry cell weight': 0.0666,
        'TP': 0.0550,
        'N conversion rate(%)': 0.0540,
        'electrical conductivity': 0.0484,
        'protein(%)': 0.0461,
        'C(%)': 0.0439
    }
    
    fig = px.bar(
        x=list(feature_importance.values()),
        y=list(feature_importance.keys()),
        orientation='h',
        title="特征重要性排序",
        labels={'x': '重要性得分', 'y': '特征名称'}
    )
    fig.update_layout(height=400)
    
    return fig

def main():
    """主函数"""
    # 页面标题
    st.title("🧬 微藻脂质含量预测系统")
    st.markdown("---")
    
    # 加载模型
    model, feature_names, model_info = load_model_and_info()
    
    if model is None:
        st.error("❌ 无法加载模型，请确保模型文件存在！")
        return
    
    # 显示模型信息
    with st.expander("📊 模型信息", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**模型参数:**")
            st.write(f"- 模型类型: {model_info['model_type'].iloc[0]}")
            st.write(f"- 核函数: {model_info['kernel'].iloc[0]}")
            st.write(f"- C参数: {model_info['C'].iloc[0]}")
            st.write(f"- Gamma: {model_info['gamma'].iloc[0]}")
        
        with col2:
            st.write("**模型性能:**")
            st.write(f"- 训练集R²: {model_info['train_r2'].iloc[0]:.4f}")
            st.write(f"- 训练集MAE: {model_info['train_mae'].iloc[0]:.4f}")
            st.write(f"- 特征数量: {model_info['n_features'].iloc[0]}")
    
    # 获取特征范围
    feature_ranges = get_feature_ranges()
    
    # 创建输入表单
    input_values = create_input_form(feature_names, feature_ranges)
    
    # 预测按钮
    if st.sidebar.button("🚀 开始预测", type="primary"):
        with st.spinner("正在预测中..."):
            prediction = make_prediction(model, feature_names, input_values)
            
            # 显示预测结果
            display_prediction_result(prediction)
            
            # 显示输入参数
            st.subheader("📋 输入参数")
            input_df = pd.DataFrame([input_values]).T
            input_df.columns = ['输入值']
            st.dataframe(input_df, use_container_width=True)
    
    # 显示特征重要性
    st.subheader("📈 特征重要性分析")
    fig = create_feature_importance_chart()
    st.plotly_chart(fig, use_container_width=True)
    
    # 使用说明
    with st.expander("📖 使用说明", expanded=False):
        st.markdown("""
        ### 如何使用本预测系统：
        
        1. **输入参数**: 在左侧边栏中输入各项特征参数
        2. **参数范围**: 每个参数都有合理的取值范围，基于训练数据确定
        3. **开始预测**: 点击"开始预测"按钮获得脂质含量预测结果
        4. **结果解读**: 
           - 预测值为脂质含量百分比
           - 等级评定：优秀(≥20%), 良好(15-20%), 一般(10-15%), 较低(<10%)
        
        ### 特征说明：
        - **DO**: 溶解氧浓度，影响微藻的呼吸和代谢
        - **电导率**: 反映水体中离子浓度
        - **硝态氮**: 重要的氮源营养元素
        - **总磷(TP)**: 重要的磷源营养元素
        - **元素含量(C%, H%, O%)**: 微藻细胞的元素组成
        - **蛋白质含量**: 反映微藻的营养状态
        """)

if __name__ == "__main__":
    main()
