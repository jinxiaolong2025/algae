"""
模型测试模块
加载训练好的模型进行测试
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_trained_model():
    """加载训练好的模型和相关信息"""
    # 加载模型
    model = joblib.load("../../results/model_training/trained_svm_model.pkl")

    # 加载特征信息
    feature_info = pd.read_csv("../../results/model_training/model_features.csv")
    feature_names = feature_info['feature_name'].tolist()

    # 加载模型信息
    model_info = pd.read_csv("../../results/model_training/model_info.csv")

    return model, feature_names, model_info

def load_test_data():
    """加载测试数据"""
    test_data = pd.read_csv("../../data/processed/test_data.csv")
    return test_data

def prepare_test_data(test_data, feature_names):
    """准备测试数据"""
    # 提取特征
    X_test = test_data[feature_names]

    # 提取目标变量
    y_test = test_data['lipid(%)']

    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """评估模型在测试集上的性能"""
    # 预测
    y_test_pred = model.predict(X_test)

    # 计算评估指标
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return {
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'y_test_pred': y_test_pred
    }

def display_test_results(test_results, model_info, feature_names):
    """显示测试结果"""
    print("="*60)
    print("SVM模型测试结果")
    print("="*60)

    print(f"\n模型信息:")
    print(f"  - 模型类型: {model_info['model_type'].iloc[0]}")
    print(f"  - 核函数: {model_info['kernel'].iloc[0]}")
    print(f"  - C: {model_info['C'].iloc[0]}")
    print(f"  - gamma: {model_info['gamma'].iloc[0]}")
    print(f"  - epsilon: {model_info['epsilon'].iloc[0]}")
    print(f"  - degree: {model_info['degree'].iloc[0]}")
    print(f"  - 特征数量: {model_info['n_features'].iloc[0]}")

    print(f"\n使用的特征:")
    for i, feature in enumerate(feature_names, 1):
        print(f"  {i}. {feature}")

    print(f"\n训练性能 :")
    print(f"  - 训练集R²: {model_info['train_r2'].iloc[0]:.4f}")
    print(f"  - 训练集MAE: {model_info['train_mae'].iloc[0]:.4f}")
    print(f"  - 训练集RMSE: {model_info['train_rmse'].iloc[0]:.4f}")

    print(f"\n测试性能:")
    print(f"  - 测试集R²: {test_results['test_r2']:.4f}")
    print(f"  - 测试集MAE: {test_results['test_mae']:.4f}")
    print(f"  - 测试集RMSE: {test_results['test_rmse']:.4f}")

    # 过拟合分析
    train_r2 = model_info['train_r2'].iloc[0]
    test_r2 = test_results['test_r2']
    overfitting = train_r2 - test_r2

    print(f"\n模型分析:")
    print(f"  - 过拟合程度: {overfitting:.4f}")

    if overfitting > 0.1:
        overfitting_level = "高"
    elif overfitting > 0.05:
        overfitting_level = "中"
    elif overfitting > -0.05:
        overfitting_level = "低"
    else:
        overfitting_level = "无 (良好泛化)"

    print(f"  - 过拟合风险: {overfitting_level}")

    # 模型质量评估
    if test_r2 >= 0.8:
        quality = "优秀"
    elif test_r2 >= 0.6:
        quality = "良好"
    elif test_r2 >= 0.4:
        quality = "一般"
    else:
        quality = "较差"

    print(f"  - 测试质量: {quality}")

def save_test_results(y_test, test_results):
    """保存测试结果"""
    # 保存测试集预测结果
    test_predictions = pd.DataFrame({
        'actual': y_test,
        'predicted': test_results['y_test_pred'],
        'residual': y_test - test_results['y_test_pred']
    })

    test_predictions.to_csv("../../results/model_testing/test_predictions.csv", index=False, float_format='%.6f')

    # 保存测试性能指标
    test_metrics = pd.DataFrame([{
        'test_r2': test_results['test_r2'],
        'test_mae': test_results['test_mae'],
        'test_rmse': test_results['test_rmse']
    }])

    test_metrics.to_csv("../../results/model_testing/test_metrics.csv", index=False, float_format='%.6f')

    print(f"\n测试结果已保存:")
    print(f"  - 测试预测结果: results/model_testing/test_predictions.csv")
    print(f"  - 测试性能指标: results/model_testing/test_metrics.csv")

def analyze_predictions(y_test, y_test_pred):
    """分析预测结果"""
    print(f"\n预测结果分析:")

    # 计算残差统计
    residuals = y_test - y_test_pred

    print(f"  - 样本数量: {len(y_test)}")
    print(f"  - 实际值范围: {y_test.min():.2f} ~ {y_test.max():.2f}")
    print(f"  - 预测值范围: {y_test_pred.min():.2f} ~ {y_test_pred.max():.2f}")
    print(f"  - 残差均值: {residuals.mean():.4f}")
    print(f"  - 残差标准差: {residuals.std():.4f}")
    print(f"  - 最大正残差: {residuals.max():.4f}")
    print(f"  - 最大负残差: {residuals.min():.4f}")

if __name__ == "__main__":
    print("微藻脂质含量预测 - SVM模型测试")
    print("="*60)

    # 1. 加载训练好的模型
    print("1. 加载训练好的模型...")
    model, feature_names, model_info = load_trained_model()

    # 2. 加载测试数据
    print("\n2. 加载测试数据...")
    test_data = load_test_data()
    X_test, y_test = prepare_test_data(test_data, feature_names)

    # 3. 进行预测和评估
    print("\n3. 进行模型测试...")
    test_results = evaluate_model(model, X_test, y_test)

    # 4. 显示测试结果
    print("\n4. 测试结果分析:")
    display_test_results(test_results, model_info, feature_names)

    # 5. 分析预测结果
    analyze_predictions(y_test, test_results['y_test_pred'])

    # 6. 保存测试结果
    print("\n5. 保存测试结果...")
    save_test_results(y_test, test_results)

    print("\n" + "="*60)
    print("SVM模型测试完成!")
    print("="*60)