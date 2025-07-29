"""
模型训练模块
使用SVM最佳参数训练模型并保存
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_training_data():
    """加载训练数据"""
    # 加载训练数据（包含目标变量）
    train_data = pd.read_csv("../../data/processed/train_data.csv")

    # 加载筛选后的特征数据
    features_common = pd.read_csv("../../results/feature_selection/best_features_list.csv")

    return train_data, features_common

def prepare_training_data(train_data, features_common):
    """准备训练数据"""
    # 获取特征的列名
    feature_names = features_common.columns.tolist()

    # 从训练数据中提取对应特征
    X_train = train_data[feature_names]

    # 提取目标变量
    y_train = train_data['lipid(%)']

    return X_train, y_train, feature_names

def train_svm_model(X_train, y_train):
    """训练SVM最佳参数模型"""
    # 使用之前找到的最佳参数
    model = SVR(C=10.0, kernel='poly', gamma=0.1, epsilon=0.1, degree=3)
    model.fit(X_train, y_train)
    return model

def evaluate_training(model, X_train, y_train):
    """评估训练性能"""
    # 预测
    y_train_pred = model.predict(X_train)

    # 计算评估指标
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    return {
        'train_r2': train_r2,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'y_train_pred': y_train_pred
    }

def display_training_results(results, feature_names):
    """显示训练结果"""
    print("="*60)
    print("SVM模型训练结果")
    print("="*60)

    print(f"\n使用特征数量: {len(feature_names)}")
    print(f"特征列表:")
    for i, feature in enumerate(feature_names, 1):
        print(f"  {i}. {feature}")

    print(f"\n模型参数:")
    print(f"  - C: 10.0")
    print(f"  - kernel: poly")
    print(f"  - gamma: 0.1")
    print(f"  - epsilon: 0.1")
    print(f"  - degree: 3")

    print(f"\n训练性能:")
    print(f"  - 训练集R²: {results['train_r2']:.4f}")
    print(f"  - 训练集MAE: {results['train_mae']:.4f}")
    print(f"  - 训练集RMSE: {results['train_rmse']:.4f}")

    # 训练质量评估
    if results['train_r2'] >= 0.8:
        quality = "优秀"
    elif results['train_r2'] >= 0.6:
        quality = "良好"
    elif results['train_r2'] >= 0.4:
        quality = "一般"
    else:
        quality = "较差"

    print(f"  - 训练质量: {quality}")

def save_model_and_info(model, feature_names, results):
    """保存训练好的模型和相关信息"""
    # 保存模型
    model_path = "../../results/model_training/trained_svm_model.pkl"
    joblib.dump(model, model_path)

    # 保存特征名称
    feature_info = pd.DataFrame({
        'feature_name': feature_names,
        'feature_index': range(len(feature_names))
    })
    feature_info.to_csv("../../results/model_training/model_features.csv", index=False)

    # 保存模型信息
    model_info = {
        'model_type': 'SVM',
        'kernel': 'poly',
        'C': 10.0,
        'gamma': 0.1,
        'epsilon': 0.1,
        'degree': 3,
        'n_features': len(feature_names),
        'train_r2': results['train_r2'],
        'train_mae': results['train_mae'],
        'train_rmse': results['train_rmse']
    }

    model_info_df = pd.DataFrame([model_info])
    model_info_df.to_csv("../../results/model_training/model_info.csv", index=False)

    # 保存训练集预测结果
    train_results = pd.DataFrame({
        'actual': results['y_train_actual'],
        'predicted': results['y_train_pred'],
        'residual': results['y_train_actual'] - results['y_train_pred']
    })
    train_results.to_csv("../../results/model_training/train_predictions.csv", index=False, float_format='%.6f')

    print(f"\n模型和相关信息已保存:")
    print(f"  - 训练好的模型: {model_path}")
    print(f"  - 特征信息: results/model_training/model_features.csv")
    print(f"  - 模型信息: results/model_training/model_info.csv")
    print(f"  - 训练预测结果: results/model_training/train_predictions.csv")

if __name__ == "__main__":
    print("微藻脂质含量预测 - SVM模型训练")
    print("="*60)

    # 1. 加载训练数据
    print("1. 加载训练数据...")
    train_data, features_common = load_training_data()
    X_train, y_train, feature_names = prepare_training_data(train_data, features_common)
    print(f"   训练数据准备完成: {X_train.shape}")
    print(f"   使用特征数量: {len(feature_names)}")

    # 2. 训练SVM模型
    print("\n2. 训练SVM模型...")
    model = train_svm_model(X_train, y_train)
    print("   SVM模型训练完成")

    # 3. 评估训练性能
    print("\n3. 评估训练性能...")
    results = evaluate_training(model, X_train, y_train)
    results['y_train_actual'] = y_train  # 添加实际值用于保存

    # 4. 显示训练结果
    print("\n4. 训练结果分析:")
    display_training_results(results, feature_names)

    # 5. 保存模型和相关信息
    print("\n5. 保存模型...")
    save_model_and_info(model, feature_names, results)

    print("\n" + "="*60)
    print("SVM模型训练完成!")
    print("模型已保存，可用于后续测试")
    print("="*60)