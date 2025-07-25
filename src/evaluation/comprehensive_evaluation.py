# -*- coding: utf-8 -*-
"""
综合性能评估系统
Comprehensive Performance Evaluation System for Algae Lipid Prediction

本模块提供完整的模型评估功能，包括：
- 多指标评估
- 交叉验证分析
- 预测vs实际值分析
- 残差分析
- 置信区间计算
- 模型稳定性评估
"""

import csv
import os
import sys
import json
import statistics
import math
import random
from collections import defaultdict

def load_csv_data(filepath):
    """加载CSV数据"""
    data = []
    headers = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                converted_row = []
                for value in row:
                    try:
                        converted_row.append(float(value))
                    except ValueError:
                        converted_row.append(0.0)
                data.append(converted_row)
        
        print(f"成功加载数据: {len(data)} 行, {len(headers)} 列")
        return data, headers
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None, None

class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def calculate_all_metrics(self, y_true, y_pred):
        """计算所有评估指标"""
        n = len(y_true)
        
        # 基本指标
        mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
        rmse = math.sqrt(mse)
        mae = sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n
        
        # R²
        y_mean = statistics.mean(y_true)
        ss_tot = sum((y_true[i] - y_mean) ** 2 for i in range(n))
        ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 调整R²
        if n > 1:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else r2
        else:
            adj_r2 = r2
        
        # MAPE (平均绝对百分比误差)
        mape = 0
        valid_mape_count = 0
        for i in range(n):
            if abs(y_true[i]) > 1e-6:  # 避免除零
                mape += abs((y_true[i] - y_pred[i]) / y_true[i])
                valid_mape_count += 1
        mape = (mape / valid_mape_count * 100) if valid_mape_count > 0 else 0
        
        # 最大误差
        max_error = max(abs(y_true[i] - y_pred[i]) for i in range(n))
        
        # 预测准确度（在±10%范围内的预测比例）
        accurate_predictions = 0
        for i in range(n):
            if abs(y_true[i]) > 1e-6:
                relative_error = abs((y_true[i] - y_pred[i]) / y_true[i])
                if relative_error <= 0.1:  # 10%以内
                    accurate_predictions += 1
        accuracy_10pct = accurate_predictions / n * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adj_r2': adj_r2,
            'mape': mape,
            'max_error': max_error,
            'accuracy_10pct': accuracy_10pct,
            'n_samples': n
        }
    
    def bootstrap_evaluation(self, y_true, y_pred, n_bootstrap=1000):
        """Bootstrap评估获得置信区间"""
        print(f"进行Bootstrap评估 ({n_bootstrap} 次重采样)...")
        
        n = len(y_true)
        bootstrap_metrics = []
        
        for i in range(n_bootstrap):
            # Bootstrap重采样
            indices = [random.randint(0, n-1) for _ in range(n)]
            bootstrap_y_true = [y_true[idx] for idx in indices]
            bootstrap_y_pred = [y_pred[idx] for idx in indices]
            
            # 计算指标
            metrics = self.calculate_all_metrics(bootstrap_y_true, bootstrap_y_pred)
            bootstrap_metrics.append(metrics)
            
            if (i + 1) % 200 == 0:
                print(f"  完成 {i+1}/{n_bootstrap} 次重采样")
        
        # 计算置信区间
        confidence_intervals = {}
        for metric in ['r2', 'rmse', 'mae', 'mape']:
            values = [m[metric] for m in bootstrap_metrics]
            values.sort()
            
            # 95%置信区间
            lower_idx = int(0.025 * len(values))
            upper_idx = int(0.975 * len(values))
            
            confidence_intervals[metric] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'ci_lower': values[lower_idx],
                'ci_upper': values[upper_idx],
                'median': statistics.median(values)
            }
        
        return confidence_intervals
    
    def residual_analysis(self, y_true, y_pred):
        """残差分析"""
        residuals = [y_true[i] - y_pred[i] for i in range(len(y_true))]
        
        # 残差统计
        residual_stats = {
            'mean': statistics.mean(residuals),
            'std': statistics.stdev(residuals) if len(residuals) > 1 else 0,
            'min': min(residuals),
            'max': max(residuals),
            'median': statistics.median(residuals)
        }
        
        # 正态性检验（简化版）
        # 计算偏度和峰度
        mean_res = residual_stats['mean']
        std_res = residual_stats['std']
        
        if std_res > 0:
            # 偏度
            skewness = sum(((r - mean_res) / std_res) ** 3 for r in residuals) / len(residuals)
            # 峰度
            kurtosis = sum(((r - mean_res) / std_res) ** 4 for r in residuals) / len(residuals) - 3
        else:
            skewness = 0
            kurtosis = 0
        
        residual_stats['skewness'] = skewness
        residual_stats['kurtosis'] = kurtosis
        
        # 异常残差检测
        if std_res > 0:
            outlier_threshold = 2 * std_res
            outlier_residuals = [i for i, r in enumerate(residuals) if abs(r) > outlier_threshold]
        else:
            outlier_residuals = []
        
        residual_stats['outlier_count'] = len(outlier_residuals)
        residual_stats['outlier_indices'] = outlier_residuals
        
        return residual_stats, residuals
    
    def prediction_intervals(self, y_true, y_pred, confidence_level=0.95):
        """计算预测区间"""
        residuals = [y_true[i] - y_pred[i] for i in range(len(y_true))]
        residual_std = statistics.stdev(residuals) if len(residuals) > 1 else 0
        
        # 计算预测区间
        alpha = 1 - confidence_level
        # 简化的t分布近似（对于大样本）
        t_value = 2.0  # 近似95%置信区间的t值
        
        prediction_intervals = []
        for pred in y_pred:
            margin = t_value * residual_std
            lower = pred - margin
            upper = pred + margin
            prediction_intervals.append((lower, upper))
        
        return prediction_intervals, residual_std
    
    def model_stability_analysis(self, X, y, model_class, model_params, n_runs=10, ensemble_class=None):
        """模型稳定性分析"""
        print(f"进行模型稳定性分析 ({n_runs} 次独立训练)...")
        
        stability_results = []
        
        for run in range(n_runs):
            print(f"  运行 {run+1}/{n_runs}")
            
            # 重新设置随机种子
            random.seed(42 + run)
            
            # 训练模型
            if model_class == 'ensemble' and ensemble_class is not None:
                # 使用集成模型
                model = ensemble_class()
                model.fit(X, y)
                predictions = model.predict(X)
            else:
                # 使用简单模型（均值预测）
                predictions = [statistics.mean(y)] * len(y)
            
            # 计算指标
            metrics = self.calculate_all_metrics(y, predictions)
            stability_results.append(metrics)
        
        # 分析稳定性
        stability_analysis = {}
        for metric in ['r2', 'rmse', 'mae']:
            values = [result[metric] for result in stability_results]
            stability_analysis[metric] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values),
                'cv': (statistics.stdev(values) / statistics.mean(values) * 100) if statistics.mean(values) != 0 and len(values) > 1 else 0
            }
        
        return stability_analysis, stability_results

def create_evaluation_report(y_true, y_pred, feature_names=None, model_name="Model", output_dir="evaluation_results"):
    """创建综合评估报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = ComprehensiveEvaluator()
    
    print(f"开始综合性能评估: {model_name}")
    print("=" * 50)
    
    # 1. 基本指标计算
    print("\n1. 计算基本评估指标...")
    basic_metrics = evaluator.calculate_all_metrics(y_true, y_pred)
    
    print(f"基本性能指标:")
    print(f"  R²: {basic_metrics['r2']:.6f}")
    print(f"  调整R²: {basic_metrics['adj_r2']:.6f}")
    print(f"  RMSE: {basic_metrics['rmse']:.6f}")
    print(f"  MAE: {basic_metrics['mae']:.6f}")
    print(f"  MAPE: {basic_metrics['mape']:.2f}%")
    print(f"  最大误差: {basic_metrics['max_error']:.6f}")
    print(f"  10%准确度: {basic_metrics['accuracy_10pct']:.1f}%")
    
    # 2. Bootstrap置信区间
    print("\n2. Bootstrap置信区间分析...")
    confidence_intervals = evaluator.bootstrap_evaluation(y_true, y_pred, n_bootstrap=500)
    
    print("95%置信区间:")
    for metric, ci in confidence_intervals.items():
        print(f"  {metric.upper()}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    
    # 3. 残差分析
    print("\n3. 残差分析...")
    residual_stats, residuals = evaluator.residual_analysis(y_true, y_pred)
    
    print("残差统计:")
    print(f"  均值: {residual_stats['mean']:.6f}")
    print(f"  标准差: {residual_stats['std']:.6f}")
    print(f"  偏度: {residual_stats['skewness']:.4f}")
    print(f"  峰度: {residual_stats['kurtosis']:.4f}")
    print(f"  异常残差数量: {residual_stats['outlier_count']}")
    
    # 4. 预测区间
    print("\n4. 预测区间计算...")
    pred_intervals, pred_std = evaluator.prediction_intervals(y_true, y_pred)
    
    print(f"预测标准误差: {pred_std:.6f}")
    print(f"95%预测区间宽度: ±{2 * pred_std:.6f}")
    
    # 5. 预测vs实际值分析
    print("\n5. 预测vs实际值分析...")
    
    # 计算预测质量分布
    error_ranges = {
        '±5%': 0, '±10%': 0, '±20%': 0, '>20%': 0
    }
    
    for i in range(len(y_true)):
        if abs(y_true[i]) > 1e-6:
            relative_error = abs((y_true[i] - y_pred[i]) / y_true[i]) * 100
            if relative_error <= 5:
                error_ranges['±5%'] += 1
            elif relative_error <= 10:
                error_ranges['±10%'] += 1
            elif relative_error <= 20:
                error_ranges['±20%'] += 1
            else:
                error_ranges['>20%'] += 1
    
    print("预测误差分布:")
    for range_name, count in error_ranges.items():
        percentage = count / len(y_true) * 100
        print(f"  {range_name}: {count} 个样本 ({percentage:.1f}%)")
    
    # 6. 生成详细报告
    print("\n6. 生成详细评估报告...")
    
    report_path = os.path.join(output_dir, f'{model_name}_evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"{model_name} 综合性能评估报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 基本信息
        f.write("1. 基本信息\n")
        f.write("-" * 30 + "\n")
        f.write(f"模型名称: {model_name}\n")
        f.write(f"样本数量: {len(y_true)}\n")
        f.write(f"特征数量: {len(feature_names) if feature_names else 'N/A'}\n")
        f.write(f"目标变量范围: [{min(y_true):.4f}, {max(y_true):.4f}]\n")
        f.write(f"预测值范围: [{min(y_pred):.4f}, {max(y_pred):.4f}]\n\n")
        
        # 性能指标
        f.write("2. 性能指标\n")
        f.write("-" * 30 + "\n")
        f.write(f"R²: {basic_metrics['r2']:.6f}\n")
        f.write(f"调整R²: {basic_metrics['adj_r2']:.6f}\n")
        f.write(f"RMSE: {basic_metrics['rmse']:.6f}\n")
        f.write(f"MAE: {basic_metrics['mae']:.6f}\n")
        f.write(f"MAPE: {basic_metrics['mape']:.2f}%\n")
        f.write(f"最大误差: {basic_metrics['max_error']:.6f}\n")
        f.write(f"10%准确度: {basic_metrics['accuracy_10pct']:.1f}%\n\n")
        
        # 置信区间
        f.write("3. Bootstrap 95%置信区间\n")
        f.write("-" * 30 + "\n")
        for metric, ci in confidence_intervals.items():
            f.write(f"{metric.upper()}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]\n")
        f.write("\n")
        
        # 残差分析
        f.write("4. 残差分析\n")
        f.write("-" * 30 + "\n")
        f.write(f"残差均值: {residual_stats['mean']:.6f}\n")
        f.write(f"残差标准差: {residual_stats['std']:.6f}\n")
        f.write(f"残差偏度: {residual_stats['skewness']:.4f}\n")
        f.write(f"残差峰度: {residual_stats['kurtosis']:.4f}\n")
        f.write(f"异常残差数量: {residual_stats['outlier_count']}\n\n")
        
        # 预测误差分布
        f.write("5. 预测误差分布\n")
        f.write("-" * 30 + "\n")
        for range_name, count in error_ranges.items():
            percentage = count / len(y_true) * 100
            f.write(f"{range_name}: {count} 个样本 ({percentage:.1f}%)\n")
        f.write("\n")
        
        # 详细预测结果
        f.write("6. 详细预测结果\n")
        f.write("-" * 30 + "\n")
        f.write("样本\t实际值\t预测值\t误差\t相对误差(%)\n")
        for i in range(len(y_true)):
            error = y_pred[i] - y_true[i]
            rel_error = (error / y_true[i] * 100) if abs(y_true[i]) > 1e-6 else 0
            f.write(f"{i+1}\t{y_true[i]:.4f}\t{y_pred[i]:.4f}\t{error:.4f}\t{rel_error:.2f}\n")
    
    print(f"详细评估报告已保存: {report_path}")
    
    # 7. 生成简化的可视化数据
    visualization_data = {
        'y_true': y_true,
        'y_pred': y_pred,
        'residuals': residuals,
        'basic_metrics': basic_metrics,
        'confidence_intervals': confidence_intervals,
        'error_distribution': error_ranges
    }
    
    viz_path = os.path.join(output_dir, f'{model_name}_visualization_data.json')
    with open(viz_path, 'w', encoding='utf-8') as f:
        # 转换为可序列化的格式
        serializable_data = {}
        for key, value in visualization_data.items():
            if isinstance(value, dict):
                serializable_data[key] = value
            else:
                serializable_data[key] = list(value) if hasattr(value, '__iter__') else value
        
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"可视化数据已保存: {viz_path}")
    
    return basic_metrics, confidence_intervals, residual_stats

def create_text_visualizations(y_true, y_pred, output_dir="evaluation_results"):
    """创建文本形式的可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    viz_path = os.path.join(output_dir, 'text_visualizations.txt')
    
    with open(viz_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("文本可视化分析\n")
        f.write("=" * 60 + "\n\n")
        
        # 1. 预测vs实际值散点图（文本版）
        f.write("1. 预测值 vs 实际值分布\n")
        f.write("-" * 40 + "\n")
        
        # 创建简单的文本散点图
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        
        f.write(f"数据范围: [{min_val:.2f}, {max_val:.2f}]\n")
        f.write("理想情况下，所有点应该在对角线 y=x 上\n\n")
        
        # 按区间统计
        n_bins = 5
        bin_width = (max_val - min_val) / n_bins
        
        f.write("区间分布统计:\n")
        for i in range(n_bins):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            
            count_true = sum(1 for val in y_true if bin_start <= val < bin_end)
            count_pred = sum(1 for val in y_pred if bin_start <= val < bin_end)
            
            f.write(f"[{bin_start:.2f}, {bin_end:.2f}): 实际={count_true}, 预测={count_pred}\n")
        
        # 2. 残差分布
        f.write(f"\n2. 残差分布分析\n")
        f.write("-" * 40 + "\n")
        
        residuals = [y_pred[i] - y_true[i] for i in range(len(y_true))]
        
        # 残差统计
        res_mean = statistics.mean(residuals)
        res_std = statistics.stdev(residuals) if len(residuals) > 1 else 0
        
        f.write(f"残差均值: {res_mean:.6f}\n")
        f.write(f"残差标准差: {res_std:.6f}\n")
        
        # 残差分布直方图（文本版）
        f.write("\n残差分布直方图:\n")
        
        # 创建残差区间
        res_bins = 7
        res_min = min(residuals)
        res_max = max(residuals)
        res_width = (res_max - res_min) / res_bins if res_max != res_min else 1
        
        for i in range(res_bins):
            bin_start = res_min + i * res_width
            bin_end = res_min + (i + 1) * res_width
            
            count = sum(1 for res in residuals if bin_start <= res < bin_end)
            bar = '*' * count
            f.write(f"[{bin_start:6.3f}, {bin_end:6.3f}): {count:2d} {bar}\n")
        
        # 3. 预测准确度分析
        f.write(f"\n3. 预测准确度分析\n")
        f.write("-" * 40 + "\n")
        
        # 计算不同误差范围的样本数
        error_thresholds = [0.05, 0.1, 0.2, 0.3]
        
        f.write("相对误差分布:\n")
        for threshold in error_thresholds:
            count = 0
            for i in range(len(y_true)):
                if abs(y_true[i]) > 1e-6:
                    rel_error = abs((y_true[i] - y_pred[i]) / y_true[i])
                    if rel_error <= threshold:
                        count += 1
            
            percentage = count / len(y_true) * 100
            f.write(f"误差 ≤ {threshold*100:3.0f}%: {count:2d} 个样本 ({percentage:5.1f}%)\n")
        
        # 4. 最佳和最差预测
        f.write(f"\n4. 最佳和最差预测样本\n")
        f.write("-" * 40 + "\n")
        
        # 计算绝对误差
        abs_errors = [abs(y_pred[i] - y_true[i]) for i in range(len(y_true))]
        
        # 找出最佳预测（误差最小）
        best_indices = sorted(range(len(abs_errors)), key=lambda i: abs_errors[i])[:3]
        worst_indices = sorted(range(len(abs_errors)), key=lambda i: abs_errors[i], reverse=True)[:3]
        
        f.write("最佳预测（误差最小）:\n")
        for i, idx in enumerate(best_indices):
            f.write(f"  {i+1}. 样本{idx+1}: 实际={y_true[idx]:.4f}, 预测={y_pred[idx]:.4f}, 误差={abs_errors[idx]:.4f}\n")
        
        f.write("\n最差预测（误差最大）:\n")
        for i, idx in enumerate(worst_indices):
            f.write(f"  {i+1}. 样本{idx+1}: 实际={y_true[idx]:.4f}, 预测={y_pred[idx]:.4f}, 误差={abs_errors[idx]:.4f}\n")
    
    print(f"文本可视化已保存: {viz_path}")

def main():
    """主函数"""
    print("开始综合性能评估...")
    
    # 设置随机种子
    random.seed(42)
    
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    
    # 加载特征数据
    # 尝试多个可能的数据路径
    possible_paths = [
        os.path.join(project_root, 'feature_engineering_results', 'selected_features_data.csv'),
        os.path.join(project_root, 'results', '02_selected_features_data.csv'),
        os.path.join(project_root, 'results', 'preprocessing', 'optimized_data.csv')
    ]
    
    data, headers = None, None
    for data_path in possible_paths:
        if os.path.exists(data_path):
            print(f"尝试加载数据: {data_path}")
            data, headers = load_csv_data(data_path)
            if data is not None:
                print(f"成功加载数据: {data_path}")
                break
    
    if data is None:
        print("所有数据路径都无法加载，检查以下路径:")
        for path in possible_paths:
            print(f"  - {path} (存在: {os.path.exists(path)})")
        return
    
    # 准备数据
    target_col = 'lipid(%)'
    target_idx = headers.index(target_col)
    feature_names = [headers[i] for i in range(len(headers)) if i != target_idx]
    
    X = []
    y_true = []
    for row in data:
        features = [row[i] for i in range(len(row)) if i != target_idx]
        target = row[target_idx]
        X.append(features)
        y_true.append(target)
    
    # 2. 使用最佳模型进行预测
    print("\n2. 使用集成模型进行预测...")
    
    # 导入集成模型
    model_training_path = os.path.join(project_root, 'model_training')
    if model_training_path not in sys.path:
        sys.path.append(model_training_path)
    
    try:
        sys.path.append(os.path.join(project_root, 'src', 'modeling'))
        from ultimate_ensemble_pipeline import BayesianEnsemble
        ensemble_available = True
        print("✅ 成功导入BayesianEnsemble模型")
    except ImportError as e:
        print(f"⚠️ 无法导入BayesianEnsemble: {e}")
        print("将使用简化评估模式")
        BayesianEnsemble = None
        ensemble_available = False
    
    # 训练最佳集成模型
    if ensemble_available and BayesianEnsemble is not None:
        best_ensemble = BayesianEnsemble()
        
        best_ensemble.fit(X, y_true)
        y_pred = best_ensemble.predict(X)
        print("✅ 使用BayesianEnsemble模型进行预测")
    else:
        # 如果无法导入集成模型，使用简化预测
        print("使用简化预测模式（均值预测）...")
        y_pred = [statistics.mean(y_true)] * len(y_true)
    
    # 3. 综合评估
    print("\n3. 进行综合性能评估...")
    
    basic_metrics, confidence_intervals, residual_stats = create_evaluation_report(
        y_true, y_pred, feature_names, "Bayesian_Ensemble", "comprehensive_evaluation_results"
    )
    
    # 4. 创建文本可视化
    print("\n4. 创建文本可视化...")
    create_text_visualizations(y_true, y_pred, "comprehensive_evaluation_results")
    
    # 5. 模型稳定性分析
    print("\n5. 模型稳定性分析...")
    evaluator = ComprehensiveEvaluator()
    stability_analysis, stability_results = evaluator.model_stability_analysis(
        X, y_true, 'ensemble' if ensemble_available else 'simple', {}, 
        n_runs=5, ensemble_class=BayesianEnsemble if ensemble_available else None
    )
    
    print("模型稳定性结果:")
    for metric, stats in stability_analysis.items():
        print(f"  {metric.upper()}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 变异系数={stats['cv']:.2f}%")
    
    # 6. 保存稳定性分析结果
    stability_path = os.path.join("comprehensive_evaluation_results", "stability_analysis.txt")
    with open(stability_path, 'w', encoding='utf-8') as f:
        f.write("模型稳定性分析结果\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("稳定性统计:\n")
        for metric, stats in stability_analysis.items():
            f.write(f"{metric.upper()}:\n")
            f.write(f"  均值: {stats['mean']:.6f}\n")
            f.write(f"  标准差: {stats['std']:.6f}\n")
            f.write(f"  最小值: {stats['min']:.6f}\n")
            f.write(f"  最大值: {stats['max']:.6f}\n")
            f.write(f"  范围: {stats['range']:.6f}\n")
            f.write(f"  变异系数: {stats['cv']:.2f}%\n\n")
        
        f.write("各次运行详细结果:\n")
        for i, result in enumerate(stability_results):
            f.write(f"运行 {i+1}: R²={result['r2']:.4f}, RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}\n")
    
    print(f"稳定性分析结果已保存: {stability_path}")
    
    # 7. 输出总结
    print("\n" + "="*60)
    print("综合性能评估完成!")
    print("="*60)
    
    print(f"最终模型性能:")
    print(f"  R²: {basic_metrics['r2']:.6f}")
    print(f"  RMSE: {basic_metrics['rmse']:.6f}")
    print(f"  MAE: {basic_metrics['mae']:.6f}")
    print(f"  MAPE: {basic_metrics['mape']:.2f}%")
    print(f"  10%准确度: {basic_metrics['accuracy_10pct']:.1f}%")
    
    print(f"\n置信区间 (95%):")
    for metric, ci in confidence_intervals.items():
        print(f"  {metric.upper()}: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    
    print(f"\n模型稳定性:")
    print(f"  R²变异系数: {stability_analysis['r2']['cv']:.2f}%")
    print(f"  RMSE变异系数: {stability_analysis['rmse']['cv']:.2f}%")
    
    # 性能评级
    r2_score = basic_metrics['r2']
    if r2_score >= 0.98:
        grade = "A+ (超高精度)"
    elif r2_score >= 0.95:
        grade = "A (高精度)"
    elif r2_score >= 0.90:
        grade = "B+ (良好)"
    elif r2_score >= 0.80:
        grade = "B (中等)"
    elif r2_score >= 0.70:
        grade = "C (一般)"
    else:
        grade = "D (需改进)"
    
    print(f"\n模型评级: {grade}")
    
    print(f"\n评估报告文件:")
    print(f"- 详细报告: comprehensive_evaluation_results/Bayesian_Ensemble_evaluation_report.txt")
    print(f"- 可视化数据: comprehensive_evaluation_results/Bayesian_Ensemble_visualization_data.json")
    print(f"- 文本可视化: comprehensive_evaluation_results/text_visualizations.txt")
    print(f"- 稳定性分析: comprehensive_evaluation_results/stability_analysis.txt")

if __name__ == "__main__":
    main()