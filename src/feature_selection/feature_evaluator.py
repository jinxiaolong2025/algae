"""
特征评估模块

该模块负责评估特征的重要性和选择结果的验证。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from .config import FeatureSelectionConfig, ModelEvaluationError
    from .model_factory import ModelFactory
    from .utils import (
        evaluate_model_performance,
        calculate_feature_importance_ranking,
        calculate_feature_correlation_matrix,
        find_highly_correlated_features,
        validate_input_data
    )
except ImportError:
    from config import FeatureSelectionConfig, ModelEvaluationError
    from model_factory import ModelFactory
    from utils import (
        evaluate_model_performance,
        calculate_feature_importance_ranking,
        calculate_feature_correlation_matrix,
        find_highly_correlated_features,
        validate_input_data
    )


class FeatureEvaluator:
    """特征评估器"""
    
    def __init__(self, config: FeatureSelectionConfig):
        """
        初始化特征评估器
        
        Args:
            config: 特征选择配置对象
        """
        self.config = config
        self.model_factory = ModelFactory(config)
        
        # 存储评估结果
        self.feature_importance_ranking = None
        self.correlation_matrix = None
        self.validation_results = {}
        self.performance_comparison = {}
    
    def calculate_baseline_performance(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: np.ndarray) -> Dict[str, Any]:
        """
        计算使用所有特征的基线性能
        
        Args:
            X: 特征数据
            y: 目标变量
            feature_names: 特征名称
            
        Returns:
            基线性能结果字典
        """
        validate_input_data(X, y, feature_names)
        
        print("计算基线性能（使用所有特征）...")
        
        # 创建验证模型
        models = self.model_factory.create_validation_models()
        baseline_results = {}
        
        for model_name, model in models.items():
            try:
                performance = evaluate_model_performance(
                    model, X, y, 
                    cv_folds=self.config.cv_folds,
                    random_state=self.config.random_state
                )
                baseline_results[model_name] = performance
                print(f"  {model_name}: R² = {performance['mean_r2']:.4f} ± {performance['std_r2']:.4f}")
                
            except Exception as e:
                print(f"  {model_name}: 评估失败 - {str(e)}")
                baseline_results[model_name] = {
                    'mean_r2': np.nan,
                    'std_r2': np.nan,
                    'error': str(e)
                }
        
        return baseline_results
    
    def evaluate_selected_features(self, X: np.ndarray, y: np.ndarray, 
                                 selected_indices: np.ndarray,
                                 feature_names: np.ndarray) -> Dict[str, Any]:
        """
        评估选择的特征性能
        
        Args:
            X: 原始特征数据
            y: 目标变量
            selected_indices: 选择的特征索引
            feature_names: 特征名称
            
        Returns:
            选择特征的性能结果字典
        """
        validate_input_data(X, y, feature_names)
        
        print(f"评估选择的特征性能（{len(selected_indices)}个特征）...")
        
        # 获取选择的特征数据
        X_selected = X[:, selected_indices]
        selected_feature_names = feature_names[selected_indices]
        
        # 创建验证模型
        models = self.model_factory.create_validation_models()
        selected_results = {}
        
        print("模型验证结果:")
        print("=" * 70)
        print(f"{'模型名称':<20} {'平均R²':<12} {'标准差':<12} {'最佳R²':<12} {'最差R²':<12}")
        print("-" * 70)
        
        for model_name, model in models.items():
            try:
                performance = evaluate_model_performance(
                    model, X_selected, y,
                    cv_folds=self.config.cv_folds,
                    random_state=self.config.random_state
                )
                selected_results[model_name] = performance
                
                print(f"{model_name:<20} {performance['mean_r2']:<12.4f} "
                      f"{performance['std_r2']:<12.4f} {performance['max_r2']:<12.4f} "
                      f"{performance['min_r2']:<12.4f}")
                
            except Exception as e:
                print(f"{model_name:<20} 训练失败: {str(e)[:30]}...")
                selected_results[model_name] = {
                    'mean_r2': np.nan,
                    'std_r2': np.nan,
                    'min_r2': np.nan,
                    'max_r2': np.nan,
                    'scores': [np.nan] * self.config.cv_folds,
                    'error': str(e)
                }
        
        print("=" * 70)
        
        # 找出最佳模型
        valid_results = {k: v for k, v in selected_results.items()
                        if not np.isnan(v['mean_r2'])}
        
        if valid_results:
            best_model = max(valid_results.keys(), key=lambda k: valid_results[k]['mean_r2'])
            print(f"最佳模型: {best_model} (R² = {valid_results[best_model]['mean_r2']:.4f})")
        else:
            print("警告: 所有模型验证都失败了")
            best_model = None
        
        self.validation_results = selected_results
        
        return {
            'results': selected_results,
            'best_model': best_model,
            'selected_features': selected_feature_names.tolist(),
            'n_features': len(selected_indices)
        }
    
    def compare_feature_selection_performance(self, X: np.ndarray, y: np.ndarray,
                                            selected_indices: np.ndarray,
                                            feature_names: np.ndarray) -> Dict[str, Any]:
        """
        比较特征选择前后的性能
        
        Args:
            X: 原始特征数据
            y: 目标变量
            selected_indices: 选择的特征索引
            feature_names: 特征名称
            
        Returns:
            性能比较结果字典
        """
        print("比较特征选择前后的性能...")
        
        # 计算基线性能
        baseline_results = self.calculate_baseline_performance(X, y, feature_names)
        
        # 计算选择特征性能
        selected_results = self.evaluate_selected_features(X, y, selected_indices, feature_names)
        
        # 计算性能改进
        comparison_results = {}
        
        for model_name in baseline_results.keys():
            if (model_name in selected_results['results'] and 
                not np.isnan(baseline_results[model_name]['mean_r2']) and
                not np.isnan(selected_results['results'][model_name]['mean_r2'])):
                
                baseline_r2 = baseline_results[model_name]['mean_r2']
                selected_r2 = selected_results['results'][model_name]['mean_r2']
                improvement = selected_r2 - baseline_r2
                improvement_pct = (improvement / abs(baseline_r2)) * 100 if baseline_r2 != 0 else 0
                
                comparison_results[model_name] = {
                    'baseline_r2': baseline_r2,
                    'selected_r2': selected_r2,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                }
        
        self.performance_comparison = {
            'baseline': baseline_results,
            'selected': selected_results,
            'comparison': comparison_results
        }
        
        # 打印比较结果
        print("\n性能比较结果:")
        print("=" * 80)
        print(f"{'模型名称':<20} {'基线R²':<12} {'选择R²':<12} {'改进':<12} {'改进%':<12}")
        print("-" * 80)
        
        for model_name, comp in comparison_results.items():
            print(f"{model_name:<20} {comp['baseline_r2']:<12.4f} "
                  f"{comp['selected_r2']:<12.4f} {comp['improvement']:<12.4f} "
                  f"{comp['improvement_pct']:<12.2f}")
        
        print("=" * 80)
        
        return self.performance_comparison
    
    def analyze_feature_importance(self, X: np.ndarray, y: np.ndarray,
                                 feature_names: np.ndarray) -> pd.DataFrame:
        """
        分析特征重要性
        
        Args:
            X: 特征数据
            y: 目标变量
            feature_names: 特征名称
            
        Returns:
            特征重要性DataFrame
        """
        print("分析特征重要性...")
        
        self.feature_importance_ranking = calculate_feature_importance_ranking(X, y, feature_names)
        
        print("特征重要性排序（前10名）:")
        print("-" * 40)
        top_10 = self.feature_importance_ranking.head(10)
        for idx, row in top_10.iterrows():
            print(f"{row['rank']:2d}. {row['feature_name']:<25} {row['importance']:.4f}")
        
        return self.feature_importance_ranking
    
    def analyze_feature_correlation(self, X: np.ndarray, 
                                  feature_names: np.ndarray,
                                  threshold: float = 0.8) -> Tuple[pd.DataFrame, List[Tuple]]:
        """
        分析特征相关性
        
        Args:
            X: 特征数据
            feature_names: 特征名称
            threshold: 高相关性阈值
            
        Returns:
            相关性矩阵和高相关特征对列表
        """
        print("分析特征相关性...")
        
        self.correlation_matrix = calculate_feature_correlation_matrix(X, feature_names)
        highly_correlated = find_highly_correlated_features(self.correlation_matrix, threshold)
        
        if highly_correlated:
            print(f"发现{len(highly_correlated)}对高度相关的特征（|r| >= {threshold}）:")
            for feature1, feature2, corr in highly_correlated:
                print(f"  {feature1} - {feature2}: {corr:.3f}")
        else:
            print(f"未发现高度相关的特征对（|r| >= {threshold}）")
        
        return self.correlation_matrix, highly_correlated
    
    def evaluate_feature_stability(self, X: np.ndarray, y: np.ndarray,
                                 selected_indices: np.ndarray,
                                 n_runs: int = 10) -> Dict[str, Any]:
        """
        评估特征选择的稳定性
        
        Args:
            X: 特征数据
            y: 目标变量
            selected_indices: 选择的特征索引
            n_runs: 运行次数
            
        Returns:
            稳定性评估结果
        """
        print(f"评估特征选择稳定性（{n_runs}次运行）...")
        
        X_selected = X[:, selected_indices]
        model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state
        )
        
        scores = []
        for i in range(n_runs):
            # 使用不同的随机种子
            kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=i)
            run_scores = cross_val_score(model, X_selected, y, cv=kf, scoring='r2')
            scores.extend(run_scores)
        
        stability_metrics = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'coefficient_of_variation': np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else np.inf,
            'n_runs': n_runs,
            'total_evaluations': len(scores)
        }
        
        print(f"稳定性评估结果:")
        print(f"  平均R²: {stability_metrics['mean_score']:.4f}")
        print(f"  标准差: {stability_metrics['std_score']:.4f}")
        print(f"  变异系数: {stability_metrics['coefficient_of_variation']:.4f}")
        print(f"  分数范围: [{stability_metrics['min_score']:.4f}, {stability_metrics['max_score']:.4f}]")
        
        return stability_metrics
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """
        生成评估报告
        
        Returns:
            完整的评估报告字典
        """
        report = {
            'feature_importance': self.feature_importance_ranking.to_dict('records') if self.feature_importance_ranking is not None else None,
            'correlation_analysis': {
                'correlation_matrix': self.correlation_matrix.to_dict() if self.correlation_matrix is not None else None,
                'highly_correlated_pairs': find_highly_correlated_features(self.correlation_matrix) if self.correlation_matrix is not None else []
            },
            'validation_results': self.validation_results,
            'performance_comparison': self.performance_comparison,
            'config': {
                'cv_folds': self.config.cv_folds,
                'random_state': self.config.random_state,
                'target_features': self.config.target_features
            }
        }
        
        return report
