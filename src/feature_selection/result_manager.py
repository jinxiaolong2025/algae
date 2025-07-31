"""
结果管理模块

该模块负责保存和管理特征选择的结果。
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from .config import FeatureSelectionConfig
    from .utils import save_results_summary
except ImportError:
    from config import FeatureSelectionConfig
    from utils import save_results_summary


class ResultManager:
    """结果管理器"""
    
    def __init__(self, config: FeatureSelectionConfig):
        """
        初始化结果管理器
        
        Args:
            config: 特征选择配置对象
        """
        self.config = config
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_selected_features(self, selected_indices: np.ndarray, 
                             feature_names: np.ndarray,
                             fitness_score: float):
        """
        保存选择的特征信息
        
        Args:
            selected_indices: 选择的特征索引
            feature_names: 所有特征名称
            fitness_score: 适应度分数
        """
        # 创建特征选择结果DataFrame
        selected_features_df = pd.DataFrame({
            'feature_index': selected_indices,
            'feature_name': feature_names[selected_indices],
            'rank': range(1, len(selected_indices) + 1)
        })
        
        # 保存到CSV文件
        output_path = self.config.get_file_path(self.config.selected_features_file)
        selected_features_df.to_csv(output_path, index=False)
        
        print(f"选择的特征已保存到: {output_path}")
        
        # 存储到结果字典
        self.results['selected_features'] = {
            'indices': selected_indices.tolist(),
            'names': feature_names[selected_indices].tolist(),
            'count': len(selected_indices),
            'fitness_score': fitness_score
        }
    
    def save_selected_data(self, data: pd.DataFrame):
        """
        保存选择特征的数据
        
        Args:
            data: 包含选择特征的数据DataFrame
        """
        output_path = self.config.get_file_path(self.config.selected_data_file)
        data.to_csv(output_path, index=False)
        
        print(f"选择特征的数据已保存到: {output_path}")
    
    def save_best_features_list(self, selected_indices: np.ndarray,
                              feature_names: np.ndarray):
        """
        保存最佳特征列表（兼容原版本格式）
        
        Args:
            selected_indices: 选择的特征索引
            feature_names: 所有特征名称
        """
        # 创建最佳特征列表
        best_features_df = pd.DataFrame({
            'feature_index': selected_indices,
            'feature_name': feature_names[selected_indices]
        })
        
        # 保存到CSV文件
        output_path = self.config.get_file_path(self.config.best_features_list_file)
        best_features_df.to_csv(output_path, index=False)
        
        print(f"最佳特征列表已保存到: {output_path}")
    
    def save_best_features_data(self, data: pd.DataFrame):
        """
        保存最佳特征数据（兼容原版本格式）
        
        Args:
            data: 包含最佳特征的数据DataFrame
        """
        output_path = self.config.get_file_path(self.config.best_features_data_file)
        data.to_csv(output_path, index=False)
        
        print(f"最佳特征数据已保存到: {output_path}")
    
    def save_validation_results(self, validation_results: Dict[str, Any]):
        """
        保存模型验证结果
        
        Args:
            validation_results: 验证结果字典
        """
        # 转换为DataFrame格式
        results_list = []
        for model_name, result in validation_results.items():
            if isinstance(result, dict) and 'mean_r2' in result:
                results_list.append({
                    'model': model_name,
                    'mean_r2': result['mean_r2'],
                    'std_r2': result['std_r2'],
                    'min_r2': result['min_r2'],
                    'max_r2': result['max_r2'],
                    'has_error': 'error' in result
                })
        
        validation_df = pd.DataFrame(results_list)
        
        # 按平均R²降序排列
        validation_df = validation_df.sort_values('mean_r2', ascending=False)
        
        # 保存到CSV文件
        output_path = self.config.get_file_path(self.config.validation_results_file)
        validation_df.to_csv(output_path, index=False)
        
        print(f"验证结果已保存到: {output_path}")
        
        # 存储到结果字典
        self.results['validation_results'] = validation_results
    
    def save_evolution_history(self, evolution_history: Dict[str, Any]):
        """
        保存进化历史数据

        Args:
            evolution_history: 进化历史字典
        """
        # 创建进化历史DataFrame
        evolution_df = pd.DataFrame({
            'generation': range(len(evolution_history['fitness_history'])),
            'mean_fitness': evolution_history['fitness_history'],
            'best_fitness': evolution_history['best_fitness_history'],
            'diversity': evolution_history['diversity_history'],
            'selection_pressure': evolution_history['selection_pressure_history']
        })

        # 保存到CSV文件（进化过程目录）
        evolution_path = os.path.join(self.config.results_dir, self.config.evolution_process_dir, 'ga_evolution_history.csv')
        evolution_df.to_csv(evolution_path, index=False)

        print(f"进化历史已保存到: {evolution_path}")

        # 保存详细统计信息（进化过程目录）
        stats_path = os.path.join(self.config.results_dir, self.config.evolution_process_dir, 'ga_generation_stats.csv')
        if evolution_history['generation_stats']:
            stats_df = pd.DataFrame(evolution_history['generation_stats'])
            stats_df.to_csv(stats_path, index=False)
            print(f"代数统计已保存到: {stats_path}")

        # 存储到结果字典
        self.results['evolution_history'] = evolution_history
    
    def save_feature_importance(self, importance_ranking: pd.DataFrame):
        """
        保存特征重要性排序

        Args:
            importance_ranking: 特征重要性DataFrame
        """
        output_path = os.path.join(self.config.results_dir, self.config.data_analysis_dir, 'feature_importance_ranking.csv')
        importance_ranking.to_csv(output_path, index=False)

        print(f"特征重要性排序已保存到: {output_path}")

        # 存储到结果字典
        self.results['feature_importance'] = importance_ranking.to_dict('records')
    
    def save_correlation_matrix(self, correlation_matrix: pd.DataFrame):
        """
        保存特征相关性矩阵

        Args:
            correlation_matrix: 相关性矩阵DataFrame
        """
        output_path = os.path.join(self.config.results_dir, self.config.data_analysis_dir, 'feature_correlation_matrix.csv')
        correlation_matrix.to_csv(output_path)

        print(f"特征相关性矩阵已保存到: {output_path}")

        # 存储到结果字典
        self.results['correlation_matrix'] = correlation_matrix.to_dict()
    
    def save_performance_comparison(self, comparison_results: Dict[str, Any]):
        """
        保存性能比较结果

        Args:
            comparison_results: 性能比较结果字典
        """
        # 保存比较结果
        if 'comparison' in comparison_results:
            comparison_list = []
            for model_name, comp in comparison_results['comparison'].items():
                comparison_list.append({
                    'model': model_name,
                    'baseline_r2': comp['baseline_r2'],
                    'selected_r2': comp['selected_r2'],
                    'improvement': comp['improvement'],
                    'improvement_pct': comp['improvement_pct']
                })

            comparison_df = pd.DataFrame(comparison_list)
            comparison_df = comparison_df.sort_values('improvement', ascending=False)

            output_path = os.path.join(self.config.results_dir, self.config.model_validation_dir, 'performance_comparison.csv')
            comparison_df.to_csv(output_path, index=False)

            print(f"性能比较结果已保存到: {output_path}")

        # 存储到结果字典
        self.results['performance_comparison'] = comparison_results
    
    def save_complete_results(self, **kwargs):
        """
        保存完整的结果集合
        
        Args:
            **kwargs: 各种结果数据
        """
        # 更新结果字典
        self.results.update(kwargs)
        
        # 添加元数据
        self.results['metadata'] = {
            'timestamp': self.timestamp,
            'config': {
                'population_size': self.config.population_size,
                'generations': self.config.generations,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate,
                'target_features': self.config.target_features,
                'cv_folds': self.config.cv_folds,
                'random_state': self.config.random_state
            }
        }
        
        # 保存为JSON文件（最终报告目录）
        json_path = os.path.join(self.config.results_dir, self.config.final_reports_dir, 'complete_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            # 处理numpy数组和其他不可序列化的对象
            json_results = self._make_json_serializable(self.results)
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        print(f"完整结果已保存到: {json_path}")

        # 保存结果摘要（最终报告目录）
        summary_path = os.path.join(self.config.results_dir, self.config.final_reports_dir, 'results_summary.txt')
        save_results_summary(self.results, summary_path)
        print(f"结果摘要已保存到: {summary_path}")
    
    def _make_json_serializable(self, obj):
        """
        将对象转换为JSON可序列化格式
        
        Args:
            obj: 要转换的对象
            
        Returns:
            JSON可序列化的对象
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def load_results(self, json_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载保存的结果
        
        Args:
            json_path: JSON文件路径，如果为None则使用默认路径
            
        Returns:
            加载的结果字典
        """
        if json_path is None:
            json_path = self.config.get_file_path('complete_results.json')
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"结果文件不存在: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded_results = json.load(f)
        
        self.results = loaded_results
        return loaded_results
    
    def print_results_summary(self):
        """打印结果摘要"""
        if not self.results:
            print("没有可用的结果")
            return
        
        print("特征选择结果摘要")
        print("=" * 60)
        
        # 选择的特征信息
        if 'selected_features' in self.results:
            sf = self.results['selected_features']
            print(f"选择的特征数量: {sf['count']}")
            print(f"最佳适应度: {sf['fitness_score']:.4f}")
            print(f"选择的特征: {', '.join(sf['names'])}")
        
        # 验证结果
        if 'validation_results' in self.results:
            print(f"\n模型验证结果:")
            for model_name, result in self.results['validation_results'].items():
                if isinstance(result, dict) and 'mean_r2' in result:
                    print(f"  {model_name}: R² = {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")
        
        # 进化历史
        if 'evolution_history' in self.results:
            eh = self.results['evolution_history']
            if 'convergence_metrics' in eh:
                cm = eh['convergence_metrics']
                print(f"\n收敛指标:")
                print(f"  收敛速度: {cm['convergence_rate']:.6f}")
                print(f"  稳定性指标: {cm['stability_index']:.6f}")
                print(f"  改进比例: {cm['improvement_ratio']:.3f}")
        
        print("=" * 60)
    
    def get_results(self) -> Dict[str, Any]:
        """
        获取所有结果
        
        Returns:
            结果字典
        """
        return self.results.copy()
    
    def export_results_to_excel(self, excel_path: Optional[str] = None):
        """
        将结果导出到Excel文件

        Args:
            excel_path: Excel文件路径
        """
        if excel_path is None:
            excel_path = os.path.join(self.config.results_dir, self.config.final_reports_dir, f'feature_selection_results_{self.timestamp}.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 选择的特征
            if 'selected_features' in self.results:
                sf = self.results['selected_features']
                selected_df = pd.DataFrame({
                    'feature_index': sf['indices'],
                    'feature_name': sf['names'],
                    'rank': range(1, len(sf['indices']) + 1)
                })
                selected_df.to_excel(writer, sheet_name='Selected_Features', index=False)
            
            # 验证结果
            if 'validation_results' in self.results:
                validation_list = []
                for model_name, result in self.results['validation_results'].items():
                    if isinstance(result, dict) and 'mean_r2' in result:
                        validation_list.append({
                            'model': model_name,
                            'mean_r2': result['mean_r2'],
                            'std_r2': result['std_r2'],
                            'min_r2': result['min_r2'],
                            'max_r2': result['max_r2']
                        })
                
                if validation_list:
                    validation_df = pd.DataFrame(validation_list)
                    validation_df.to_excel(writer, sheet_name='Validation_Results', index=False)
            
            # 特征重要性
            if 'feature_importance' in self.results:
                importance_df = pd.DataFrame(self.results['feature_importance'])
                importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
            
            # 进化历史
            if 'evolution_history' in self.results:
                eh = self.results['evolution_history']
                if 'fitness_history' in eh:
                    evolution_df = pd.DataFrame({
                        'generation': range(len(eh['fitness_history'])),
                        'mean_fitness': eh['fitness_history'],
                        'best_fitness': eh['best_fitness_history'],
                        'diversity': eh['diversity_history']
                    })
                    evolution_df.to_excel(writer, sheet_name='Evolution_History', index=False)
        
        print(f"结果已导出到Excel文件: {excel_path}")
