"""
特征选择模块

该模块提供基于遗传算法的特征选择功能，包括完整的特征选择流程、
模型验证、可视化分析和结果管理。

主要功能:
- 遗传算法特征选择
- 多模型性能验证
- 特征重要性分析
- 可视化图表生成
- 结果保存和管理

使用示例:
    # 基本使用
    from src.feature_selection import main
    main()
    
    # 自定义配置
    from src.feature_selection import FeatureSelectionConfig, main
    config = FeatureSelectionConfig(population_size=50, generations=200)
    main(config)
    
    # 编程接口
    from src.feature_selection import create_feature_selection_pipeline
    selector, evaluator, visualizer, result_manager = create_feature_selection_pipeline()
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# 导入核心模块
try:
    from .config import FeatureSelectionConfig, get_default_config
    from .genetic_algorithm import GeneticAlgorithmFeatureSelector
    from .model_factory import ModelFactory
    from .feature_evaluator import FeatureEvaluator
    from .visualization import FeatureSelectionVisualizer
    from .data_handler import DataHandler
    from .result_manager import ResultManager
    from .utils import print_section_header, log_processing_step
except ImportError:
    # 支持直接运行
    from config import FeatureSelectionConfig, get_default_config
    from genetic_algorithm import GeneticAlgorithmFeatureSelector
    from model_factory import ModelFactory
    from feature_evaluator import FeatureEvaluator
    from visualization import FeatureSelectionVisualizer
    from data_handler import DataHandler
    from result_manager import ResultManager
    from utils import print_section_header, log_processing_step


def create_feature_selection_pipeline(config: Optional[FeatureSelectionConfig] = None) -> Tuple:
    """
    创建特征选择处理管道
    
    Args:
        config: 特征选择配置，如果为None则使用默认配置
        
    Returns:
        包含所有组件的元组 (selector, evaluator, visualizer, result_manager, data_handler, model_factory)
    """
    if config is None:
        config = get_default_config()
    
    # 验证配置
    config.validate_config()
    
    # 创建各个组件
    data_handler = DataHandler(config)
    model_factory = ModelFactory(config)
    selector = GeneticAlgorithmFeatureSelector(config)
    evaluator = FeatureEvaluator(config)
    visualizer = FeatureSelectionVisualizer(config)
    result_manager = ResultManager(config)
    
    return selector, evaluator, visualizer, result_manager, data_handler, model_factory


def main(config: Optional[FeatureSelectionConfig] = None) -> Dict[str, Any]:
    """
    执行完整的特征选择流程
    
    Args:
        config: 特征选择配置，如果为None则使用默认配置
        
    Returns:
        包含所有结果的字典
    """
    # 使用默认配置或传入的配置
    if config is None:
        config = get_default_config()
    
    print_section_header("遗传算法特征选择系统", 80)
    config.print_config_summary()
    
    # 创建处理管道
    selector, evaluator, visualizer, result_manager, data_handler, model_factory = create_feature_selection_pipeline(config)
    
    try:
        # 步骤1: 加载数据
        log_processing_step(1, "加载训练数据")
        X, y, feature_names = data_handler.load_training_data()
        data_handler.print_data_info()
        
        # 步骤2: 数据质量检查
        log_processing_step(2, "数据质量检查")
        quality_report = data_handler.check_data_quality()
        print(f"数据质量检查完成，发现 {quality_report['missing_values']['total']} 个缺失值")
        
        # 步骤3: 特征重要性分析
        log_processing_step(3, "分析特征重要性")
        importance_ranking = evaluator.analyze_feature_importance(X, y, feature_names)
        
        # 步骤4: 特征相关性分析
        log_processing_step(4, "分析特征相关性")
        correlation_matrix, highly_correlated = evaluator.analyze_feature_correlation(X, feature_names)
        
        # 步骤5: 创建评估模型
        log_processing_step(5, "创建评估模型")
        evaluation_model = model_factory.create_evaluation_model()
        print(f"使用评估模型: {type(evaluation_model).__name__}")
        
        # 步骤6: 运行遗传算法
        log_processing_step(6, "运行遗传算法特征选择")
        selector.fit(X, y, evaluation_model)
        selected_indices = selector.get_selected_features()
        evolution_history = selector.get_evolution_history()
        
        print(f"遗传算法完成，选择了 {len(selected_indices)} 个特征")
        print(f"选择的特征: {feature_names[selected_indices].tolist()}")
        
        # 步骤7: 模型验证
        log_processing_step(7, "验证选择的特征")
        validation_results = evaluator.evaluate_selected_features(X, y, selected_indices, feature_names)
        
        # 步骤8: 性能比较
        log_processing_step(8, "比较特征选择前后的性能")
        performance_comparison = evaluator.compare_feature_selection_performance(X, y, selected_indices, feature_names)
        
        # 步骤9: 稳定性评估
        log_processing_step(9, "评估特征选择稳定性")
        stability_metrics = evaluator.evaluate_feature_stability(X, y, selected_indices)
        
        # 步骤10: 生成可视化图表
        log_processing_step(10, "生成可视化图表")
        visualizer.create_all_visualizations(
            evolution_history=evolution_history,
            validation_results=validation_results['results'],
            selected_indices=selected_indices,
            feature_names=feature_names,
            importance_ranking=importance_ranking,
            correlation_matrix=correlation_matrix
        )
        
        # 步骤11: 保存结果
        log_processing_step(11, "保存结果")
        
        # 保存选择的特征
        result_manager.save_selected_features(selected_indices, feature_names, selector.best_fitness_)
        result_manager.save_best_features_list(selected_indices, feature_names)
        
        # 保存数据
        selected_data = data_handler.create_feature_dataframe(selected_indices)
        result_manager.save_selected_data(selected_data)
        result_manager.save_best_features_data(selected_data)
        
        # 保存验证结果
        result_manager.save_validation_results(validation_results['results'])
        
        # 保存进化历史
        result_manager.save_evolution_history(evolution_history)
        
        # 保存特征重要性和相关性
        result_manager.save_feature_importance(importance_ranking)
        result_manager.save_correlation_matrix(correlation_matrix)
        
        # 保存性能比较
        result_manager.save_performance_comparison(performance_comparison)
        
        # 保存完整结果
        complete_results = {
            'selected_features': {
                'indices': selected_indices.tolist(),
                'names': feature_names[selected_indices].tolist(),
                'count': len(selected_indices),
                'fitness_score': selector.best_fitness_
            },
            'evolution_history': evolution_history,
            'validation_results': validation_results,
            'performance_comparison': performance_comparison,
            'stability_metrics': stability_metrics,
            'feature_importance': importance_ranking.to_dict('records'),
            'correlation_analysis': {
                'correlation_matrix': correlation_matrix.to_dict(),
                'highly_correlated_pairs': highly_correlated
            },
            'data_quality': quality_report
        }
        
        result_manager.save_complete_results(**complete_results)
        
        # 导出Excel报告
        result_manager.export_results_to_excel()

        # 打印结果摘要
        print_section_header("特征选择完成", 80)
        result_manager.print_results_summary()

        print(f"\n所有结果已保存到: {config.results_dir}")
        print(" 目录结构:")
        print("  01_data_analysis/      - 数据分析阶段")
        print("  02_evolution_process/  - 遗传算法进化过程")
        print("  03_convergence_analysis/ - 收敛分析")
        print("  04_feature_selection/  - 特征选择结果")
        print("  05_model_validation/   - 模型验证结果")
        print("  06_correlation_analysis/ - 相关性分析")
        print("特征选择流程执行完成！")
        
        return complete_results
        
    except Exception as e:
        print(f"特征选择过程中出现错误: {str(e)}")
        raise


def run_quick_selection(target_features: int = 5, generations: int = 50) -> Dict[str, Any]:
    """
    运行快速特征选择
    
    Args:
        target_features: 目标特征数量
        generations: 进化代数
        
    Returns:
        选择结果字典
    """
    config = FeatureSelectionConfig(
        target_features=target_features,
        generations=generations,
        population_size=20
    )
    
    return main(config)


def run_thorough_selection(target_features: int = 8, generations: int = 200) -> Dict[str, Any]:
    """
    运行深度特征选择
    
    Args:
        target_features: 目标特征数量
        generations: 进化代数
        
    Returns:
        选择结果字典
    """
    config = FeatureSelectionConfig(
        target_features=target_features,
        generations=generations,
        population_size=50,
        cv_folds=10
    )
    
    return main(config)


# 直接运行支持
if __name__ == "__main__":
    print("直接运行特征选择模块...")
    
    # 检查是否有命令行参数
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            print("运行快速特征选择...")
            run_quick_selection()
        elif sys.argv[1] == "thorough":
            print("运行深度特征选择...")
            run_thorough_selection()
        else:
            print("未知参数，运行默认配置...")
            main()
    else:
        # 使用默认配置
        main()
