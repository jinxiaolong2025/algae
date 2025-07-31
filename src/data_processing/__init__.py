"""数据预处理模块

模块化的数据预处理解决方案，提供从原始数据加载到处理完成的全流程自动化。

主要功能：
- 数据加载和基本验证
- 数据质量分析
- 缺失值智能处理
- 异常值检测和处理
- 数据分布分析
- 可视化报告生成
- 数据集分割和保存

使用方法：
    # 方法1: 使用便捷接口
    from data_processing import main
    main()
    
    # 方法2: 自定义配置
    from data_processing import ProcessingConfig, create_processing_pipeline
    config = ProcessingConfig()
    config.test_size = 0.3  # 自定义测试集比例
    
    # 创建处理管道
    loader, quality_analyzer, missing_handler, outlier_handler, scaler, visualizer = create_processing_pipeline(config)
    
    # 执行处理流程
    data = loader.load_raw_data()
    quality_analyzer.analyze_data_quality(data)
    # ... 其他步骤
"""

try:
    from .config import ProcessingConfig, DataProcessingError, DataLoadError, MissingValueError, OutlierHandlingError
    from .data_loader import DataLoader, DataValidator
    from .quality_analyzer import DataQualityAnalyzer
    from .missing_value_handler import MissingValueHandler
    from .outlier_handler import OutlierHandler
    from .data_scaler import DataScaler
    from .visualization import DataVisualization
    from .utils import (
        setup_matplotlib_chinese,
        calculate_skewness_kurtosis,
        interpret_skewness,
        interpret_kurtosis,
        print_section_header,
        log_processing_step
    )
except ImportError:
    # 处理直接运行时的导入
    from config import ProcessingConfig, DataProcessingError, DataLoadError, MissingValueError, OutlierHandlingError
    from data_loader import DataLoader, DataValidator
    from quality_analyzer import DataQualityAnalyzer
    from missing_value_handler import MissingValueHandler
    from outlier_handler import OutlierHandler
    from data_scaler import DataScaler
    from visualization import DataVisualization
    from utils import (
        setup_matplotlib_chinese,
        calculate_skewness_kurtosis,
        interpret_skewness,
        interpret_kurtosis,
        print_section_header,
        log_processing_step
    )

__version__ = "1.0.0"
__author__ = "Data Processing Team"

# 导出主要类和函数
__all__ = [
    # 配置和异常
    'ProcessingConfig',
    'DataProcessingError',
    'DataLoadError', 
    'MissingValueError',
    'OutlierHandlingError',
    
    # 核心处理类
    'DataLoader',
    'DataValidator',
    'DataQualityAnalyzer',
    'MissingValueHandler',
    'OutlierHandler',
    'DataScaler',
    'DataVisualization',
    
    # 便捷函数
    'main',
    'create_processing_pipeline',
    
    # 工具函数
    'setup_matplotlib_chinese',
    'calculate_skewness_kurtosis',
    'interpret_skewness',
    'interpret_kurtosis',
    'print_section_header',
    'log_processing_step'
]


def create_processing_pipeline(config: ProcessingConfig = None):
    """创建数据预处理管道
    
    Args:
        config: 配置对象，如果为None则使用默认配置
        
    Returns:
        tuple: 包含所有处理器的元组
    """
    if config is None:
        config = ProcessingConfig()
    
    # 创建各个处理器
    data_loader = DataLoader(config)
    quality_analyzer = DataQualityAnalyzer(config)
    missing_handler = MissingValueHandler(config)
    outlier_handler = OutlierHandler(config)
    data_scaler = DataScaler(config)
    visualizer = DataVisualization(config)
    
    return data_loader, quality_analyzer, missing_handler, outlier_handler, data_scaler, visualizer


def main(config: ProcessingConfig = None) -> None:
    """数据预处理主函数 - 执行完整的数据预处理流程
    
    Args:
        config: 配置对象，如果为None则使用默认配置
    """
    try:
        print_section_header("微藻数据预处理系统", 80)
        
        # 使用默认配置或传入的配置
        if config is None:
            config = ProcessingConfig()

        # 确保目录存在
        config.ensure_directories()

        # 创建处理管道
        data_loader, quality_analyzer, missing_handler, outlier_handler, data_scaler, visualizer = create_processing_pipeline(config)
        
        # 1. 加载原始数据
        raw_data = data_loader.load_raw_data()
        
        # 2. 数据质量分析
        quality_analysis = quality_analyzer.analyze_data_quality(raw_data)
        
        # 3. 缺失值处理
        data_after_missing = missing_handler.handle_missing_values(raw_data)
        
        # 4. 异常值处理
        data_after_outliers = outlier_handler.handle_outliers(data_after_missing)

        # 生成异常值处理前后对比图
        visualizer.visualize_outlier_treatment(data_after_missing, data_after_outliers)

        # 5. 数据分布分析和标准化
        data_scaled = data_scaler.analyze_and_scale_data(data_after_outliers)

        # 6. 生成可视化图表
        visualizer.create_all_visualizations(data_scaled)
        
        # 7. 数据集分割
        train_data, test_data = data_loader.split_dataset(data_scaled)

        # 8. Robust标准化
        train_scaled, test_scaled, full_scaled = data_scaler.apply_robust_scaling(train_data, test_data, data_scaled)

        # 9. 保存处理后的数据
        data_loader.save_processed_data(train_scaled, test_scaled, full_scaled)
        
        # 10. 生成处理摘要
        _print_processing_summary(raw_data, full_scaled, quality_analysis)
        
        print_section_header("数据预处理完成", 80)
        
    except Exception as e:
        print(f"\n❌ 数据预处理失败: {str(e)}")
        raise


def _print_processing_summary(raw_data, processed_data, quality_analysis):
    """打印处理摘要
    
    Args:
        raw_data: 原始数据
        processed_data: 处理后数据
        quality_analysis: 质量分析结果
    """
    print_section_header("数据预处理摘要")
    
    print(f" 数据概览:")
    print(f"   原始数据形状: {raw_data.shape}")
    print(f"   处理后形状: {processed_data.shape}")
    print(f"   数据完整性: {'✓ 完整' if processed_data.isnull().sum().sum() == 0 else '⚠ 仍有缺失值'}")
    
    print(f"\n 质量分析:")
    missing_info = quality_analysis['missing_info']
    print(f"   原始缺失值: {missing_info['total_missing']} 个")
    print(f"   缺失值比例: {missing_info['missing_percentage']:.2f}%")
    
    skew_info = quality_analysis['skewness_info']
    print(f"   高度偏斜特征: {skew_info['severe_skew_count']} 个")
    print(f"   中度偏斜特征: {skew_info['moderate_skew_count']} 个")
    
    print(f"\n 输出文件:")
    import os
    config = ProcessingConfig()
    output_paths = config.get_output_paths()
    viz_paths = config.get_visualization_paths()

    print(f"   数据文件:")
    for name, path in output_paths.items():
        if os.path.exists(path):
            print(f"     ✓ {name}: {path}")

    print(f"   可视化文件:")
    for name, path in viz_paths.items():
        if os.path.exists(path):
            print(f"     ✓ {name}: {path}")
    
    print(f"\n 处理建议:")
    if missing_info['total_missing'] > 0:
        print(f"   - 原始数据存在缺失值，已使用智能填充策略处理")
    
    if skew_info['severe_skew_count'] > 0:
        print(f"   - 发现{skew_info['severe_skew_count']}个高度偏斜特征，建议在建模前考虑变换")
    
    print(f"   - 数据已准备就绪，可用于特征选择和模型训练")


# 支持直接运行模块
if __name__ == "__main__":
    # 处理直接运行时的导入问题
    import sys
    import os

    # 直接运行时，模块已经通过try-except导入了
    pass

    # 重新定义main函数用于直接运行
    def main_direct():
        """直接运行时的主函数"""
        try:
            print_section_header("微藻数据预处理系统", 80)

            # 使用默认配置，但修改路径以适应直接运行
            config = ProcessingConfig()
            # 修改路径为从src/data_processing目录运行时的相对路径
            config.raw_data_path = "../../data/raw/数据.xlsx"
            config.results_dir = "../../results/data_preprocess/"
            config.raw_analysis_dir = "../../results/data_preprocess/raw_analysis/"
            config.after_filling_dir = "../../results/data_preprocess/after_filling/"
            config.processed_data_dir = "../../data/processed/"

            # 确保目录存在
            config.ensure_directories()

            # 创建处理管道
            data_loader = DataLoader(config)
            quality_analyzer = DataQualityAnalyzer(config)
            missing_handler = MissingValueHandler(config)
            outlier_handler = OutlierHandler(config)
            data_scaler = DataScaler(config)
            visualizer = DataVisualization(config)

            # 1. 加载原始数据
            raw_data = data_loader.load_raw_data()

            # 2. 数据质量分析
            quality_analysis = quality_analyzer.analyze_data_quality(raw_data)

            # 3. 缺失值处理
            data_after_missing = missing_handler.handle_missing_values(raw_data)

            # 4. 异常值处理
            data_after_outliers = outlier_handler.handle_outliers(data_after_missing)

            # 生成异常值处理前后对比图
            visualizer.visualize_outlier_treatment(data_after_missing, data_after_outliers)

            # 5. 数据分布分析和标准化
            data_scaled = data_scaler.analyze_and_scale_data(data_after_outliers)

            # 6. 生成可视化图表
            visualizer.create_all_visualizations(data_scaled)

            # 7. 数据集分割
            train_data, test_data = data_loader.split_dataset(data_scaled)

            # 8. Robust标准化
            train_scaled, test_scaled, full_scaled = data_scaler.apply_robust_scaling(train_data, test_data, data_scaled)

            # 9. 保存处理后的数据
            data_loader.save_processed_data(train_scaled, test_scaled, full_scaled)

            print_section_header("数据预处理完成", 80)

        except Exception as e:
            print(f"\n 数据预处理失败: {str(e)}")
            raise

    # 运行主函数
    main_direct()
