"""模型训练模块

提供完整的机器学习模型训练、评估和保存功能。

主要组件：
- TrainingConfig: 训练配置管理
- DataLoader: 数据加载和预处理
- DataAugmentor: 数据增强
- ModelFactory: 模型创建工厂
- ModelTrainer: 模型训练器
- ResultAnalyzer: 结果分析器
- ModelSaver: 模型保存器

便捷函数：
- create_training_pipeline: 创建完整的训练管道
- main: 主训练流程
"""

# 导入所有主要组件
try:
    # 相对导入（作为包使用时）
    from .config import TrainingConfig, ModelTrainingError, DataLoadError
    from .data_loader import DataLoader, DataPreprocessor
    from .data_augmentor import DataAugmentor
    from .model_factory import ModelFactory
    from .model_trainer import ModelTrainer
    from .result_handler import ResultAnalyzer, ModelSaver
    from .visualization import ModelVisualization
except ImportError:
    # 绝对导入（直接运行时）
    from config import TrainingConfig, ModelTrainingError, DataLoadError
    from data_loader import DataLoader, DataPreprocessor
    from data_augmentor import DataAugmentor
    from model_factory import ModelFactory
    from model_trainer import ModelTrainer
    from result_handler import ResultAnalyzer, ModelSaver
    from visualization import ModelVisualization

# 版本信息
__version__ = "1.0.0"

# 公开的API
__all__ = [
    'TrainingConfig',
    'ModelTrainingError',
    'DataLoadError',
    'DataLoader',
    'DataPreprocessor',
    'DataAugmentor',
    'ModelFactory',
    'ModelTrainer',
    'ResultAnalyzer',
    'ModelSaver',
    'ModelVisualization',
    'create_training_pipeline',
    'main'
]


def create_training_pipeline(config: TrainingConfig = None):
    """
    创建完整的训练管道

    Args:
        config: 训练配置，如果为None则使用默认配置

    Returns:
        Tuple: (data_loader, model_trainer, result_analyzer, model_saver, visualizer)
    """
    if config is None:
        config = TrainingConfig()

    # 创建各个组件
    data_loader = DataLoader(config)
    model_trainer = ModelTrainer(config)
    result_analyzer = ResultAnalyzer(config)
    model_saver = ModelSaver(config)
    visualizer = ModelVisualization(config)

    return data_loader, model_trainer, result_analyzer, model_saver, visualizer


def main():
    """
    主训练流程
    
    执行完整的模型训练、评估和保存流程
    """
    print("="*100)
    print("多模型训练和对比系统")
    print("="*100)
    
    try:
        # 1. 初始化配置和组件
        print("\n=== 1. 初始化配置 ===")
        config = TrainingConfig()
        data_loader, model_trainer, result_analyzer, model_saver, visualizer = create_training_pipeline(config)
        print("   配置和组件初始化完成")
        
        # 2. 加载数据
        print("\n=== 2. 加载数据 ===")
        train_data, selected_features, data_source = data_loader.load_training_data()
        X_train, y_train, feature_names = DataPreprocessor.prepare_training_data(
            train_data, selected_features, data_source
        )
        
        # 3. 训练模型
        print("\n=== 3. 训练和对比模型 ===")
        model_results, trained_models = model_trainer.train_and_compare_models(
            X_train, y_train, feature_names
        )
        
        # 4. 分析结果
        print("\n=== 4. 模型对比分析 ===")
        comparison_df, best_model_name = result_analyzer.display_model_comparison(
            model_results, feature_names
        )
        
        # 5. 保存最佳模型
        print("\n=== 5. 保存最佳模型 ===")
        print("保存最佳模型...")
        if best_model_name:
            model_saver.save_best_model(model_results, best_model_name, feature_names)
            # 显示特征重要性（如果可用）
            best_result = model_results[best_model_name]
            if hasattr(best_result['model'], 'feature_importances_'):
                print(f"\n   特征重要性 (Top 5):")
                importances = best_result['model'].feature_importances_
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                for i, (feature, importance) in enumerate(feature_importance[:5], 1):
                    print(f"     {i}. {feature}: {importance:.4f}")
        else:
            print("   没有找到可保存的最佳模型")

        # 6. 生成可视化报告
        print("\n=== 6. 生成可视化报告 ===")
        try:
            # 获取增强数据（如果使用了数据增强）
            X_augmented, y_augmented = None, None
            if config.use_augmentation and best_model_name:
                try:
                    from .data_augmentor import DataAugmentor
                except ImportError:
                    from data_augmentor import DataAugmentor
                augmentor = DataAugmentor(config)
                X_augmented, y_augmented = augmentor.augment_data(
                    X_train, y_train, feature_names,
                    random_state=config.cv_random_state, verbose=False
                )

            # 生成完整的可视化报告
            visualizer.create_complete_visualization_report(
                results=model_results,
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names,
                best_model_name=best_model_name,
                X_augmented=X_augmented,
                y_augmented=y_augmented
            )
        except Exception as e:
            print(f"   可视化生成失败: {str(e)}")
        
        print("\n" + "="*100)
        print("训练完成！")
        print("="*100)
        
    except Exception as e:
        print(f"\n 训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
