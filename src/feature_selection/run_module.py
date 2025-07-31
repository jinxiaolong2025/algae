"""
特征选择模块运行脚本

该脚本提供了运行特征选择模块的便捷方式，支持不同的配置预设。

使用方法:
    python run_module.py                    # 使用默认配置
    python run_module.py --preset quick     # 使用快速配置
    python run_module.py --preset thorough  # 使用深度配置
    python run_module.py --custom           # 使用自定义配置
"""

import sys
import os
import argparse
from typing import Optional

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import FeatureSelectionConfig, get_preset_config
    import __init__ as feature_selection_module
    main = feature_selection_module.main
    run_quick_selection = feature_selection_module.run_quick_selection
    run_thorough_selection = feature_selection_module.run_thorough_selection
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有必要的模块文件都存在")
    sys.exit(1)


def create_custom_config() -> FeatureSelectionConfig:
    """
    创建自定义配置
    
    Returns:
        自定义的特征选择配置
    """
    print("创建自定义配置...")
    print("=" * 50)
    
    try:
        # 获取用户输入
        population_size = int(input("种群大小 (默认30): ") or "30")
        generations = int(input("进化代数 (默认100): ") or "100")
        target_features = int(input("目标特征数 (默认6): ") or "6")
        mutation_rate = float(input("变异率 (默认0.15): ") or "0.15")
        crossover_rate = float(input("交叉率 (默认0.85): ") or "0.85")
        cv_folds = int(input("交叉验证折数 (默认5): ") or "5")
        
        config = FeatureSelectionConfig(
            population_size=population_size,
            generations=generations,
            target_features=target_features,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            cv_folds=cv_folds
        )
        
        print("\n自定义配置创建完成:")
        config.print_config_summary()
        
        return config
        
    except ValueError as e:
        print(f"输入错误: {e}")
        print("使用默认配置...")
        return FeatureSelectionConfig()
    except KeyboardInterrupt:
        print("\n用户取消，使用默认配置...")
        return FeatureSelectionConfig()


def run_with_preset(preset_name: str):
    """
    使用预设配置运行
    
    Args:
        preset_name: 预设名称
    """
    try:
        if preset_name == "quick":
            print("使用快速配置运行特征选择...")
            run_quick_selection()
        elif preset_name == "thorough":
            print("使用深度配置运行特征选择...")
            run_thorough_selection()
        else:
            config = get_preset_config(preset_name)
            print(f"使用预设配置 '{preset_name}' 运行特征选择...")
            main(config)
    except ValueError as e:
        print(f"错误: {e}")
        print("可用的预设配置: quick, thorough, default, experimental")
        sys.exit(1)


def run_interactive_mode():
    """
    运行交互模式
    """
    print("特征选择交互模式")
    print("=" * 50)
    print("请选择运行模式:")
    print("1. 默认配置")
    print("2. 快速配置 (较少代数，适合测试)")
    print("3. 深度配置 (较多代数，更准确)")
    print("4. 自定义配置")
    print("5. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-5): ").strip()
            
            if choice == "1":
                print("使用默认配置...")
                main()
                break
            elif choice == "2":
                print("使用快速配置...")
                run_quick_selection()
                break
            elif choice == "3":
                print("使用深度配置...")
                run_thorough_selection()
                break
            elif choice == "4":
                config = create_custom_config()
                main(config)
                break
            elif choice == "5":
                print("退出程序")
                sys.exit(0)
            else:
                print("无效选择，请输入1-5之间的数字")
                
        except KeyboardInterrupt:
            print("\n\n用户取消，退出程序")
            sys.exit(0)
        except Exception as e:
            print(f"发生错误: {e}")


def main_script():
    """
    主脚本函数
    """
    parser = argparse.ArgumentParser(
        description="遗传算法特征选择模块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_module.py                    # 默认配置
  python run_module.py --preset quick     # 快速配置
  python run_module.py --preset thorough  # 深度配置
  python run_module.py --custom           # 自定义配置
  python run_module.py --interactive      # 交互模式
        """
    )
    
    parser.add_argument(
        "--preset", 
        choices=["quick", "thorough", "default", "experimental"],
        help="使用预设配置"
    )
    
    parser.add_argument(
        "--custom",
        action="store_true",
        help="使用自定义配置"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="运行交互模式"
    )
    
    parser.add_argument(
        "--population-size",
        type=int,
        help="种群大小"
    )
    
    parser.add_argument(
        "--generations",
        type=int,
        help="进化代数"
    )
    
    parser.add_argument(
        "--target-features",
        type=int,
        help="目标特征数量"
    )
    
    parser.add_argument(
        "--mutation-rate",
        type=float,
        help="变异率"
    )
    
    parser.add_argument(
        "--crossover-rate",
        type=float,
        help="交叉率"
    )
    
    args = parser.parse_args()
    
    try:
        # 检查数据文件是否存在
        config = FeatureSelectionConfig()
        if not os.path.exists(config.train_data_path):
            print(f"错误: 训练数据文件不存在: {config.train_data_path}")
            print("请确保数据预处理模块已经运行并生成了训练数据")
            sys.exit(1)
        
        # 根据参数选择运行模式
        if args.interactive:
            run_interactive_mode()
        elif args.custom:
            config = create_custom_config()
            main(config)
        elif args.preset:
            run_with_preset(args.preset)
        elif any([args.population_size, args.generations, args.target_features, 
                 args.mutation_rate, args.crossover_rate]):
            # 使用命令行参数创建配置
            config_params = {}
            if args.population_size:
                config_params['population_size'] = args.population_size
            if args.generations:
                config_params['generations'] = args.generations
            if args.target_features:
                config_params['target_features'] = args.target_features
            if args.mutation_rate:
                config_params['mutation_rate'] = args.mutation_rate
            if args.crossover_rate:
                config_params['crossover_rate'] = args.crossover_rate
            
            config = FeatureSelectionConfig(**config_params)
            print("使用命令行参数配置...")
            config.print_config_summary()
            main(config)
        else:
            # 默认配置
            print("使用默认配置运行特征选择...")
            main()
            
    except KeyboardInterrupt:
        print("\n\n用户中断，程序退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main_script()
