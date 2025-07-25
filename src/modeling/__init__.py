# -*- coding: utf-8 -*-
"""
建模模块
Modeling Module

包含模型训练、集成学习、小样本优化等功能
"""

import sys
import os

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .fixed_optimized_training import FixedSmallSampleOptimizer
except ImportError:
    from src.modeling.fixed_optimized_training import FixedSmallSampleOptimizer

__all__ = [
    'FixedSmallSampleOptimizer'
]

def main():
    """建模模块主函数"""
    print(" 运行建模模块...")
    print("选择要运行的模块:")
    print("1. 固定优化训练 (fixed_optimized_training.py)")
    print("2. 终极集成管道 (ultimate_ensemble_pipeline.py)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        try:
            from .fixed_optimized_training import main as training_main
        except ImportError:
            from src.modeling.fixed_optimized_training import main as training_main
        training_main()
    elif choice == "2":
        try:
            from .ultimate_ensemble_pipeline import main as pipeline_main
        except ImportError:
            from src.modeling.ultimate_ensemble_pipeline import main as pipeline_main
        pipeline_main()
    else:
        print(" 无效选择，默认运行固定优化训练")
        try:
            from .fixed_optimized_training import main as training_main
        except ImportError:
            from src.modeling.fixed_optimized_training import main as training_main
        training_main()

if __name__ == "__main__":
    main()