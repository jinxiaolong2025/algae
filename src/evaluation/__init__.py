# -*- coding: utf-8 -*-
"""
模型评估模块
Model Evaluation Module

包含综合评估、性能分析等功能
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
    from .comprehensive_evaluation import *
except ImportError:
    from src.evaluation.comprehensive_evaluation import *

__all__ = []

def main():
    """主函数 - 运行评估模块"""
    try:
        from .comprehensive_evaluation import main as eval_main
    except ImportError:
        from src.evaluation.comprehensive_evaluation import main as eval_main
    eval_main()

if __name__ == "__main__":
    print("启动综合性能评估模块...")
    main()