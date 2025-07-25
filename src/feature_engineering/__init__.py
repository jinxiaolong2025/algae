# -*- coding: utf-8 -*-
"""
特征工程模块
Feature Engineering Module

包含特征选择、特征工程、特征重要性分析等功能
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
    from .selection import AdvancedFeatureSelector
    from .engineering import *
except ImportError:
    from src.feature_engineering.selection import AdvancedFeatureSelector
    from src.feature_engineering.engineering import *

__all__ = [
    'AdvancedFeatureSelector'
]

def main():
    """特征工程模块主函数"""
    print("🚀 运行特征工程模块...")
    print("选择要运行的模块:")
    print("1. 高级特征选择 (selection.py)")
    print("2. 基础特征工程 (engineering.py)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        try:
            from .selection import main as selection_main
        except ImportError:
            from src.feature_engineering.selection import main as selection_main
        selection_main()
    elif choice == "2":
        try:
            from .engineering import main as engineering_main
        except ImportError:
            from src.feature_engineering.engineering import main as engineering_main
        engineering_main()
    else:
        print("❌ 无效选择，默认运行高级特征选择")
        try:
            from .selection import main as selection_main
        except ImportError:
            from src.feature_engineering.selection import main as selection_main
        selection_main()

if __name__ == "__main__":
    main()