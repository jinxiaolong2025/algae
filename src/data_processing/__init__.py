# -*- coding: utf-8 -*-
"""
数据处理模块
Data Processing Module

包含数据预处理、清洗、诊断等功能
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
    from .intelligent_data_processor import OptimizedPreprocessor
except ImportError:
    from src.data_processing.intelligent_data_processor import OptimizedPreprocessor

__all__ = [
    'OptimizedPreprocessor'
]

def main():
    """主函数 - 运行数据处理模块"""
    try:
        from .intelligent_data_processor import main as processor_main
    except ImportError:
        from src.data_processing.intelligent_data_processor import main as processor_main
    processor_main()

if __name__ == "__main__":
    main()