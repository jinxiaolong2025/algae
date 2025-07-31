#!/usr/bin/env python3
"""
数据预处理模块直接运行脚本

使用方法：
    cd src/data_processing
    python run_module.py
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入并运行主函数
from src.data_processing import ProcessingConfig, main

if __name__ == "__main__":
    # 创建适合从src/data_processing目录运行的配置
    config = ProcessingConfig()
    config.raw_data_path = "../../data/raw/数据.xlsx"
    config.results_dir = "../../results/data_preprocess/"
    config.raw_analysis_dir = "../../results/data_preprocess/raw_analysis/"
    config.after_filling_dir = "../../results/data_preprocess/after_filling/"
    config.processed_data_dir = "../../data/processed/"

    main(config)
