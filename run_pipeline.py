#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主运行脚本 - 集成学习管道
小样本高维数据优化的完整解决方案
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from modeling.ultimate_ensemble_pipeline import UltimateEnsemblePipeline

def main():
    """
    主函数 - 运行终极集成学习管道
    """
    print(" 启动集成学习管道...")
    print(" 专为小样本高维数据设计的机器学习解决方案")
    print(" 结合SMOGN数据增强、贝叶斯集成、深度学习等先进技术")
    print("="*80)
    
    # 创建管道实例
    pipeline = UltimateEnsemblePipeline(target_features=3, random_state=42)
    
    # 运行管道
    try:
        results, best_combination = pipeline.run_ultimate_pipeline('data/raw/数据.xlsx')
        
        if best_combination:
            model_name, dataset_name, result = best_combination
            print(f"\n 成功完成！最佳解决方案: {model_name} + {dataset_name}")
            print(f" 训练集 R²: {result['train_r2']:.4f}")
            print(f" 交叉验证 R²: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}")
            print(f" 详细结果已保存到 'results/ultimate_ensemble_results.txt'")
        else:
            print("\n️ 未找到满意的解决方案，建议进一步调优")
            
    except Exception as e:
        print(f" 管道运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)