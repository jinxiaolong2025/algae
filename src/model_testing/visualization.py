"""
模型测试可视化模块

该模块负责生成模型测试结果的可视化图表。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import warnings

# 设置中文字体和警告过滤
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class ModelTestVisualizer:
    """模型测试可视化器"""
    
    def __init__(self, results_dir: str = "../../results/model_testing/"):
        """
        初始化可视化器
        
        Args:
            results_dir: 结果保存目录
        """
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 r2_score: float, save_path: Optional[str] = None):
        """
        绘制预测值vs真实值散点图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            r2_score: R²分数
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制散点图
        ax.scatter(y_true, y_pred, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        
        # 绘制理想预测线 (y=x)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线')
        
        # 设置标签和标题
        ax.set_xlabel('真实值 (lipid %)', fontsize=12)
        ax.set_ylabel('预测值 (lipid %)', fontsize=12)
        ax.set_title(f'预测值 vs 真实值\nR² = {r2_score:.4f}', fontsize=14, fontweight='bold')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 设置坐标轴范围
        ax.set_xlim(min_val * 0.95, max_val * 1.05)
        ax.set_ylim(min_val * 0.95, max_val * 1.05)
        
        # 添加统计信息
        ax.text(0.05, 0.95, f'样本数: {len(y_true)}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, "prediction_vs_actual.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 预测vs真实值图已保存: {save_path}")
    
    def plot_residuals_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                               save_path: Optional[str] = None):
        """
        绘制残差分析图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            save_path: 保存路径
        """
        residuals = y_true - y_pred
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 残差vs预测值散点图
        ax1.scatter(y_pred, residuals, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('预测值')
        ax1.set_ylabel('残差')
        ax1.set_title('残差 vs 预测值')
        ax1.grid(True, alpha=0.3)
        
        # 2. 残差直方图
        ax2.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('残差')
        ax2.set_ylabel('频次')
        ax2.set_title('残差分布直方图')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q图（正态性检验）
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('残差Q-Q图（正态性检验）')
        ax3.grid(True, alpha=0.3)
        
        # 4. 残差vs样本索引
        ax4.scatter(range(len(residuals)), residuals, alpha=0.6, s=60, 
                   edgecolors='black', linewidth=0.5)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('样本索引')
        ax4.set_ylabel('残差')
        ax4.set_title('残差 vs 样本索引')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, "residuals_analysis.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 残差分析图已保存: {save_path}")
    
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                               save_path: Optional[str] = None):
        """
        绘制误差分布图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            save_path: 保存路径
        """
        absolute_errors = np.abs(y_true - y_pred)
        relative_errors = np.abs((y_true - y_pred) / y_true) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 绝对误差分布
        ax1.hist(absolute_errors, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax1.set_xlabel('绝对误差')
        ax1.set_ylabel('频次')
        ax1.set_title('绝对误差分布')
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        mae = np.mean(absolute_errors)
        ax1.axvline(x=mae, color='red', linestyle='--', linewidth=2, label=f'MAE = {mae:.4f}')
        ax1.legend()
        
        # 2. 相对误差分布
        ax2.hist(relative_errors, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('相对误差 (%)')
        ax2.set_ylabel('频次')
        ax2.set_title('相对误差分布')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        mape = np.mean(relative_errors)
        ax2.axvline(x=mape, color='red', linestyle='--', linewidth=2, label=f'MAPE = {mape:.2f}%')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, "error_distribution.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"误差分布图已保存: {save_path}")
    
    def plot_performance_summary(self, metrics: dict, save_path: Optional[str] = None):
        """
        绘制性能指标总结图
        
        Args:
            metrics: 性能指标字典
            save_path: 保存路径
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 主要指标柱状图
        main_metrics = ['R²', 'RMSE', 'MAE']
        main_values = [metrics.get(m, 0) for m in main_metrics]
        
        bars1 = ax1.bar(main_metrics, main_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('主要性能指标')
        ax1.set_ylabel('指标值')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars1, main_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 2. MAPE单独显示
        ax2.bar(['MAPE'], [metrics.get('MAPE', 0)], color='orange')
        ax2.set_title('平均绝对百分比误差')
        ax2.set_ylabel('MAPE (%)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        mape_value = metrics.get('MAPE', 0)
        ax2.text(0, mape_value + mape_value*0.01, f'{mape_value:.2f}%', 
                ha='center', va='bottom')
        
        # 3. 相关系数和样本数
        other_metrics = ['Correlation', 'Test_Samples']
        other_values = [metrics.get(m, 0) for m in other_metrics]
        
        bars3 = ax3.bar(other_metrics, other_values, color=['purple', 'gold'])
        ax3.set_title('其他指标')
        ax3.set_ylabel('指标值')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars3, other_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}' if value < 100 else f'{int(value)}', 
                    ha='center', va='bottom')
        
        # 4. 性能等级评估
        r2 = metrics.get('R²', 0)
        if r2 >= 0.8:
            performance_level = "优秀"
            color = 'green'
        elif r2 >= 0.6:
            performance_level = "良好"
            color = 'blue'
        elif r2 >= 0.4:
            performance_level = "一般"
            color = 'orange'
        else:
            performance_level = "较差"
            color = 'red'
        
        ax4.text(0.5, 0.6, f'模型性能等级', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=16, fontweight='bold')
        ax4.text(0.5, 0.4, performance_level, ha='center', va='center',
                transform=ax4.transAxes, fontsize=24, fontweight='bold', color=color)
        ax4.text(0.5, 0.2, f'R² = {r2:.4f}', ha='center', va='center',
                transform=ax4.transAxes, fontsize=14)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, "performance_summary.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 性能总结图已保存: {save_path}")
    
    def create_all_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 metrics: dict):
        """
        创建所有可视化图表
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            metrics: 性能指标字典
        """
        print("\n8. 生成可视化图表...")
        
        try:
            # 1. 预测vs真实值图
            self.plot_prediction_vs_actual(y_true, y_pred, metrics.get('R²', 0))
            
            # 2. 残差分析图
            self.plot_residuals_analysis(y_true, y_pred)
            
            # 3. 误差分布图
            self.plot_error_distribution(y_true, y_pred)
            
            # 4. 性能总结图
            self.plot_performance_summary(metrics)
            
            print(" 所有可视化图表生成完成")
            
        except Exception as e:
            print(f"可视化生成过程中出现错误: {str(e)}")
            raise
