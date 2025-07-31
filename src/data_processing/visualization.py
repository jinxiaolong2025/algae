"""数据预处理可视化模块

负责生成数据预处理过程中的各种可视化图表。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Tuple
try:
    from .config import ProcessingConfig
    from .utils import (
        setup_matplotlib_chinese,
        setup_subplot_layout,
        hide_extra_subplots,
        save_plot,
        calculate_correlation_matrix,
        log_processing_step
    )
except ImportError:
    from config import ProcessingConfig
    from utils import (
        setup_matplotlib_chinese,
        setup_subplot_layout,
        hide_extra_subplots,
        save_plot,
        calculate_correlation_matrix,
        log_processing_step
    )


class DataVisualization:
    """数据可视化器类"""
    
    def __init__(self, config: ProcessingConfig):
        """初始化数据可视化器
        
        Args:
            config: 配置对象
        """
        self.config = config
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def create_all_visualizations(self, data: pd.DataFrame) -> None:
        """创建所有可视化图表

        Args:
            data: 输入数据
        """
        log_processing_step("6. 生成可视化图表")

        print("   生成可视化图表:")

        # 箱线图
        self.create_boxplots(data)

        # 散点图
        self.create_scatter_plots(data)

        # 直方图
        self.create_histograms(data)

        print("   所有可视化图表已生成完成")

    def visualize_outlier_treatment(self, data_before: pd.DataFrame, data_after: pd.DataFrame) -> None:
        """可视化异常值处理前后的对比

        Args:
            data_before: 处理前数据
            data_after: 处理后数据
        """
        print("\n 生成异常值处理前后对比图...")

        # 直接计算偏度，避免导入问题
        numeric_cols = data_before.select_dtypes(include=[np.number]).columns
        skewness_before = data_before[numeric_cols].skew()
        skewness_after = data_after[numeric_cols].skew()

        # 排除目标变量
        target_variable = self.config.target_column
        skewness_before = skewness_before.drop(target_variable, errors='ignore')
        skewness_after = skewness_after.drop(target_variable, errors='ignore')
        numeric_cols = numeric_cols.drop(target_variable, errors='ignore')

        # 定义基于偏度的特征组
        light_skew = skewness_before[abs(skewness_before) < 1].index.tolist()
        moderate_skew = skewness_before[(abs(skewness_before) >= 1) & (abs(skewness_before) < 2)].index.tolist()
        heavy_skew = skewness_before[abs(skewness_before) >= 2].index.tolist()

        feature_groups = {
            'Light_Skewness': light_skew,
            'Moderate_Skewness': moderate_skew,
            'Heavy_Skewness': heavy_skew
        }

        for group_name, features in feature_groups.items():
            # 筛选存在的特征
            existing_features = [f for f in features if f in data_before.columns and f in data_after.columns]

            if not existing_features:
                continue

            self._create_group_comparison_plot(data_before, data_after, existing_features, group_name)

        print("    异常值处理可视化完成")

        # 生成偏度改善分析图
        self.create_skewness_improvement_analysis(data_before, data_after)

    def _create_group_comparison_plot(self, data_before: pd.DataFrame, data_after: pd.DataFrame,
                                    features: List[str], group_name: str) -> None:
        """创建特征组的对比图

        Args:
            data_before: 处理前数据
            data_after: 处理后数据
            features: 特征列表
            group_name: 组名
        """
        try:
            # 计算需要的行数和列数
            n_features = len(features)
            n_cols = min(3, n_features)  # 最多3列
            n_rows = (n_features + n_cols - 1) // n_cols  # 向上取整

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            fig.suptitle(f'{group_name} - 处理前后对比', fontsize=16, fontweight='bold')

            # 确保axes是二维数组
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            for idx, feature in enumerate(features):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col]

                # 创建箱线图对比
                data_to_plot = [data_before[feature].dropna(), data_after[feature].dropna()]
                labels = ['处理前', '处理后']

                box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

                # 设置颜色
                colors = ['lightblue', 'lightgreen']
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)

                # 计算偏度改善
                skew_before = data_before[feature].skew()
                skew_after = data_after[feature].skew()
                improvement = abs(skew_before) - abs(skew_after)

                ax.set_title(f'{feature}\n偏度: {skew_before:.3f} → {skew_after:.3f} (改善: {improvement:+.3f})',
                           fontweight='bold', fontsize=10)
                ax.grid(True, alpha=0.3)

            # 隐藏多余的子图
            for idx in range(n_features, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                if row < n_rows and col < n_cols:
                    axes[row, col].set_visible(False)

            plt.tight_layout()

            # 保存图片 - 使用简洁的文件名格式，避免特殊字符
            filename = f"outlier_treatment_{group_name}.png"
            output_path = os.path.join(self.config.results_dir, filename)
            plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()

            print(f"    {group_name}对比图已保存: {output_path}")

        except Exception as e:
            print(f"    ✗ {group_name}对比图生成失败: {str(e)}")

    def create_skewness_improvement_analysis(self, data_before: pd.DataFrame, data_after: pd.DataFrame) -> None:
        """生成偏度改善分析图

        Args:
            data_before: 处理前数据
            data_after: 处理后数据
        """
        try:
            print("\n 生成偏度改善分析图...")

            # 直接计算偏度，避免导入问题
            numeric_cols = data_before.select_dtypes(include=[np.number]).columns
            skew_before = data_before[numeric_cols].skew()
            skew_after = data_after[numeric_cols].skew()

            # 排除目标变量
            target_variable = self.config.target_column
            skew_before = skew_before.drop(target_variable, errors='ignore')
            skew_after = skew_after.drop(target_variable, errors='ignore')
            numeric_cols = numeric_cols.drop(target_variable, errors='ignore')

            skew_improvement = abs(skew_before) - abs(skew_after)

            # 创建偏度对比图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

            # 左图：处理前后偏度对比
            x_pos = np.arange(len(numeric_cols))
            width = 0.35

            ax1.bar(x_pos - width/2, abs(skew_before), width, label='Before', alpha=0.7, color='lightcoral')
            ax1.bar(x_pos + width/2, abs(skew_after), width, label='After', alpha=0.7, color='lightblue')

            ax1.set_xlabel('Features')
            ax1.set_ylabel('|Skewness|')
            ax1.set_title('Skewness Comparison Before/After Outlier Treatment')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(numeric_cols, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 右图：偏度改善程度
            colors = ['green' if x > 0 else 'red' for x in skew_improvement]
            bars = ax2.bar(x_pos, skew_improvement, color=colors, alpha=0.7)

            ax2.set_xlabel('Features')
            ax2.set_ylabel('Skewness Improvement')
            ax2.set_title('Skewness Improvement (Positive = Better)')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(numeric_cols, rotation=45, ha='right')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, improvement in zip(bars, skew_improvement):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.05,
                        f'{improvement:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

            plt.tight_layout()

            # 保存图片 - 直接构建路径避免导入问题
            output_path = os.path.join(self.config.results_dir, 'skewness_improvement_analysis.png')
            plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()

            print(f"    偏度改善分析图已保存: {output_path}")

            # 打印改善统计
            improved_features = skew_improvement[skew_improvement > 0]
            worsened_features = skew_improvement[skew_improvement < 0]

            print(f"\n 偏度改善统计:")
            print(f"   - 改善的特征: {len(improved_features)}个")
            print(f"   - 恶化的特征: {len(worsened_features)}个")
            print(f"   - 平均改善程度: {skew_improvement.mean():.3f}")

            if len(improved_features) > 0:
                print(f"   - 最大改善: {improved_features.max():.3f} ({improved_features.idxmax()})")
            if len(worsened_features) > 0:
                print(f"   - 最大恶化: {worsened_features.min():.3f} ({worsened_features.idxmin()})")

        except Exception as e:
            print(f"    ✗ 偏度改善分析图生成失败: {str(e)}")


    
    def create_boxplots(self, data: pd.DataFrame) -> None:
        """创建箱线图
        
        Args:
            data: 输入数据
        """
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            n_features = len(numeric_cols)
            
            if n_features == 0:
                print("     跳过箱线图: 没有数值特征")
                return
            
            # 设置子图布局
            n_rows, n_cols = setup_subplot_layout(n_features, self.config.subplot_cols)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
            fig.suptitle('数据分布箱线图 (缺失值填充后)', fontsize=16, fontweight='bold')
            
            # 确保axes是二维数组
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for idx, col in enumerate(numeric_cols):
                row = idx // n_cols
                col_idx = idx % n_cols
                ax = axes[row, col_idx]
                
                # 创建箱线图
                data[col].plot(kind='box', ax=ax)
                ax.set_title(f'{col}', fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            hide_extra_subplots(axes, n_features, n_rows, n_cols)
            
            plt.tight_layout()
            
            # 保存图片
            viz_paths = self.config.get_visualization_paths()
            plt.savefig(viz_paths['boxplots'], dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            
            print(f"     ✓ 箱线图已保存: {viz_paths['boxplots']}")
            
        except Exception as e:
            print(f"     ✗ 箱线图生成失败: {str(e)}")
    
    def create_scatter_plots(self, data: pd.DataFrame) -> None:
        """创建散点图矩阵
        
        Args:
            data: 输入数据
        """
        try:
            # 计算相关性矩阵
            corr_matrix, high_corr_pairs = calculate_correlation_matrix(
                data, self.config.correlation_threshold_viz
            )
            
            if len(high_corr_pairs) == 0:
                print("     跳过散点图: 没有高相关性特征对")
                return
            
            # 限制显示的特征对数量
            pairs_to_plot = high_corr_pairs[:self.config.max_scatter_pairs]
            n_pairs = len(pairs_to_plot)
            
            # 设置子图布局
            n_rows, n_cols = setup_subplot_layout(n_pairs, self.config.subplot_cols)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
            fig.suptitle('高相关性特征散点图 (缺失值填充后)', fontsize=16, fontweight='bold')
            
            # 确保axes是二维数组
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for idx, (feature1, feature2, corr_val) in enumerate(pairs_to_plot):
                row = idx // n_cols
                col_idx = idx % n_cols
                ax = axes[row, col_idx]
                
                # 创建散点图
                ax.scatter(data[feature1], data[feature2], alpha=0.6)
                ax.set_xlabel(feature1)
                ax.set_ylabel(feature2)
                ax.set_title(f'{feature1} vs {feature2}\n相关系数: {corr_val:.3f}', fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            hide_extra_subplots(axes, n_pairs, n_rows, n_cols)
            
            plt.tight_layout()
            
            # 保存图片
            viz_paths = self.config.get_visualization_paths()
            plt.savefig(viz_paths['scatter_plots'], dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            
            print(f"     ✓ 散点图已保存: {viz_paths['scatter_plots']}")
            
        except Exception as e:
            print(f"     ✗ 散点图生成失败: {str(e)}")
    
    def create_histograms(self, data: pd.DataFrame) -> None:
        """创建直方图
        
        Args:
            data: 输入数据
        """
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            n_features = len(numeric_cols)
            
            if n_features == 0:
                print("     跳过直方图: 没有数值特征")
                return
            
            # 设置子图布局
            n_rows, n_cols = setup_subplot_layout(n_features, self.config.subplot_cols)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
            fig.suptitle('数据分布直方图 (缺失值填充后)', fontsize=16, fontweight='bold')
            
            # 确保axes是二维数组
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for idx, col in enumerate(numeric_cols):
                row = idx // n_cols
                col_idx = idx % n_cols
                ax = axes[row, col_idx]
                
                # 创建直方图
                data[col].hist(bins=20, ax=ax, alpha=0.7, edgecolor='black')
                ax.set_title(f'{col}', fontweight='bold')
                ax.set_xlabel('值')
                ax.set_ylabel('频次')
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            hide_extra_subplots(axes, n_features, n_rows, n_cols)
            
            plt.tight_layout()
            
            # 保存图片
            viz_paths = self.config.get_visualization_paths()
            plt.savefig(viz_paths['histograms'], dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            
            print(f"     ✓ 直方图已保存: {viz_paths['histograms']}")
            
        except Exception as e:
            print(f"     ✗ 直方图生成失败: {str(e)}")
    
    def create_skewness_improvement_analysis(self, before_data: pd.DataFrame, 
                                           after_data: pd.DataFrame) -> None:
        """创建偏度改善分析图
        
        Args:
            before_data: 处理前数据
            after_data: 处理后数据
        """
        try:
            from .utils import calculate_skewness_kurtosis
            
            # 计算处理前后的偏度
            skew_before, _, numeric_cols = calculate_skewness_kurtosis(before_data)
            skew_after, _, _ = calculate_skewness_kurtosis(after_data)
            
            # 创建对比图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # 偏度对比条形图
            x_pos = np.arange(len(numeric_cols))
            width = 0.35
            
            ax1.bar(x_pos - width/2, skew_before, width, label='处理前', alpha=0.8)
            ax1.bar(x_pos + width/2, skew_after, width, label='处理后', alpha=0.8)
            
            ax1.set_xlabel('特征')
            ax1.set_ylabel('偏度值')
            ax1.set_title('偏度改善对比', fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(numeric_cols, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # 偏度改善散点图
            ax2.scatter(skew_before, skew_after, alpha=0.7)
            
            # 添加对角线
            min_val = min(skew_before.min(), skew_after.min())
            max_val = max(skew_before.max(), skew_after.max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='无改善线')
            
            ax2.set_xlabel('处理前偏度')
            ax2.set_ylabel('处理后偏度')
            ax2.set_title('偏度改善散点图', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 添加特征标签
            for i, col in enumerate(numeric_cols):
                ax2.annotate(col, (skew_before[col], skew_after[col]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            
            # 保存图片
            viz_paths = self.config.get_visualization_paths()
            plt.savefig(viz_paths['skewness_improvement'], dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            
            print(f"     ✓ 偏度改善分析图已保存: {viz_paths['skewness_improvement']}")
            
        except Exception as e:
            print(f"     ✗ 偏度改善分析图生成失败: {str(e)}")
    
    def create_correlation_heatmap(self, data: pd.DataFrame, title: str = "特征相关性热力图") -> None:
        """创建相关性热力图
        
        Args:
            data: 输入数据
            title: 图表标题
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) < 2:
                print("     跳过相关性热力图: 数值特征不足")
                return
            
            # 计算相关性矩阵
            corr_matrix = numeric_data.corr()
            
            # 创建热力图
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            
            plt.title(title, fontweight='bold', fontsize=14)
            plt.tight_layout()
            
            # 保存图片
            filename = f"correlation_heatmap_{title.replace(' ', '_').lower()}.png"
            output_path = os.path.join(self.config.results_dir, filename)
            plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            
            print(f"     ✓ 相关性热力图已保存: {output_path}")
            
        except Exception as e:
            print(f"     ✗ 相关性热力图生成失败: {str(e)}")
