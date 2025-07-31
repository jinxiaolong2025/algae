"""可视化模块

提供模型训练过程和结果的专业可视化功能，参考机器学习和数据科学领域的最佳实践。

主要功能：
- 训练过程可视化：交叉验证性能对比、模型性能雷达图、学习曲线
- 结果分析可视化：特征重要性、预测vs真实值、残差分析、性能热力图
- 数据增强可视化：数据分布对比、特征相关性变化
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    from .config import TrainingConfig
except ImportError:
    from config import TrainingConfig

# 设置中文字体和样式
import matplotlib
import matplotlib.font_manager as fm

# 强制重新加载字体缓存
try:
    fm._rebuild()
except AttributeError:
    pass

# 检查并设置可用的中文字体
available_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = ['Heiti TC', 'STHeiti', 'PingFang SC', 'Songti SC', 'Kaiti SC', 'SimSong', 'Arial Unicode MS']
selected_font = None

for font in chinese_fonts:
    if font in available_fonts:
        selected_font = font
        break

if selected_font:
    # 强制设置字体
    matplotlib.rcParams['font.sans-serif'] = [selected_font]
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.dpi'] = 300
    matplotlib.rcParams['savefig.dpi'] = 300
    print(f"全局设置中文字体: {selected_font}")
else:
    print("警告: 未找到合适的中文字体，可能显示为方框")

# 专业配色方案
COLORS = {
    'primary': '#2E86AB',      # 主色调：深蓝
    'secondary': '#A23B72',    # 次色调：紫红
    'accent': '#F18F01',       # 强调色：橙色
    'success': '#C73E1D',      # 成功色：红色
    'info': '#5D737E',         # 信息色：灰蓝
    'light': '#F5F5F5',        # 浅色
    'dark': '#2C3E50',         # 深色
    'palette': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5D737E', 
                '#8E44AD', '#27AE60', '#E67E22', '#34495E', '#95A5A6']
}


class ModelVisualization:
    """模型可视化类，提供全面的训练过程和结果可视化功能"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.results_dir = os.path.abspath(config.results_dir)
        self.viz_dir = os.path.join(self.results_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)

        # 重新设置中文字体
        self._setup_chinese_font()

        # 设置seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette(COLORS['palette'])

    def _setup_chinese_font(self):
        """设置中文字体"""
        import matplotlib.font_manager as fm

        available_fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = ['Heiti TC', 'STHeiti', 'Songti SC', 'Kaiti SC', 'SimSong', 'Arial Unicode MS']

        selected_font = None
        for font in chinese_fonts:
            if font in available_fonts:
                selected_font = font
                break

        if selected_font:
            # 强制设置字体 - 使用更强制的方式
            plt.rcParams['font.sans-serif'] = [selected_font]
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.size'] = 12

            # 存储选择的字体供后续使用
            self.chinese_font = selected_font

            print(f"   设置中文字体: {selected_font}")

            # 测试字体是否生效
            self._test_font_display()
        else:
            print("   警告: 未找到合适的中文字体")
            self.chinese_font = None

    def _test_font_display(self):
        """测试字体显示是否正常"""
        try:
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试', ha='center', va='center')
            plt.close(fig)
            print(f"   字体测试通过")
        except Exception as e:
            print(f"   字体测试失败: {e}")

    def _ensure_chinese_font(self):
        """确保中文字体设置生效"""
        if hasattr(self, 'chinese_font') and self.chinese_font:
            plt.rcParams['font.sans-serif'] = [self.chinese_font]
            plt.rcParams['axes.unicode_minus'] = False
    
    def create_complete_visualization_report(self, 
                                           results: Dict[str, Any],
                                           X_train: pd.DataFrame,
                                           y_train: pd.Series,
                                           feature_names: List[str],
                                           best_model_name: str = None,
                                           X_augmented: pd.DataFrame = None,
                                           y_augmented: pd.Series = None) -> None:
        """
        创建完整的可视化报告
        
        Args:
            results: 模型训练结果
            X_train: 原始训练特征数据
            y_train: 原始训练目标变量
            feature_names: 特征名称列表
            best_model_name: 最佳模型名称
            X_augmented: 增强后的特征数据（可选）
            y_augmented: 增强后的目标变量（可选）
        """
        print(f"\n=== 生成可视化报告 ===")
        print(f"   保存路径: {self.viz_dir}")
        
        try:
            # 1. 训练过程可视化
            print("   生成训练过程可视化...")
            self.plot_cross_validation_comparison(results)
            self.plot_model_performance_radar(results)
            
            # 2. 结果分析可视化
            print("   生成结果分析可视化...")
            if best_model_name and best_model_name in results:
                best_model = results[best_model_name]['model']
                
                # 特征重要性
                if hasattr(best_model, 'feature_importances_'):
                    self.plot_feature_importance(best_model, feature_names, best_model_name)
                
                # 预测vs真实值
                self.plot_prediction_vs_actual(best_model, X_train, y_train, best_model_name)
                
                # 残差分析
                self.plot_residual_analysis(best_model, X_train, y_train, best_model_name)
            
            # 性能热力图
            self.plot_performance_heatmap(results)
            
            # 3. 数据增强可视化
            if X_augmented is not None and y_augmented is not None:
                print("   生成数据增强可视化...")
                self.plot_data_distribution_comparison(X_train, X_augmented, y_train, y_augmented)
                self.plot_feature_correlation_heatmap(X_train, X_augmented, feature_names)
            
            print("   ✅ 可视化报告生成完成！")
            
        except Exception as e:
            print(f"   ❌ 生成可视化报告时发生错误: {str(e)}")
    
    def plot_cross_validation_comparison(self, results: Dict[str, Any]) -> None:
        """绘制交叉验证性能对比图"""
        # 确保中文字体设置
        self._ensure_chinese_font()

        # 准备数据
        valid_results = {k: v for k, v in results.items()
                        if not np.isnan(v.get('r2', np.nan))}

        if not valid_results:
            return
        
        models = list(valid_results.keys())
        metrics = ['r2', 'mae', 'rmse']
        metric_names = ['R² Score', 'MAE', 'RMSE']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('交叉验证性能对比', fontsize=16, fontweight='bold')

        metric_names_cn = ['R² 分数', 'MAE', 'RMSE']

        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names_cn)):
            ax = axes[i]

            # 准备数据
            means = [valid_results[model][metrics[i]] for model in models]
            stds = [valid_results[model][f'{metrics[i]}_std'] for model in models]

            # 创建条形图
            bars = ax.bar(range(len(models)), means, yerr=stds,
                         capsize=5, alpha=0.8, color=COLORS['palette'][:len(models)])

            # 设置样式
            ax.set_xlabel('模型', fontweight='bold')
            ax.set_ylabel(metric_name, fontweight='bold')
            ax.set_title(f'{metric_name} 对比', fontweight='bold')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')

            # 添加数值标签
            for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                       f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'cv_performance_comparison.png'), 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_model_performance_radar(self, results: Dict[str, Any]) -> None:
        """绘制模型性能雷达图"""
        # 确保中文字体设置
        self._ensure_chinese_font()

        valid_results = {k: v for k, v in results.items()
                        if not np.isnan(v.get('r2', np.nan))}

        if len(valid_results) < 2:
            return
        
        # 准备数据 - 标准化指标到0-1范围
        models = list(valid_results.keys())
        
        # 收集所有指标
        r2_scores = [valid_results[model]['r2'] for model in models]
        mae_scores = [valid_results[model]['mae'] for model in models]
        rmse_scores = [valid_results[model]['rmse'] for model in models]
        
        # 标准化 (R²越大越好，MAE和RMSE越小越好)
        r2_norm = [(score - min(r2_scores)) / (max(r2_scores) - min(r2_scores)) 
                   if max(r2_scores) != min(r2_scores) else 0.5 for score in r2_scores]
        mae_norm = [1 - (score - min(mae_scores)) / (max(mae_scores) - min(mae_scores)) 
                    if max(mae_scores) != min(mae_scores) else 0.5 for score in mae_scores]
        rmse_norm = [1 - (score - min(rmse_scores)) / (max(rmse_scores) - min(rmse_scores)) 
                     if max(rmse_scores) != min(rmse_scores) else 0.5 for score in rmse_scores]
        
        # 设置雷达图
        categories = ['R² 分数', 'MAE (反向)', 'RMSE (反向)']
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合图形

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # 为每个模型绘制雷达图
        for i, model in enumerate(models[:5]):  # 限制显示前5个模型
            values = [r2_norm[i], mae_norm[i], rmse_norm[i]]
            values += values[:1]  # 闭合图形

            ax.plot(angles, values, 'o-', linewidth=2,
                   label=model, color=COLORS['palette'][i])
            ax.fill(angles, values, alpha=0.25, color=COLORS['palette'][i])

        # 设置样式
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)

        plt.title('模型性能雷达图', size=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'model_performance_radar.png'), 
                   bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_feature_importance(self, model: Any, feature_names: List[str], model_name: str) -> None:
        """绘制特征重要性图"""
        # 确保中文字体设置
        self._ensure_chinese_font()

        if not hasattr(model, 'feature_importances_'):
            return

        importances = model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        # 取前15个最重要的特征
        top_features = feature_importance[:15]
        features, importance_values = zip(*top_features)

        plt.figure(figsize=(12, 8))

        # 创建水平条形图
        y_pos = np.arange(len(features))
        bars = plt.barh(y_pos, importance_values, color=COLORS['primary'], alpha=0.8)

        # 设置样式
        plt.xlabel('特征重要性', fontweight='bold', fontsize=12)
        plt.ylabel('特征', fontweight='bold', fontsize=12)
        plt.title(f'特征重要性 - {model_name}', fontweight='bold', fontsize=14)
        plt.yticks(y_pos, features)

        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, importance_values)):
            plt.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center', fontsize=10)

        # 反转y轴，使最重要的特征在顶部
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(self.viz_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.png'),
                   bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_prediction_vs_actual(self, model: Any, X_test: pd.DataFrame,
                                 y_test: pd.Series, model_name: str) -> None:
        """绘制预测值vs真实值散点图"""
        # 确保中文字体设置
        self._ensure_chinese_font()

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        plt.figure(figsize=(10, 8))

        # 创建散点图
        plt.scatter(y_test, y_pred, alpha=0.6, color=COLORS['primary'], s=50)

        # 添加完美预测线
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')

        # 设置样式
        plt.xlabel('真实值', fontweight='bold', fontsize=12)
        plt.ylabel('预测值', fontweight='bold', fontsize=12)
        plt.title(f'预测值 vs 真实值 - {model_name}\nR² = {r2:.4f}',
                 fontweight='bold', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 设置相等的坐标轴比例
        plt.axis('equal')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f'prediction_vs_actual_{model_name.lower().replace(" ", "_")}.png'),
                   bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_residual_analysis(self, model: Any, X_test: pd.DataFrame,
                              y_test: pd.Series, model_name: str) -> None:
        """绘制残差分析图"""
        # 确保中文字体设置
        self._ensure_chinese_font()

        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'残差分析 - {model_name}', fontsize=16, fontweight='bold')

        # 1. 残差vs预测值
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color=COLORS['primary'])
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('预测值', fontweight='bold')
        axes[0, 0].set_ylabel('残差', fontweight='bold')
        axes[0, 0].set_title('残差 vs 预测值', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 残差分布直方图
        axes[0, 1].hist(residuals, bins=20, alpha=0.7, color=COLORS['secondary'], edgecolor='black')
        axes[0, 1].axvline(x=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('残差', fontweight='bold')
        axes[0, 1].set_ylabel('频次', fontweight='bold')
        axes[0, 1].set_title('残差分布', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q图
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q图 (正态分布)', fontweight='bold')
        axes[1, 0].set_xlabel('理论分位数', fontweight='bold')
        axes[1, 0].set_ylabel('样本分位数', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 残差vs实际值
        axes[1, 1].scatter(y_test, residuals, alpha=0.6, color=COLORS['accent'])
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('真实值', fontweight='bold')
        axes[1, 1].set_ylabel('残差', fontweight='bold')
        axes[1, 1].set_title('残差 vs 真实值', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f'residual_analysis_{model_name.lower().replace(" ", "_")}.png'),
                   bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_performance_heatmap(self, results: Dict[str, Any]) -> None:
        """绘制模型性能热力图"""
        # 确保中文字体设置
        self._ensure_chinese_font()

        valid_results = {k: v for k, v in results.items()
                        if not np.isnan(v.get('r2', np.nan))}

        if not valid_results:
            return

        # 准备数据
        models = list(valid_results.keys())
        metrics = ['r2', 'mae', 'rmse']
        metric_names = ['R² Score', 'MAE', 'RMSE']

        # 创建数据矩阵
        data_matrix = []
        for model in models:
            row = [valid_results[model][metric] for metric in metrics]
            data_matrix.append(row)

        # 标准化数据 (0-1范围)
        data_matrix = np.array(data_matrix)
        for j in range(len(metrics)):
            col = data_matrix[:, j]
            if j == 0:  # R² - 越大越好
                normalized = (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else np.ones_like(col) * 0.5
            else:  # MAE, RMSE - 越小越好
                normalized = 1 - (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else np.ones_like(col) * 0.5
            data_matrix[:, j] = normalized

        plt.figure(figsize=(10, 8))

        # 创建热力图
        metric_names_cn = ['R² 分数', 'MAE', 'RMSE']
        sns.heatmap(data_matrix,
                   xticklabels=metric_names_cn,
                   yticklabels=models,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0.5,
                   square=True,
                   cbar_kws={'label': '标准化性能'})

        plt.title('模型性能热力图\n(标准化分数)', fontweight='bold', fontsize=14)
        plt.xlabel('指标', fontweight='bold')
        plt.ylabel('模型', fontweight='bold')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'performance_heatmap.png'),
                   bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_data_distribution_comparison(self, X_original: pd.DataFrame, X_augmented: pd.DataFrame,
                                        y_original: pd.Series, y_augmented: pd.Series) -> None:
        """绘制数据分布对比图"""
        # 确保中文字体设置
        self._ensure_chinese_font()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('数据分布对比：原始数据 vs 增强数据',
                    fontsize=16, fontweight='bold')

        # 1. 目标变量分布对比
        axes[0, 0].hist(y_original, bins=15, alpha=0.7, label='原始数据',
                       color=COLORS['primary'], density=True)
        axes[0, 0].hist(y_augmented, bins=30, alpha=0.5, label='增强数据',
                       color=COLORS['secondary'], density=True)
        axes[0, 0].set_xlabel('目标变量值', fontweight='bold')
        axes[0, 0].set_ylabel('密度', fontweight='bold')
        axes[0, 0].set_title('目标变量分布', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 样本数量对比
        categories = ['原始数据', '增强数据']
        counts = [len(X_original), len(X_augmented)]
        bars = axes[0, 1].bar(categories, counts, color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
        axes[0, 1].set_ylabel('样本数量', fontweight='bold')
        axes[0, 1].set_title('样本数量对比', fontweight='bold')

        # 添加数值标签
        for bar, count in zip(bars, counts):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                           str(count), ha='center', va='bottom', fontweight='bold')

        # 3. 特征均值对比（选择前8个特征）
        n_features_to_show = min(8, len(X_original.columns))
        features_to_show = X_original.columns[:n_features_to_show]

        orig_means = [X_original[col].mean() for col in features_to_show]
        aug_means = [X_augmented[col].mean() for col in features_to_show]

        x = np.arange(len(features_to_show))
        width = 0.35

        axes[1, 0].bar(x - width/2, orig_means, width, label='原始数据',
                      color=COLORS['primary'], alpha=0.8)
        axes[1, 0].bar(x + width/2, aug_means, width, label='增强数据',
                      color=COLORS['secondary'], alpha=0.8)

        axes[1, 0].set_xlabel('特征', fontweight='bold')
        axes[1, 0].set_ylabel('均值', fontweight='bold')
        axes[1, 0].set_title('特征均值对比', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(features_to_show, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 特征标准差对比
        orig_stds = [X_original[col].std() for col in features_to_show]
        aug_stds = [X_augmented[col].std() for col in features_to_show]

        axes[1, 1].bar(x - width/2, orig_stds, width, label='原始数据',
                      color=COLORS['primary'], alpha=0.8)
        axes[1, 1].bar(x + width/2, aug_stds, width, label='增强数据',
                      color=COLORS['secondary'], alpha=0.8)

        axes[1, 1].set_xlabel('特征', fontweight='bold')
        axes[1, 1].set_ylabel('标准差', fontweight='bold')
        axes[1, 1].set_title('特征标准差对比', fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(features_to_show, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'data_distribution_comparison.png'),
                   bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_feature_correlation_heatmap(self, X_original: pd.DataFrame, X_augmented: pd.DataFrame,
                                       feature_names: List[str]) -> None:
        """绘制特征相关性热力图对比"""
        # 确保中文字体设置
        self._ensure_chinese_font()

        # 限制特征数量以保持可读性
        n_features_to_show = min(12, len(feature_names))
        features_to_show = feature_names[:n_features_to_show]

        # 计算相关性矩阵
        corr_original = X_original[features_to_show].corr()
        corr_augmented = X_augmented[features_to_show].corr()

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('特征相关性对比', fontsize=16, fontweight='bold')

        # 1. 原始数据相关性
        sns.heatmap(corr_original, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, ax=axes[0], cbar_kws={'shrink': 0.8})
        axes[0].set_title('原始数据相关性', fontweight='bold')

        # 2. 增强数据相关性
        sns.heatmap(corr_augmented, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, ax=axes[1], cbar_kws={'shrink': 0.8})
        axes[1].set_title('增强数据相关性', fontweight='bold')

        # 3. 相关性差异
        corr_diff = corr_augmented - corr_original
        sns.heatmap(corr_diff, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   square=True, ax=axes[2], cbar_kws={'shrink': 0.8})
        axes[2].set_title('相关性差异\n(增强数据 - 原始数据)', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'feature_correlation_comparison.png'),
                   bbox_inches='tight', facecolor='white')
        plt.close()
