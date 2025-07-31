"""
可视化模块

该模块负责生成特征选择过程中的各种可视化图表。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings

try:
    from .config import FeatureSelectionConfig, VisualizationError
    from .utils import setup_matplotlib_chinese, get_color_palette
except ImportError:
    from config import FeatureSelectionConfig, VisualizationError
    from utils import setup_matplotlib_chinese, get_color_palette

# 设置中文字体和警告过滤
setup_matplotlib_chinese()
warnings.filterwarnings('ignore')


class FeatureSelectionVisualizer:
    """特征选择可视化器"""
    
    def __init__(self, config: FeatureSelectionConfig):
        """
        初始化可视化器
        
        Args:
            config: 特征选择配置对象
        """
        self.config = config
        setup_matplotlib_chinese()
    
    def plot_evolution_process(self, evolution_history: Dict[str, Any]):
        """
        绘制遗传算法进化过程图
        
        Args:
            evolution_history: 进化历史数据
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_size_large)
            
            generations = range(len(evolution_history['fitness_history']))
            
            # 1. 适应度进化曲线
            ax1.plot(generations, evolution_history['fitness_history'], 
                    label='平均适应度', color='blue', alpha=0.7)
            ax1.plot(generations, evolution_history['best_fitness_history'], 
                    label='最佳适应度', color='red', linewidth=2)
            ax1.set_xlabel('代数')
            ax1.set_ylabel('适应度 (R²)')
            ax1.set_title('遗传算法适应度进化曲线')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 种群多样性变化
            ax2.plot(generations, evolution_history['diversity_history'], 
                    color='green', linewidth=2)
            ax2.set_xlabel('代数')
            ax2.set_ylabel('种群多样性')
            ax2.set_title('种群多样性变化')
            ax2.grid(True, alpha=0.3)
            
            # 3. 选择压力变化
            ax3.plot(generations, evolution_history['selection_pressure_history'], 
                    color='orange', linewidth=2)
            ax3.set_xlabel('代数')
            ax3.set_ylabel('选择压力')
            ax3.set_title('选择压力变化')
            ax3.grid(True, alpha=0.3)
            
            # 4. 适应度分布箱线图（最后10代）
            if len(evolution_history['generation_stats']) >= 10:
                last_10_stats = evolution_history['generation_stats'][-10:]
                fitness_data = []
                generation_labels = []
                
                for stat in last_10_stats:
                    # 模拟适应度分布（基于均值和标准差）
                    mean_fit = stat['mean_fitness']
                    std_fit = stat['std_fitness']
                    simulated_fitness = np.random.normal(mean_fit, std_fit, 30)
                    fitness_data.append(simulated_fitness)
                    generation_labels.append(f"第{stat['generation']+1}代")
                
                ax4.boxplot(fitness_data, labels=generation_labels)
                ax4.set_xlabel('代数')
                ax4.set_ylabel('适应度分布')
                ax4.set_title('最后10代适应度分布')
                ax4.tick_params(axis='x', rotation=45)
            else:
                ax4.text(0.5, 0.5, '数据不足\n(需要至少10代)', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('最后10代适应度分布')
            
            plt.tight_layout()
            
            # 保存图片
            output_path = self.config.get_file_path(self.config.evolution_process_file)
            plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            
            print(f"进化过程图已保存: {output_path}")
            
        except Exception as e:
            raise VisualizationError(f"绘制进化过程图失败: {str(e)}")
    
    def plot_feature_importance_comparison(self, importance_ranking: pd.DataFrame,
                                         selected_indices: np.ndarray,
                                         feature_names: np.ndarray):
        """
        绘制特征重要性对比图
        
        Args:
            importance_ranking: 特征重要性排序DataFrame
            selected_indices: 选择的特征索引
            feature_names: 所有特征名称
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size_large)
            
            # 1. 所有特征重要性排序
            top_20 = importance_ranking.head(20)
            colors = ['red' if idx in selected_indices else 'lightblue' 
                     for idx in range(len(feature_names)) if feature_names[idx] in top_20['feature_name'].values]
            
            bars1 = ax1.barh(range(len(top_20)), top_20['importance'], color=colors)
            ax1.set_yticks(range(len(top_20)))
            ax1.set_yticklabels(top_20['feature_name'], fontsize=10)
            ax1.set_xlabel('重要性分数')
            ax1.set_title('特征重要性排序（前20名）')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', label='已选择特征'),
                             Patch(facecolor='lightblue', label='未选择特征')]
            ax1.legend(handles=legend_elements, loc='lower right')
            
            # 2. 选择特征的重要性分布
            selected_feature_names = feature_names[selected_indices]
            selected_importance = importance_ranking[
                importance_ranking['feature_name'].isin(selected_feature_names)
            ].sort_values('importance', ascending=True)
            
            bars2 = ax2.barh(range(len(selected_importance)), 
                           selected_importance['importance'], 
                           color='darkred', alpha=0.7)
            ax2.set_yticks(range(len(selected_importance)))
            ax2.set_yticklabels(selected_importance['feature_name'], fontsize=10)
            ax2.set_xlabel('重要性分数')
            ax2.set_title(f'选择特征的重要性分布（{len(selected_indices)}个）')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # 添加数值标签
            for i, bar in enumerate(bars2):
                width = bar.get_width()
                ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=8)
            
            plt.tight_layout()
            
            # 保存图片
            output_path = self.config.get_file_path(self.config.feature_importance_file)
            plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            
            print(f"特征重要性对比图已保存: {output_path}")
            
        except Exception as e:
            raise VisualizationError(f"绘制特征重要性对比图失败: {str(e)}")
    
    def plot_model_comparison(self, validation_results: Dict[str, Any]):
        """
        绘制模型性能对比图
        
        Args:
            validation_results: 模型验证结果
        """
        try:
            # 过滤有效结果
            valid_results = {}
            for model_name, result in validation_results.items():
                if isinstance(result, dict) and 'mean_r2' in result and not np.isnan(result['mean_r2']):
                    valid_results[model_name] = result
            
            if not valid_results:
                print("警告: 没有有效的模型验证结果用于绘图")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size_large)
            
            # 准备数据
            model_names = list(valid_results.keys())
            mean_scores = [valid_results[name]['mean_r2'] for name in model_names]
            std_scores = [valid_results[name]['std_r2'] for name in model_names]
            min_scores = [valid_results[name]['min_r2'] for name in model_names]
            max_scores = [valid_results[name]['max_r2'] for name in model_names]
            
            # 按平均分数排序
            sorted_indices = np.argsort(mean_scores)[::-1]
            model_names = [model_names[i] for i in sorted_indices]
            mean_scores = [mean_scores[i] for i in sorted_indices]
            std_scores = [std_scores[i] for i in sorted_indices]
            min_scores = [min_scores[i] for i in sorted_indices]
            max_scores = [max_scores[i] for i in sorted_indices]
            
            # 1. 平均R²分数对比
            colors = get_color_palette(len(model_names))
            bars = ax1.bar(range(len(model_names)), mean_scores, 
                          yerr=std_scores, capsize=5, color=colors, alpha=0.7)
            ax1.set_xticks(range(len(model_names)))
            ax1.set_xticklabels(model_names, rotation=45, ha='right')
            ax1.set_ylabel('R² 分数')
            ax1.set_title('模型性能对比（平均R² ± 标准差）')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, (bar, score, std) in enumerate(zip(bars, mean_scores, std_scores)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            
            # 2. R²分数范围对比
            y_pos = np.arange(len(model_names))
            ax2.barh(y_pos, [max_val - min_val for max_val, min_val in zip(max_scores, min_scores)],
                    left=min_scores, color=colors, alpha=0.7)
            
            # 添加平均值点
            ax2.scatter(mean_scores, y_pos, color='red', s=50, zorder=5, label='平均值')
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(model_names)
            ax2.set_xlabel('R² 分数')
            ax2.set_title('模型性能范围对比')
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.legend()
            
            plt.tight_layout()
            
            # 保存图片
            output_path = self.config.get_file_path(self.config.model_comparison_file)
            plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            
            print(f"模型对比图已保存: {output_path}")
            
        except Exception as e:
            raise VisualizationError(f"绘制模型对比图失败: {str(e)}")
    
    def plot_convergence_analysis(self, evolution_history: Dict[str, Any]):
        """
        绘制收敛分析图
        
        Args:
            evolution_history: 进化历史数据
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_size_large)
            
            generations = range(len(evolution_history['fitness_history']))
            best_fitness = evolution_history['best_fitness_history']
            
            # 1. 收敛曲线
            ax1.plot(generations, best_fitness, 'b-', linewidth=2, label='最佳适应度')
            
            # 添加收敛趋势线
            if len(best_fitness) > 10:
                z = np.polyfit(generations, best_fitness, 2)
                p = np.poly1d(z)
                ax1.plot(generations, p(generations), 'r--', alpha=0.7, label='趋势线')
            
            ax1.set_xlabel('代数')
            ax1.set_ylabel('最佳适应度')
            ax1.set_title('收敛曲线分析')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 改进速度分析
            if len(best_fitness) > 1:
                improvements = np.diff(best_fitness)
                ax2.plot(range(1, len(best_fitness)), improvements, 'g-', linewidth=2)
                ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                ax2.set_xlabel('代数')
                ax2.set_ylabel('适应度改进')
                ax2.set_title('每代改进速度')
                ax2.grid(True, alpha=0.3)
            
            # 3. 滑动窗口平均
            window_size = min(10, len(best_fitness) // 4)
            if window_size > 1:
                moving_avg = pd.Series(best_fitness).rolling(window=window_size).mean()
                ax3.plot(generations, best_fitness, 'b-', alpha=0.5, label='原始数据')
                ax3.plot(generations, moving_avg, 'r-', linewidth=2, label=f'{window_size}代滑动平均')
                ax3.set_xlabel('代数')
                ax3.set_ylabel('适应度')
                ax3.set_title('滑动平均收敛分析')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 4. 收敛指标显示
            if 'convergence_metrics' in evolution_history:
                metrics = evolution_history['convergence_metrics']
                
                # 创建指标文本
                metrics_text = f"""收敛指标:
收敛速度: {metrics['convergence_rate']:.6f}
稳定性指标: {metrics['stability_index']:.6f}
改进比例: {metrics['improvement_ratio']:.3f}

最终适应度: {best_fitness[-1]:.4f}
最大适应度: {max(best_fitness):.4f}
总代数: {len(best_fitness)}"""
                
                ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
                ax4.axis('off')
                ax4.set_title('收敛指标总结')
            
            plt.tight_layout()
            
            # 保存图片
            output_path = self.config.get_file_path(self.config.convergence_analysis_file)
            plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            
            print(f"收敛分析图已保存: {output_path}")
            
        except Exception as e:
            raise VisualizationError(f"绘制收敛分析图失败: {str(e)}")

    def plot_feature_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                       selected_indices: np.ndarray,
                                       feature_names: np.ndarray):
        """
        绘制特征相关性热力图

        Args:
            correlation_matrix: 相关性矩阵
            selected_indices: 选择的特征索引
            feature_names: 所有特征名称
        """
        try:
            selected_feature_names = feature_names[selected_indices]

            # 创建两个子图：全部特征和选择特征
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # 1. 选择特征的相关性热力图
            selected_corr = correlation_matrix.loc[selected_feature_names, selected_feature_names]

            mask1 = np.triu(np.ones_like(selected_corr, dtype=bool))
            sns.heatmap(selected_corr, mask=mask1, annot=True, cmap='RdBu_r', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax1)
            ax1.set_title(f'选择特征相关性热力图 ({len(selected_indices)}个特征)')

            # 2. 所有特征相关性热力图（只显示上三角）
            mask2 = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask2, cmap='RdBu_r', center=0,
                       square=True, cbar_kws={"shrink": .8}, ax=ax2)
            ax2.set_title(f'所有特征相关性热力图 ({len(feature_names)}个特征)')

            plt.tight_layout()

            # 保存图片
            output_path = self.config.get_file_path(self.config.feature_correlation_file)
            plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()

            print(f"特征相关性热力图已保存: {output_path}")

        except Exception as e:
            raise VisualizationError(f"绘制特征相关性热力图失败: {str(e)}")

    def plot_population_diversity_analysis(self, evolution_history: Dict[str, Any]):
        """
        绘制种群多样性分析图

        Args:
            evolution_history: 进化历史数据
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_size_large)

            generations = range(len(evolution_history['diversity_history']))
            diversity = evolution_history['diversity_history']
            selection_pressure = evolution_history['selection_pressure_history']

            # 1. 多样性变化趋势
            ax1.plot(generations, diversity, 'g-', linewidth=2, label='种群多样性')
            ax1.set_xlabel('代数')
            ax1.set_ylabel('多样性指标')
            ax1.set_title('种群多样性变化趋势')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # 2. 多样性与选择压力关系
            ax2.scatter(diversity, selection_pressure, alpha=0.6, c=generations, cmap='viridis')
            ax2.set_xlabel('种群多样性')
            ax2.set_ylabel('选择压力')
            ax2.set_title('多样性与选择压力关系')
            ax2.grid(True, alpha=0.3)

            # 添加颜色条
            cbar = plt.colorbar(ax2.collections[0], ax=ax2)
            cbar.set_label('代数')

            # 3. 多样性分布直方图
            ax3.hist(diversity, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(np.mean(diversity), color='red', linestyle='--',
                       label=f'平均值: {np.mean(diversity):.2f}')
            ax3.set_xlabel('多样性值')
            ax3.set_ylabel('频次')
            ax3.set_title('多样性分布直方图')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. 多样性统计信息
            diversity_stats = {
                '平均多样性': np.mean(diversity),
                '多样性标准差': np.std(diversity),
                '最大多样性': np.max(diversity),
                '最小多样性': np.min(diversity),
                '多样性范围': np.max(diversity) - np.min(diversity)
            }

            stats_text = '\n'.join([f'{key}: {value:.3f}' for key, value in diversity_stats.items()])
            ax4.text(0.1, 0.9, f'多样性统计:\n{stats_text}', transform=ax4.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('多样性统计信息')

            plt.tight_layout()

            # 保存图片
            output_path = self.config.get_file_path(self.config.population_diversity_file)
            plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()

            print(f"种群多样性分析图已保存: {output_path}")

        except Exception as e:
            raise VisualizationError(f"绘制种群多样性分析图失败: {str(e)}")

    def plot_optimal_feature_distribution(self, selected_indices: np.ndarray,
                                        feature_names: np.ndarray,
                                        importance_ranking: pd.DataFrame):
        """
        绘制最优特征分布图

        Args:
            selected_indices: 选择的特征索引
            feature_names: 所有特征名称
            importance_ranking: 特征重要性排序
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_size_large)

            selected_feature_names = feature_names[selected_indices]

            # 1. 特征选择分布饼图
            selected_count = len(selected_indices)
            unselected_count = len(feature_names) - selected_count

            sizes = [selected_count, unselected_count]
            labels = [f'选择特征\n({selected_count}个)', f'未选择特征\n({unselected_count}个)']
            colors = ['lightcoral', 'lightblue']
            explode = (0.1, 0)

            ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
            ax1.set_title('特征选择分布')

            # 2. 选择特征在重要性排序中的位置
            selected_ranks = []
            for feature_name in selected_feature_names:
                rank = importance_ranking[importance_ranking['feature_name'] == feature_name]['rank'].iloc[0]
                selected_ranks.append(rank)

            ax2.hist(selected_ranks, bins=min(10, len(selected_ranks)),
                    alpha=0.7, color='orange', edgecolor='black')
            ax2.set_xlabel('重要性排名')
            ax2.set_ylabel('特征数量')
            ax2.set_title('选择特征的重要性排名分布')
            ax2.grid(True, alpha=0.3)

            # 3. 特征重要性分数分布
            selected_importance_scores = []
            for feature_name in selected_feature_names:
                score = importance_ranking[importance_ranking['feature_name'] == feature_name]['importance'].iloc[0]
                selected_importance_scores.append(score)

            ax3.scatter(range(len(selected_importance_scores)),
                       sorted(selected_importance_scores, reverse=True),
                       c='red', s=100, alpha=0.7)
            ax3.set_xlabel('特征索引（按重要性排序）')
            ax3.set_ylabel('重要性分数')
            ax3.set_title('选择特征重要性分数分布')
            ax3.grid(True, alpha=0.3)

            # 4. 特征选择质量评估
            # 计算选择特征在前N名中的比例
            top_percentages = []
            top_ns = [5, 10, 15, 20]

            for top_n in top_ns:
                if top_n <= len(feature_names):
                    top_features = set(importance_ranking.head(top_n)['feature_name'])
                    selected_in_top = len(set(selected_feature_names) & top_features)
                    percentage = (selected_in_top / min(top_n, len(selected_feature_names))) * 100
                    top_percentages.append(percentage)
                else:
                    top_percentages.append(0)

            bars = ax4.bar([f'前{n}名' for n in top_ns], top_percentages,
                          color='purple', alpha=0.7)
            ax4.set_ylabel('选择比例 (%)')
            ax4.set_title('选择特征在重要性排名中的分布')
            ax4.grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for bar, percentage in zip(bars, top_percentages):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{percentage:.1f}%', ha='center', va='bottom')

            plt.tight_layout()

            # 保存图片
            output_path = self.config.get_file_path(self.config.feature_distribution_file)
            plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()

            print(f"最优特征分布图已保存: {output_path}")

        except Exception as e:
            raise VisualizationError(f"绘制最优特征分布图失败: {str(e)}")

    def create_all_visualizations(self, evolution_history: Dict[str, Any],
                                validation_results: Dict[str, Any],
                                selected_indices: np.ndarray,
                                feature_names: np.ndarray,
                                importance_ranking: pd.DataFrame,
                                correlation_matrix: pd.DataFrame):
        """
        创建所有可视化图表

        Args:
            evolution_history: 进化历史数据
            validation_results: 验证结果
            selected_indices: 选择的特征索引
            feature_names: 所有特征名称
            importance_ranking: 特征重要性排序
            correlation_matrix: 相关性矩阵
        """
        print("开始生成可视化图表...")

        try:
            # 1. 进化过程图
            self.plot_evolution_process(evolution_history)

            # 2. 特征重要性对比图
            self.plot_feature_importance_comparison(importance_ranking, selected_indices, feature_names)

            # 3. 模型性能对比图
            self.plot_model_comparison(validation_results)

            # 4. 收敛分析图
            self.plot_convergence_analysis(evolution_history)

            # 5. 特征相关性热力图
            self.plot_feature_correlation_heatmap(correlation_matrix, selected_indices, feature_names)

            # 6. 种群多样性分析图
            self.plot_population_diversity_analysis(evolution_history)

            # 7. 最优特征分布图
            self.plot_optimal_feature_distribution(selected_indices, feature_names, importance_ranking)

            print("所有可视化图表生成完成！")

        except Exception as e:
            print(f"生成可视化图表时出现错误: {str(e)}")
            raise VisualizationError(f"生成可视化图表失败: {str(e)}")
