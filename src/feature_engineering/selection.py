# -*- coding: utf-8 -*-
"""
02_特征选择模块 - 藻类脂质预测
Feature Selection Module for Algae Lipid Prediction

目标：选择最优特征组合，提升模型性能至R² ≥ 0.9
策略：
1. 多种特征选择方法集成
2. 特征重要性深度分析
3. 递归特征消除优化
4. 特征稳定性验证
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, RFECV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedFeatureSelector:
    """高级特征选择器"""
    
    def __init__(self):
        self.selected_features = {}
        self.feature_scores = {}
        self.selection_history = []
        
    def load_processed_data(self, filepath='results/01_processed_data.csv'):
        """加载预处理后的数据"""
        print(" 加载预处理后的数据...")
        
        try:
            df = pd.read_csv(filepath)
            print(f" 成功加载数据: {df.shape}")
            return df
        except Exception as e:
            print(f" 加载失败: {e}")
            return None
    
    def correlation_analysis(self, df, target_col='lipid(%)', threshold=0.1):
        """相关性分析"""
        print(f"\n 相关性分析 (阈值: {threshold})...")
        
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Pearson相关性
        pearson_corrs = {}
        for col in feature_cols:
            try:
                corr, p_value = pearsonr(df[col], df[target_col])
                pearson_corrs[col] = {'correlation': corr, 'p_value': p_value}
            except:
                pearson_corrs[col] = {'correlation': 0, 'p_value': 1}
        
        # Spearman相关性
        spearman_corrs = {}
        for col in feature_cols:
            try:
                corr, p_value = spearmanr(df[col], df[target_col])
                spearman_corrs[col] = {'correlation': corr, 'p_value': p_value}
            except:
                spearman_corrs[col] = {'correlation': 0, 'p_value': 1}
        
        # 选择显著相关的特征
        selected_features = []
        for col in feature_cols:
            pearson_corr = abs(pearson_corrs[col]['correlation'])
            spearman_corr = abs(spearman_corrs[col]['correlation'])
            
            if pearson_corr > threshold or spearman_corr > threshold:
                selected_features.append(col)
        
        self.selected_features['correlation'] = selected_features
        self.feature_scores['pearson'] = pearson_corrs
        self.feature_scores['spearman'] = spearman_corrs
        
        print(f"相关性选择: {len(selected_features)} 个特征")
        return selected_features
    
    def mutual_information_selection(self, df, target_col='lipid(%)', k_best=20):
        """互信息特征选择"""
        print(f"\n 互信息特征选择 (选择前{k_best}个)...")
        
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        y = df[target_col].values
        
        # 计算互信息
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # 创建特征-分数映射
        mi_dict = dict(zip(feature_cols, mi_scores))
        
        # 选择前k个特征
        sorted_features = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, score in sorted_features[:k_best]]
        
        self.selected_features['mutual_info'] = selected_features
        self.feature_scores['mutual_info'] = mi_dict
        
        print(f"互信息选择: {len(selected_features)} 个特征")
        return selected_features
    
    def univariate_selection(self, df, target_col='lipid(%)', k_best=15):
        """单变量特征选择"""
        print(f"\n 单变量特征选择 (F-regression, 前{k_best}个)...")
        
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        y = df[target_col].values
        
        # F-regression选择
        selector = SelectKBest(score_func=f_regression, k=k_best)
        X_selected = selector.fit_transform(X, y)
        
        # 获取选中的特征
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_cols[i] for i in selected_indices]
        
        # 获取F分数
        f_scores = dict(zip(feature_cols, selector.scores_))
        
        self.selected_features['univariate'] = selected_features
        self.feature_scores['f_regression'] = f_scores
        
        print(f"单变量选择: {len(selected_features)} 个特征")
        return selected_features
    
    def tree_based_selection(self, df, target_col='lipid(%)', n_features=15):
        """基于树模型的特征选择"""
        print(f"\n 基于树模型的特征选择 (前{n_features}个)...")
        
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        y = df[target_col].values
        
        # 随机森林特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = dict(zip(feature_cols, rf.feature_importances_))
        
        # 极端随机树特征重要性
        et = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        et.fit(X, y)
        et_importance = dict(zip(feature_cols, et.feature_importances_))
        
        # 综合重要性
        combined_importance = {}
        for col in feature_cols:
            combined_importance[col] = (rf_importance[col] + et_importance[col]) / 2
        
        # 选择前n个特征
        sorted_features = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, score in sorted_features[:n_features]]
        
        self.selected_features['tree_based'] = selected_features
        self.feature_scores['rf_importance'] = rf_importance
        self.feature_scores['et_importance'] = et_importance
        self.feature_scores['combined_tree'] = combined_importance
        
        print(f"树模型选择: {len(selected_features)} 个特征")
        return selected_features
    
    def lasso_selection(self, df, target_col='lipid(%)', alpha_range=None):
        """Lasso特征选择"""
        print("\n Lasso特征选择...")
        
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        y = df[target_col].values
        
        if alpha_range is None:
            alpha_range = np.logspace(-4, 1, 50)
        
        # Lasso交叉验证
        lasso_cv = LassoCV(alphas=alpha_range, cv=5, random_state=42, max_iter=2000)
        lasso_cv.fit(X, y)
        
        # 获取非零系数的特征
        selected_indices = np.where(lasso_cv.coef_ != 0)[0]
        selected_features = [feature_cols[i] for i in selected_indices]
        
        # 特征系数
        lasso_coefs = dict(zip(feature_cols, lasso_cv.coef_))
        
        self.selected_features['lasso'] = selected_features
        self.feature_scores['lasso_coef'] = lasso_coefs
        
        print(f"Lasso选择: {len(selected_features)} 个特征 (alpha={lasso_cv.alpha_:.4f})")
        return selected_features
    
    def recursive_feature_elimination(self, df, target_col='lipid(%)', n_features=12):
        """递归特征消除"""
        print(f"\n 递归特征消除 (目标{n_features}个特征)...")
        
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        y = df[target_col].values
        
        # 使用随机森林作为基础估计器
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        # RFE
        rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)
        
        # 获取选中的特征
        selected_indices = rfe.get_support(indices=True)
        selected_features = [feature_cols[i] for i in selected_indices]
        
        # 特征排名
        feature_ranking = dict(zip(feature_cols, rfe.ranking_))
        
        self.selected_features['rfe'] = selected_features
        self.feature_scores['rfe_ranking'] = feature_ranking
        
        print(f"RFE选择: {len(selected_features)} 个特征")
        return selected_features
    
    def rfecv_selection(self, df, target_col='lipid(%)', min_features=8):
        """带交叉验证的递归特征消除"""
        print(f"\n RFECV特征选择 (最少{min_features}个特征)...")
        
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        y = df[target_col].values
        
        # 使用随机森林作为基础估计器
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        # RFECV
        rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='r2', 
                      min_features_to_select=min_features, n_jobs=-1)
        rfecv.fit(X, y)
        
        # 获取选中的特征
        selected_indices = rfecv.get_support(indices=True)
        selected_features = [feature_cols[i] for i in selected_indices]
        
        # 交叉验证分数
        cv_scores = rfecv.cv_results_['mean_test_score'] if hasattr(rfecv, 'cv_results_') else rfecv.grid_scores_
        
        self.selected_features['rfecv'] = selected_features
        self.feature_scores['rfecv_scores'] = cv_scores
        
        print(f"RFECV选择: {len(selected_features)} 个特征 (最佳CV分数: {max(cv_scores):.4f})")
        return selected_features
    
    def ensemble_feature_selection(self, df, target_col='lipid(%)', voting_threshold=3):
        """集成特征选择"""
        print(f"\n 集成特征选择 (投票阈值: {voting_threshold})...")
        
        # 运行所有特征选择方法
        methods = {
            'correlation': lambda: self.correlation_analysis(df, target_col, threshold=0.05),
            'mutual_info': lambda: self.mutual_information_selection(df, target_col, k_best=20),
            'univariate': lambda: self.univariate_selection(df, target_col, k_best=18),
            'tree_based': lambda: self.tree_based_selection(df, target_col, n_features=15),
            'lasso': lambda: self.lasso_selection(df, target_col),
            'rfe': lambda: self.recursive_feature_elimination(df, target_col, n_features=12),
            'rfecv': lambda: self.rfecv_selection(df, target_col, min_features=8)
        }
        
        # 执行所有方法
        all_selections = {}
        for method_name, method_func in methods.items():
            try:
                selected = method_func()
                all_selections[method_name] = set(selected)
                print(f"  {method_name}: {len(selected)} 个特征")
            except Exception as e:
                print(f"  {method_name}: 失败 ({e})")
                all_selections[method_name] = set()
        
        # 投票统计
        feature_votes = {}
        all_features = set()
        for features in all_selections.values():
            all_features.update(features)
        
        for feature in all_features:
            votes = sum(1 for features in all_selections.values() if feature in features)
            feature_votes[feature] = votes
        
        # 选择得票数 >= 阈值的特征
        ensemble_features = [feat for feat, votes in feature_votes.items() if votes >= voting_threshold]
        
        # 按得票数排序
        ensemble_features.sort(key=lambda x: feature_votes[x], reverse=True)
        
        self.selected_features['ensemble'] = ensemble_features
        self.feature_scores['feature_votes'] = feature_votes
        
        print(f"集成选择: {len(ensemble_features)} 个特征")
        
        # 显示投票结果
        print("\n投票结果 (前15个):")
        sorted_votes = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, votes) in enumerate(sorted_votes[:15], 1):
            status = "✓" if feat in ensemble_features else "✗"
            print(f"  {i:2d}. {status} {feat}: {votes} 票")
        
        return ensemble_features, all_selections, feature_votes
    
    def validate_feature_stability(self, df, selected_features, target_col='lipid(%)', n_splits=5):
        """验证特征稳定性"""
        print(f"\n 特征稳定性验证 ({n_splits}折交叉验证)...")
        
        X = df[selected_features].values
        y = df[target_col].values
        
        # 使用简单的随机森林进行验证
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # 交叉验证
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf, X, y, cv=kf, scoring='r2')
        
        stability_metrics = {
            'mean_r2': cv_scores.mean(),
            'std_r2': cv_scores.std(),
            'cv_scores': cv_scores,
            'stability_score': 1 - (cv_scores.std() / abs(cv_scores.mean())) if cv_scores.mean() != 0 else 0
        }
        
        print(f"稳定性验证结果:")
        print(f"  平均R²: {stability_metrics['mean_r2']:.4f}")
        print(f"  标准差: {stability_metrics['std_r2']:.4f}")
        print(f"  稳定性分数: {stability_metrics['stability_score']:.4f}")
        
        return stability_metrics
    
    def create_feature_analysis_visualization(self, df, ensemble_features, feature_votes, output_dir='results'):
        """创建特征分析可视化"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Advanced Feature Selection Analysis', fontsize=16, weight='bold')
        
        target_col = 'lipid(%)'
        
        # 1. 特征投票结果
        sorted_votes = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        top_20_votes = sorted_votes[:20]
        
        features = [f.split('(')[0][:15] for f, v in top_20_votes]
        votes = [v for f, v in top_20_votes]
        colors = ['green' if f in ensemble_features else 'gray' for f, v in top_20_votes]
        
        axes[0, 0].barh(range(len(features)), votes, color=colors, alpha=0.7)
        axes[0, 0].set_yticks(range(len(features)))
        axes[0, 0].set_yticklabels(features)
        axes[0, 0].set_title('Feature Voting Results (Top 20)')
        axes[0, 0].set_xlabel('Number of Votes')
        axes[0, 0].invert_yaxis()
        
        # 2. 选中特征的相关性
        if len(ensemble_features) > 1:
            selected_corr = df[ensemble_features + [target_col]].corr()
            sns.heatmap(selected_corr, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1], fmt='.2f')
            axes[0, 1].set_title('Selected Features Correlation Matrix')
        
        # 3. 特征重要性对比
        if 'combined_tree' in self.feature_scores:
            tree_scores = self.feature_scores['combined_tree']
            selected_tree_scores = {k: v for k, v in tree_scores.items() if k in ensemble_features}
            
            if selected_tree_scores:
                sorted_tree = sorted(selected_tree_scores.items(), key=lambda x: x[1], reverse=True)
                tree_features = [f.split('(')[0][:15] for f, s in sorted_tree]
                tree_scores_vals = [s for f, s in sorted_tree]
                
                axes[0, 2].bar(range(len(tree_features)), tree_scores_vals, color='lightblue', alpha=0.7)
                axes[0, 2].set_xticks(range(len(tree_features)))
                axes[0, 2].set_xticklabels(tree_features, rotation=45, ha='right')
                axes[0, 2].set_title('Tree-based Feature Importance')
                axes[0, 2].set_ylabel('Importance Score')
        
        # 4. 不同方法选择的特征数量
        method_counts = {}
        for method, features in self.selected_features.items():
            if method != 'ensemble':
                method_counts[method] = len(features)
        
        if method_counts:
            axes[1, 0].bar(method_counts.keys(), method_counts.values(), color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Features Selected by Each Method')
            axes[1, 0].set_ylabel('Number of Features')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. 特征与目标变量的散点图（选择最相关的特征）
        if ensemble_features:
            correlations = df[ensemble_features].corrwith(df[target_col]).abs()
            best_feature = correlations.idxmax()
            
            axes[1, 1].scatter(df[best_feature], df[target_col], alpha=0.6, color='red')
            axes[1, 1].set_xlabel(best_feature.split('(')[0])
            axes[1, 1].set_ylabel('lipid(%)')
            axes[1, 1].set_title(f'Best Feature vs Target\n(r = {correlations[best_feature]:.3f})')
            
            # 添加趋势线
            z = np.polyfit(df[best_feature], df[target_col], 1)
            p = np.poly1d(z)
            axes[1, 1].plot(df[best_feature], p(df[best_feature]), "r--", alpha=0.8)
        
        # 6. 特征选择摘要
        summary_text = f"Total Original Features: {len(df.columns) - 1}\n"
        summary_text += f"Selected Features: {len(ensemble_features)}\n"
        summary_text += f"Selection Ratio: {len(ensemble_features)/(len(df.columns)-1)*100:.1f}%\n"
        summary_text += f"Average Votes: {np.mean(list(feature_votes.values())):.1f}\n"
        summary_text += f"Max Votes: {max(feature_votes.values())}\n"
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, transform=axes[1, 2].transAxes,
                        verticalalignment='center')
        axes[1, 2].set_title('Feature Selection Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_feature_selection_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" 特征分析可视化已保存: {output_dir}/02_feature_selection_analysis.png")
    
    def save_selected_features(self, df, ensemble_features, target_col='lipid(%)', output_dir='results'):
        """保存选中的特征数据"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建选中特征的数据集
        selected_data = df[ensemble_features + [target_col]].copy()
        
        # 保存数据
        selected_data.to_csv(f'{output_dir}/02_selected_features_data.csv', index=False)
        
        # 保存特征列表
        with open(f'{output_dir}/02_selected_features_list.txt', 'w', encoding='utf-8') as f:
            f.write("选中的特征列表\n")
            f.write("=" * 30 + "\n\n")
            for i, feat in enumerate(ensemble_features, 1):
                f.write(f"{i:2d}. {feat}\n")
        
        print(f" 选中特征数据已保存: {output_dir}/02_selected_features_data.csv")
        print(f" 特征列表已保存: {output_dir}/02_selected_features_list.txt")
        
        return selected_data
    
    def generate_feature_report(self, df, ensemble_features, all_selections, feature_votes, stability_metrics, output_dir='results'):
        """生成特征选择报告"""
        
        with open(f'{output_dir}/02_feature_selection_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("特征选择详细报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1. 数据基本信息\n")
            f.write("-" * 30 + "\n")
            f.write(f"原始特征数: {len(df.columns) - 1}\n")
            f.write(f"选中特征数: {len(ensemble_features)}\n")
            f.write(f"选择比例: {len(ensemble_features)/(len(df.columns)-1)*100:.1f}%\n\n")
            
            f.write("2. 各方法选择结果\n")
            f.write("-" * 30 + "\n")
            for method, features in all_selections.items():
                f.write(f"{method}: {len(features)} 个特征\n")
            f.write("\n")
            
            f.write("3. 最终选中特征 (按投票数排序)\n")
            f.write("-" * 30 + "\n")
            sorted_votes = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            for i, (feat, votes) in enumerate(sorted_votes, 1):
                if feat in ensemble_features:
                    f.write(f"{i:2d}. ✓ {feat}: {votes} 票\n")
            f.write("\n")
            
            f.write("4. 特征稳定性验证\n")
            f.write("-" * 30 + "\n")
            f.write(f"平均R²: {stability_metrics['mean_r2']:.4f}\n")
            f.write(f"标准差: {stability_metrics['std_r2']:.4f}\n")
            f.write(f"稳定性分数: {stability_metrics['stability_score']:.4f}\n")
            f.write("交叉验证分数: " + ", ".join([f"{score:.3f}" for score in stability_metrics['cv_scores']]) + "\n\n")
            
            f.write("5. 特征相关性分析\n")
            f.write("-" * 30 + "\n")
            target_col = 'lipid(%)'
            correlations = df[ensemble_features].corrwith(df[target_col]).abs().sort_values(ascending=False)
            for i, (feat, corr) in enumerate(correlations.items(), 1):
                f.write(f"{i:2d}. {feat}: {corr:.3f}\n")
        
        print(f" 特征选择报告已保存: {output_dir}/02_feature_selection_report.txt")
    
    def select_features(self, input_file='results/01_processed_data.csv', output_dir='results'):
        """完整特征选择流程"""
        print(" 开始高级特征选择...")
        print("=" * 60)
        
        # 1. 加载数据
        df = self.load_processed_data(input_file)
        if df is None:
            return None
        
        # 2. 集成特征选择
        ensemble_features, all_selections, feature_votes = self.ensemble_feature_selection(df)
        
        # 3. 特征稳定性验证
        stability_metrics = self.validate_feature_stability(df, ensemble_features)
        
        # 4. 创建可视化
        self.create_feature_analysis_visualization(df, ensemble_features, feature_votes, output_dir)
        
        # 5. 保存选中特征数据
        selected_data = self.save_selected_features(df, ensemble_features, output_dir=output_dir)
        
        # 6. 生成报告
        self.generate_feature_report(df, ensemble_features, all_selections, feature_votes, stability_metrics, output_dir)
        
        print("\n" + "=" * 60)
        print(" 特征选择完成!")
        print(f" 原始特征: {len(df.columns) - 1}")
        print(f" 选中特征: {len(ensemble_features)}")
        print(f" 稳定性R²: {stability_metrics['mean_r2']:.4f} ± {stability_metrics['std_r2']:.4f}")
        
        return selected_data, ensemble_features

def main():
    """主函数"""
    import os
    from pathlib import Path
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    
    selector = AdvancedFeatureSelector()
    
    # 执行特征选择
    selected_data, selected_features = selector.select_features(
        input_file=str(project_root / 'results' / 'preprocessing' / 'optimized_data.csv'),
        output_dir=str(project_root / 'results')
    )
    
    if selected_data is not None:
        print(f"\n 特征选择成功完成!")
        print(f" 结果文件:")
        print(f"   - 选中特征数据: results/02_selected_features_data.csv")
        print(f"   - 特征列表: results/02_selected_features_list.txt")
        print(f"   - 分析可视化: results/02_feature_selection_analysis.png")
        print(f"   - 详细报告: results/02_feature_selection_report.txt")

if __name__ == "__main__":
    main()