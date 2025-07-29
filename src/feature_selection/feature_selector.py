
"""
高级特征选择系统
结合多种特征选择方法，提供全面的特征筛选和分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    RFE, RFECV, SelectKBest, SelectPercentile,
    f_regression, mutual_info_regression, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedFeatureSelector:
    """高级特征选择器"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.feature_scores = {}
        self.selected_features = {}

    def load_data(self):
        """加载预处理后的数据"""
        print(" 加载预处理后的数据...")

        # 加载完整数据
        processed_data = pd.read_csv("../../data/processed/processed_data.csv")

        # 分离特征和目标
        self.y = processed_data['lipid(%)']
        self.X = processed_data.drop('lipid(%)', axis=1)

        print(f"   原始数据形状: {processed_data.shape}")
        print(f"   原始特征数量: {self.X.shape[1]}")
        print(f"   样本数量: {self.X.shape[0]}")
        print(f"   目标变量范围: {self.y.min():.3f} ~ {self.y.max():.3f}")

        # 创建衍生特征
        self.X = self.create_derived_features(self.X)

        print(f"   特征工程后特征数量: {self.X.shape[1]}")

        # 数据质量检查
        self.validate_data_quality()

        return self.X, self.y

    def validate_data_quality(self):
        """验证数据质量"""
        print("\n 数据质量验证...")

        # 检查无穷大和NaN
        inf_count = np.isinf(self.X).sum().sum()
        nan_count = np.isnan(self.X).sum().sum()

        if inf_count > 0:
            print(f"     发现{inf_count}个无穷大值，将替换为极大值")
            self.X = self.X.replace([np.inf, -np.inf], [1e6, -1e6])

        if nan_count > 0:
            print(f"     发现{nan_count}个NaN值，将用中位数填充")
            self.X = self.X.fillna(self.X.median())

        # 检查目标变量
        y_range = self.y.max() - self.y.min()
        y_std = self.y.std()
        print(f"   目标变量范围: {self.y.min():.3f} ~ {self.y.max():.3f}")
        print(f"   目标变量标准差: {y_std:.3f}")
        print(f"   目标变量变异系数: {y_std/self.y.mean():.3f}")

        # 检查特征方差
        low_var_features = []
        for col in self.X.columns:
            if self.X[col].var() < 1e-6:
                low_var_features.append(col)

        if low_var_features:
            print(f"     发现{len(low_var_features)}个近零方差特征: {low_var_features}")
            self.X = self.X.drop(columns=low_var_features)
            print(f"   移除后特征数: {self.X.shape[1]}")

        print(f"    数据质量验证完成")



    def remove_low_variance_features(self, threshold=0.01):
        """移除低方差特征"""
        print(f"\n 移除低方差特征 (阈值: {threshold})...")

        selector = VarianceThreshold(threshold=threshold)
        X_filtered = selector.fit_transform(self.X)

        # 获取保留的特征
        selected_mask = selector.get_support()
        selected_features = self.X.columns[selected_mask]
        removed_features = self.X.columns[~selected_mask]

        print(f"   移除特征数: {len(removed_features)}")
        print(f"   保留特征数: {len(selected_features)}")

        if len(removed_features) > 0:
            print(f"   移除的特征: {list(removed_features)}")

        self.results['variance_threshold'] = {
            'selected_features': selected_features,
            'removed_features': removed_features,
            'selector': selector
        }

        return selected_features

    def correlation_analysis(self, threshold=0.95):
        """相关性分析和高相关特征移除"""
        print(f"\n 相关性分析 (阈值: {threshold})...")

        # 计算相关性矩阵
        corr_matrix = self.X.corr().abs()

        # 找到高相关的特征对
        high_corr_pairs = []
        features_to_remove = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]

                    high_corr_pairs.append((feature1, feature2, corr_value))

                    # 选择移除与目标变量相关性较低的特征
                    corr_with_target1 = abs(self.X[feature1].corr(self.y))
                    corr_with_target2 = abs(self.X[feature2].corr(self.y))

                    if corr_with_target1 < corr_with_target2:
                        features_to_remove.add(feature1)
                    else:
                        features_to_remove.add(feature2)

        # 保留的特征
        selected_features = [f for f in self.X.columns if f not in features_to_remove]

        print(f"   发现高相关特征对: {len(high_corr_pairs)}")
        print(f"   移除特征数: {len(features_to_remove)}")
        print(f"   保留特征数: {len(selected_features)}")

        if high_corr_pairs:
            print(f"   高相关特征对:")
            for f1, f2, corr in high_corr_pairs[:5]:  # 只显示前5个
                print(f"     {f1} - {f2}: {corr:.3f}")

        self.results['correlation'] = {
            'selected_features': selected_features,
            'removed_features': list(features_to_remove),
            'high_corr_pairs': high_corr_pairs,
            'corr_matrix': corr_matrix
        }

        return selected_features

    def univariate_selection(self, k=15):
        """单变量特征选择 - 基于统计显著性"""
        print(f"\n 单变量特征选择 (选择前{k}个)...")

        # 计算每个特征与目标的相关性
        correlations = {}
        p_values = {}

        for col in self.X.columns:
            from scipy.stats import pearsonr
            corr, p_val = pearsonr(self.X[col], self.y)
            correlations[col] = abs(corr)  # 使用绝对值
            p_values[col] = p_val

        # 选择显著相关的特征 (p < 0.1)
        significant_features = [col for col, p in p_values.items() if p < 0.1]
        print(f"   统计显著特征数 (p<0.1): {len(significant_features)}")

        # F-regression (只在显著特征中选择)
        if len(significant_features) > 0:
            X_significant = self.X[significant_features]
            k_adjusted = min(k, len(significant_features))

            f_selector = SelectKBest(score_func=f_regression, k=k_adjusted)
            f_selector.fit(X_significant, self.y)
            f_selected = X_significant.columns[f_selector.get_support()]
            f_scores = f_selector.scores_
        else:
            # 如果没有显著特征，使用相关性最高的
            sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            f_selected = [item[0] for item in sorted_corr[:k]]
            f_scores = [correlations[col] for col in f_selected]

        # Mutual Information (更保守的选择)
        mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(self.X.columns)//2))
        mi_selector.fit(self.X, self.y)
        mi_selected = self.X.columns[mi_selector.get_support()]
        mi_scores = mi_selector.scores_

        print(f"   F-regression选择的特征数: {len(f_selected)}")
        print(f"   互信息选择的特征数: {len(mi_selected)}")

        # 保存结果
        self.results['f_regression'] = {
            'selected_features': f_selected,
            'scores': dict(zip(self.X.columns, f_scores)) if isinstance(f_scores, (list, np.ndarray)) else correlations,
            'correlations': correlations,
            'p_values': p_values,
            'significant_features': significant_features
        }

        self.results['mutual_info'] = {
            'selected_features': mi_selected,
            'scores': dict(zip(self.X.columns, mi_scores)),
            'selector': mi_selector
        }

        return f_selected, mi_selected

    def tree_based_selection(self, n_features=15):
        """基于树模型的特征选择"""
        print(f"\n 基于树模型的特征选择 (选择前{n_features}个)...")

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        rf.fit(self.X, self.y)
        rf_importances = rf.feature_importances_

        # Extra Trees
        et = ExtraTreesRegressor(n_estimators=100, random_state=self.random_state)
        et.fit(self.X, self.y)
        et_importances = et.feature_importances_

        # 选择重要性最高的特征
        rf_indices = np.argsort(rf_importances)[::-1][:n_features]
        et_indices = np.argsort(et_importances)[::-1][:n_features]

        rf_selected = self.X.columns[rf_indices]
        et_selected = self.X.columns[et_indices]

        print(f"   Random Forest选择的特征数: {len(rf_selected)}")
        print(f"   Extra Trees选择的特征数: {len(et_selected)}")

        # 保存结果
        self.results['random_forest'] = {
            'selected_features': rf_selected,
            'importances': dict(zip(self.X.columns, rf_importances)),
            'model': rf
        }

        self.results['extra_trees'] = {
            'selected_features': et_selected,
            'importances': dict(zip(self.X.columns, et_importances)),
            'model': et
        }

        return rf_selected, et_selected

    def lasso_selection(self, alpha_range=None):
        """基于Lasso回归的特征选择"""
        print(f"\n 基于Lasso回归的特征选择...")

        # 针对小样本调整alpha范围
        if alpha_range is None:
            alpha_range = np.logspace(-6, -1, 50)  # 更小的alpha范围，避免过度正则化

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # 针对小样本调整CV折数
        n_samples = len(self.y)
        cv_folds = min(5, max(3, n_samples // 5))

        # Lasso CV
        lasso_cv = LassoCV(alphas=alpha_range, cv=cv_folds, random_state=self.random_state)
        lasso_cv.fit(X_scaled, self.y)

        # 获取非零系数的特征
        selected_mask = np.abs(lasso_cv.coef_) > 1e-6  # 使用更小的阈值
        selected_features = self.X.columns[selected_mask]

        print(f"   最优alpha: {lasso_cv.alpha_:.6f}")
        print(f"   选择的特征数: {len(selected_features)}")
        print(f"   使用{cv_folds}折交叉验证")

        # 如果没有选择任何特征，尝试更小的alpha
        if len(selected_features) == 0:
            print(f"   未选择任何特征，尝试更小的alpha...")
            smaller_alphas = np.logspace(-8, -3, 30)
            lasso_cv2 = LassoCV(alphas=smaller_alphas, cv=cv_folds, random_state=self.random_state)
            lasso_cv2.fit(X_scaled, self.y)

            selected_mask2 = np.abs(lasso_cv2.coef_) > 1e-8
            selected_features = self.X.columns[selected_mask2]

            print(f"   重新选择 - alpha: {lasso_cv2.alpha_:.8f}, 特征数: {len(selected_features)}")

            # 使用新的结果
            lasso_cv = lasso_cv2

        # 保存结果
        self.results['lasso'] = {
            'selected_features': selected_features,
            'coefficients': dict(zip(self.X.columns, lasso_cv.coef_)),
            'alpha': lasso_cv.alpha_,
            'model': lasso_cv,
            'scaler': scaler
        }

        return selected_features

    def rfe_selection(self, n_features=15):
        """递归特征消除"""
        print(f"\n 递归特征消除 (选择{n_features}个特征)...")

        # 使用Random Forest作为基础估计器
        estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state)

        # RFE
        rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
        rfe.fit(self.X, self.y)

        selected_features = self.X.columns[rfe.support_]

        print(f"   选择的特征数: {len(selected_features)}")

        # 保存结果
        self.results['rfe'] = {
            'selected_features': selected_features,
            'ranking': dict(zip(self.X.columns, rfe.ranking_)),
            'selector': rfe
        }

        return selected_features

    def rfecv_selection(self, cv=5):
        """带交叉验证的递归特征消除"""
        print(f"\n RFECV特征选择 (CV={cv})...")

        # 使用Random Forest作为基础估计器
        estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state)

        # RFECV
        rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring='r2')
        rfecv.fit(self.X, self.y)

        selected_features = self.X.columns[rfecv.support_]

        print(f"   最优特征数: {rfecv.n_features_}")
        print(f"   选择的特征数: {len(selected_features)}")

        # 保存结果
        self.results['rfecv'] = {
            'selected_features': selected_features,
            'ranking': dict(zip(self.X.columns, rfecv.ranking_)),
            'cv_scores': rfecv.cv_results_,
            'optimal_n_features': rfecv.n_features_,
            'selector': rfecv
        }

        return selected_features

    def ensemble_selection(self, methods=None, min_votes=3):
        """集成多种方法的特征选择结果"""
        print(f"\n 集成特征选择 (最少{min_votes}票)...")

        if methods is None:
            methods = ['f_regression', 'mutual_info', 'random_forest', 'extra_trees', 'lasso', 'rfe']

        # 统计每个特征被选择的次数
        feature_votes = {}
        for feature in self.X.columns:
            feature_votes[feature] = 0

        # 计算投票
        for method in methods:
            if method in self.results:
                selected = self.results[method]['selected_features']
                for feature in selected:
                    feature_votes[feature] += 1

        # 选择得票数 >= min_votes 的特征
        ensemble_features = [f for f, votes in feature_votes.items() if votes >= min_votes]

        print(f"   参与投票的方法: {len(methods)}")
        print(f"   集成选择的特征数: {len(ensemble_features)}")

        # 按得票数排序
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)

        # 保存结果
        self.results['ensemble'] = {
            'selected_features': ensemble_features,
            'feature_votes': feature_votes,
            'sorted_features': sorted_features,
            'min_votes': min_votes
        }

        return ensemble_features

    def evaluate_feature_sets(self, feature_sets=None):
        """评估不同特征集的性能"""
        print(f"\n 评估不同特征集的性能...")

        if feature_sets is None:
            feature_sets = {
                'all_features': list(self.X.columns),
                'f_regression': self.results.get('f_regression', {}).get('selected_features', []),
                'random_forest': self.results.get('random_forest', {}).get('selected_features', []),
                'lasso': self.results.get('lasso', {}).get('selected_features', []),
                'rfe': self.results.get('rfe', {}).get('selected_features', []),
                'rfecv': self.results.get('rfecv', {}).get('selected_features', []),
                'ensemble': self.results.get('ensemble', {}).get('selected_features', [])
            }

        # 评估每个特征集
        evaluation_results = {}

        # 针对小样本数据，使用Leave-One-Out交叉验证
        n_samples = len(self.y)

        if n_samples <= 50:
            cv_strategy = LeaveOneOut()
            cv_name = "Leave-One-Out"
        else:
            cv_strategy = 5
            cv_name = "5-fold"

        print(f"   使用{cv_name}交叉验证 (样本数: {n_samples})")

        for name, features in feature_sets.items():
            if len(features) == 0:
                print(f"   {name}: 跳过 (无特征)")
                continue

            # 检查特征数量是否合理
            if len(features) >= n_samples:
                print(f"   {name}: 跳过 (特征数{len(features)} >= 样本数{n_samples})")
                continue

            try:
                # 使用交叉验证评估
                X_subset = self.X[features]

                # 针对小样本使用更简单的模型
                # 使用岭回归（更适合小样本）
                model = Ridge(alpha=1.0, random_state=self.random_state)

                # 交叉验证评估 - 添加异常处理
                try:
                    cv_scores = cross_val_score(model, X_subset, self.y, cv=cv_strategy, scoring='r2')

                    # 检查是否有NaN值
                    if np.isnan(cv_scores).any():
                        print(f"     Ridge回归产生NaN，尝试简单线性回归...")
                        from sklearn.linear_model import LinearRegression
                        model_lr = LinearRegression()
                        cv_scores = cross_val_score(model_lr, X_subset, self.y, cv=cv_strategy, scoring='r2')
                        model_name = "LinearRegression"

                        # 如果还是NaN，使用3折交叉验证
                        if np.isnan(cv_scores).any():
                            print(f"     LOO产生NaN，改用3折交叉验证...")
                            cv_scores = cross_val_score(model_lr, X_subset, self.y, cv=3, scoring='r2')
                    else:
                        model_name = "Ridge"

                except Exception as e:
                    print(f"     交叉验证失败: {e}，使用简单训练测试分割...")
                    # 简单的训练测试分割
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_subset, self.y, test_size=0.3, random_state=self.random_state
                    )
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    cv_scores = np.array([score])
                    model_name = "Ridge(split)"

                best_score = {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores
                }

                evaluation_results[name] = {
                    'n_features': len(features),
                    'cv_mean': best_score['mean'],
                    'cv_std': best_score['std'],
                    'model': model_name,
                    'features': features
                }

                print(f"   {name}: {len(features)}个特征, R²={best_score['mean']:.4f}±{best_score['std']:.4f} ({model_name})")

            except Exception as e:
                print(f"   {name}: 评估失败 - {e}")
                continue

        # 保存评估结果
        self.results['evaluation'] = evaluation_results

        return evaluation_results

    def visualize_results(self):
        """可视化特征选择结果"""
        print(f"\n 生成可视化图表...")

        # 1. 特征重要性对比图
        self._plot_feature_importance()

        # 2. 特征选择方法对比图
        self._plot_method_comparison()

        # 3. 特征投票图
        self._plot_feature_votes()

        # 4. 性能评估图
        self._plot_performance_comparison()

        print(f"   可视化图表已保存到 results/feature_selection/")

    def _plot_feature_importance(self):
        """绘制特征重要性图"""
        if 'random_forest' not in self.results:
            return

        importances = self.results['random_forest']['importances']

        # 排序
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features[:15])  # 只显示前15个

        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance (Top 15)')
        plt.gca().invert_yaxis()

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')

        plt.tight_layout()
        plt.savefig("../../results/feature_selection/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_method_comparison(self):
        """绘制不同方法选择的特征数量对比"""
        methods = []
        n_features = []

        for method, result in self.results.items():
            if 'selected_features' in result:
                methods.append(method)
                n_features.append(len(result['selected_features']))

        if not methods:
            return

        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, n_features, alpha=0.7)
        plt.xlabel('Feature Selection Method')
        plt.ylabel('Number of Selected Features')
        plt.title('Feature Selection Methods Comparison')
        plt.xticks(rotation=45)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig("../../results/feature_selection/method_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_votes(self):
        """绘制特征投票图"""
        if 'ensemble' not in self.results:
            return

        feature_votes = self.results['ensemble']['feature_votes']
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)

        # 只显示有投票的特征
        voted_features = [(f, v) for f, v in sorted_features if v > 0]

        if not voted_features:
            return

        features, votes = zip(*voted_features)

        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), votes)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Number of Votes')
        plt.title('Feature Selection Votes Across Methods')
        plt.gca().invert_yaxis()

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', ha='left', va='center')

        plt.tight_layout()
        plt.savefig("../../results/feature_selection/feature_votes.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_comparison(self):
        """绘制性能对比图"""
        if 'evaluation' not in self.results:
            return

        evaluation = self.results['evaluation']

        methods = []
        scores = []
        errors = []
        n_features = []

        for method, result in evaluation.items():
            methods.append(method)
            scores.append(result['cv_mean'])
            errors.append(result['cv_std'])
            n_features.append(result['n_features'])

        if not methods:
            return

        # 性能对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # R²得分
        bars1 = ax1.bar(methods, scores, yerr=errors, capsize=5, alpha=0.7)
        ax1.set_xlabel('Feature Set')
        ax1.set_ylabel('Cross-Validation R² Score')
        ax1.set_title('Performance Comparison')
        ax1.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        # 特征数量 vs 性能
        ax2.scatter(n_features, scores, s=100, alpha=0.7)
        for i, method in enumerate(methods):
            ax2.annotate(method, (n_features[i], scores[i]),
                        xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cross-Validation R² Score')
        ax2.set_title('Features vs Performance')

        plt.tight_layout()
        plt.savefig("../../results/feature_selection/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self):
        """保存特征选择结果"""
        print(f"\n 保存特征选择结果...")

        # 保存所有方法的选择结果
        summary = {}
        for method, result in self.results.items():
            if 'selected_features' in result:
                summary[method] = {
                    'n_features': len(result['selected_features']),
                    'features': list(result['selected_features'])
                }

        # 保存摘要
        summary_df = pd.DataFrame([
            {
                'method': method,
                'n_features': info['n_features'],
                'features': ', '.join(info['features'])
            }
            for method, info in summary.items()
        ])
        summary_df.to_csv("../../results/feature_selection/selection_summary.csv", index=False)

        # 保存最佳特征集（基于性能评估）
        if 'evaluation' in self.results and len(self.results['evaluation']) > 0:
            try:
                best_method = max(self.results['evaluation'].items(),
                                key=lambda x: x[1]['cv_mean'])
                best_features = best_method[1]['features']

                # 保存最佳特征的数据
                X_best = self.X[best_features]
                X_best['lipid(%)'] = self.y  # 添加目标变量
                X_best.to_csv("../../results/feature_selection/best_features_data.csv", index=False, float_format='%.6f')

                # 保存最佳特征列表
                best_features_df = pd.DataFrame({'feature': best_features})
                best_features_df.to_csv("../../results/feature_selection/best_features_list.csv", index=False)

                print(f"   最佳方法: {best_method[0]} (R²={best_method[1]['cv_mean']:.4f})")
                print(f"   最佳特征数: {len(best_features)}")
            except Exception as e:
                print(f"   ️  无法确定最佳特征集: {e}")
        else:
            print(f"     性能评估失败，使用RFE结果作为默认最佳特征集")
            if 'rfe' in self.results:
                rfe_features = self.results['rfe']['selected_features']
                X_rfe = self.X[rfe_features]
                X_rfe['lipid(%)'] = self.y
                X_rfe.to_csv("../../results/feature_selection/best_features_data.csv", index=False, float_format='%.6f')

                rfe_df = pd.DataFrame({'feature': rfe_features})
                rfe_df.to_csv("../../results/feature_selection/best_features_list.csv", index=False)
                print(f"   使用RFE特征集: {len(rfe_features)}个特征")

        # 保存集成特征（如果存在）
        if 'ensemble' in self.results:
            ensemble_features = self.results['ensemble']['selected_features']
            if ensemble_features:
                X_ensemble = self.X[ensemble_features]
                X_ensemble['lipid(%)'] = self.y
                X_ensemble.to_csv("../../results/feature_selection/ensemble_features_data.csv", index=False, float_format='%.6f')

                ensemble_df = pd.DataFrame({'feature': ensemble_features})
                ensemble_df.to_csv("../../results/feature_selection/ensemble_features_list.csv", index=False)

        print(f"   结果已保存到 results/feature_selection/")

    def run_complete_analysis(self, n_features=10, min_votes=2):
        """运行完整的特征选择分析 - 针对微藻脂质含量预测优化"""
        print(" 微藻脂质含量预测 - 特征选择分析")
        print("="*80)
        print("项目目标: 基于环境和生物参数预测微藻脂质含量")
        print("数据特点: 小样本高维数据，需要谨慎的特征选择策略")

        # 1. 加载数据
        self.load_data()

        # 检查数据合理性
        n_samples, n_features_orig = self.X.shape
        print(f"\n数据检查:")
        print(f"   样本数/特征数比例: {n_samples}/{n_features_orig} = {n_samples/n_features_orig:.2f}")
        if n_samples/n_features_orig < 5:
            print(f"   ️  警告: 样本数相对特征数较少，容易过拟合")
            n_features = min(n_features, n_samples//3)  # 调整特征数
            print(f"   调整目标特征数为: {n_features}")

        # 2. 基础过滤
        variance_features = self.remove_low_variance_features(threshold=0.001)  # 更小的阈值
        correlation_features = self.correlation_analysis(threshold=0.90)  # 稍微宽松的相关性阈值

        # 3. 单变量选择 - 针对回归问题优化
        f_features, mi_features = self.univariate_selection(k=n_features)

        # 4. 基于树的选择 - 使用更保守的参数
        rf_features, et_features = self.tree_based_selection(n_features=n_features)

        # 5. 正则化选择 - 针对小样本优化
        lasso_features = self.lasso_selection()

        # 6. 递归特征消除 - 使用更少的特征
        rfe_features = self.rfe_selection(n_features=n_features)
        rfecv_features = self.rfecv_selection()

        # 7. 集成选择 - 降低投票阈值
        ensemble_features = self.ensemble_selection(min_votes=min_votes)

        # 8. 性能评估
        evaluation_results = self.evaluate_feature_sets()

        # 9. 可视化
        self.visualize_results()

        # 10. 保存结果
        self.save_results()

        # 11. 总结报告
        self.print_summary()

        print("\n 微藻脂质含量预测特征选择分析完成！")
        return self.results

    def print_summary(self):
        """打印总结报告"""
        print("\n" + "="*80)
        print("特征选择总结报告")
        print("="*80)

        print(f"\n 原始数据:")
        print(f"   - 总特征数: {self.X.shape[1]}")
        print(f"   - 样本数: {self.X.shape[0]}")

        print(f"\n 特征选择方法结果:")
        for method, result in self.results.items():
            if 'selected_features' in result:
                n_features = len(result['selected_features'])
                print(f"   - {method}: {n_features}个特征")

        if 'evaluation' in self.results:
            print(f"\n 性能评估 (5折交叉验证 R²):")
            evaluation = self.results['evaluation']
            sorted_eval = sorted(evaluation.items(), key=lambda x: x[1]['cv_mean'], reverse=True)

            for method, result in sorted_eval:
                print(f"   - {method}: {result['cv_mean']:.4f}±{result['cv_std']:.4f} ({result['n_features']}个特征)")

        if 'ensemble' in self.results:
            ensemble_features = self.results['ensemble']['selected_features']
            print(f"\n 集成选择:")
            print(f"   - 集成特征数: {len(ensemble_features)}")
            print(f"   - 最少投票数: {self.results['ensemble']['min_votes']}")

            if ensemble_features:
                print(f"   - 集成特征: {', '.join(ensemble_features)}")

        print(f"\n 输出文件:")
        print(f"   - selection_summary.csv: 所有方法的选择结果")
        print(f"   - best_features_data.csv: 最佳特征集数据")
        print(f"   - ensemble_features_data.csv: 集成特征数据")
        print(f"   - 可视化图表: feature_importance.png, method_comparison.png 等")

if __name__ == "__main__":
    # 创建高级特征选择器
    selector = AdvancedFeatureSelector(random_state=42)

    # 运行完整分析 - 针对微藻脂质含量预测优化
    results = selector.run_complete_analysis(
        n_features=4,   # 极度保守的特征数，确保样本/特征比例 > 9:1
        min_votes=2     # 降低投票阈值，增加特征多样性
    )
