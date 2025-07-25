# -*- coding: utf-8 -*-
"""
修复版优化模型训练模块
Fixed Optimized Model Training for Small Sample Algae Lipid Prediction

修复LOOCV中的nan问题，增强错误处理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import (
    cross_val_score, LeaveOneOut, KFold, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
import os
from typing import Dict, List, Tuple
import joblib
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FixedSmallSampleOptimizer:
    """修复版小样本专用模型优化器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.cv_results = {}
        self.bootstrap_results = {}
        
    def create_robust_models(self):
        """创建更稳健的模型"""
        print("🤖 创建稳健的小样本模型...")
        
        models = {
            # 线性模型 - 更保守的正则化
            'Ridge_Conservative': Ridge(alpha=5.0, random_state=self.random_state),
            'Ridge_Moderate': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso_Conservative': Lasso(alpha=0.5, random_state=self.random_state, max_iter=3000),
            'Lasso_Moderate': Lasso(alpha=0.1, random_state=self.random_state, max_iter=3000),
            'ElasticNet_Conservative': ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=self.random_state, max_iter=3000),
            
            # 树模型 - 极度保守
            'RF_Ultra_Conservative': RandomForestRegressor(
                n_estimators=50, max_depth=2, min_samples_split=8, 
                min_samples_leaf=4, random_state=self.random_state, n_jobs=1
            ),
            'RF_Conservative': RandomForestRegressor(
                n_estimators=30, max_depth=3, min_samples_split=6, 
                min_samples_leaf=3, random_state=self.random_state, n_jobs=1
            ),
            
            # 梯度提升 - 极度保守
            'GBM_Ultra_Conservative': GradientBoostingRegressor(
                n_estimators=20, max_depth=2, learning_rate=0.01,
                subsample=0.8, random_state=self.random_state
            )
        }
        
        self.models = models
        print(f"创建了 {len(models)} 个稳健模型")
        return models
    
    def safe_loocv_evaluation(self, model, X, y, model_name):
        """安全的LOOCV评估"""
        print(f"   安全LOOCV评估 {model_name}...")
        
        loo = LeaveOneOut()
        cv_scores = []
        failed_folds = 0
        
        for i, (train_idx, test_idx) in enumerate(loo.split(X)):
            try:
                X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
                y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                y_test = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
                
                # 检查训练数据
                if len(np.unique(y_train)) < 2:
                    print(f"    警告: 折{i+1}训练数据方差过小")
                    failed_folds += 1
                    continue
                
                # 创建模型副本
                model_copy = type(model)(**model.get_params())
                
                # 训练模型
                model_copy.fit(X_train, y_train)
                
                # 预测
                y_pred = model_copy.predict(X_test)
                
                # 计算R²
                if len(y_test) == 1:
                    # 单样本情况，使用简化计算
                    y_mean = y_train.mean()
                    ss_tot = ((y_test - y_mean) ** 2).sum()
                    ss_res = ((y_test - y_pred) ** 2).sum()
                    
                    if ss_tot == 0:
                        r2 = 0.0  # 如果方差为0，设为0
                    else:
                        r2 = 1 - (ss_res / ss_tot)
                else:
                    r2 = r2_score(y_test, y_pred)
                
                # 检查结果有效性
                if np.isfinite(r2):
                    cv_scores.append(r2)
                else:
                    failed_folds += 1
                    
            except Exception as e:
                print(f"    折{i+1}失败: {e}")
                failed_folds += 1
                continue
        
        if len(cv_scores) == 0:
            print(f"     所有LOOCV折都失败")
            return np.nan, np.nan
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        print(f"    成功折数: {len(cv_scores)}/{len(X)}, 失败: {failed_folds}")
        print(f"    LOOCV R²: {cv_mean:.4f} (±{cv_std:.4f})")
        
        return cv_mean, cv_std
    
    def robust_bootstrap_validation(self, model, X, y, n_bootstrap=500):
        """稳健的Bootstrap验证"""
        print(f"     稳健Bootstrap验证 (n={n_bootstrap})...")
        
        n_samples = len(X)
        bootstrap_scores = []
        failed_bootstraps = 0
        
        for i in range(n_bootstrap):
            try:
                # Bootstrap采样
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_boot = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
                y_boot = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
                
                # 检查Bootstrap样本的多样性
                if len(np.unique(y_boot)) < 2:
                    failed_bootstraps += 1
                    continue
                
                # 训练模型
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_boot, y_boot)
                
                # 在原始数据上评估
                y_pred = model_copy.predict(X)
                score = r2_score(y, y_pred)
                
                if np.isfinite(score):
                    bootstrap_scores.append(score)
                else:
                    failed_bootstraps += 1
                    
            except Exception as e:
                failed_bootstraps += 1
                continue
        
        if len(bootstrap_scores) == 0:
            print(f"     所有Bootstrap都失败")
            return {
                'mean': np.nan, 'std': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan,
                'scores': []
            }
        
        bootstrap_scores = np.array(bootstrap_scores)
        mean_score = bootstrap_scores.mean()
        std_score = bootstrap_scores.std()
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        
        print(f"    成功Bootstrap: {len(bootstrap_scores)}/{n_bootstrap}, 失败: {failed_bootstraps}")
        print(f"    Bootstrap R²: {mean_score:.4f} (±{std_score:.4f})")
        print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return {
            'mean': mean_score,
            'std': std_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'scores': bootstrap_scores
        }
    
    def train_and_evaluate(self, X, y):
        """训练和评估所有模型"""
        print("\n🏋️ 开始稳健小样本训练...")
        print(f"训练数据: {X.shape}, 目标变量范围: [{y.min():.2f}, {y.max():.2f}]")
        
        # 数据质量检查
        print(f"\n 数据质量检查:")
        print(f"   特征数量: {X.shape[1]}")
        print(f"   样本数量: {X.shape[0]}")
        print(f"   特征/样本比: {X.shape[1]/X.shape[0]:.3f}")
        print(f"   目标变量方差: {y.var():.4f}")
        print(f"   目标变量标准差: {y.std():.4f}")
        
        if y.var() < 1e-10:
            print("    警告: 目标变量方差极小")
        
        models = self.create_robust_models()
        results = {}
        
        for model_name, model in models.items():
            print(f"\n 训练 {model_name}...")
            
            try:
                # 训练完整模型
                model.fit(X, y)
                y_pred = model.predict(X)
                
                # 计算训练指标
                train_r2 = r2_score(y, y_pred)
                train_rmse = np.sqrt(mean_squared_error(y, y_pred))
                train_mae = mean_absolute_error(y, y_pred)
                
                print(f"   训练R²: {train_r2:.4f}")
                
                # 安全LOOCV评估
                cv_score, cv_std = self.safe_loocv_evaluation(model, X, y, model_name)
                
                # 稳健Bootstrap验证
                bootstrap_result = self.robust_bootstrap_validation(model, X, y, n_bootstrap=300)
                
                results[model_name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'train_rmse': train_rmse,
                    'train_mae': train_mae,
                    'loocv_r2': cv_score,
                    'loocv_std': cv_std,
                    'bootstrap': bootstrap_result
                }
                
                self.best_models[model_name] = model
                self.cv_results[model_name] = cv_score
                self.bootstrap_results[model_name] = bootstrap_result
                
                if not np.isnan(cv_score):
                    print(f"   LOOCV R²: {cv_score:.4f} (±{cv_std:.4f})")
                else:
                    print(f"   LOOCV失败，使用Bootstrap结果")
                
            except Exception as e:
                print(f"   {model_name} 训练失败: {e}")
                continue
        
        return results
    
    def create_simple_ensemble(self, X, y, results, top_n=3):
        """创建简单集成模型"""
        print(f"\n 创建简单集成模型 (Top {top_n})...")
        
        # 按Bootstrap性能排序（因为LOOCV可能有nan）
        valid_results = {name: result for name, result in results.items() 
                        if not np.isnan(result['bootstrap']['mean'])}
        
        if len(valid_results) == 0:
            print(" 没有有效的模型用于集成")
            return None
        
        sorted_models = sorted(valid_results.items(), 
                             key=lambda x: x[1]['bootstrap']['mean'], reverse=True)
        top_models = sorted_models[:min(top_n, len(sorted_models))]
        
        print("选择的模型 (按Bootstrap性能):")
        for i, (name, result) in enumerate(top_models, 1):
            print(f"  {i}. {name}: {result['bootstrap']['mean']:.4f}")
        
        # 简单平均集成
        ensemble_predictions = []
        for name, _ in top_models:
            model = results[name]['model']
            pred = model.predict(X)
            ensemble_predictions.append(pred)
        
        ensemble_pred = np.mean(ensemble_predictions, axis=0)
        ensemble_r2 = r2_score(y, ensemble_pred)
        
        # Bootstrap评估集成模型
        ensemble_bootstrap = self.evaluate_ensemble_bootstrap(X, y, top_models, results)
        
        print(f"集成模型性能:")
        print(f"  训练R²: {ensemble_r2:.4f}")
        print(f"  Bootstrap R²: {ensemble_bootstrap['mean']:.4f} (±{ensemble_bootstrap['std']:.4f})")
        
        return {
            'models': [results[name]['model'] for name, _ in top_models],
            'model_names': [name for name, _ in top_models],
            'weights': [1/len(top_models)] * len(top_models),
            'train_r2': ensemble_r2,
            'bootstrap': ensemble_bootstrap
        }
    
    def evaluate_ensemble_bootstrap(self, X, y, top_models, results, n_bootstrap=300):
        """评估集成模型的Bootstrap性能"""
        n_samples = len(X)
        bootstrap_scores = []
        
        for i in range(n_bootstrap):
            try:
                # Bootstrap采样
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_boot = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
                y_boot = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
                
                # 训练每个模型并预测
                predictions = []
                for name, _ in top_models:
                    model_type = type(results[name]['model'])
                    model_params = results[name]['model'].get_params()
                    temp_model = model_type(**model_params)
                    temp_model.fit(X_boot, y_boot)
                    pred = temp_model.predict(X)
                    predictions.append(pred)
                
                # 集成预测
                ensemble_pred = np.mean(predictions, axis=0)
                score = r2_score(y, ensemble_pred)
                
                if np.isfinite(score):
                    bootstrap_scores.append(score)
                    
            except:
                continue
        
        if len(bootstrap_scores) == 0:
            return {'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
        
        bootstrap_scores = np.array(bootstrap_scores)
        return {
            'mean': bootstrap_scores.mean(),
            'std': bootstrap_scores.std(),
            'ci_lower': np.percentile(bootstrap_scores, 2.5),
            'ci_upper': np.percentile(bootstrap_scores, 97.5)
        }
    
    def create_performance_visualization(self, results, ensemble_result=None, output_dir='results/modeling'):
        """创建性能可视化"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('修复版小样本模型性能', fontsize=16, weight='bold')
        
        # 准备数据
        model_names = list(results.keys())
        train_r2s = [results[name]['train_r2'] for name in model_names]
        bootstrap_means = [results[name]['bootstrap']['mean'] for name in model_names]
        bootstrap_stds = [results[name]['bootstrap']['std'] for name in model_names]
        
        # 1. 训练 vs Bootstrap R²
        x_pos = np.arange(len(model_names))
        axes[0, 0].bar(x_pos - 0.2, train_r2s, 0.4, label='训练R²', alpha=0.7, color='blue')
        axes[0, 0].bar(x_pos + 0.2, bootstrap_means, 0.4, label='Bootstrap R²', alpha=0.7, color='red')
        axes[0, 0].set_xlabel('模型')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].set_title('训练 vs Bootstrap 性能')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([name.split('_')[0] for name in model_names], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='目标(0.8)')
        
        # 2. Bootstrap性能排序
        valid_indices = [i for i, val in enumerate(bootstrap_means) if not np.isnan(val)]
        if valid_indices:
            valid_names = [model_names[i] for i in valid_indices]
            valid_means = [bootstrap_means[i] for i in valid_indices]
            valid_stds = [bootstrap_stds[i] for i in valid_indices]
            
            sorted_data = sorted(zip(valid_names, valid_means, valid_stds), 
                               key=lambda x: x[1], reverse=True)
            sorted_names, sorted_means, sorted_stds = zip(*sorted_data)
            
            axes[0, 1].barh(range(len(sorted_names)), sorted_means, 
                           xerr=sorted_stds, alpha=0.7, color='green')
            axes[0, 1].set_yticks(range(len(sorted_names)))
            axes[0, 1].set_yticklabels([name.split('_')[0] for name in sorted_names])
            axes[0, 1].set_xlabel('Bootstrap R²')
            axes[0, 1].set_title('模型性能排序')
            axes[0, 1].axvline(x=0.8, color='red', linestyle='--', alpha=0.7)
        
        # 3. 过拟合分析
        overfitting = [train_r2s[i] - bootstrap_means[i] if not np.isnan(bootstrap_means[i]) else np.nan 
                      for i in range(len(model_names))]
        valid_overfitting = [of for of in overfitting if not np.isnan(of)]
        valid_names_of = [model_names[i] for i, of in enumerate(overfitting) if not np.isnan(of)]
        
        if valid_overfitting:
            colors = ['red' if of > 0.2 else 'orange' if of > 0.1 else 'green' for of in valid_overfitting]
            axes[1, 0].bar(range(len(valid_names_of)), valid_overfitting, color=colors, alpha=0.7)
            axes[1, 0].set_xticks(range(len(valid_names_of)))
            axes[1, 0].set_xticklabels([name.split('_')[0] for name in valid_names_of], rotation=45)
            axes[1, 0].set_ylabel('训练R² - Bootstrap R²')
            axes[1, 0].set_title('过拟合分析')
            axes[1, 0].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='警告线')
            axes[1, 0].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='危险线')
            axes[1, 0].legend()
        
        # 4. 性能摘要
        summary_text = "性能摘要:\n\n"
        if valid_indices:
            for i, (name, mean, std) in enumerate(sorted_data[:5], 1):
                train_r2 = results[name]['train_r2']
                summary_text += f"{i}. {name.split('_')[0]}:\n"
                summary_text += f"   训练R²: {train_r2:.3f}\n"
                summary_text += f"   Bootstrap R²: {mean:.3f}±{std:.3f}\n\n"
        
        if ensemble_result:
            summary_text += f"集成模型:\n"
            summary_text += f"   训练R²: {ensemble_result['train_r2']:.3f}\n"
            summary_text += f"   Bootstrap R²: {ensemble_result['bootstrap']['mean']:.3f}\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Top 5 模型')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fixed_optimized_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" 修复版性能可视化已保存: {output_dir}/fixed_optimized_performance.png")
    
    def save_detailed_results(self, results, ensemble_result=None, output_dir='results/modeling'):
        """保存详细结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f'{output_dir}/fixed_optimized_results.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("修复版小样本优化模型结果\n")
            f.write("=" * 60 + "\n\n")
            
            # 按Bootstrap性能排序
            valid_results = {name: result for name, result in results.items() 
                           if not np.isnan(result['bootstrap']['mean'])}
            
            if valid_results:
                sorted_results = sorted(valid_results.items(), 
                                      key=lambda x: x[1]['bootstrap']['mean'], reverse=True)
                
                f.write("模型性能排序 (按Bootstrap R²):\n")
                f.write("-" * 40 + "\n")
                for i, (name, result) in enumerate(sorted_results, 1):
                    f.write(f"{i:2d}. {name}\n")
                    f.write(f"    训练R²: {result['train_r2']:.4f}\n")
                    if not np.isnan(result['loocv_r2']):
                        f.write(f"    LOOCV R²: {result['loocv_r2']:.4f} (±{result['loocv_std']:.4f})\n")
                    else:
                        f.write(f"    LOOCV R²: 失败\n")
                    f.write(f"    Bootstrap R²: {result['bootstrap']['mean']:.4f} ")
                    f.write(f"[{result['bootstrap']['ci_lower']:.4f}, {result['bootstrap']['ci_upper']:.4f}]\n")
                    overfitting = result['train_r2'] - result['bootstrap']['mean']
                    f.write(f"    过拟合程度: {overfitting:.4f}\n\n")
            
            if ensemble_result:
                f.write("\n集成模型结果:\n")
                f.write("-" * 40 + "\n")
                f.write(f"组成模型: {', '.join(ensemble_result['model_names'])}\n")
                f.write(f"训练R²: {ensemble_result['train_r2']:.4f}\n")
                f.write(f"Bootstrap R²: {ensemble_result['bootstrap']['mean']:.4f} ")
                f.write(f"(±{ensemble_result['bootstrap']['std']:.4f})\n\n")
        
        print(f" 修复版详细结果已保存: {output_dir}/fixed_optimized_results.txt")
        
        # 保存最佳模型
        if valid_results:
            best_model_name = sorted_results[0][0]
            best_model = results[best_model_name]['model']
            joblib.dump(best_model, f'{output_dir}/fixed_best_model.pkl')
            print(f" 修复版最佳模型已保存: {output_dir}/fixed_best_model.pkl")

def main():
    """主函数"""
    # 获取项目根目录
    import os
    
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录（向上两级）
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # 构建优化数据文件的绝对路径
    data_file = os.path.join(project_root, 'results', 'preprocessing', 'optimized_data.csv')
    
    # 加载优化后的数据
    try:
        df = pd.read_csv(data_file)
        print(f"加载优化数据: {df.shape}")
    except Exception as e:
        print(f"未找到优化数据，请先运行优化预处理。错误: {e}")
        print(f"尝试加载的文件路径: {data_file}")
        return
    
    # 分离特征和目标
    target_col = 'lipid(%)'
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(df)}")
    print(f"特征/样本比: {len(feature_cols)/len(df):.3f}")
    
    # 创建修复版优化器并训练
    optimizer = FixedSmallSampleOptimizer(random_state=42)
    results = optimizer.train_and_evaluate(X, y)
    
    if results:
        # 创建集成模型
        ensemble_result = optimizer.create_simple_ensemble(X, y, results, top_n=3)
        
        # 创建可视化
        optimizer.create_performance_visualization(results, ensemble_result)
        
        # 保存结果
        optimizer.save_detailed_results(results, ensemble_result)
        
        print("\n 修复版小样本优化训练完成!")
        
        # 显示最佳结果
        valid_results = {name: result for name, result in results.items() 
                        if not np.isnan(result['bootstrap']['mean'])}
        if valid_results:
            best_model = max(valid_results.items(), key=lambda x: x[1]['bootstrap']['mean'])
            print(f"\n 最佳模型: {best_model[0]}")
            print(f"   训练R²: {best_model[1]['train_r2']:.4f}")
            print(f"   Bootstrap R²: {best_model[1]['bootstrap']['mean']:.4f}")
            if ensemble_result:
                print(f"   集成模型Bootstrap R²: {ensemble_result['bootstrap']['mean']:.4f}")

if __name__ == "__main__":
    main()