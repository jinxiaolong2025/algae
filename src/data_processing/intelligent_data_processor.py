# -*- coding: utf-8 -*-
"""
优化数据预处理模块
Optimized Data Preprocessing for Small Sample Algae Lipid Prediction

针对小样本问题的优化策略：
1. 保守的特征工程
2. 智能特征选择
3. 适合小样本的数据处理
4. 防止过拟合的预处理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class OptimizedPreprocessor:
    """针对小样本优化的数据预处理器"""
    
    def __init__(self, max_features_ratio=0.3):
        """
        初始化
        
        Args:
            max_features_ratio: 最大特征数与样本数的比例
        """
        self.max_features_ratio = max_features_ratio
        self.scaler = None
        self.selected_features = None
        self.feature_importance = {}
        
    def load_and_clean_data(self, filepath):
        """加载和基础清洗数据"""
        print("🔄 加载原始数据...")
        
        try:
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)
                
            print(f"✅ 成功加载数据: {df.shape}")
            
            # 清理列名
            df.columns = df.columns.str.strip()
            
            # 删除完全空的列和行
            df = df.dropna(how='all', axis=1)
            df = df.dropna(how='all', axis=0)
            
            # 删除非数值列（除了可能的ID列）
            numeric_cols = []
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    numeric_cols.append(col)
                else:
                    # 尝试转换为数值
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        numeric_cols.append(col)
                    except:
                        print(f"  删除非数值列: {col}")
            
            df = df[numeric_cols]
            
            # 删除Unnamed列
            unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
                print(f"  删除无用列: {unnamed_cols}")
            
            print(f"清洗后数据形状: {df.shape}")
            return df
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return None
    
    def handle_missing_values(self, df, target_col='lipid(%)'):
        """智能处理缺失值"""
        print("\n🔧 处理缺失值...")
        
        df_filled = df.copy()
        
        # 检查缺失值情况
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if len(missing_cols) > 0:
            print(f"发现 {len(missing_cols)} 列有缺失值:")
            for col, count in missing_cols.items():
                pct = count / len(df) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        
        # 对于小样本，使用保守的填充策略
        for col in df.columns:
            if col == target_col:
                continue
                
            missing_count = df_filled[col].isnull().sum()
            if missing_count > 0:
                # 如果缺失值过多（>30%），考虑删除该特征
                if missing_count / len(df) > 0.3:
                    print(f"  删除高缺失率特征: {col} ({missing_count/len(df)*100:.1f}%)")
                    df_filled = df_filled.drop(columns=[col])
                else:
                    # 使用中位数填充（对小样本更稳健）
                    median_val = df_filled[col].median()
                    df_filled[col] = df_filled[col].fillna(median_val)
                    print(f"  填充 {col}: {missing_count} 个缺失值 (中位数: {median_val:.3f})")
        
        return df_filled
    
    def conservative_feature_engineering(self, df, target_col='lipid(%)'):
        """保守的特征工程（避免过度复杂化）"""
        print("\n⚙️ 保守特征工程...")
        
        df_enhanced = df.copy()
        original_features = len(df.columns) - 1  # 减去目标变量
        
        # 只创建最有意义的比率特征
        ratio_features = []
        
        # 1. C/N比率（生物学上重要）
        if 'C(%)' in df.columns and 'N(%)' in df.columns:
            df_enhanced['C_N_ratio'] = df['C(%)'] / (df['N(%)'] + 1e-6)
            ratio_features.append('C_N_ratio')
        
        # 2. 蛋白质/多糖比率
        if 'protein(%)' in df.columns and 'polysaccharide(%)' in df.columns:
            df_enhanced['protein_polysaccharide_ratio'] = df['protein(%)'] / (df['polysaccharide(%)'] + 1e-6)
            ratio_features.append('protein_polysaccharide_ratio')
        
        # 3. TN/TP比率
        if 'TN' in df.columns and 'TP' in df.columns:
            df_enhanced['TN_TP_ratio'] = df['TN'] / (df['TP'] + 1e-6)
            ratio_features.append('TN_TP_ratio')
        
        # 只对最重要的特征进行平方变换
        important_features = ['protein(%)', 'H(%)', 'O(%)']
        squared_features = []
        
        for feat in important_features:
            if feat in df.columns:
                # 检查与目标变量的相关性
                corr = abs(df[feat].corr(df[target_col]))
                if corr > 0.2:  # 只对相关性较高的特征进行变换
                    df_enhanced[f"{feat}_squared"] = df[feat] ** 2
                    squared_features.append(f"{feat}_squared")
        
        new_features = ratio_features + squared_features
        print(f"创建了 {len(new_features)} 个新特征: {new_features}")
        print(f"特征数量: {original_features} → {len(df_enhanced.columns) - 1}")
        
        return df_enhanced
    
    def intelligent_feature_selection(self, df, target_col='lipid(%)', max_features=None):
        """智能特征选择"""
        print("\n🎯 智能特征选择...")
        
        if max_features is None:
            max_features = max(8, int(len(df) * self.max_features_ratio))
        
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        print(f"目标特征数: {max_features} (样本数: {len(df)})")
        
        # 1. 相关性筛选
        correlations = {}
        for col in feature_cols:
            corr = abs(X[col].corr(y))
            correlations[col] = corr
        
        # 按相关性排序
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # 选择相关性最高的特征
        correlation_selected = [feat for feat, corr in sorted_corr[:max_features*2] if corr > 0.1]
        print(f"相关性筛选: {len(correlation_selected)} 个特征 (阈值: 0.1)")
        
        # 2. 方差筛选
        variance_threshold = X[correlation_selected].var().quantile(0.1)
        variance_selected = []
        for col in correlation_selected:
            if X[col].var() > variance_threshold:
                variance_selected.append(col)
        
        print(f"方差筛选: {len(variance_selected)} 个特征")
        
        # 3. 递归特征消除
        if len(variance_selected) > max_features:
            ridge = Ridge(alpha=1.0, random_state=42)
            rfe = RFE(ridge, n_features_to_select=max_features, step=1)
            rfe.fit(X[variance_selected], y)
            
            final_features = [variance_selected[i] for i in range(len(variance_selected)) if rfe.support_[i]]
        else:
            final_features = variance_selected
        
        print(f"最终选择: {len(final_features)} 个特征")
        
        # 保存特征重要性
        for i, feat in enumerate(final_features):
            self.feature_importance[feat] = correlations[feat]
        
        self.selected_features = final_features
        
        # 显示选择的特征
        print("\n选择的特征:")
        for i, feat in enumerate(final_features, 1):
            corr = correlations[feat]
            print(f"  {i:2d}. {feat}: {corr:.3f}")
        
        return df[final_features + [target_col]]
    
    def smart_scaling(self, df, target_col='lipid(%)', method='standard'):
        """智能缩放"""
        print(f"\n📏 特征缩放 ({method})...")
        
        feature_cols = [col for col in df.columns if col != target_col]
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            print("使用标准缩放")
            self.scaler = StandardScaler()
        
        # 只缩放特征，不缩放目标变量
        X = df[feature_cols].values
        y = df[target_col].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        # 重新组合
        df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
        df_scaled[target_col] = y
        
        print(f"缩放完成: {len(feature_cols)} 个特征")
        
        return df_scaled
    
    def create_diagnostic_plots(self, df_original, df_processed, output_dir='results/preprocessing'):
        """创建诊断图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('优化预处理结果', fontsize=16, weight='bold')
        
        target_col = 'lipid(%)'
        
        # 1. 目标变量分布对比
        axes[0, 0].hist(df_original[target_col], bins=10, alpha=0.7, color='red', label='原始')
        axes[0, 0].hist(df_processed[target_col], bins=10, alpha=0.7, color='blue', label='处理后')
        axes[0, 0].set_title('目标变量分布')
        axes[0, 0].legend()
        
        # 2. 特征数量对比
        axes[0, 1].bar(['原始', '处理后'], [df_original.shape[1]-1, df_processed.shape[1]-1], 
                      color=['red', 'blue'], alpha=0.7)
        axes[0, 1].set_title('特征数量对比')
        axes[0, 1].set_ylabel('特征数量')
        
        # 3. 数据质量改进
        orig_missing = df_original.isnull().sum().sum()
        proc_missing = df_processed.isnull().sum().sum()
        axes[0, 2].bar(['原始', '处理后'], [orig_missing, proc_missing], 
                      color=['red', 'blue'], alpha=0.7)
        axes[0, 2].set_title('缺失值对比')
        axes[0, 2].set_ylabel('缺失值数量')
        
        # 4. 特征重要性
        if self.feature_importance:
            features = list(self.feature_importance.keys())[:10]
            importances = [self.feature_importance[f] for f in features]
            
            axes[1, 0].barh(range(len(features)), importances, color='green', alpha=0.7)
            axes[1, 0].set_yticks(range(len(features)))
            axes[1, 0].set_yticklabels([f.split('(')[0][:15] for f in features])
            axes[1, 0].set_title('特征重要性 (相关性)')
            axes[1, 0].set_xlabel('绝对相关系数')
        
        # 5. 相关性矩阵
        if len(df_processed.columns) <= 15:
            corr_matrix = df_processed.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1, 1], fmt='.2f', cbar_kws={'shrink': 0.8})
            axes[1, 1].set_title('特征相关性矩阵')
        else:
            # 只显示选择的特征
            selected_cols = self.selected_features[:10] + [target_col]
            corr_matrix = df_processed[selected_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1, 1], fmt='.2f', cbar_kws={'shrink': 0.8})
            axes[1, 1].set_title('核心特征相关性')
        
        # 6. 处理摘要
        summary_text = f"""处理摘要:
原始数据: {df_original.shape}
处理后: {df_processed.shape}
特征减少: {df_original.shape[1]-1} → {df_processed.shape[1]-1}
缺失值: {orig_missing} → {proc_missing}
样本保留: {len(df_processed)}/{len(df_original)}
特征选择比例: {(df_processed.shape[1]-1)/len(df_processed):.2f}"""
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, 
                        transform=axes[1, 2].transAxes, verticalalignment='center')
        axes[1, 2].set_title('处理摘要')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/optimized_preprocessing_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 诊断图表已保存: {output_dir}/optimized_preprocessing_results.png")
    
    def process_data(self, filepath, output_dir='results/preprocessing', target_col='lipid(%)'):
        """完整的优化预处理流程"""
        print("🚀 开始优化数据预处理...")
        print("=" * 60)
        
        # 1. 加载和清洗
        df_original = self.load_and_clean_data(filepath)
        if df_original is None:
            return None
        
        # 2. 处理缺失值
        df_clean = self.handle_missing_values(df_original, target_col)
        
        # 3. 保守特征工程
        df_enhanced = self.conservative_feature_engineering(df_clean, target_col)
        
        # 4. 智能特征选择
        df_selected = self.intelligent_feature_selection(df_enhanced, target_col)
        
        # 5. 智能缩放
        df_final = self.smart_scaling(df_selected, target_col, method='standard')
        
        # 6. 保存结果
        os.makedirs(output_dir, exist_ok=True)
        df_final.to_csv(f'{output_dir}/optimized_data.csv', index=False)
        print(f"💾 优化数据已保存: {output_dir}/optimized_data.csv")
        
        # 7. 创建诊断图表
        self.create_diagnostic_plots(df_original, df_final, output_dir)
        
        # 8. 保存处理报告
        self.save_optimization_report(df_original, df_final, output_dir)
        
        print("\n" + "=" * 60)
        print("✅ 优化预处理完成!")
        print(f"📊 原始数据: {df_original.shape}")
        print(f"📈 优化后数据: {df_final.shape}")
        print(f"🎯 特征优化: {df_original.shape[1]-1} → {df_final.shape[1]-1} 个特征")
        print(f"📏 特征/样本比: {(df_final.shape[1]-1)/len(df_final):.2f}")
        
        return df_final
    
    def save_optimization_report(self, df_orig, df_final, output_dir):
        """保存优化报告"""
        with open(f'{output_dir}/optimization_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("优化预处理报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1. 数据优化摘要\n")
            f.write("-" * 30 + "\n")
            f.write(f"原始数据形状: {df_orig.shape}\n")
            f.write(f"优化后数据形状: {df_final.shape}\n")
            f.write(f"特征优化: {df_orig.shape[1]-1} → {df_final.shape[1]-1}\n")
            f.write(f"特征/样本比: {(df_final.shape[1]-1)/len(df_final):.3f}\n\n")
            
            f.write("2. 数据质量改进\n")
            f.write("-" * 30 + "\n")
            f.write(f"原始缺失值: {df_orig.isnull().sum().sum()}\n")
            f.write(f"处理后缺失值: {df_final.isnull().sum().sum()}\n")
            f.write(f"样本保留率: {len(df_final)/len(df_orig)*100:.1f}%\n\n")
            
            f.write("3. 选择的特征 (按重要性)\n")
            f.write("-" * 30 + "\n")
            if self.feature_importance:
                sorted_features = sorted(self.feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
                for i, (feat, importance) in enumerate(sorted_features, 1):
                    f.write(f"{i:2d}. {feat}: {importance:.3f}\n")
            
            f.write("\n4. 优化策略\n")
            f.write("-" * 30 + "\n")
            f.write("- 保守特征工程: 避免过度复杂化\n")
            f.write("- 智能特征选择: 相关性+方差+RFE\n")
            f.write("- 标准化缩放: 只缩放特征，保留目标变量原始尺度\n")
            f.write("- 小样本优化: 特征数 < 样本数/3\n")
        
        print(f"📄 优化报告已保存: {output_dir}/optimization_report.txt")

def main():
    """主函数"""
    import os
    from pathlib import Path
    
    # 获取项目根目录
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    preprocessor = OptimizedPreprocessor(max_features_ratio=0.25)
    
    # 处理数据
    df_processed = preprocessor.process_data(
        filepath=str(project_root / 'data' / 'raw' / '数据.xlsx'),
        output_dir=str(project_root / 'results' / 'preprocessing')
    )
    
    if df_processed is not None:
        print(f"\n🎉 优化预处理成功完成!")
        print(f"📁 结果文件:")
        print(f"   - 优化数据: results/preprocessing/optimized_data.csv")
        print(f"   - 诊断图表: results/preprocessing/optimized_preprocessing_results.png") 
        print(f"   - 优化报告: results/preprocessing/optimization_report.txt")

if __name__ == "__main__":
    main()