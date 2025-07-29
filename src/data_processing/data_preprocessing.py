import pandas as pd
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def load_data():
    data = pd.read_excel("../../data/raw/数据.xlsx")
    # 删除S(%)列
    data = data.drop('S(%)', axis=1)
    return data

def analyze_data_quality(data):

    print("\n" + "="*80)
    print("数据质量分析报告")
    print("="*80)

    # 基本信息
    print(f"\n 数据基本信息:")
    print(f"   - 数据形状: {data.shape}")
    print(f"   - 样本数量: {data.shape[0]}")
    print(f"   - 特征数量: {data.shape[1]}")

    # 数据类型统计
    print(f"\n 数据类型分布:")
    dtype_counts = data.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   - {dtype}: {count}个特征")

    # 创建质量分析DataFrame
    quality_analysis = pd.DataFrame(index=data.columns)

    # 1. 缺失值分析
    print(f"\n 缺失值分析:")
    missing_counts = data.isnull().sum()
    missing_rates = (missing_counts / len(data) * 100).round(2)
    quality_analysis['missing_count'] = missing_counts
    quality_analysis['missing_rate(%)'] = missing_rates

    total_missing = missing_counts.sum()
    print(f"   - 总缺失值数量: {total_missing}")
    print(f"   - 有缺失值的特征数: {(missing_counts > 0).sum()}")

    if total_missing > 0:
        print(f"   - 缺失值最多的特征:")
        top_missing = missing_rates[missing_rates > 0].sort_values(ascending=False).head(5)
        for feature, rate in top_missing.items():
            print(f"     * {feature}: {rate}% ({missing_counts[feature]}个)")
    else:
        print(f"   -  无缺失值")

    # 2. 零值分析
    print(f"\n 零值分析:")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    zero_counts = (data[numeric_cols] == 0).sum()
    zero_rates = (zero_counts / len(data) * 100).round(2)

    # 添加到质量分析表
    for col in data.columns:
        if col in numeric_cols:
            quality_analysis.loc[col, 'zero_count'] = zero_counts[col]
            quality_analysis.loc[col, 'zero_rate(%)'] = zero_rates[col]
        else:
            quality_analysis.loc[col, 'zero_count'] = 0
            quality_analysis.loc[col, 'zero_rate(%)'] = 0.0

    total_zeros = zero_counts.sum()
    print(f"   - 总零值数量: {total_zeros}")
    print(f"   - 有零值的特征数: {(zero_counts > 0).sum()}")

    if total_zeros > 0:
        print(f"   - 零值最多的特征:")
        top_zeros = zero_rates[zero_rates > 0].sort_values(ascending=False).head(5)
        for feature, rate in top_zeros.items():
            print(f"     * {feature}: {rate}% ({zero_counts[feature]}个)")

    # 3. 偏度分析
    print(f"\n 偏度分析:")
    skewness_values = data[numeric_cols].skew().round(3)
    quality_analysis.loc[numeric_cols, 'skewness'] = skewness_values

    # 偏度解释
    highly_skewed = skewness_values[abs(skewness_values) > 2]
    moderately_skewed = skewness_values[(abs(skewness_values) > 1) & (abs(skewness_values) <= 2)]

    print(f"   - 高度偏斜特征 (|偏度| > 2): {len(highly_skewed)}个")
    if len(highly_skewed) > 0:
        for feature, skew in highly_skewed.items():
            direction = "右偏" if skew > 0 else "左偏"
            print(f"     * {feature}: {skew} ({direction})")

    print(f"   - 中度偏斜特征 (1 < |偏度| <= 2): {len(moderately_skewed)}个")
    if len(moderately_skewed) > 0:
        for feature, skew in moderately_skewed.items():
            direction = "右偏" if skew > 0 else "左偏"
            print(f"     * {feature}: {skew} ({direction})")

    # 4. 数值范围分析
    print(f"\n 数值范围分析:")
    print(f"{'特征名称':<25} {'最小值':<10} {'最大值':<10} {'均值':<10} {'标准差':<10}")
    print("-" * 70)

    for col in numeric_cols:
        min_val = data[col].min()
        max_val = data[col].max()
        mean_val = data[col].mean()
        std_val = data[col].std()
        print(f"{col:<25} {min_val:<10.2f} {max_val:<10.2f} {mean_val:<10.2f} {std_val:<10.2f}")

    # 5. 保存质量分析报告
    quality_analysis = quality_analysis.fillna(0)
    quality_analysis.to_csv("../../results/data_preprocess/data_quality_analysis.csv", float_format='%.3f')

    print(f"\n 数据质量分析报告已保存:")
    print(f"   - results/data_preprocess/data_quality_analysis.csv")

    # 6. 总结建议
    print(f"\n 数据质量总结:")

    # 缺失值建议
    if total_missing > 0:
        print(f"     发现{total_missing}个缺失值，建议先对其进行填充处理")
    else:
        print(f"    无缺失值，数据完整性良好")

    # 零值建议
    high_zero_features = zero_rates[zero_rates > 50].index.tolist()
    if len(high_zero_features) > 0:
        print(f"   ️  {len(high_zero_features)}个特征零值占比超过50%，可能需要特殊处理")

    # 偏度建议
    if len(highly_skewed) > 0:
        print(f"     {len(highly_skewed)}个特征高度偏斜，建议考虑对数变换或其他标准化方法")

    print(f"\n" + "="*80)

    return quality_analysis

def visualize_outlier_treatment(data_before, data_after):
    """可视化异常值处理前后的对比"""
    print("\n 生成异常值处理前后对比图...")

    # 基于偏度计算特征组
    skewness_before = data_before.select_dtypes(include=[np.number]).skew()
    skewness_after = data_after.select_dtypes(include=[np.number]).skew()

    # 定义基于偏度的特征组
    light_skew = skewness_before[abs(skewness_before) < 1].index.tolist()
    moderate_skew = skewness_before[(abs(skewness_before) >= 1) & (abs(skewness_before) < 2)].index.tolist()
    heavy_skew = skewness_before[abs(skewness_before) >= 2].index.tolist()

    feature_groups = {
        'Light Skewness (No Treatment)': light_skew,
        'Moderate Skewness (5%-95% Winsorize)': moderate_skew,
        'Heavy Skewness (10%-90% Winsorize)': heavy_skew
    }

    for group_name, features in feature_groups.items():
        # 筛选存在的特征
        existing_features = [f for f in features if f in data_before.columns and f in data_after.columns]

        if not existing_features:
            continue

        # 计算需要的行数和列数
        n_features = len(existing_features)
        n_cols = min(3, n_features)  # 最多3列
        n_rows = (n_features + n_cols - 1) // n_cols  # 向上取整

        # 创建图形
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(5*n_cols, 4*n_rows*2))
        fig.suptitle(f'{group_name} - Outlier Treatment Comparison', fontsize=16, y=0.98)

        # 确保axes是二维数组
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        elif n_rows == 1:
            axes = axes.reshape(2, n_cols)
        elif n_cols == 1:
            axes = axes.reshape(n_rows*2, 1)

        for idx, feature in enumerate(existing_features):
            row = (idx // n_cols) * 2
            col = idx % n_cols

            # 计算偏度
            skew_before = data_before[feature].skew()
            skew_after = data_after[feature].skew()
            skew_improvement = abs(skew_before) - abs(skew_after)

            # 上方：箱线图对比
            ax_box = axes[row, col]
            box_data = [data_before[feature].dropna(), data_after[feature].dropna()]
            box_labels = ['Before', 'After']

            bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightcoral')
            bp['boxes'][1].set_facecolor('lightblue')

            ax_box.set_title(f'{feature}\nSkewness: {skew_before:.3f} -> {skew_after:.3f} (Improve: {skew_improvement:+.3f})',
                           fontsize=10)
            ax_box.grid(True, alpha=0.3)

            # 下方：直方图对比
            ax_hist = axes[row + 1, col]

            # 处理前的直方图
            ax_hist.hist(data_before[feature].dropna(), bins=15, alpha=0.6,
                        color='lightcoral', label='Before', density=True)

            # 处理后的直方图
            ax_hist.hist(data_after[feature].dropna(), bins=15, alpha=0.6,
                        color='lightblue', label='After', density=True)

            ax_hist.set_xlabel('Value')
            ax_hist.set_ylabel('Density')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(n_features, n_rows * n_cols):
            row = (idx // n_cols) * 2
            col = idx % n_cols
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].set_visible(False)
                axes[row + 1, col].set_visible(False)

        plt.tight_layout()

        # 保存图片
        filename = f"../../results/data_preprocess/outlier_treatment_{group_name.replace('+', '_').replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    {group_name}对比图已保存: {filename}")

    print("    异常值处理可视化完成")

def visualize_skewness_improvement(data_before, data_after):
    """可视化偏度改善情况"""
    print("\n 生成偏度改善分析图...")

    # 计算所有数值特征的偏度
    numeric_cols = data_before.select_dtypes(include=[np.number]).columns

    skew_before = data_before[numeric_cols].skew()
    skew_after = data_after[numeric_cols].skew()
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

    # 保存图片
    filename = "../../results/data_preprocess/skewness_improvement_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    偏度改善分析图已保存: {filename}")

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

def handle_missing_values(data):
    """处理缺失值 - 针对性处理策略"""
    print("\n 缺失值处理详情:")

    data_filled = data.copy()

    # 1. 分析phosphate和TP的相关性
    if 'phosphate' in data.columns and 'TP' in data.columns:
        # 获取两列都不为空的数据
        valid_mask = data['phosphate'].notna() & data['TP'].notna()
        if valid_mask.sum() > 0:
            correlation = data.loc[valid_mask, 'phosphate'].corr(data.loc[valid_mask, 'TP'])
            print(f"   phosphate与TP的相关系数: {correlation:.4f}")

            # 2. 使用TP对phosphate进行线性回归填充
            if data['phosphate'].isna().sum() > 0:

                # 准备训练数据（phosphate和TP都不为空的样本）
                train_mask = data['phosphate'].notna() & data['TP'].notna()
                X_train = data.loc[train_mask, 'TP'].values.reshape(-1, 1)
                y_train = data.loc[train_mask, 'phosphate'].values

                # 训练线性回归模型
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)

                # 预测缺失值
                missing_mask = data['phosphate'].isna() & data['TP'].notna()
                if missing_mask.sum() > 0:
                    X_pred = data.loc[missing_mask, 'TP'].values.reshape(-1, 1)
                    y_pred = lr_model.predict(X_pred)

                    # 填充缺失值
                    data_filled.loc[missing_mask, 'phosphate'] = y_pred

                    print(f"    使用TP线性回归填充phosphate缺失值: {missing_mask.sum()}个")
                    print(f"   回归方程: phosphate = {lr_model.coef_[0]:.4f} * TP + {lr_model.intercept_:.4f}")
                    print(f"   R²得分: {lr_model.score(X_train, y_train):.4f}")

    # 3. 处理N(%)和C(%)的缺失值 - 使用KNN填充
    remaining_missing = data_filled.isnull().sum()
    if remaining_missing.sum() > 0:
        print(f"\n   处理N(%)和C(%)缺失值:")

        # 只对N(%)和C(%)使用KNN填充
        missing_cols = remaining_missing[remaining_missing > 0].index.tolist()
        print(f"   需要处理的特征: {missing_cols}")

        if len(missing_cols) > 0:
            # 使用KNN填充器，选择相关特征作为参考
            knn_imputer = KNNImputer(n_neighbors=5)

            # 选择相关特征进行KNN填充
            reference_cols = ['H(%)', 'O(%)', 'P(%)', 'protein(%)']
            available_refs = [col for col in reference_cols if col in data_filled.columns]

            # 构建用于KNN的特征集
            knn_features = missing_cols + available_refs
            data_subset = data_filled[knn_features].copy()

            # 执行KNN填充
            data_subset_filled = knn_imputer.fit_transform(data_subset)

            # 将填充结果放回原数据
            for i, col in enumerate(missing_cols):
                missing_count = remaining_missing[col]
                data_filled[col] = data_subset_filled[:, i]
                print(f"    {col}: 用KNN填充{missing_count}个缺失值")

    # 4. 验证填充结果
    final_missing = data_filled.isnull().sum().sum()
    print(f"\n    填充结果:")
    print(f"   - 填充前总缺失值: {data.isnull().sum().sum()}")
    print(f"   - 填充后总缺失值: {final_missing}")

    if final_missing == 0:
        print(f"    所有缺失值已成功填充")
    else:
        print(f"     仍有{final_missing}个缺失值未处理")

    return data_filled

def visualize_data(data):
    """可视化数据分布"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建子图
    n_cols = len(data.columns)
    n_rows = (n_cols + 4) // 5  # 每行5个图

    plt.figure(figsize=(20, 4*n_rows))

    for i, col in enumerate(data.columns):
        plt.subplot(n_rows, 5, i+1)
        plt.hist(data[col], bins=10, alpha=0.7)
        plt.title(f'{col}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('../../results/data_preprocess/data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("数据分布图已保存到 results/data_distribution.png")

def handle_outliers(data):
    """基于偏度的异常值处理"""
    print("\n 异常值处理详情:")
    data_before = data.copy()  # 保存处理前的数据用于对比
    data_processed = data.copy()

    # 计算所有特征的偏度
    skewness = data_processed.select_dtypes(include=[np.number]).skew()

    # 1. 轻度偏态（|偏度|<1）：不做裁切
    light_skew_cols = skewness[abs(skewness) < 1].index.tolist()
    print(f"    轻度偏态(|偏度|<1)，不处理: {len(light_skew_cols)}个")
    for col in light_skew_cols:
        print(f"     • {col}: 偏度={skewness[col]:.3f}")

    # 2. 中度偏态（1≤|偏度|<2）：分位数限幅
    moderate_skew_cols = skewness[(abs(skewness) >= 1) & (abs(skewness) < 2)].index.tolist()
    print(f"    中度偏态(1≤|偏度|<2)，分位数限幅: {len(moderate_skew_cols)}个")

    for col in moderate_skew_cols:
        if col in data_processed.columns:
            # 使用5%-95%分位数限幅
            lower = data_processed[col].quantile(0.05)
            upper = data_processed[col].quantile(0.95)

            # 统计异常值
            outliers_count = ((data_processed[col] < lower) | (data_processed[col] > upper)).sum()
            outlier_rate = outliers_count / len(data_processed) * 100

            # Winsorize限幅
            data_processed[col] = data_processed[col].clip(lower, upper)

            print(f"     • {col}: 偏度={skewness[col]:.3f}, {outliers_count}个异常值({outlier_rate:.1f}%) → 5%-95%分位数限幅")

    # 3. 重度偏态（|偏度|≥2）：分位数限幅
    heavy_skew_cols = skewness[abs(skewness) >= 2].index.tolist()
    print(f"    重度偏态(|偏度|≥2)，分位数限幅: {len(heavy_skew_cols)}个")

    for col in heavy_skew_cols:
        if col in data_processed.columns:
            # 使用10%-90%分位数限幅
            lower = data_processed[col].quantile(0.10)
            upper = data_processed[col].quantile(0.90)

            # 统计异常值
            outliers_count = ((data_processed[col] < lower) | (data_processed[col] > upper)).sum()
            outlier_rate = outliers_count / len(data_processed) * 100

            # Winsorize限幅
            data_processed[col] = data_processed[col].clip(lower, upper)

            print(f"     • {col}: 偏度={skewness[col]:.3f}, {outliers_count}个异常值({outlier_rate:.1f}%) → 10%-90%分位数限幅")

    print(f"    异常值处理完成，数据形状保持: {data_processed.shape}")

    # 生成处理前后对比图
    visualize_outlier_treatment(data_before, data_processed)

    # 生成偏度改善分析图
    visualize_skewness_improvement(data_before, data_processed)

    return data_processed

def robust_scaling(data, scaler=None, fit=True):

    data = data.copy()

    # 保存lipid(%)列，不做标准化
    lipid_col = None
    if 'lipid(%)' in data.columns:
        lipid_col = data['lipid(%)'].copy()
        data_to_scale = data.drop('lipid(%)', axis=1)
    else:
        data_to_scale = data

    # 对其他列做标准化
    if scaler is None:
        scaler = RobustScaler()

    if fit:
        data_scaled = scaler.fit_transform(data_to_scale)
    else:
        data_scaled = scaler.transform(data_to_scale)

    data_scaled = pd.DataFrame(data_scaled, columns=data_to_scale.columns, index=data_to_scale.index)

    # 把lipid(%)列加回来
    if lipid_col is not None:
        data_scaled['lipid(%)'] = lipid_col

    return data_scaled, scaler

def split_dataset(data, test_size=0.2, random_state=42):
    """分割数据集为训练集和测试集"""
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

if __name__ == "__main__":
    print("微藻数据预处理系统")
    print("="*60)

    # 1. 加载原始数据
    print("\n1. 加载原始数据...")
    data = load_data()
    print(f"   原始数据形状: {data.shape}")

    # 2. 数据质量分析
    print("\n2. 进行数据质量分析...")
    quality_analysis = analyze_data_quality(data)

    # 3. 处理缺失值
    print("\n3. 处理缺失值...")
    processed_data = handle_missing_values(data)
    print(f"   缺失值处理后数据形状: {processed_data.shape}")

    # 4. 数据可视化
    print("\n4. 生成数据可视化...")
    processed_data.to_csv("../../data/processed/processed_data1.csv", index=False, float_format='%.6f')
    # 5. 处理异常值
    print("\n5. 处理异常值...")
    processed_data = handle_outliers(processed_data)
    print(f"   异常值处理后数据形状: {processed_data.shape}")
    visualize_data(processed_data)
    processed_data.to_csv("../../data/processed/processed_data2.csv", index=False, float_format='%.6f')
    # 6. 分割数据集（在标准化之前）
    print("\n6. 分割数据集...")
    train_data, test_data = split_dataset(processed_data)
    print(f"   训练集大小: {train_data.shape}")
    print(f"   测试集大小: {test_data.shape}")

    # 7. Robust标准化（避免数据泄露）
    print("\n7. 进行Robust标准化...")
    print("   - 在训练集上拟合scaler")
    train_data_scaled, scaler = robust_scaling(train_data, fit=True)
    print("   - 将scaler应用到测试集")
    test_data_scaled, _ = robust_scaling(test_data, scaler=scaler, fit=False)

    processed_data_scaled = pd.concat([train_data_scaled, test_data_scaled], ignore_index=True)

    print(f"   标准化后数据形状: 训练集{train_data_scaled.shape}, 测试集{test_data_scaled.shape}")
    print(f"   完整数据集形状: {processed_data_scaled.shape}")

    # 验证没有缺失值
    missing_count = processed_data_scaled.isnull().sum().sum()
    if missing_count > 0:
        print(f"     警告: 发现{missing_count}个缺失值")
    else:
        print(f"    数据完整，无缺失值")

    # 8. 保存数据集
    print("\n8. 保存处理后的数据...")
    train_data_scaled.to_csv("../../data/processed/train_data.csv", index=False, float_format='%.6f')
    test_data_scaled.to_csv("../../data/processed/test_data.csv", index=False, float_format='%.6f')
    processed_data_scaled.to_csv("../../data/processed/processed_data.csv", index=False, float_format='%.6f')

    print("   数据已保存到:")
    print("     - data/processed/train_data.csv")
    print("     - data/processed/test_data.csv")
    print("     - data/processed/processed_data.csv")

    print("\n" + "="*60)
    print("数据预处理完成！")
    print("="*60)