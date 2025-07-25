# -*- coding: utf-8 -*-

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import numpy as np

def load_clean_data(path: str) -> pd.DataFrame:

    ext = os.path.splitext(path)[1].lower()
    if ext in ['.csv', '.txt']:
        return pd.read_csv(path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    else:
        raise ValueError(f'Unsupported file type: {ext}')

def select_features_by_correlation(df: pd.DataFrame, target_col: str = 'lipid(%)', threshold: float = 0.3):

    corr_matrix = df.corr()
    corr_with_target = corr_matrix[target_col].abs()

    selected = corr_with_target[corr_with_target > threshold].index.tolist()

    if target_col in selected:
        selected.remove(target_col)

    return df[selected].copy(), selected

def remove_multicollinearity(df: pd.DataFrame, threshold: float = 10.0):

    X = df.copy()

    vif_data = pd.DataFrame({
        'feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })

    kept_features = vif_data[vif_data['VIF'] < threshold]['feature'].tolist()
    df_vif = X[kept_features].copy()
    return df_vif, kept_features, vif_data

def select_features_by_wrapper(
        df: pd.DataFrame,
        target_col: str = 'lipid(%)',
        n_features: int = 10,
        estimator=None,
        step: int = 1,
        random_state: int = 42):

    if estimator is None:
        estimator = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    selector = RFE(estimator, n_features_to_select=n_features, step=step)
    selector.fit(X, y)

    selected_cols = X.columns[selector.support_].tolist()
    return df[selected_cols].copy(), selected_cols, selector


# === RFECV feature selection ===
def select_features_by_rfecv(
        df: pd.DataFrame,
        target_col: str = 'lipid(%)',
        min_features: int = 10,
        estimator=None,
        cv_splits: int = 5,
        cv_repeats: int = 10,
        scoring: str = 'neg_mean_squared_error',
        random_state: int = 42):

    if estimator is None:
        estimator = RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    cv = RepeatedKFold(
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=random_state
    )

    selector = RFECV(
        estimator=estimator,
        step=1,
        min_features_to_select=min_features,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    selector.fit(X, y)
    selected_cols = X.columns[selector.support_].tolist()
    return df[selected_cols].copy(), selected_cols, selector

if __name__ == '__main__':

    # Determine project root and data file absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    data_file = os.path.join(project_root, 'data', 'processed', 'clean_data.csv')
    df = load_clean_data(data_file)

    # === Wrapper + CV (RFECV) feature selection ===
    X_rfecv, feats_rfecv, rfecv_obj = select_features_by_rfecv(
        df,
        target_col='lipid(%)',
        min_features=10
    )

    mean_scores = (-rfecv_obj.cv_results_['mean_test_score']  # 取正的 MSE
                   if 'mean_test_score' in rfecv_obj.cv_results_
                   else -rfecv_obj.grid_scores_)  # 兼容旧版 sklearn

    plt.figure()
    plt.plot(
        range(rfecv_obj.min_features_to_select,
              len(mean_scores) + rfecv_obj.min_features_to_select),
        mean_scores,
        marker='o'
    )
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Cross-Validated MSE')
    plt.title('RFECV Progress Curve')
    plt.grid(True)
    plt.tight_layout()

    # 保存到项目 /data 目录，运行时自动覆盖
    plot_path = os.path.join(project_root, 'feature_visual', 'rfecv_curve.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f'RFECV 曲线已经保存到 {plot_path}')
    print(f'RFECV-selected features ({len(feats_rfecv)}): {feats_rfecv}')

    # Optionally, save selected feature names to a text file
    out_path = os.path.join(project_root, 'data', 'rfecv_selected_features.txt')
    with open(out_path, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(feats_rfecv))
    print(f'Feature list saved to {out_path}')

    # === Additional visualizations ===
    # 1) Bar chart of feature importances
    importances = rfecv_obj.estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_feats = [feats_rfecv[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), sorted_feats, rotation=45, ha='right')
    plt.ylabel('Feature Importance')
    plt.title('RandomForest Feature Importances (Selected Features)')
    plt.tight_layout()
    bar_path = os.path.join(project_root, 'feature_visual', 'rfecv_importance_bar.png')
    os.makedirs(os.path.dirname(bar_path), exist_ok=True)
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f'Feature importance bar chart saved to {bar_path}')

    # 2) Cumulative importance curve
    cum_imp = np.cumsum(importances[indices])
    plt.figure(figsize=(8, 4))
    plt.step(range(len(importances)), cum_imp, where='mid', marker='o')
    plt.xlabel('Top N Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.grid(True)
    plt.tight_layout()
    cum_path = os.path.join(project_root, 'feature_visual', 'rfecv_cumulative_importance.png')
    os.makedirs(os.path.dirname(cum_path), exist_ok=True)
    plt.savefig(cum_path, dpi=300)
    plt.close()
    print(f'Cumulative importance curve saved to {cum_path}')

    # 3) Correlation heatmap for selected features
    corr = df[feats_rfecv].corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(feats_rfecv)), feats_rfecv, rotation=45, ha='right')
    plt.yticks(range(len(feats_rfecv)), feats_rfecv)
    plt.title('Correlation Heatmap (Selected Features)')
    plt.tight_layout()
    heat_path = os.path.join(project_root, 'feature_visual', 'selected_features_corr.png')
    os.makedirs(os.path.dirname(heat_path), exist_ok=True)
    plt.savefig(heat_path, dpi=300)
    plt.close()
    print(f'Correlation heatmap saved to {heat_path}')

def main():
    """主函数 - 运行特征工程流程"""
    print("🚀 开始基础特征工程...")
    
    # Determine project root and data file absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    data_file = os.path.join(project_root, 'data', 'processed', 'clean_data.csv')
    df = load_clean_data(data_file)

    # === Wrapper + CV (RFECV) feature selection ===
    X_rfecv, feats_rfecv, rfecv_obj = select_features_by_rfecv(
        df,
        target_col='lipid(%)',
        min_features=10
    )

    mean_scores = (-rfecv_obj.cv_results_['mean_test_score']  # 取正的 MSE
                   if 'mean_test_score' in rfecv_obj.cv_results_
                   else -rfecv_obj.grid_scores_)  # 兼容旧版 sklearn

    plt.figure()
    plt.plot(
        range(rfecv_obj.min_features_to_select,
              len(mean_scores) + rfecv_obj.min_features_to_select),
        mean_scores,
        marker='o'
    )
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Cross-Validated MSE')
    plt.title('RFECV Progress Curve')
    plt.grid(True)
    plt.tight_layout()

    # 保存到项目 /data 目录，运行时自动覆盖
    plot_path = os.path.join(project_root, 'feature_visual', 'rfecv_curve.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f'RFECV 曲线已经保存到 {plot_path}')
    print(f'RFECV-selected features ({len(feats_rfecv)}): {feats_rfecv}')

    # Optionally, save selected feature names to a text file
    out_path = os.path.join(project_root, 'data', 'rfecv_selected_features.txt')
    with open(out_path, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(feats_rfecv))
    print(f'Feature list saved to {out_path}')

    # === Additional visualizations ===
    # 1) Bar chart of feature importances
    importances = rfecv_obj.estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_feats = [feats_rfecv[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), sorted_feats, rotation=45, ha='right')
    plt.ylabel('Feature Importance')
    plt.title('RandomForest Feature Importances (Selected Features)')
    plt.tight_layout()
    bar_path = os.path.join(project_root, 'feature_visual', 'rfecv_importance_bar.png')
    os.makedirs(os.path.dirname(bar_path), exist_ok=True)
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f'Feature importance bar chart saved to {bar_path}')

    # 2) Cumulative importance curve
    cum_imp = np.cumsum(importances[indices])
    plt.figure(figsize=(8, 4))
    plt.step(range(len(importances)), cum_imp, where='mid', marker='o')
    plt.xlabel('Top N Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.grid(True)
    plt.tight_layout()
    cum_path = os.path.join(project_root, 'feature_visual', 'rfecv_cumulative_importance.png')
    os.makedirs(os.path.dirname(cum_path), exist_ok=True)
    plt.savefig(cum_path, dpi=300)
    plt.close()
    print(f'Cumulative importance curve saved to {cum_path}')

    # 3) Correlation heatmap for selected features
    corr = df[feats_rfecv].corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(feats_rfecv)), feats_rfecv, rotation=45, ha='right')
    plt.yticks(range(len(feats_rfecv)), feats_rfecv)
    plt.title('Correlation Heatmap (Selected Features)')
    plt.tight_layout()
    heat_path = os.path.join(project_root, 'feature_visual', 'selected_features_corr.png')
    os.makedirs(os.path.dirname(heat_path), exist_ok=True)
    plt.savefig(heat_path, dpi=300)
    plt.close()
    print(f'Correlation heatmap saved to {heat_path}')
    
    print("✅ 基础特征工程完成!")
