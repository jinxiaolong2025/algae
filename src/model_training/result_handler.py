"""结果处理和保存模块

包含结果分析器和模型保存器类
"""
import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, List, Tuple, Optional, Any

try:
    from .config import TrainingConfig
except ImportError:
    from config import TrainingConfig


class ResultAnalyzer:
    """结果分析器类，负责分析和展示训练结果"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config

    def display_model_comparison(self, results: Dict[str, Any],
                               feature_names: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """显示模型对比结果

        显示内容：
        - 按交叉验证R²分数降序排列的模型排名
        - 交叉验证指标：R²、MAE、RMSE及其标准差
        - 训练集指标：R²、MAE、RMSE
        - 性能等级评定（优秀/良好/一般/较差）

        排序规则：优先按R²分数，其次按MAE分数

        Args:
            results: 模型训练结果字典
            feature_names: 特征名称列表

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: (对比结果DataFrame, 最佳模型名称)
        """
        print("\n" + "="*100)
        print("多模型性能对比结果（5折交叉验证）")
        print("="*100)
        
        if not results:
            print("   没有可用的训练结果")
            return None, None
        
        # 创建结果DataFrame
        comparison_data = []
        for model_name, result in results.items():
            if not np.isnan(result['r2']):
                data_row = {
                    'Model': model_name,
                    'CV_R²': result['r2'],
                    'CV_R²_Std': result['r2_std'],
                    'CV_MAE': result['mae'],
                    'CV_MAE_Std': result['mae_std'],
                    'CV_RMSE': result['rmse'],
                    'CV_RMSE_Std': result['rmse_std']
                }
                
                # 添加最终模型评估指标（如果存在）
                if 'final_cv_r2' in result:
                    data_row.update({
                        'Final_R²': result['final_cv_r2'],
                        'Final_R²_Std': result['final_cv_r2_std'],
                        'Final_MAE': result['final_cv_mae'],
                        'Final_MAE_Std': result['final_cv_mae_std'],
                        'Final_RMSE': result['final_cv_rmse'],
                        'Final_RMSE_Std': result['final_cv_rmse_std']
                    })
                else:
                    # 如果没有最终评估指标，使用NaN填充
                    data_row.update({
                        'Final_R²': np.nan,
                        'Final_R²_Std': np.nan,
                        'Final_MAE': np.nan,
                        'Final_MAE_Std': np.nan,
                        'Final_RMSE': np.nan,
                        'Final_RMSE_Std': np.nan
                    })
                
                comparison_data.append(data_row)
        
        if not comparison_data:
            print("   所有模型训练都失败了")
            return None, None
        
        # 创建DataFrame并排序
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(['CV_R²', 'CV_MAE'], ascending=[False, True])
        comparison_df = comparison_df.reset_index(drop=True)
        
        # 打印对比表格
        self._print_comparison_table(comparison_df)
        
        # 获取最佳模型
        best_model_name = self._display_best_model(comparison_df)
        
        # 保存对比结果
        self._save_comparison_results(comparison_df)
        
        return comparison_df, best_model_name
    
    def _print_comparison_table(self, comparison_df: pd.DataFrame) -> None:
        """打印对比表格"""
        # 打印表头
        print(f"\n{'排名':<4} {'模型名称':<20} {'交叉验证R²':<15} {'交叉验证MAE':<15} {'交叉验证RMSE':<16} {'最终评估R²':<15} {'最终评估MAE':<15} {'最终评估RMSE':<16} {'等级':<8}")
        print("-" * 140)
        
        for i, row in comparison_df.iterrows():
            rank = comparison_df.index.get_loc(i) + 1
            # 性能等级判断
            level = self._get_performance_level(row['CV_R²'])
            
            # 交叉验证指标
            cv_r2_str = f"{row['CV_R²']:.4f}±{row['CV_R²_Std']:.3f}"
            cv_mae_str = f"{row['CV_MAE']:.3f}±{row['CV_MAE_Std']:.3f}"
            cv_rmse_str = f"{row['CV_RMSE']:.3f}±{row['CV_RMSE_Std']:.3f}"
            
            # 最终评估指标
            if not np.isnan(row['Final_R²']):
                final_r2_str = f"{row['Final_R²']:.4f}±{row['Final_R²_Std']:.3f}"
                final_mae_str = f"{row['Final_MAE']:.3f}±{row['Final_MAE_Std']:.3f}"
                final_rmse_str = f"{row['Final_RMSE']:.3f}±{row['Final_RMSE_Std']:.3f}"
            else:
                final_r2_str = "N/A"
                final_mae_str = "N/A"
                final_rmse_str = "N/A"
            
            print(f"{rank:<4} {row['Model']:<20} {cv_r2_str:<15} {cv_mae_str:<15} {cv_rmse_str:<16} {final_r2_str:<15} {final_mae_str:<15} {final_rmse_str:<16} {level:<8}")
    
    def _get_performance_level(self, r2_score: float) -> str:
        """根据R²分数判断性能等级"""
        if r2_score >= self.config.excellent_r2_threshold:
            return "优秀"
        elif r2_score >= self.config.good_r2_threshold:
            return "良好"
        elif r2_score >= self.config.fair_r2_threshold:
            return "一般"
        else:
            return "较差"
    
    def _display_best_model(self, comparison_df: pd.DataFrame) -> str:
        """获取最佳模型名称（不显示详细信息）"""
        best_model = comparison_df.iloc[0]
        return best_model['Model']
    
    def _save_comparison_results(self, comparison_df: pd.DataFrame) -> None:
        """保存对比结果到CSV文件"""
        os.makedirs(self.config.results_dir, exist_ok=True)
        comparison_path = os.path.join(self.config.results_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n   模型对比结果已保存: {comparison_path}")


class ModelSaver:
    """模型保存器类，负责保存训练好的模型和相关信息"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config

    def save_best_model(self, results: Dict[str, Any], best_model_name: str,
                       feature_names: List[str]) -> Optional[str]:
        """保存最佳模型及相关信息

        保存内容：
        - 模型文件：使用pickle格式保存训练好的模型
        - 模型信息：JSON格式保存模型性能指标和配置
        - 特征信息：保存特征名称列表和重要性（如果可用）

        文件命名规则：
        - 模型文件：best_model_{model_name}.pkl
        - 信息文件：best_model_info.json

        Args:
            results: 模型训练结果字典
            best_model_name: 最佳模型名称
            feature_names: 特征名称列表

        Returns:
            Optional[str]: 成功保存时返回模型名称，失败时返回None
        """
        if not best_model_name or best_model_name not in results:
            return None
        
        best_result = results[best_model_name]
        
        # 显示最佳模型详细信息
        print(f"\n   最佳模型: {best_model_name}")
        print(f"   交叉验证 R² = {best_result['r2']:.4f} (±{best_result['r2_std']:.4f})")
        print(f"   交叉验证 MAE = {best_result['mae']:.3f} (±{best_result['mae_std']:.3f})")
        print(f"   交叉验证 RMSE = {best_result['rmse']:.3f} (±{best_result['rmse_std']:.3f})")
        
        # 显示最终评估指标（如果存在）
        if 'final_cv_r2' in best_result and not np.isnan(best_result['final_cv_r2']):
            print(f"   最终评估 R² = {best_result['final_cv_r2']:.4f} (±{best_result['final_cv_r2_std']:.4f})")
            print(f"   最终评估 MAE = {best_result['final_cv_mae']:.3f} (±{best_result['final_cv_mae_std']:.3f})")
            print(f"   最终评估 RMSE = {best_result['final_cv_rmse']:.3f} (±{best_result['final_cv_rmse_std']:.3f})")
        
        try:
            # 创建保存目录
            os.makedirs(self.config.results_dir, exist_ok=True)
            
            # 保存模型
            model_filename = f"best_model_{best_model_name.lower().replace(' ', '_')}.pkl"
            model_path = os.path.join(self.config.results_dir, model_filename)
            joblib.dump(best_result['model'], model_path)
            
            # 保存模型信息
            model_info = {
                'model_name': best_model_name,
                'cv_r2': best_result['r2'],
                'cv_mae': best_result['mae'],
                'cv_rmse': best_result['rmse'],
                'cv_r2_std': best_result['r2_std'],
                'cv_mae_std': best_result['mae_std'],
                'cv_rmse_std': best_result['rmse_std'],
                'n_features': len(feature_names),
                'feature_names': feature_names
            }
            
            # 添加最终评估信息（如果存在）
            if 'final_cv_r2' in best_result and not np.isnan(best_result['final_cv_r2']):
                model_info.update({
                    'final_cv_r2': best_result['final_cv_r2'],
                    'final_cv_mae': best_result['final_cv_mae'],
                    'final_cv_rmse': best_result['final_cv_rmse'],
                    'final_cv_r2_std': best_result['final_cv_r2_std'],
                    'final_cv_mae_std': best_result['final_cv_mae_std'],
                    'final_cv_rmse_std': best_result['final_cv_rmse_std']
                })
            
            model_info_path = os.path.join(self.config.results_dir, "best_model_info.csv")
            pd.DataFrame([model_info]).to_csv(model_info_path, index=False)
            
            print(f"\n   最佳模型已保存:")
            print(f"   模型文件: {model_path}")
            print(f"   模型信息: {model_info_path}")
            
            # 如果是XGBoost，保存额外的兼容性信息
            if best_model_name.lower() == 'xgboost':
                self.save_xgboost_model_for_compatibility(results, feature_names)
            
            return best_model_name
            
        except Exception as e:
            print(f"   保存模型时发生错误: {str(e)}")
            return None

    def save_xgboost_model_for_compatibility(self, results: Dict[str, Any], feature_names: List[str]) -> None:
        """为XGBoost模型保存额外的兼容性信息"""
        if 'XGBoost' not in results:
            return

        try:
            xgb_result = results['XGBoost']
            xgb_model = xgb_result['model']

            # 保存模型信息
            model_info = {
                'model_type': 'XGBoost',
                'n_estimators': xgb_model.n_estimators,
                'max_depth': xgb_model.max_depth,
                'learning_rate': xgb_model.learning_rate,
                'subsample': xgb_model.subsample,
                'colsample_bytree': xgb_model.colsample_bytree,
                'reg_alpha': xgb_model.reg_alpha,
                'reg_lambda': xgb_model.reg_lambda,
                'min_child_weight': xgb_model.min_child_weight,
                'gamma': xgb_model.gamma,
                'n_features': len(feature_names),
                'cv_r2': xgb_result['r2'],
                'cv_mae': xgb_result['mae'],
                'cv_rmse': xgb_result['rmse']
            }

            # 保存特征信息
            feature_info = pd.DataFrame({
                'feature': feature_names,
                'importance': xgb_model.feature_importances_
            })

            # 保存文件
            model_info_path = os.path.join(self.config.results_dir, "xgboost_model_info.csv")
            feature_info_path = os.path.join(self.config.results_dir, "xgboost_feature_importance.csv")

            pd.DataFrame([model_info]).to_csv(model_info_path, index=False)
            feature_info.to_csv(feature_info_path, index=False)

            print(f"   XGBoost兼容性文件已保存:")
            print(f"   模型信息: {model_info_path}")
            print(f"   特征重要性: {feature_info_path}")

        except Exception as e:
            print(f"   保存XGBoost兼容性信息时发生错误: {str(e)}")
