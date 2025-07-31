"""
模型测试模块

该模块负责对训练好的模型进行全面测试，包括性能评估、预测分析和可视化输出。
使用特征选择模块选出的最优特征集进行测试。
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# 导入可视化模块
try:
    from .visualization import ModelTestVisualizer
except ImportError:
    from visualization import ModelTestVisualizer

# 设置中文字体和警告过滤
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class ModelTester:
    """模型测试器"""

    def __init__(self,
                 model_path: str = "../../results/model_training/best_model_xgboost.pkl",
                 test_data_path: str = "../../data/processed/test_data.csv",
                 selected_features_path: str = "../../results/feature_selection/04_feature_selection/ga_selected_features.csv",
                 results_dir: str = "../../results/model_testing/",
                 scaler_path: str = "../../data/processed/scalers/robust_scaler.pkl"):
        """
        初始化模型测试器

        Args:
            model_path: 训练好的模型文件路径
            test_data_path: 测试数据文件路径
            selected_features_path: 选择特征文件路径
            results_dir: 结果保存目录
            scaler_path: 训练时使用的标准化器路径
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.selected_features_path = selected_features_path
        self.results_dir = results_dir
        self.scaler_path = scaler_path

        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)

        # 存储数据和结果
        self.model = None
        self.test_data = None
        self.selected_features = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.performance_metrics = {}
        self.scaler_info = None  # 存储标准化器信息

        # 创建可视化器
        self.visualizer = ModelTestVisualizer(self.results_dir)

        # 目标变量名称
        self.target_column = "lipid(%)"

        print("模型测试器初始化完成")
        print(f"模型路径: {self.model_path}")
        print(f"测试数据路径: {self.test_data_path}")
        print(f"选择特征路径: {self.selected_features_path}")
        print(f"标准化器路径: {self.scaler_path}")
        print(f"结果保存目录: {self.results_dir}")

    def load_model(self):
        """加载训练好的模型"""
        print("\n1. 加载训练好的模型...")

        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ 模型加载成功: {type(self.model).__name__}")

            # 如果模型有feature_names_in_属性，打印特征信息
            if hasattr(self.model, 'feature_names_in_'):
                print(f"  模型训练特征数: {len(self.model.feature_names_in_)}")

        except Exception as e:
            raise Exception(f"模型加载失败: {str(e)}")

    def load_scaler(self):
        """加载训练时使用的标准化器"""
        print("\n1.5. 加载训练时的标准化器...")

        try:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler_info = pickle.load(f)
                print(f"✓ 标准化器加载成功")
                print(f"  标准化特征数: {len(self.scaler_info['feature_cols'])}")
                print(f"  目标变量: {self.scaler_info['target_column']}")
                print("  ✓ 将确保测试数据使用相同的标准化参数")
            else:
                print(f"⚠ 警告: 标准化器文件不存在: {self.scaler_path}")
                print("  将跳过数据标准化步骤（可能导致性能下降）")
                self.scaler_info = None

        except Exception as e:
            print(f"⚠ 警告: 标准化器加载失败: {str(e)}")
            print("  将跳过数据标准化步骤")
            self.scaler_info = None

    def load_test_data(self):
        """加载测试数据"""
        print("\n2. 加载测试数据...")

        try:
            self.test_data = pd.read_csv(self.test_data_path)
            print(f"✓ 测试数据加载成功")
            print(f"  数据形状: {self.test_data.shape}")
            print(f"  列名: {list(self.test_data.columns)}")

            # 检查目标变量是否存在
            if self.target_column not in self.test_data.columns:
                raise ValueError(f"目标变量 '{self.target_column}' 不存在于测试数据中")

        except Exception as e:
            raise Exception(f"测试数据加载失败: {str(e)}")

    def load_selected_features(self):
        """加载选择的特征"""
        print("\n3. 加载选择的特征...")

        try:
            features_df = pd.read_csv(self.selected_features_path)
            self.selected_features = features_df['feature_name'].tolist()
            print(f"✓ 选择特征加载成功")
            print(f"  选择特征数: {len(self.selected_features)}")
            print(f"  特征列表: {self.selected_features}")

            # 检查特征是否存在于测试数据中
            missing_features = [f for f in self.selected_features if f not in self.test_data.columns]
            if missing_features:
                raise ValueError(f"以下特征不存在于测试数据中: {missing_features}")

        except Exception as e:
            raise Exception(f"选择特征加载失败: {str(e)}")

    def prepare_test_data(self):
        """准备测试数据并应用训练时的标准化"""
        print("\n4. 准备测试数据...")

        try:
            # 提取特征和目标变量
            X_test_df = self.test_data[self.selected_features].copy()
            self.y_test = self.test_data[self.target_column].values

            print(f"  原始测试数据形状: {X_test_df.shape}")
            print(f"  目标数据形状: {self.y_test.shape}")

            # 关键修复：应用训练时的标准化
            if self.scaler_info is not None:
                print("  🔧 应用训练时的标准化参数...")
                scaler = self.scaler_info['scaler']
                feature_cols = self.scaler_info['feature_cols']

                # 确保特征顺序和名称一致
                available_features = [f for f in feature_cols if f in X_test_df.columns]
                missing_features = [f for f in feature_cols if f not in X_test_df.columns]

                if missing_features:
                    print(f"  ⚠ 警告: 测试数据中缺少以下标准化特征: {missing_features}")

                if available_features:
                    # 应用相同的标准化变换
                    X_test_scaled = X_test_df.copy()
                    X_test_scaled[available_features] = scaler.transform(X_test_df[available_features])
                    self.X_test = X_test_scaled.values
                    print(f"  ✅ 成功应用标准化到 {len(available_features)} 个特征")
                    print("  📊 数据预处理一致性已确保")
                else:
                    print("  ❌ 错误: 没有可用的特征进行标准化")
                    self.X_test = X_test_df.values
            else:
                print("  ⚠ 跳过标准化步骤（标准化器不可用）")
                print("  ⚠ 这可能导致模型性能严重下降！")
                self.X_test = X_test_df.values

            print(f"✓ 测试数据准备完成")
            print(f"  最终特征数据形状: {self.X_test.shape}")

            # 检查数据中是否有缺失值
            if np.any(np.isnan(self.X_test)) or np.any(np.isnan(self.y_test)):
                print("⚠ 警告: 测试数据中存在缺失值")

            # 打印数据统计信息
            print(f"  目标变量统计:")
            print(f"    均值: {np.mean(self.y_test):.4f}")
            print(f"    标准差: {np.std(self.y_test):.4f}")
            print(f"    范围: [{np.min(self.y_test):.4f}, {np.max(self.y_test):.4f}]")

        except Exception as e:
            raise Exception(f"测试数据准备失败: {str(e)}")

    def make_predictions(self):
        """进行预测"""
        print("\n5. 进行模型预测...")

        try:
            self.y_pred = self.model.predict(self.X_test)
            print(f"✓ 预测完成")
            print(f"  预测结果形状: {self.y_pred.shape}")

            # 打印预测统计信息
            print(f"  预测值统计:")
            print(f"    均值: {np.mean(self.y_pred):.4f}")
            print(f"    标准差: {np.std(self.y_pred):.4f}")
            print(f"    范围: [{np.min(self.y_pred):.4f}, {np.max(self.y_pred):.4f}]")

        except Exception as e:
            raise Exception(f"模型预测失败: {str(e)}")

    def calculate_performance_metrics(self):
        """计算性能指标"""
        print("\n6. 计算性能指标...")

        try:
            # 计算各种回归指标
            r2 = r2_score(self.y_test, self.y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
            mae = mean_absolute_error(self.y_test, self.y_pred)

            # 计算MAPE（平均绝对百分比误差）
            # 避免除零错误
            non_zero_mask = self.y_test != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((self.y_test[non_zero_mask] - self.y_pred[non_zero_mask]) / self.y_test[non_zero_mask])) * 100
            else:
                mape = np.inf

            # 计算其他指标
            residuals = self.y_test - self.y_pred
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)

            # 计算相关系数
            correlation = np.corrcoef(self.y_test, self.y_pred)[0, 1]

            self.performance_metrics = {
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'Mean_Residual': mean_residual,
                'Std_Residual': std_residual,
                'Correlation': correlation,
                'Test_Samples': len(self.y_test)
            }

            print(f"✓ 性能指标计算完成")
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  相关系数: {correlation:.4f}")

        except Exception as e:
            raise Exception(f"性能指标计算失败: {str(e)}")

    def save_test_results(self):
        """保存测试结果"""
        print("\n7. 保存测试结果...")

        try:
            # 创建结果DataFrame
            results_df = pd.DataFrame({
                'Sample_Index': range(len(self.y_test)),
                'True_Value': self.y_test,
                'Predicted_Value': self.y_pred,
                'Residual': self.y_test - self.y_pred,
                'Absolute_Error': np.abs(self.y_test - self.y_pred),
                'Relative_Error_Percent': np.abs((self.y_test - self.y_pred) / self.y_test) * 100
            })

            # 保存预测结果
            results_path = os.path.join(self.results_dir, "test_predictions.csv")
            results_df.to_csv(results_path, index=False)
            print(f"✓ 预测结果已保存: {results_path}")

            # 保存性能指标
            metrics_df = pd.DataFrame([self.performance_metrics])
            metrics_path = os.path.join(self.results_dir, "performance_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
            print(f"性能指标已保存: {metrics_path}")

            # 保存详细报告
            self.save_detailed_report()

        except Exception as e:
            raise Exception(f"结果保存失败: {str(e)}")

    def save_detailed_report(self):
        """保存详细测试报告"""
        report_path = os.path.join(self.results_dir, "test_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("模型测试详细报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型类型: {type(self.model).__name__}\n")
            f.write(f"测试样本数: {len(self.y_test)}\n")
            f.write(f"使用特征数: {len(self.selected_features)}\n\n")

            f.write("使用的特征:\n")
            for i, feature in enumerate(self.selected_features, 1):
                f.write(f"  {i}. {feature}\n")
            f.write("\n")

            f.write("性能指标:\n")
            for metric, value in self.performance_metrics.items():
                if metric == 'MAPE':
                    f.write(f"  {metric}: {value:.2f}%\n")
                else:
                    f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")

            f.write("数据统计:\n")
            f.write(f"  真实值范围: [{np.min(self.y_test):.4f}, {np.max(self.y_test):.4f}]\n")
            f.write(f"  预测值范围: [{np.min(self.y_pred):.4f}, {np.max(self.y_pred):.4f}]\n")
            f.write(f"  真实值均值: {np.mean(self.y_test):.4f}\n")
            f.write(f"  预测值均值: {np.mean(self.y_pred):.4f}\n")
            f.write(f"  真实值标准差: {np.std(self.y_test):.4f}\n")
            f.write(f"  预测值标准差: {np.std(self.y_pred):.4f}\n")

        print(f"✓ 详细报告已保存: {report_path}")

    def diagnose_data_consistency(self):
        """诊断数据预处理一致性（用于验证修复效果）"""
        print("\n🔍 数据预处理一致性诊断:")

        # 检查标准化器状态
        if self.scaler_info is not None:
            print("✅ 标准化器已加载 - 数据预处理将保持一致")
            print(f"   标准化特征数: {len(self.scaler_info['feature_cols'])}")
        else:
            print("❌ 标准化器未加载 - 存在数据预处理不一致风险")

        # 检查特征一致性
        if hasattr(self.model, 'feature_names_in_') and self.selected_features:
            model_features = set(self.model.feature_names_in_)
            test_features = set(self.selected_features)

            if model_features == test_features:
                print("✅ 模型和测试特征完全一致")
            else:
                print("⚠️  模型和测试特征存在差异")
                missing = model_features - test_features
                extra = test_features - model_features
                if missing:
                    print(f"   测试中缺少: {missing}")
                if extra:
                    print(f"   测试中多余: {extra}")

        return self.scaler_info is not None

    def run_complete_test(self):
        """运行完整的模型测试流程"""
        print("🚀 开始修复后的模型测试流程...")
        print("=" * 60)

        try:
            # 执行测试步骤（关键修复：添加标准化器加载）
            self.load_model()
            self.load_scaler()  # 🔧 关键修复：加载训练时的标准化器
            self.load_test_data()
            self.load_selected_features()
            self.prepare_test_data()  # 现在会应用正确的标准化
            self.make_predictions()
            self.calculate_performance_metrics()
            self.save_test_results()

            # 生成可视化图表
            self.visualizer.create_all_visualizations(self.y_test, self.y_pred, self.performance_metrics)

            print("\n" + "=" * 60)
            print("✅ 修复后的模型测试完成！")
            print(f"📊 结果已保存到: {self.results_dir}")

            # 打印关键性能指标
            r2 = self.performance_metrics.get('R²', 'N/A')
            rmse = self.performance_metrics.get('RMSE', 'N/A')
            print(f"\n📈 关键性能指标:")
            print(f"   R² Score: {r2}")
            print(f"   RMSE: {rmse}")

            if isinstance(r2, float):
                if r2 >= 0:
                    print("✅ R²为正值，模型性能已改善！")
                else:
                    print("⚠️  R²仍为负值，可能需要进一步调整")

            return self.performance_metrics

        except Exception as e:
            print(f"\n❌ 测试过程中出现错误: {str(e)}")
            raise


def main():
    """主函数 - 修复后的模型测试"""
    print("🔧 启动修复后的模型测试程序")
    print("主要修复：确保训练和测试时数据预处理一致性")
    print("=" * 60)

    try:
        # 创建模型测试器
        tester = ModelTester()

        # 运行诊断检查
        print("\n🔍 运行预处理一致性诊断...")
        tester.load_model()
        tester.load_scaler()
        tester.load_test_data()
        tester.load_selected_features()

        # 诊断数据一致性
        is_consistent = tester.diagnose_data_consistency()

        if is_consistent:
            print("✅ 数据预处理一致性检查通过")
        else:
            print("⚠️  数据预处理存在不一致，可能影响性能")

        # 运行完整测试
        print("\n" + "=" * 60)
        metrics = tester.run_complete_test()

        print(f"\n📊 最终性能指标:")
        print("=" * 40)
        for metric, value in metrics.items():
            if metric == 'MAPE':
                print(f"  {metric}: {value:.2f}%")
            else:
                print(f"  {metric}: {value:.4f}")

        # 性能改善分析
        r2 = metrics.get('R²', 0)
        print(f"\n📈 性能分析:")
        if r2 > -0.2810:  # 原来的R²值
            improvement = r2 - (-0.2810)
            print(f"✅ R²相比修复前改善了 {improvement:.4f}")
        else:
            print("⚠️  R²仍需进一步改善")

    except Exception as e:
        print(f"❌ 程序执行失败: {str(e)}")
        print("请检查文件路径和数据完整性")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

    print("\n" + "="*60)
    print("模型测试完成!")
    print("="*60)