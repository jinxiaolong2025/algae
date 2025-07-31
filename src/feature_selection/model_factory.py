"""
模型工厂模块

该模块负责创建和管理用于特征选择和验证的机器学习模型。
"""

from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

try:
    from .config import FeatureSelectionConfig
    from .utils import check_xgboost_availability
except ImportError:
    from config import FeatureSelectionConfig
    from utils import check_xgboost_availability


class ModelFactory:
    """模型工厂类"""
    
    def __init__(self, config: FeatureSelectionConfig):
        """
        初始化模型工厂
        
        Args:
            config: 特征选择配置对象
        """
        self.config = config
        self.xgboost_available = check_xgboost_availability()
        
        if not self.xgboost_available:
            print("警告: XGBoost未安装，将跳过XGBoost相关功能")
    
    def create_evaluation_model(self, model_type: str = 'auto') -> Any:
        """
        创建用于遗传算法评估的模型
        
        Args:
            model_type: 模型类型 ('auto', 'xgboost', 'random_forest', 'gradient_boosting')
            
        Returns:
            机器学习模型实例
        """
        if model_type == 'auto':
            # 自动选择最佳可用模型
            if self.xgboost_available:
                model_type = 'xgboost'
            else:
                model_type = 'gradient_boosting'
        
        if model_type == 'xgboost' and self.xgboost_available:
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbosity=0
            )
        
        elif model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state
            )
        
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def create_validation_models(self) -> Dict[str, Any]:
        """
        创建用于验证的模型集合
        
        Returns:
            包含多个模型的字典
        """
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1
            ),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(
                alpha=1.0, 
                random_state=self.config.random_state
            ),
            'Lasso Regression': Lasso(
                alpha=0.1, 
                random_state=self.config.random_state,
                max_iter=2000
            ),
            'ElasticNet': ElasticNet(
                alpha=0.1, 
                l1_ratio=0.5, 
                random_state=self.config.random_state,
                max_iter=2000
            ),
            'SVR (RBF)': SVR(
                kernel='rbf', 
                C=1.0, 
                gamma='scale'
            ),
            'SVR (Linear)': SVR(
                kernel='linear', 
                C=1.0
            ),
            'Decision Tree': DecisionTreeRegressor(
                max_depth=self.config.max_depth,
                min_samples_split=5,
                random_state=self.config.random_state
            ),
            'KNN': KNeighborsRegressor(
                n_neighbors=5
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state
            )
        }
        
        # 添加XGBoost（如果可用）
        if self.xgboost_available:
            import xgboost as xgb
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbosity=0
            )
        
        return models
    
    def create_ensemble_models(self) -> Dict[str, Any]:
        """
        创建集成模型
        
        Returns:
            包含集成模型的字典
        """
        models = {}
        
        # 随机森林变体
        models['RF_Deep'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        models['RF_Wide'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        # 梯度提升变体
        models['GB_Conservative'] = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            random_state=self.config.random_state
        )
        
        models['GB_Aggressive'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.2,
            random_state=self.config.random_state
        )
        
        # XGBoost变体（如果可用）
        if self.xgboost_available:
            import xgboost as xgb
            
            models['XGB_Conservative'] = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.05,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbosity=0
            )
            
            models['XGB_Aggressive'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.2,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbosity=0
            )
        
        return models
    
    def create_linear_models(self) -> Dict[str, Any]:
        """
        创建线性模型集合
        
        Returns:
            包含线性模型的字典
        """
        models = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Alpha_0.1': Ridge(alpha=0.1, random_state=self.config.random_state),
            'Ridge_Alpha_1.0': Ridge(alpha=1.0, random_state=self.config.random_state),
            'Ridge_Alpha_10.0': Ridge(alpha=10.0, random_state=self.config.random_state),
            'Lasso_Alpha_0.01': Lasso(alpha=0.01, random_state=self.config.random_state, max_iter=2000),
            'Lasso_Alpha_0.1': Lasso(alpha=0.1, random_state=self.config.random_state, max_iter=2000),
            'Lasso_Alpha_1.0': Lasso(alpha=1.0, random_state=self.config.random_state, max_iter=2000),
            'ElasticNet_L1_0.1': ElasticNet(alpha=0.1, l1_ratio=0.1, random_state=self.config.random_state, max_iter=2000),
            'ElasticNet_L1_0.5': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.config.random_state, max_iter=2000),
            'ElasticNet_L1_0.9': ElasticNet(alpha=0.1, l1_ratio=0.9, random_state=self.config.random_state, max_iter=2000)
        }
        
        return models
    
    def create_nonlinear_models(self) -> Dict[str, Any]:
        """
        创建非线性模型集合
        
        Returns:
            包含非线性模型的字典
        """
        models = {
            'SVR_RBF_C_0.1': SVR(kernel='rbf', C=0.1, gamma='scale'),
            'SVR_RBF_C_1.0': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'SVR_RBF_C_10.0': SVR(kernel='rbf', C=10.0, gamma='scale'),
            'SVR_Poly_Degree_2': SVR(kernel='poly', degree=2, C=1.0),
            'SVR_Poly_Degree_3': SVR(kernel='poly', degree=3, C=1.0),
            'KNN_3': KNeighborsRegressor(n_neighbors=3),
            'KNN_5': KNeighborsRegressor(n_neighbors=5),
            'KNN_7': KNeighborsRegressor(n_neighbors=7),
            'Decision_Tree_Depth_3': DecisionTreeRegressor(
                max_depth=3, 
                min_samples_split=5, 
                random_state=self.config.random_state
            ),
            'Decision_Tree_Depth_5': DecisionTreeRegressor(
                max_depth=5, 
                min_samples_split=5, 
                random_state=self.config.random_state
            ),
            'Decision_Tree_Depth_10': DecisionTreeRegressor(
                max_depth=10, 
                min_samples_split=5, 
                random_state=self.config.random_state
            )
        }
        
        return models
    
    def get_model_by_name(self, model_name: str) -> Any:
        """
        根据名称获取特定模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型实例
        """
        all_models = {}
        all_models.update(self.create_validation_models())
        all_models.update(self.create_ensemble_models())
        all_models.update(self.create_linear_models())
        all_models.update(self.create_nonlinear_models())
        
        if model_name not in all_models:
            available_models = list(all_models.keys())
            raise ValueError(f"未知的模型名称: {model_name}. 可用模型: {available_models}")
        
        return all_models[model_name]
    
    def get_recommended_evaluation_model(self) -> str:
        """
        获取推荐的评估模型名称
        
        Returns:
            推荐的模型名称
        """
        if self.xgboost_available:
            return 'xgboost'
        else:
            return 'gradient_boosting'
    
    def print_available_models(self):
        """打印所有可用的模型"""
        print("可用的模型:")
        print("=" * 50)
        
        print("\n验证模型:")
        validation_models = self.create_validation_models()
        for i, model_name in enumerate(validation_models.keys(), 1):
            print(f"  {i:2d}. {model_name}")
        
        print("\n集成模型:")
        ensemble_models = self.create_ensemble_models()
        for i, model_name in enumerate(ensemble_models.keys(), 1):
            print(f"  {i:2d}. {model_name}")
        
        print("\n线性模型:")
        linear_models = self.create_linear_models()
        for i, model_name in enumerate(linear_models.keys(), 1):
            print(f"  {i:2d}. {model_name}")
        
        print("\n非线性模型:")
        nonlinear_models = self.create_nonlinear_models()
        for i, model_name in enumerate(nonlinear_models.keys(), 1):
            print(f"  {i:2d}. {model_name}")
        
        print("=" * 50)
