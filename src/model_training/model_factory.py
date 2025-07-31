"""模型工厂模块

包含模型工厂类，负责创建和配置各种机器学习模型
"""
import warnings
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

try:
    from .config import TrainingConfig
except ImportError:
    from config import TrainingConfig

warnings.filterwarnings('ignore')


class ModelFactory:
    """模型工厂类，负责创建和配置各种机器学习模型"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def create_model_suite(self) -> Dict[str, Any]:
        """创建多种模型进行对比

        Returns:
            Dict[str, Any]: 包含各种配置好的模型的字典
        """
        models = {}

        # XGBoost模型
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=45,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_alpha=0.7,
            reg_lambda=1.2,
            min_child_weight=3,
            gamma=0.3,
            random_state=self.config.model_random_state,
            n_jobs=1,
            verbosity=0
        )

        # 集成学习模型
        models.update({
            'Random Forest': RandomForestRegressor(
                n_estimators=30,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=1,
                max_features=0.6,
                random_state=self.config.model_random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.2,
                subsample=0.6,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=self.config.model_random_state
            ),
            'Support Vector Regression': SVR(
                kernel='rbf',
                C=3.0,
                gamma='scale',
                epsilon=0.2
            ),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(
                alpha=3.0,
                random_state=self.config.model_random_state
            ),
            'Lasso Regression': Lasso(
                alpha=0.5,
                random_state=self.config.model_random_state,
                max_iter=2000
            ),
            'ElasticNet': ElasticNet(
                alpha=0.5,
                l1_ratio=0.5,
                random_state=self.config.model_random_state,
                max_iter=2000
            ),
            'Decision Tree': DecisionTreeRegressor(
                max_depth=4,
                min_samples_split=12,
                min_samples_leaf=6,
                max_features=0.6,
                random_state=self.config.model_random_state
            ),
            'K-Nearest Neighbors': KNeighborsRegressor(
                n_neighbors=8,
                weights='uniform',
                metric='manhattan'
            )
        })

        return models
