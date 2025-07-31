"""
遗传算法核心模块

该模块实现了用于特征选择的遗传算法，包括种群初始化、选择、交叉、变异等操作。
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Callable
from sklearn.model_selection import cross_val_score, KFold

try:
    from .config import FeatureSelectionConfig, GeneticAlgorithmError
    from .utils import (
        calculate_population_diversity,
        calculate_convergence_metrics,
        calculate_selection_pressure,
        detect_premature_convergence,
        validate_input_data,
        create_progress_bar
    )
except ImportError:
    from config import FeatureSelectionConfig, GeneticAlgorithmError
    from utils import (
        calculate_population_diversity,
        calculate_convergence_metrics,
        calculate_selection_pressure,
        detect_premature_convergence,
        validate_input_data,
        create_progress_bar
    )


class GeneticAlgorithmFeatureSelector:
    """基于遗传算法的特征选择器"""
    
    def __init__(self, config: FeatureSelectionConfig):
        """
        初始化遗传算法特征选择器
        
        Args:
            config: 特征选择配置对象
        """
        self.config = config
        
        # 设置随机种子
        random.seed(config.random_state)
        np.random.seed(config.random_state)
        
        # 存储进化过程数据
        self.fitness_history = []
        self.best_fitness_history = []
        self.diversity_history = []
        self.selection_pressure_history = []
        self.convergence_metrics = {}
        
        # 存储最佳结果
        self.best_individual_ = None
        self.best_fitness_ = -np.inf
        self.selected_features_ = None
        
        # 进化统计
        self.generation_stats = []
    
    def initialize_population(self, n_features: int) -> np.ndarray:
        """
        初始化种群
        
        Args:
            n_features: 特征总数
            
        Returns:
            初始化的种群数组
        """
        if n_features < self.config.target_features:
            raise GeneticAlgorithmError(
                f"特征总数({n_features})小于目标特征数({self.config.target_features})"
            )
        
        population = []
        for _ in range(self.config.population_size):
            # 随机选择target_features个特征
            individual = np.zeros(n_features, dtype=bool)
            selected_indices = np.random.choice(
                n_features, 
                self.config.target_features, 
                replace=False
            )
            individual[selected_indices] = True
            population.append(individual)
        
        return np.array(population)
    
    def calculate_fitness(self, individual: np.ndarray, X: np.ndarray, 
                         y: np.ndarray, model) -> float:
        """
        计算个体适应度
        
        Args:
            individual: 个体（特征选择掩码）
            X: 特征数据
            y: 目标变量
            model: 评估模型
            
        Returns:
            适应度值（R²分数）
        """
        # 获取选中的特征
        selected_features = X[:, individual]
        
        if selected_features.shape[1] == 0:
            return 0.0
        
        try:
            # 使用交叉验证评估模型性能
            kf = KFold(
                n_splits=self.config.cv_folds, 
                shuffle=True, 
                random_state=self.config.random_state
            )
            scores = cross_val_score(model, selected_features, y, cv=kf, scoring='r2')
            
            # 返回平均R²分数作为适应度
            fitness = np.mean(scores)
            
            # 如果R²为负数，设为0
            return max(0.0, fitness)
            
        except Exception as e:
            print(f"计算适应度时出错: {e}")
            return 0.0
    
    def selection(self, population: np.ndarray,
                  fitness_scores: np.ndarray) -> np.ndarray:
        """
        选择操作（锦标赛选择）

        Args:
            population: 当前种群
            fitness_scores: 适应度分数

        Returns:
            选择的个体数组
        """
        return self.tournament_selection(population, fitness_scores)

    def tournament_selection(self, population: np.ndarray,
                           fitness_scores: np.ndarray) -> np.ndarray:
        """
        锦标赛选择
        
        Args:
            population: 当前种群
            fitness_scores: 适应度分数
            
        Returns:
            选择的个体数组
        """
        selected = []
        
        for _ in range(self.config.population_size - self.config.elite_size):
            # 锦标赛选择
            tournament_indices = np.random.choice(
                len(population), 
                self.config.tournament_size, 
                replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return np.array(selected)
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        交叉操作（均匀交叉）

        Args:
            parent1: 父代1
            parent2: 父代2

        Returns:
            两个子代个体
        """
        return self.uniform_crossover(parent1, parent2)

    def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        均匀交叉操作
        
        Args:
            parent1: 父代1
            parent2: 父代2
            
        Returns:
            两个子代个体
        """
        if np.random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # 均匀交叉
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        
        # 确保每个子代都有正确数量的特征
        child1 = self.repair_individual(child1)
        child2 = self.repair_individual(child2)
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        变异操作（交换变异）

        Args:
            individual: 个体

        Returns:
            变异后的个体
        """
        return self.swap_mutation(individual)

    def swap_mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        交换变异操作
        
        Args:
            individual: 个体
            
        Returns:
            变异后的个体
        """
        if np.random.random() > self.config.mutation_rate:
            return individual
        
        mutated = individual.copy()
        
        # 随机选择一个选中的特征和一个未选中的特征进行交换
        selected_indices = np.where(mutated)[0]
        unselected_indices = np.where(~mutated)[0]
        
        if len(selected_indices) > 0 and len(unselected_indices) > 0:
            # 随机选择要交换的特征
            selected_idx = np.random.choice(selected_indices)
            unselected_idx = np.random.choice(unselected_indices)
            
            # 交换
            mutated[selected_idx] = False
            mutated[unselected_idx] = True
        
        return mutated
    
    def repair_individual(self, individual: np.ndarray) -> np.ndarray:
        """
        修复个体，确保有正确数量的特征
        
        Args:
            individual: 需要修复的个体
            
        Returns:
            修复后的个体
        """
        selected_count = np.sum(individual)
        
        if selected_count == self.config.target_features:
            return individual
        
        repaired = individual.copy()
        
        if selected_count < self.config.target_features:
            # 需要添加特征
            unselected_indices = np.where(~repaired)[0]
            need_to_add = self.config.target_features - selected_count
            if len(unselected_indices) >= need_to_add:
                add_indices = np.random.choice(unselected_indices, need_to_add, replace=False)
                repaired[add_indices] = True
        
        elif selected_count > self.config.target_features:
            # 需要移除特征
            selected_indices = np.where(repaired)[0]
            need_to_remove = selected_count - self.config.target_features
            remove_indices = np.random.choice(selected_indices, need_to_remove, replace=False)
            repaired[remove_indices] = False
        
        return repaired
    
    def elite_preservation(self, population: np.ndarray, 
                          fitness_scores: np.ndarray) -> np.ndarray:
        """
        精英保留策略
        
        Args:
            population: 当前种群
            fitness_scores: 适应度分数
            
        Returns:
            精英个体数组
        """
        elite_indices = np.argsort(fitness_scores)[-self.config.elite_size:]
        return population[elite_indices]
    
    def evaluate_population(self, population: np.ndarray, X: np.ndarray, 
                          y: np.ndarray, model) -> np.ndarray:
        """
        评估整个种群的适应度
        
        Args:
            population: 种群
            X: 特征数据
            y: 目标变量
            model: 评估模型
            
        Returns:
            适应度分数数组
        """
        fitness_scores = []
        for individual in population:
            fitness = self.calculate_fitness(individual, X, y, model)
            fitness_scores.append(fitness)
        
        return np.array(fitness_scores)

    def calculate_diversity(self, population: np.ndarray) -> float:
        """
        计算种群多样性（兼容原版本接口）

        Args:
            population: 种群

        Returns:
            多样性值
        """
        return calculate_population_diversity(population)
    
    def record_generation_stats(self, generation: int, population: np.ndarray, 
                              fitness_scores: np.ndarray):
        """
        记录代数统计信息
        
        Args:
            generation: 当前代数
            population: 当前种群
            fitness_scores: 适应度分数
        """
        # 基本统计
        mean_fitness = np.mean(fitness_scores)
        best_fitness = np.max(fitness_scores)
        diversity = calculate_population_diversity(population)
        selection_pressure = calculate_selection_pressure(fitness_scores)
        
        # 更新历史记录
        self.fitness_history.append(mean_fitness)
        self.best_fitness_history.append(best_fitness)
        self.diversity_history.append(diversity)
        self.selection_pressure_history.append(selection_pressure)
        
        # 记录详细统计
        stats = {
            'generation': generation,
            'mean_fitness': mean_fitness,
            'best_fitness': best_fitness,
            'worst_fitness': np.min(fitness_scores),
            'std_fitness': np.std(fitness_scores),
            'diversity': diversity,
            'selection_pressure': selection_pressure
        }
        self.generation_stats.append(stats)
    
    def check_termination_criteria(self, generation: int) -> bool:
        """
        检查终止条件
        
        Args:
            generation: 当前代数
            
        Returns:
            是否应该终止
        """
        # 达到最大代数
        if generation >= self.config.generations - 1:
            return True
        
        # 检查过早收敛
        if detect_premature_convergence(self.best_fitness_history):
            print(f"检测到过早收敛，在第{generation+1}代终止")
            return True
        
        return False

    def fit(self, X: np.ndarray, y: np.ndarray, model=None):
        """
        运行遗传算法进行特征选择

        Args:
            X: 特征数据
            y: 目标变量
            model: 评估模型

        Returns:
            self
        """
        # 验证输入数据
        validate_input_data(X, y)

        if model is None:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state
            )

        print("开始遗传算法特征选择...")
        print(f"种群大小: {self.config.population_size}")
        print(f"进化代数: {self.config.generations}")
        print(f"目标特征数: {self.config.target_features}")
        print(f"总特征数: {X.shape[1]}")
        print("=" * 60)

        # 初始化种群
        population = self.initialize_population(X.shape[1])

        # 存储最佳个体
        best_individual = None
        best_fitness = -np.inf

        for generation in range(self.config.generations):
            # 评估种群适应度
            fitness_scores = self.evaluate_population(population, X, y, model)

            # 更新最佳个体
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()

            # 记录统计信息
            self.record_generation_stats(generation, population, fitness_scores)

            # 打印进度
            if generation % 10 == 0 or generation == self.config.generations - 1:
                progress_bar = create_progress_bar(generation + 1, self.config.generations)
                print(f"第 {generation+1:3d} 代: {progress_bar}")
                print(f"         平均适应度={np.mean(fitness_scores):.4f}, "
                      f"最佳适应度={best_fitness:.4f}, 多样性={self.diversity_history[-1]:.2f}")

            # 检查终止条件
            if self.check_termination_criteria(generation):
                break

            # 进化操作
            if generation < self.config.generations - 1:
                # 精英保留
                elite_population = self.elite_preservation(population, fitness_scores)

                # 选择
                selected_population = self.selection(population, fitness_scores)

                # 生成新种群
                new_population = []

                # 保留精英
                for elite in elite_population:
                    new_population.append(elite.copy())

                # 交叉和变异生成新个体
                while len(new_population) < self.config.population_size:
                    if len(new_population) < self.config.population_size - 1:
                        # 选择两个父代
                        parent1 = selected_population[np.random.randint(len(selected_population))]
                        parent2 = selected_population[np.random.randint(len(selected_population))]

                        # 交叉
                        child1, child2 = self.crossover(parent1, parent2)

                        # 变异
                        child1 = self.mutate(child1)
                        child2 = self.mutate(child2)

                        new_population.append(child1)
                        if len(new_population) < self.config.population_size:
                            new_population.append(child2)
                    else:
                        # 只需要一个个体
                        parent = selected_population[np.random.randint(len(selected_population))]
                        child = self.mutate(parent.copy())
                        new_population.append(child)

                population = np.array(new_population)

        # 保存结果
        self.best_individual_ = best_individual
        self.best_fitness_ = best_fitness
        self.selected_features_ = np.where(best_individual)[0]

        # 计算收敛指标
        self.convergence_metrics = calculate_convergence_metrics(self.best_fitness_history)

        print("=" * 60)
        print(f"遗传算法完成！最佳适应度: {best_fitness:.4f}")
        print(f"选择的特征数量: {len(self.selected_features_)}")
        print(f"收敛指标: {self.convergence_metrics}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        使用选择的特征转换数据

        Args:
            X: 输入数据

        Returns:
            转换后的数据
        """
        if not hasattr(self, 'selected_features_'):
            raise ValueError("请先运行fit方法进行特征选择")

        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, model=None) -> np.ndarray:
        """
        拟合并转换数据

        Args:
            X: 特征数据
            y: 目标变量
            model: 评估模型

        Returns:
            转换后的数据
        """
        return self.fit(X, y, model).transform(X)

    def get_selected_features(self) -> np.ndarray:
        """
        获取选择的特征索引

        Returns:
            选择的特征索引数组
        """
        if not hasattr(self, 'selected_features_'):
            raise ValueError("请先运行fit方法进行特征选择")

        return self.selected_features_

    def get_evolution_history(self) -> dict:
        """
        获取进化历史数据

        Returns:
            包含进化历史的字典
        """
        return {
            'fitness_history': self.fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'diversity_history': self.diversity_history,
            'selection_pressure_history': self.selection_pressure_history,
            'generation_stats': self.generation_stats,
            'convergence_metrics': self.convergence_metrics
        }
