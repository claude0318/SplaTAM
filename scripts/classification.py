import numpy as np
from collections import defaultdict
import torch

class GaussianManager:
    def __init__(self, params):
        """
        初始化 GaussianManager 类，并构建基于类别的字典。
        :param params: 一个字典，包含所有 Gaussian 的参数信息。
        """
        self.params = params
        self.gaussians_by_category = defaultdict(list)
        self._initialize_gaussians()

    def _initialize_gaussians(self):
        """根据初始的 params['sem'] 将 Gaussians 分类存储到字典中。"""
        sem_categories = self.params['sem']
        for i in range(sem_categories.shape[0]):
            category = tuple(sem_categories[i].tolist())
            if category not in self.gaussians_by_category:
                self.gaussians_by_category[category] = []
            self.gaussians_by_category[category].append(i)

    def find_gaussians_by_category(self, target_category, epsilon=1e-3):
        """
        查找接近目标类别的 Gaussians。
        :param target_category: 目标类别 RGB 值，列表或元组形式，例如 [0.502, 0.0, 0.251]
        :param epsilon: 允许的浮点数比较容差
        :return: 属于该类别的 Gaussians 的索引
        """
        target_category = torch.tensor(target_category, device=self.params['sem'].device)
        diffs = torch.abs(self.params['sem'] - target_category)
        mask = torch.all(diffs < epsilon, dim=1)
        indices = torch.nonzero(mask).squeeze()
        return indices
    
    def update_gaussians(self, new_params):
        """更新分类字典"""
        current_size = self.params['sem'].shape[0]
        for i in range(new_params['sem'].shape[0]):
            category = tuple(new_params['sem'][i].tolist())
            if category not in self.gaussians_by_category:
                self.gaussians_by_category[category] = []
            self.gaussians_by_category[category].append(current_size + i)

    # def update_gaussian_category(self, index, new_category):
    #     """
    #     更新某个 Gaussian 的类别信息，并在字典中移动它。
    #     :param index: 需要更新的 Gaussian 在 params 中的索引。
    #     :param new_category: 新的类别，类型为 [R, G, B]。
    #     """
    #     old_category = tuple(self.params['sem'][index])

    #     # 从旧类别中移除
    #     self.gaussians_by_category[old_category].remove(index)
    #     if not self.gaussians_by_category[old_category]:
    #         del self.gaussians_by_category[old_category]

    #     # 更新 Gaussian 的类别
    #     self.params['sem'][index] = new_category

    #     # 添加到新类别
    #     new_category_tuple = tuple(new_category)
    #     self.gaussians_by_category[new_category_tuple].append(index)

    def remove_gaussians_by_indices(self, indices):
        """
        从管理器中删除指定索引的 Gaussians。
        :param indices: 需要删除的 Gaussians 的索引列表。
        """
        # 如果 indices 是一个标量，将其转换为一个包含单个元素的列表
        if isinstance(indices, np.ndarray) and indices.ndim == 0:
            indices = [indices.item()]
        elif isinstance(indices, int):
            indices = [indices]
        else:
            indices = sorted(indices, reverse=True)  # 反向排序，保证在移除时索引不会错位

        for index in indices:
            category = tuple(self.params['sem'][index])

            # 从类别列表中移除
            if category in self.gaussians_by_category:
                self.gaussians_by_category[category].remove(index)
                if not self.gaussians_by_category[category]:  # 如果这个类别没有 Gaussians 了，删除这个类别
                    del self.gaussians_by_category[category]

    def get_gaussians_by_category(self, category):
        """
        获取特定类别的 Gaussians 的索引列表。
        :param category: 类别的 RGB 值 [R, G, B]。
        :return: 属于该类别的 Gaussians 的索引列表。
        """
        category_tuple = tuple(category)
        return self.gaussians_by_category.get(category_tuple, [])

    def get_gaussian_params_by_category(self, category):
        """
        获取特定类别的 Gaussians 的参数。
        :param category: 类别的 RGB 值 [R, G, B]。
        :return: 属于该类别的 Gaussians 的参数。
        """
        indices = self.get_gaussians_by_category(category)
        return {key: self.params[key][indices] for key in self.params}

