from collections import defaultdict
import torch
from torch.nn import Parameter

class Gaussian:
    def __init__(self, mean3d, color, rotation, opacity, scale, cam_rot, cam_trans, sem):
        self.mean3d = mean3d
        self.color = color
        self.rotation = rotation
        self.opacity = opacity
        self.scale = scale
        self.cam_rot = cam_rot
        self.cam_trans = cam_trans
        self.sem = sem  # 将类别信息存储在对象中

    def update_sem(self, new_sem):
        """更新高斯对象的类别"""
        self.sem = new_sem

    def __repr__(self):
        return (f"Gaussian(Mean3D: {self.mean3d}, Color: {self.color}, Rotation: {self.rotation}, "
                f"Opacity: {self.opacity}, Scale: {self.scale}, "
                f"CamRot: {self.cam_rot}, CamTrans: {self.cam_trans}, Sem: {self.sem})")

class GaussianManager:
    def __init__(self):
        self.gaussians = []
        self.category_index = defaultdict(list)

    def initialize_or_update_gaussians(self, params):
        means3D = params['means3D']
        rgb_colors = params['rgb_colors']
        unnorm_rotations = params['unnorm_rotations']
        logit_opacities = params['logit_opacities']
        log_scales = params['log_scales']
        cam_unnorm_rots = params['cam_unnorm_rots']
        cam_trans = params['cam_trans']
        sem = params['Sem']

        num_gaussians = means3D.size(0)

        if not self.gaussians:
            # 首次初始化
            for i in range(num_gaussians):
                gaussian = Gaussian(
                    mean3d=means3D[i],
                    color=rgb_colors[i],
                    rotation=unnorm_rotations[i],
                    opacity=logit_opacities[i],
                    scale=log_scales[i],
                    cam_rot=cam_unnorm_rots[:, i],
                    cam_trans=cam_trans[:, i],
                    sem=sem[i]
                )
                self.gaussians.append(gaussian)
                category = sem[i].item()
                self.category_index[category].append(i)
        else:
            # 仅更新已有对象的参数
            for i in range(num_gaussians):
                gaussian = self.gaussians[i]
                gaussian.mean3d = means3D[i]
                gaussian.color = rgb_colors[i]
                gaussian.rotation = unnorm_rotations[i]
                gaussian.opacity = logit_opacities[i]
                gaussian.scale = log_scales[i]
                gaussian.cam_rot = cam_unnorm_rots[:, i]
                gaussian.cam_trans = cam_trans[:, i]
                # 检查和更新类别
                old_category = gaussian.sem.item()
                new_category = sem[i].item()
                if old_category != new_category:
                    self.category_index[old_category].remove(i)
                    if not self.category_index[old_category]:
                        del self.category_index[old_category]
                    self.category_index[new_category].append(i)
                # 更新高斯对象的类别
                gaussian.update_sem(sem[i])


    def __repr__(self):
        return "\n".join([f"Category: {cat}, Number of Gaussians: {len(indices)}"
                          for cat, indices in self.category_index.items()])
    
    
    # TODO： 更新已存在的gaussian的类别（删除类别发生改变的gaussian）
    # TODO:  给新增的gaussian分配类别
    # TODO： 边缘部分的gaussian如何精细化处理
