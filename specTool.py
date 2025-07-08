import torch
# import os
import numpy as np


def equivalent_mu(spec1,spec2,atten_m1,atten_m2):
    # 计算不同能谱下不同物质的等效线性衰减系数
    mu_m1E1 = - np.log((spec1 * np.exp(-atten_m1[:56])).sum())
    mu_m2E1 = - np.log((spec1 * np.exp(-atten_m2[:56])).sum())
    mu_m1E2 = - np.log((spec2 * np.exp(-atten_m1)).sum())
    mu_m2E2 = - np.log((spec2 * np.exp(-atten_m2)).sum())
    return np.array(([mu_m1E1, mu_m2E1],[mu_m1E2,mu_m2E2]))

def ls_mu(m1,m2,ct_img):
    # 最小二乘法拟合系数，但是偷看数据
    # y = k1 * m1 + k2 * m2
    x1_flat = m1.ravel()
    x2_flat = m2.ravel()
    y_flat = ct_img.ravel()

    # 构建设计矩阵 X
    X = np.vstack((x1_flat, x2_flat)).T
    # 使用最小二乘法求解 k1, k2
    k, _, _, _ = np.linalg.lstsq(X, y_flat, rcond=None)

    return k
def DEsino_Syn(m1,m2,spec,atten_m1,atten_m2,projector):
    # 计算由m1,m2组成的物体在给定能谱下的弦图
    projector.train()
    # m1(256, 256), m2(256, 256), spec_E1(x, ), atten_m1(x, ), atten_m2(x, )
    # 注意spec大小要跟atten_m1和atten_m2大小一致，且均为32位tensor
    m1m2 = torch.stack((m1, m2)).reshape(2, 1, 256, 256)
    sino_m1m2 = projector(m1m2)[:, :, 0, :] / 1000
    # sino_m2 = projector(m2.reshape(1, 1, 256, 256))[0, :, 0, :].clone()
    # sino = torch.zeros_like(sino_m1)
    atten_m1 = atten_m1.unsqueeze(0).unsqueeze(-1)  # 形状变为 (1, 56, 1)
    atten_m2 = atten_m2.unsqueeze(0).unsqueeze(-1)  # 形状变为 (1, 56, 1)
    spec = spec.unsqueeze(0).unsqueeze(-1)
    attenuation = atten_m1 * sino_m1m2[0].unsqueeze(1) + atten_m2 * sino_m1m2[1].unsqueeze(1)  # (1000, 56, 384) 算出e指数上的
    exponent = torch.exp(-attenuation)  # (1000, 56, 384)
    spec_integration = (spec * exponent).sum(dim=1) + 1e-9  # (1000, 384)
    sino = -torch.log(spec_integration )
    return sino

def DECT_Syn(m1,m2,spec,atten_m1,atten_m2,projector_FP,projector_FBP, img_size = 256):
    # 计算由m1,m2组成的物体在给定能谱下的CT图
    # m1(256, 256), m2(256, 256), spec_E1(x, ), atten_m1(x, ), atten_m2(x, )
    # 注意spec大小要跟atten_m1和atten_m2大小一致，且均为32位tensor
    projector_FP.train()
    projector_FBP.train()
    m1m2 = torch.stack((m1, m2)).reshape(2, 1, img_size, img_size)
    sino_m1m2 = projector_FP(m1m2)[:, :, 0, :] / 1000
    # sino_m2 = projector(m2.reshape(1, 1, 256, 256))[0, :, 0, :].clone()
    # sino = torch.zeros_like(sino_m1)
    atten_m1 = atten_m1.unsqueeze(0).unsqueeze(-1)  # 形状变为 (1, 56, 1)
    atten_m2 = atten_m2.unsqueeze(0).unsqueeze(-1)  # 形状变为 (1, 56, 1)
    spec = spec.unsqueeze(0).unsqueeze(-1)
    attenuation = atten_m1 * sino_m1m2[0].unsqueeze(1) + atten_m2 * sino_m1m2[1].unsqueeze(1)  # (1000, 56, 384) 算出e指数上的
    exponent = torch.exp(-attenuation)  # (1000, 56, 384)
    spec_integration = (spec * exponent).sum(dim=1) + 1e-9  # (1000, 384)
    sino = -torch.log(spec_integration)
    CT = projector_FBP(sino.unsqueeze(1).unsqueeze(0))[0,0,:,:]
    return CT

