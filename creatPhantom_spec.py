import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
import datetime
import astra
from skimage.metrics import normalized_root_mse as nrmse
from scipy.ndimage import distance_transform_edt, map_coordinates
import tifffile

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from leaptorch import Projector
from specTool import *

def mean_vale(x, y, img, window_size):
    half_window_size = window_size // 2
    window = img[x - half_window_size:x + half_window_size + 1, y - half_window_size:y + half_window_size + 1]
    non_zero_elements = window[window.nonzero()]
    mean_val = non_zero_elements.mean()
    return mean_val

# def replace_with_nearest(img, mask):
#     dist, coords = distance_transform_edt(mask, return_distances=True, return_indices=True)
#     img_replaced = map_coordinates(img, coords, order=0)
#     return img_replaced
slice_num = 35
spec_low = torch.from_numpy(np.load('raw_data/spec_data/spec_low.npy')[25:81]).float().cuda()  # 80kV
spec_high = torch.from_numpy(np.load('raw_data/spec_data/spec_high.npy')[25:141]).float().cuda()  # 140kV
atten_iodine = torch.from_numpy(np.load('raw_data/spec_data/atten_iodine.npy')).float().cuda()
atten_water = torch.from_numpy(np.load('raw_data/spec_data/atten_water.npy')).float().cuda()
mask = tifffile.imread('raw_data/DECT_human/mask_ROI_noBone.tif')[70+slice_num, :, :] / 255
# mask = np.load('generated_data/slice35brain/mask.npy')  # 256大小的mask
baseline_water = np.load(f'generated_data/spec{slice_num}/water1500GT.npy')* mask  # 选定（256，256）的区域
baseline_iodine = np.load(f'generated_data/spec{slice_num}/iodine1500GT.npy')* mask
# vessel_mask = tifffile.imread(f'generated_data/slice{slice_num}brain/vessel_mask36.tif') / 255
vessel_mask = tifffile.imread('raw_data/DECT_human/vessel_waterMask.tif')[slice_num] / 255
numAngles = 1000
numRows = 1
numCols = 384
pixelSize = 1
proj_2d = Projector(forward_project=True, use_static=True, use_gpu=True, gpu_device=torch.device('cuda:0'),
                    batch_size=2)
proj_2d.leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5 * (numRows - 1),
                                0.5 * (numCols - 1), proj_2d.leapct.setAngleArray(numAngles, 180.0))
proj_2d.leapct.set_volume(256, 256, 1, pixelSize * 384.0 / 256.0, pixelSize)
proj_2d.allocate_batch_data()
# E_spec_low = ((spec_low * (torch.arange(56).cuda() + 25))/spec_high.sum()).sum().item()  # 80kV平均能量
# E_spec_high = ((spec_high * (torch.arange(116).cuda() + 25))/spec_high.sum()).sum().item()  # 140kV平均能量
# mu_water_low_avg = (atten_water[:56] * spec_low).sum().item()
# mu_water_high_avg = (atten_water * spec_high).sum().item()
# mu_iodine_low_avg = (atten_iodine[:56] * spec_low).sum().item()
# mu_iodine_high_avg = (atten_iodine * spec_high).sum().item()
# mu_loader = np.array([[mu_water_low_avg, mu_iodine_low_avg],[mu_water_high_avg, mu_iodine_high_avg]])
# decom_matrix = np.linalg.inv(mu_loader)
# np.dot(decom_matrix, mu_loader)
# mu_water_E1 = 0.0319  # 60 keV
# mu_water_E2 = 0.0259  # 82 keV
# mu_iodine_E1 = 4.518
# mu_iodine_E2 = 2.321
# matrix_avg = np.array([[0.0319,4.518],[0.0259,2.321 ]])
# decom_avg = np.linalg.inv(matrix_avg)
# np.dot(matrix_avg, decom_avg)
E1CT = proj_2d.fbp(DEsino_Syn(torch.from_numpy(baseline_water).float().cuda(), torch.from_numpy(baseline_iodine).float().cuda(),spec_high,atten_water, atten_iodine,proj_2d).unsqueeze(0).unsqueeze(0)).cpu().numpy()[0][0]

T0_water = baseline_water
T0_iodine = np.where((vessel_mask == 0) & (E1CT > 0) & (baseline_iodine > 0), baseline_iodine * 0.8, baseline_iodine)  # 不替血管用这两句。 软组织部分碘浓度变为2.5倍
T0_iodine = np.where(vessel_mask == 1, T0_iodine * 0.8, T0_iodine)  # 血管部分碘浓度变为4倍
T3000_iodine = np.where((vessel_mask == 0) & (E1CT > 0) & (baseline_iodine > 0), baseline_iodine * 1.2, baseline_iodine)  # 不替血管用这两句。 软组织部分碘浓度变为2.5倍
T3000_iodine = np.where(vessel_mask == 1, T3000_iodine * 1.2, T3000_iodine)  # 血管部分碘浓度变为4倍

t = np.linspace(0, 1, 3000)
iodine_loader = np.array([(1 - ti) * T0_iodine + ti * T3000_iodine for ti in t])
# T1500E1 = (T0_water + (mu_iodine_E1 / mu_water_E1) * iodine_loader[1499, :, :] - 1000) * mask
# T1500E2 = (T0_water + (mu_iodine_E2 / mu_water_E2) * iodine_loader[1499, :, :] - 1000) * mask
# T2500E1 = (T0_water + (mu_iodine_E1 / mu_water_E1) * iodine_loader[2499, :, :] - 1000) * mask
# T2500E2 = (T0_water + (mu_iodine_E2 / mu_water_E2) * iodine_loader[2499, :, :] - 1000) * mask

sino_E1_list = []
sino_E2_list = []
CT_E1 = []
CT_E2 = []
sino_E1_train = []
sino_E2_train = []
for i in range(1500):
    sino_E1 = DEsino_Syn(torch.from_numpy(T0_water).float().cuda(), torch.from_numpy(iodine_loader[i, :, :]).float().cuda(),spec_low,atten_water[:56], atten_iodine[:56],proj_2d)
    sino_E2 = DEsino_Syn(torch.from_numpy(T0_water).float().cuda(), torch.from_numpy(iodine_loader[1500+i, :, :]).float().cuda(),spec_high,atten_water, atten_iodine,proj_2d)
    CT_E1E2 = proj_2d.fbp(torch.stack((sino_E1.unsqueeze(0).unsqueeze(0),sino_E2.unsqueeze(0).unsqueeze(0))))
    CT_E1.append(CT_E1E2[0,0,:,:].cpu().numpy() )
    CT_E2.append(CT_E1E2[1,0,:,:].cpu().numpy() )
    sino_E1_list.append(sino_E1.cpu().numpy())
    sino_E2_list.append(sino_E2.cpu().numpy())
for i in range(1000):
    sino_E1_train.append(sino_E1_list[i][i,:])
    sino_E2_train.append(sino_E2_list[i+500][i,:])

CT_E1_loader = np.array(CT_E1)
CT_E2_loader = np.array(CT_E2)  #
sino_E1_train = np.array(sino_E1_train)
sino_E2_train = np.array(sino_E2_train)
sino_E1_loader = np.array(sino_E1_list)
sino_E2_loader = np.array(sino_E2_list)

mu_e1 = ls_mu(T0_water, iodine_loader[1499],CT_E1_loader[1499]) * 1000
mu_e2 = ls_mu(T0_water, iodine_loader[1499],CT_E2_loader[0]) * 1000

syt_mu = equivalent_mu(spec_low.cpu().numpy(),spec_high.cpu().numpy(),atten_water.cpu().numpy(),atten_iodine.cpu().numpy()) # 正常方法算出等效线性衰减系数
decom_mu = np.linalg.inv(syt_mu)

mu_equ =np.stack((mu_e1,mu_e2),axis=0)  # 最小二乘根据一致数据算线性衰减系数（效果好但偷看了数据）
dec = np.linalg.inv(mu_equ)

dec_final = (decom_mu + dec)/2

proj_geom = astra.create_proj_geom('parallel', 256.0 / 384.0, 384,
                                   np.linspace(0, - np.pi, 1000, False))  # 配置投影几何参数（proj_geom中只存储参数，此时还没有创建出投影图形）
vol_geom = astra.create_vol_geom(256, 256)  # 配置被扫物体参数（vol_geom中只存储参数，此时还没有创建出被扫物体图形）
#   astra.create_proj_geom和astra.create_vol_geom只创建参数载体，不创建图形实体
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)  # 创建投影器, 如果要使用cuda，这里需要改为cuda
recon_id = astra.data2d.create('-vol', vol_geom, 0)  # 创建被投影数据实体
sinogram_id = astra.data2d.create('-sino', proj_geom, sino_E1_train)  # 创建投影数据实体
#   使用id创建才会创建出实体，算法运行前需要三个数据实体及其id（实体都有id）：被投影物体、投影数据和算法实体

# 创建重建算法配置字典
cfg = astra.astra_dict('SIRT_CUDA')
cfg['ProjectorId'] = proj_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ReconstructionDataId'] = recon_id
# cfg['option'] = { 'FilterType': 'gaussian' }

# 运行recon算法
algorithm_id = astra.algorithm.create(cfg)
astra.algorithm.run(algorithm_id, 1000)
recon_E1 = astra.data2d.get(recon_id) / 1.5

# 清理资源
astra.algorithm.delete(algorithm_id)
error_nr_E1 = nrmse(CT_E1_loader[499, :, :], recon_E1 * mask)
plt.imshow((recon_E1- 0.025) * 50, cmap='gray', vmin=0, vmax=1)
plt.title('algorithm:{}|E1_NRMSE:{:.4g}'.format(cfg['type'], error_nr_E1))
plt.show()
print(error_nr_E1)

proj_geom = astra.create_proj_geom('parallel', 256.0 / 384.0, 384,
                                   np.linspace(0, - np.pi, 1000, False))  # 配置投影几何参数（proj_geom中只存储参数，此时还没有创建出投影图形）
vol_geom = astra.create_vol_geom(256, 256)  # 配置被扫物体参数（vol_geom中只存储参数，此时还没有创建出被扫物体图形）
#   astra.create_proj_geom和astra.create_vol_geom只创建参数载体，不创建图形实体
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)  # 创建投影器, 如果要使用cuda，这里需要改为cuda
recon_id = astra.data2d.create('-vol', vol_geom, 0)  # 创建被投影数据实体
sinogram_id = astra.data2d.create('-sino', proj_geom, sino_E2_train)  # 创建投影数据实体
#   使用id创建才会创建出实体，算法运行前需要三个数据实体及其id（实体都有id）：被投影物体、投影数据和算法实体

# 创建重建算法配置字典
cfg = astra.astra_dict('SIRT_CUDA')
cfg['ProjectorId'] = proj_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ReconstructionDataId'] = recon_id
# cfg['option'] = { 'FilterType': 'gaussian' }

# 运行recon算法
algorithm_id = astra.algorithm.create(cfg)
astra.algorithm.run(algorithm_id, 1000)
recon_E2 = astra.data2d.get(recon_id) / 1.5
# 清理资源
astra.algorithm.delete(algorithm_id)

water_FBP = (dec_final[0][0] * recon_E1  + dec_final[0][1] * recon_E2 ) * mask
iodine_FBP = (dec_final[1][0] * recon_E1  + dec_final[1][1] * recon_E2 ) * mask
water_ID = (dec_final[0][0] * CT_E1_loader[1499]  + dec_final[0][1] * CT_E2_loader[0] ) * mask
iodine_ID = (dec_final[1][0] * CT_E1_loader[1499]  + dec_final[1][1] * CT_E2_loader[0] ) * mask

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(baseline_water, cmap='gray', vmin=1000, vmax=1100)
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(water_FBP * 1000, cmap='gray', vmin=1000, vmax=1100)
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(water_ID * 1000, cmap='gray', vmin=1000, vmax=1100)
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(iodine_loader[1499], cmap='hot', vmin=0, vmax=3)
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(((iodine_FBP * 1000)+1)* mask, cmap='hot', vmin=0, vmax=3)
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(((iodine_ID * 1000)+1)* mask, cmap='hot', vmin=0, vmax=3)
plt.axis('off')
plt.tight_layout()
plt.show()

error_nr_E2 = nrmse(CT_E2_loader[499, :, :], recon_E2 * mask)
plt.imshow((recon_E2- 0.025) * 50, cmap='gray', vmin=0, vmax=1)
plt.title('algorithm:{}|E2_NRMSE:{:.4g}'.format(cfg['type'], error_nr_E2))
plt.show()
print(error_nr_E2)

np.save(f'generated_data/spec{slice_num}/T1500_E1.npy', CT_E1_loader[1499] * mask)  # (256, 256 )
np.save(f'generated_data/spec{slice_num}/T1500_E2.npy', CT_E2_loader[0]* mask)  # (256, 256 )
np.save(f'generated_data/spec{slice_num}/T1500_water.npy', T0_water* mask)  # (256, 256 )
np.save(f'generated_data/spec{slice_num}/T1500_iodine.npy', iodine_loader[1499]* mask)  # (256, 256 )
np.save(f'generated_data/spec{slice_num}/CT_E1_loader.npy', CT_E1_loader* mask)  # 保存的是scale过的 (1000, 256, 256 )
np.save(f'generated_data/spec{slice_num}/CT_E2_loader.npy', CT_E2_loader* mask)  # (1000, 256, 256 )
np.save(f'generated_data/spec{slice_num}/sinogram_E1_train.npy', sino_E1_train)  # (1000, 384 )
np.save(f'generated_data/spec{slice_num}/sinogram_E2_train.npy', sino_E2_train)  # (1000, 384 )
np.save(f'generated_data/spec{slice_num}/recon_E1_T1.npy', recon_E1* mask)  # (256, 256 )
np.save(f'generated_data/spec{slice_num}/recon_E2_T2.npy', recon_E2* mask)  # (256, 256 )
np.save(f'generated_data/spec{slice_num}/water_FBP.npy', water_FBP* mask)  # (256, 256 )
np.save(f'generated_data/spec{slice_num}/iodine_FBP.npy', iodine_FBP* mask)  # (256, 256 )
np.save(f'generated_data/spec{slice_num}/water_ID.npy', water_ID* mask)  # (256, 256 )
np.save(f'generated_data/spec{slice_num}/iodine_ID.npy', iodine_ID* mask)  # (256, 256 )
np.save(f'generated_data/spec{slice_num}/iodine_loader.npy', iodine_loader* mask)  # (256, 256 )
np.save(f'generated_data/spec{slice_num}/sys_ID.npy', syt_mu)
np.save(f'generated_data/spec{slice_num}/dec_ID.npy', decom_mu)
np.save(f'generated_data/spec{slice_num}/sys_IDlsq.npy', mu_equ)
np.save(f'generated_data/spec{slice_num}/dec_IDlsq.npy', dec)
np.save(f'generated_data/spec{slice_num}/dec_final.npy', dec_final)
np.save(f'generated_data/spec{slice_num}/mask.npy', mask)


# np.save('generated_data/slice35brain9/FBP_E1.npy', FBP_E1)  # (256, 256 )

# print('NRMSE(249|749):{}'.format(nrmse(CT_E1_loader[249, :, :], CT_E1_loader[749, :, :])))
# print('iodine/water E1:{} | E2:{}'.format(mu_iodine_E1 / mu_water_E1, mu_iodine_E2 / mu_water_E2))  # 决定碘好不好算
# print('mean E1:{} | E2:{}'.format(T1500E1.mean(), T1500E2.mean()))
# print('mean mu_E1M1muE2M2CTE1:{} | muE1M2muE2M1CTE2:{}'.format(mu_water_E1 * mu_iodine_E2 * T1500E1.mean(),
#                                                                mu_water_E2 * mu_iodine_E1 * T1500E2.mean()))    # 决定水好不好算
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(np.rot90(CT_E1_loader[249]*scale1,3), cmap='gray', vmin=10, vmax=100)
# plt.axis('off')
# # plt.title('E1_gt')
# plt.subplot(1, 2, 2)
# plt.imshow(np.rot90(CT_E1_loader[749]*scale1,3), cmap='gray', vmin=10, vmax=100)
# plt.axis('off')
# # plt.title('E1_recon')
# plt.tight_layout()
# plt.show()

# # 生成用于FBP对比实验所用sino（仅外面的骨头）
# CT_masked = np.load('generated_data/slice35brain/CTE1_masked.npy') / scale1
# sino_masked = proj_2d(torch.from_numpy(CT_masked.reshape(1, 1, 256, 256)).cuda().float())[0,
#                               :, 0, :].clone().cpu().numpy()
# np.save('generated_data/slice35brain9/sino_masked.npy', sino_masked)
