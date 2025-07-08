import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
import datetime
import astra
from skimage.metrics import normalized_root_mse as nrmse
from scipy.ndimage import distance_transform_edt, map_coordinates
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import tifffile
from leaptorch import Projector
from specTool import DEsino_Syn, DECT_Syn

def replace_with_nearest(img, mask):
    dist, coords = distance_transform_edt(mask, return_distances=True, return_indices=True)
    img_replaced = map_coordinates(img, coords, order=0)
    return img_replaced

# usage: correct water basis images using generated mask for every slice

# 读取tif文件
mask = tifffile.imread('raw_data/DECT_human/mask_ROI_noBone.tif')[70:145, :, :] / 255
vessel_mask = tifffile.imread('raw_data/DECT_human/vessel_waterMask.tif') / 255  # 这个vessel_mask很不准确，很多骨头附近的比较大的值也被算作血管了

images_water = []
images_iodine = []
# 打开二进制文件
with open("raw_data/DECT_human/human_cerebral_ex1_140kV_m1.dsi", "rb") as f:  # 一共160个slice
    for _ in range(160):
        image = np.fromfile(f, dtype=np.float32, count=512 * 512)
        image = image.reshape((512, 512))
        images_water.append(image)
images_water = np.array(images_water)[70:145, 156:412, 100:356]  # 取脑部结构比较清晰的75个slice

with open("raw_data/DECT_human/human_cerebral_ex1_140kV_m2.dsi", "rb") as f:
    for _ in range(160):
        image = np.fromfile(f, dtype=np.float32, count=512 * 512)
        image = image.reshape((512, 512))
        images_iodine.append(image)
images_iodine = np.array(images_iodine)[70:145, 156:412, 100:356]  # 取脑部结构比较清晰的75个slice

water_masked = np.zeros_like(mask)
iodine_masked = np.zeros_like(mask)
for i in range(75):
    water_masked[i, :, :] = images_water[i, :, :] * (1- mask[i, :, :])
    iodine_masked[i, :, :] = images_iodine[i, :, :] * (1- mask[i, :, :])
    images_water[i, :, :] = images_water[i, :, :] * mask[i, :, :]  # 去掉外面的一圈头骨
    images_iodine[i, :, :] = images_iodine[i, :, :] * mask[i, :, :]

# # 保存为tif文件
# tifffile.imwrite('raw_data/DECT_human/water_masked.tif', images_water)
# tifffile.imwrite('raw_data/DECT_human/iodine_masked.tif', images_iodine)


for i in range(75):
    images_water[i, :, :] = replace_with_nearest(images_water[i, :, :],
                                                 vessel_mask[i, :, :] + (1 - mask[i, :, :])) * mask[i, :,
                                                                                               :]  # 使用vessel_mask修改水基，把血管部分抹平
for i in range(75):
    images_iodine[i, :, :] = np.where(images_iodine[i, :, :] > 2, 2 + images_iodine[i, :, :] / 10,
                                      images_iodine[i, :, :])  # 限制iodine的最大值，避免跟软组织相差过大难以重建

# for i in range(75):
#     if i % 5 == 0:
#         plt.figure(figsize=(12, 6))
#         plt.subplot(1, 2, 1)
#         plt.imshow(images_water[i,:,:], cmap='gray', vmin=1000, vmax=1100)
#         plt.axis('off')
#         plt.subplot(1, 2, 2)
#         plt.imshow(images_iodine[i, :, :], cmap='hot', vmin=0, vmax=3)
#         plt.axis('off')
#         plt.tight_layout()
#         plt.show()

# mu_water_E1 = 0.0259  # 82 keV
# mu_water_E2 = 0.0319  # 60 keV
# mu_iodine_E1 = 2.321
# mu_iodine_E2 = 4.518
# m1 = np.zeros([75,256,256])
# CTiodine = np.zeros([75, 2, 256, 256])
# for i in range(75):
#     CTiodine[i, 0, :, :] = ((images_water[i, :, :] - 1000 + mu_iodine_E1 / mu_water_E1 * images_iodine[i, :, :])) * mask[i]
#     CTiodine[i, 1, :, :] = images_iodine[i, :, :]
#     m1[i,:,:] = (images_water - 1000)[i] * mask[i]
#
# slice_num = 35
#
# train_CTiodine = np.concatenate((CTiodine[0:slice_num,:,:,:], CTiodine[slice_num + 1:75,:,:,:]))
# train_m1 = np.concatenate((m1[0:slice_num], m1[slice_num + 1:75]))
#
# test_CTiodine = CTiodine[slice_num,:,:,:]
# test_m1 = m1[slice_num]  # test_slice用作DECT的动态模拟

spec_low = torch.from_numpy(np.load('raw_data/spec_data/spec_low.npy')[25:81]).float().cuda()  # 80kV
spec_high = torch.from_numpy(np.load('raw_data/spec_data/spec_high.npy')[25:141]).float().cuda()  # 140kV
atten_iodine = torch.from_numpy(np.load('raw_data/spec_data/atten_iodine.npy')).float().cuda()
atten_water = torch.from_numpy(np.load('raw_data/spec_data/atten_water.npy')).float().cuda()
#
# E_spec_low = ((spec_low * (torch.arange(56).cuda() + 25))/spec_high.sum()).sum()  # 80kV平均能量
# E_spec_high = ((spec_high * (torch.arange(116).cuda() + 25))/spec_high.sum()).sum()  # 140kV平均能量
# mu_water_low_avg = (atten_water[:56] * spec_low).sum()
# mu_water_high_avg = (atten_water * spec_high).sum()
# mu_iodine_low_avg = (atten_iodine[:56] * spec_low).sum()
# mu_iodine_high_avg = (atten_iodine * spec_high).sum()
# mu_loader = [mu_water_low_avg, mu_water_high_avg, mu_iodine_low_avg, mu_iodine_high_avg]

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
sino_low_list = []
sino_high_list = []
for i in range(75):
    sino_low = DEsino_Syn(torch.from_numpy(images_water[i, :, :]).float().cuda(), torch.from_numpy(images_iodine[i, :, :]).float().cuda(),spec_low,atten_water[:56], atten_iodine[:56],proj_2d)
    sino_high = DEsino_Syn(torch.from_numpy(images_water[i, :, :]).float().cuda(), torch.from_numpy(images_iodine[i, :, :]).float().cuda(),spec_high,atten_water, atten_iodine,proj_2d)
    sino_low_list.append(sino_low)
    sino_high_list.append(sino_high)
CT_low = []
CT_high = []
for i in range(75):
    CT_lowhigh = proj_2d.fbp(torch.stack((sino_low_list[i].unsqueeze(0).unsqueeze(0), sino_high_list[i].unsqueeze(0).unsqueeze(0))).reshape(2,1000,1,384))
    CT_low.append(CT_lowhigh[0,0,:,:].cpu().numpy())
    CT_high.append(CT_lowhigh[1,0,:,:].cpu().numpy())

# E1E2iodine = np.zeros([75, 3, 256, 256])
# for i in range(75):
#     E1E2iodine[i, 0, :, :] = CT_low[i] - 0.028
#     E1E2iodine[i, 1, :, :] = CT_high[i] - 0.028
#     E1E2iodine[i, 2, :, :] = images_iodine[i]

slice_num = 35
low_sino_masked = DEsino_Syn(torch.from_numpy(water_masked[slice_num,:,:]).float().cuda(), torch.from_numpy(iodine_masked[slice_num,:,:]).float().cuda(),spec_low,atten_water[:56], atten_iodine[:56],proj_2d)
high__sino_masked = DEsino_Syn(torch.from_numpy(water_masked[slice_num,:,:]).float().cuda(), torch.from_numpy(iodine_masked[slice_num,:,:]).float().cuda(),spec_high,atten_water, atten_iodine,proj_2d)
masked_lowhigh = proj_2d.fbp(torch.stack((low_sino_masked.unsqueeze(0).unsqueeze(0), high__sino_masked.unsqueeze(0).unsqueeze(0))).reshape(2,1000,1,384))
low_masked = masked_lowhigh[0,0,:,:].cpu().numpy()
high_masked = masked_lowhigh[1,0,:,:].cpu().numpy()

np.save(f'generated_data/spec{slice_num}/water1500GT.npy',images_water[slice_num])
np.save(f'generated_data/spec{slice_num}/iodine1500GT.npy',images_iodine[slice_num])
np.save('generated_data/spec35/mask.npy',mask[slice_num])

# np.save('generated_data/slice35brain/E1iodine_train.npy',
#         train_CTiodine)  # 碘基图和CT图作为输入 (2627,2,256,256),第一个channel是CT，第二个channel是iodine
# np.save('generated_data/slice35brain/E1iodine_test.npy', test_CTiodine)

# np.save('generated_data/slice35brain/water_train.npy', train_m1)  # 水基图和CT图作为label
# np.save('generated_data/slice35brain/water_test.npy', test_m1)

np.save(f'generated_data/spec{slice_num}/water_masked.npy', water_masked[slice_num,:,:])
np.save(f'generated_data/spec{slice_num}/iodine_masked.npy', iodine_masked[slice_num,:,:])

np.save(f'generated_data/spec{slice_num}/CTE1_masked.npy', low_masked)
np.save(f'generated_data/spec{slice_num}/CTE2_masked.npy', high_masked)

#
# np.save('generated_data/slice35brain/water1500GT.npy', images_water[slice_num])
# np.save('generated_data/slice35brain/iodine1500GT.npy', images_iodine[slice_num])
# np.save('generated_data/slice35brain/mask.npy', mask[slice_num])
