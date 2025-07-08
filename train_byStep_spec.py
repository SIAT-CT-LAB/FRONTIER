import os
import argparse
import shutil

import torch
import torch.nn.functional as F
import torch.nn as nn
# import torchvision
# import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
# import tensorboardX
import numpy as np
import scipy.io as sio
import skimage
import collections
from skimage.metrics import normalized_root_mse as nrmse

from networks import Positional_Encoder, FFN, SIREN
from utils import get_config, prepare_sub_folder

import matplotlib.pyplot as plt

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from leaptorch import Projector

device_name = "cuda:0"
device = torch.device(device_name)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='cfgs/train_byStep_spec.yaml',
                    help='path to the config file.')
parser.add_argument('--output_path', type=str, default='', help="outputs path")
parser.add_argument('--pretrain', action='store_true', help="load pretrained model weights")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']

cudnn.benchmark = True

# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]
output_subfolder = config['data']
model_name = os.path.join(output_folder,
                          output_subfolder + config[
                              'energy'] + '/time_num{}_img{}_proj{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}' \
                          .format(config['time_num'], config['img_size'], config['num_proj'], config['model'], \
                                  config['net']['network_input_size'], config['net']['network_width'], \
                                  config['net']['network_depth'], config['loss'], config['lr'],
                                  config['encoder']['embedding']))
if not (config['encoder']['embedding'] == 'none'):
    model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
print(model_name)

# train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "logs", model_name))
output_directory = os.path.join(opts.output_path + "outputs", model_name)

checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
print(image_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

# Setup input encoder:
encoder = Positional_Encoder(config['encoder'])

# Setup model
if config['model'] == 'SIREN':
    model = SIREN(config['net'])
elif config['model'] == 'FFN':
    model = FFN(config['net'])
else:
    raise NotImplementedError

model.cuda()
model.train()

# Load pretrain model

model_path = config['pretrain_model_path']
state_dict = torch.load(model_path)

encoder.B = state_dict['enc']
final_step = []
# Setup optimizer
# optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']),
#                          weight_decay=config['weight_decay'])

# Setup loss function
if config['loss'] == 'L2':
    loss_fn = torch.nn.MSELoss()
elif config['loss'] == 'L1':
    loss_fn = torch.nn.L1Loss()
else:
    NotImplementedError
# # Setup data loader
# print('Load image: {}'.format(config['img_path']))
# data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], config['img_slice'], train=True,
#                               batch_size=config['batch_size'])

# # Setup data loader
# print('Load image:{}',format(config['img_path']))
# data_loader = DataLoader()
time_num = config['time_num']
proj_num = config['num_proj']
proj_every = proj_num // time_num
data = config['data']
energy = config['energy']
# scaler = np.load(f'generated_data/{data}/scale_E1.npy').item()
mask = np.load(f'generated_data/{data}/mask.npy').astype(np.float32)
image_raw = (np.load(f'generated_data/{data}/CT_E{energy}_loader.npy').astype(np.float32)- 0.025) * mask* 50
proir_image = (np.load(f'generated_data/{data}/recon_E{energy}_T{energy}.npy').astype(np.float32) * mask - 0.025) * mask * 50
proir_image_t = torch.from_numpy(proir_image).float().cuda()
train_data = np.load(f'generated_data/{data}/sinogram_E{energy}_train.npy').astype(np.float32) * 10
# image_FBP = (np.load(f'generated_data/{data}/recon_E1_T1.npy') * mask - 0.025) * mask * 50
# image_raw = np.transpose(image_raw, (2, 0, 1))
indices = np.arange(1, 2 * time_num, 2) * (proj_num / (2 * time_num))
indices = np.ceil(indices).astype(np.int32)
indices_left = np.where(indices > (proj_num / 2), 2 * indices - proj_num + 1, 0).astype(int)
indices_right = np.where(indices > (proj_num / 2), proj_num, 2 * indices).astype(int)
if time_num == 1000:  # time_num=1000时
    indices = np.arange(1, 2 * time_num, 2) * (proj_num / (2 * time_num)) - 1
    indices = np.ceil(indices).astype(np.int32)
    indices_left = np.where(indices >= (proj_num / 2), 2 * indices - proj_num + 1, 0).astype(int)
    indices_right = np.where(indices >= (proj_num / 2), 999, 2 * indices).astype(int) + 1
test_data = image_raw[indices]
# Input coordinates (x,y) grid and target image
grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=config['img_size']),
                                 torch.linspace(0, 1, steps=config['img_size'])])
grid = torch.stack([grid_y, grid_x], dim=-1).cuda()
test_data = torch.from_numpy(test_data).cuda()
train_data = torch.from_numpy(train_data).cuda()
# FBP_t = torch.from_numpy(image_FBP).cuda().float()
mask = torch.from_numpy(mask).cuda()
grid_embedding = encoder.embedding(grid).cuda()
final_images = np.zeros([time_num, config['img_size'], config['img_size']])
proj_2d = Projector(forward_project=True, use_static=True, use_gpu=True, gpu_device=device, batch_size=1)
numRows = 1
numCols = 384  # 是为了让探测器更密一些，探测器大小小于像素大小
pixelSize = 1
angels = proj_2d.leapct.setAngleArray(proj_num, 180.0)

def sobel_filter(img):
    # img: [H, W]
    # 将img扩展为[1,1,H,W]
    img = img.unsqueeze(0).unsqueeze(0)

    # 定义Sobel核
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=img.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=img.device).unsqueeze(0).unsqueeze(0)

    # 对图像进行Sobel卷积
    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)

    # 计算梯度幅值图
    mag = torch.sqrt(gx**2 + gy**2)
    # 去掉批次和通道维度，返回[H, W]
    return mag.squeeze(0).squeeze(0)

def generate_learning_rates(max_iter, start_lr, mid_lr):
    learning_rates = []
    for i in range(max_iter):
        if i < max_iter / 2:
            lr = start_lr + (mid_lr - start_lr) * (2 * i / max_iter)
        else:
            lr = mid_lr - (mid_lr - start_lr) * (2 * (i - max_iter / 2) / max_iter)
        learning_rates.append(lr)
    return learning_rates

lr_list = generate_learning_rates(time_num, config['lr'], config['lr_max'])
for i in range(time_num):
    time_point = indices[i]
    test_data_temp = test_data[i, :, :]
    train_data_temp = train_data[indices_left[i]:indices_right[i]]
    thetas_temp = angels[indices_left[i]:indices_right[i]]
    numAngles = indices_right[i] - indices_left[i]
    iter_final = 0
    loss_last = 999
    optim = torch.optim.Adam(model.parameters(), lr=lr_list[i], betas=(config['beta1'], config['beta2']),
                             weight_decay=config['weight_decay'])  # model.parameters 表示模型需要更新的参数
    model.load_state_dict(state_dict['net'])  # 每次都从FBP得到的CT图出发

    proj_2d.leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5 * (numRows - 1),
                                    0.5 * (numCols - 1), thetas_temp)
    proj_2d.leapct.set_volume(256, 256, 1, pixelSize * 384.0 / 256.0, pixelSize)
    proj_2d.allocate_batch_data()
    for iterations in range(max_iter):
        model.train()
        proj_2d.train()
        optim.zero_grad()
        train_output = model(grid_embedding)[:, :, 0]  # tensor(256,256)
        train_output_mu = (train_output / 50 + 0.025)* mask
        train_projs = proj_2d(train_output_mu.unsqueeze(0).unsqueeze(0)).float() * 10
        # train_loss = 0.5 * loss_fn(train_projs[0, :, 0, :], train_data_temp)
        train_loss =loss_fn(train_projs[0, :, 0, :], train_data_temp) + 5000 * loss_fn(sobel_filter(train_output), sobel_filter(proir_image_t))
        # train_loss =loss_fn(train_projs[0, :, 0, :], train_data_temp) + 5 * loss_fn(train_output, test_data_temp)+ 500 * loss_fn(sobel_filter(train_output), sobel_filter(test_data_temp))
        # train_loss = 0.5 * loss_fn(train_projs[0, :, 0, :], train_data_temp) + 500000 * loss_fn(sobel_filter(train_output), sobel_filter(test_data_temp))
        train_loss.backward()
        optim.step()

        # Compute testing psnr
        if iterations == 0 or (iterations + 1) % config['val_iter'] == 0:
            model.eval()
            proj_2d.eval()
            with torch.no_grad():
                test_output = model(grid_embedding)[:, :, 0] * mask

                # torchvision.utils.save_image(test_output.detach(),
                #                              os.path.join(image_directory,
                #                                           'time_point{}_iter{}.png'.format(
                #                                               time_point, iterations + 1)))

                test_loss = 0.5 * loss_fn(test_output, test_data_temp)
                test_NRMSE = torch.sqrt(2 * test_loss) / torch.norm(test_data_temp) * 256
                test_loss = test_loss.item()

                if ((loss_last < test_loss) & (iterations > 100)) | (iterations + 1 == max_iter):
                    print('time_point:{}|best iter:{}|NRMSE:{:.4g}'.format(time_point, iterations - config['val_iter'],
                                                                           test_NRMSE))
                    iter_final = iterations
                    final_images[i, :, :] = final_img_t.cpu().numpy()
                    # plt.imshow((test_output).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                    # plt.colorbar()
                    # plt.title(
                    #     'time_num{}|time_point{}|iter_final:{}'.format(time_num, time_point, iter_final - config['val_iter']))
                    # plt.show()
                    break
                loss_last = test_loss

            # train_writer.add_scalar('test_loss{}'.format(time_point), test_loss, iterations + 1)
            # train_writer.add_scalar('test_NRMSE{}'.format(time_point), test_NRMSE, iterations + 1)
            final_img_t = test_output.clone()
            print("[time_point:{}][Validation Iteration: {}] Test loss: {:.4g} | Test NRMSE: {:.4g}".format(time_point,
                                                                                                            iterations + 1,
                                                                                                            test_loss,
                                                                                                            test_NRMSE))

        # torch.save({'net': model.state_dict(), 'enc': encoder.B, 'opt': optim.state_dict(), },
        #            os.path.join(checkpoint_directory, 'time_point{}model_final.pt'.format(time_point)))
        # torch.save(test_output, os.path.join(image_directory, 'time_point{}_final.pth'.format(time_point)))

    # print("[TimePoint{}]:total_iter:{}|RMSE:{}".format(time_point,iter_final,np.sqrt(test_loss)))
    final_step.append(iter_final - config['val_iter'])
    # train_writer.add_scalar('NRMSE', test_NRMSE, time_point)
with open(os.path.join(image_directory, 'final_step_list.txt'), 'w') as f:
    for item in final_step:
        f.write("%d\n" % item)

test_data = test_data.cpu().numpy()
slope_truth = np.zeros([256, 256])
color_list = ['gray', 'blue', 'black', 'red', 'green', 'orange', 'pink', 'yellow', 'purple']
coordinates = [[133, 202], [116, 216], [67, 107], [149, 185]]  # 第一个是血管

def ROI_value(imgs, coordinate, window_size):
    x, y = coordinate[0], coordinate[1]
    half_window_size = window_size // 2
    mean_vals = np.mean(
        imgs[:, x - half_window_size:x + half_window_size + 1, y - half_window_size:y + half_window_size + 1],
        axis=(1, 2))
    return mean_vals


if energy == '1':
    pred_img1500 = np.zeros([256, 256])
    # pred_img2500 = np.zeros([256, 256])
    truth_img1500 = np.zeros([256, 256])
    # truth_img2500 = np.zeros([256, 256])
    for i in range(256):  # 算出外插图像的每个像素值
        for j in range(256):
            y = test_data[:, i, j]
            k, b = np.polyfit(indices, y, 1)
            slope_truth[i, j] = k
            # truth_img2500[i, j] = k * 2499 + b
            truth_img1500[i, j] = k * 1499 + b
    slope_pred = np.zeros([256, 256])
    for i in range(256):  # 算出外插图像的每个像素值
        for j in range(256):
            y = final_images[:, i, j]
            k, b = np.polyfit(indices, y, 1)
            slope_pred[i, j] = k
            # pred_img2500[i, j] = k * 2499 + b
            pred_img1500[i, j] = k * 1499 + b

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(slope_truth * scaler, cmap='hot', vmin=0, vmax=0.5)
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(slope_pred * scaler, cmap='hot', vmin=0, vmax=0.5)
    # plt.axis('off')
    # # plt.colorbar()
    # plt.tight_layout()
    # plt.suptitle('time_num:{}_error{:.4g}'.format(time_num, nrmse(slope_truth, slope_pred)), color='green')
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(truth_img1500, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(pred_img1500 , cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    # plt.colorbar()
    plt.tight_layout()
    plt.suptitle('E1time_num:{}_error1500:{:.4g} '.format(time_num, nrmse(truth_img1500, pred_img1500)),
                 color='green')
    plt.savefig('paper_plot/fig6/spec35pred1500E1/TR{}nrmse{}.jpg'.format(time_num,nrmse(truth_img1500, pred_img1500)/1.5 ))
    plt.show()

    for i in range(len(coordinates)):
        coord = coordinates[i]
        y = ROI_value(final_images, coord, 4)
        plt.scatter(indices, y, color=color_list[i], s=10)
        k, b = np.polyfit(indices, y, 1)
        x = np.arange(0, 1.5 * proj_num)
        plt.plot(x, k * x + b, linestyle="--", color=color_list[i])
        y = ROI_value(test_data, coord, 4)
        k, b = np.polyfit(indices, y, 1)
        plt.plot(x, k * x + b, color=color_list[i], label='[{},{}]'.format(coord[0], coord[1]), linewidth=3.5)
    plt.legend()
    plt.suptitle('E1recon_num:{}'.format(time_num))
    plt.show()
    np.save(os.path.join(image_directory, 'recon_imgs.npy'), (final_images/50+0.025-0.03127156)/0.03127156*1000)  # (1000, 256, 256)
    np.save(os.path.join(image_directory, 'extrapolation1500.npy'), (pred_img1500/50+0.025-0.03127156)/0.03127156*1000)  # (256, 256)
    # np.save(os.path.join(image_directory, 'extrapolation2500.npy'), pred_img2500)  # (256, 256)

if energy == '2':
    pred_imgs1500 = np.zeros([256, 256])
    truth_imgs1500 = np.zeros([256, 256])
    for i in range(256):  # 算出truth外插图像的每个像素值
        for j in range(256):
            y = test_data[:, i, j]
            k, b = np.polyfit(indices + 2000, y, 1)
            slope_truth[i, j] = k
            truth_imgs1500[i, j] = k * 1499 + b

    slope_pred = np.zeros([256, 256])
    for i in range(256):  # 算出recon外插图像的每个像素值
        for j in range(256):
            y = final_images[:, i, j]
            k, b = np.polyfit(indices + 2000, y, 1)
            slope_pred[i, j] = k
            pred_imgs1500[i, j] = k * 1499 + b

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(truth_imgs1500 , cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(pred_imgs1500 , cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    # plt.colorbar()
    plt.tight_layout()
    plt.suptitle('E2time_num:{}_error{:.4g}'.format(time_num, nrmse(truth_imgs1500, pred_imgs1500)), color='green')
    plt.savefig('paper_plot/fig6/spec35pred1500E2/TR{}nrmse{}.jpg'.format(time_num,nrmse(truth_imgs1500, pred_imgs1500)/1.5 ))
    plt.show()

    for i in range(len(coordinates)):
        coord = coordinates[i]
        y = ROI_value(final_images, coord, 4)
        plt.scatter(indices + 2000, y, color=color_list[i], s=5)    # 重建结果
        k, b = np.polyfit(indices + 2000, y, 1)
        x = np.arange(0, 1.5 * proj_num) + 1500
        plt.plot(x, k * x + b, linestyle="--", color=color_list[i],label='[{},{}]'.format(coord[0], coord[1]))  # 重建拟合结果
        y = ROI_value(test_data, coord, 4)
        k, b = np.polyfit(indices + 2000, y, 1)
        plt.plot(x, k * x + b, color=color_list[i], linewidth=3.5)  # truth结果
    plt.legend()
    plt.suptitle('E2recon_num:{}'.format(time_num))
    plt.show()
    np.save(os.path.join(image_directory, 'recon_imgs.npy'), (final_images/50+0.025-0.02727755625413329)/0.02727755625413329*1000)  # (1000, 256, 256)
    np.save(os.path.join(image_directory, 'extrapolation1500.npy'), (pred_imgs1500/50+0.025-0.02727755625413329)/0.02727755625413329*1000)  # (256, 256)

# image = np.load('generated_data/DECT6082keV6/recon_E1_T1.npy') * mask.cpu().numpy()  # 加载预训练数据
# fig, axes = plt.subplots(2, 5, figsize=(15, 6))
#
# for i in range(5):
#     axes[0, i].imshow(np.rot90(test_data[i, :, :],k=3) * scaler, cmap='gray', vmin=20, vmax=180)
#     axes[0, i].axis('off')  # 关闭坐标轴
#
# # 第二行显示array2的图像
# for i in range(5):
#     axes[1, i].imshow(np.rot90(final_images[i, :, :],k=3) * scaler, cmap='gray', vmin=20, vmax=180)
#     axes[1, i].axis('off')  # 关闭坐标轴
# axes[1, 2].imshow(np.rot90(image,k=3) * scaler, cmap='gray', vmin=0, vmax=200)
# plt.tight_layout()
# plt.show()
