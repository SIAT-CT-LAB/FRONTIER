import os
import argparse
import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import tensorboardX
from skimage.metrics import normalized_root_mse as nrmse

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from networks import Positional_Encoder, FFN, SIREN
from utils import get_config, prepare_sub_folder

# 设置使用第四张卡
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 添加配置文件路径
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='cfgs/pre_train_spec.yaml',
                    help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='', help="outputs path")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']

cudnn.benchmark = True

# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]
model_name = os.path.join(output_folder, config['data'] + config['energy'] + '/img{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}' \
                          .format(config['img_size'], config['model'], \
                                  config['net']['network_input_size'], config['net']['network_width'], \
                                  config['net']['network_depth'], config['loss'], config['lr'],
                                  config['encoder']['embedding']))
if not (config['encoder']['embedding'] == 'none'):
    model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
print(model_name)

# 保存训练日志
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "logs", model_name))
output_directory = os.path.join(opts.output_path + "outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

# Setup input encoder:
# Fourier特征编码（gauss）
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

# Setup optimizer   模型参数都在optim里面
if config['optimizer'] == 'Adam':
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']),
                             weight_decay=config['weight_decay'])
else:
    NotImplementedError

# Setup loss function
if config['loss'] == 'L2':
    loss_fn = torch.nn.MSELoss()
elif config['loss'] == 'L1':
    loss_fn = torch.nn.L1Loss()
else:
    raise NotImplementedError

# Setup data loader
# print('Load image: {}'.format(config['img_path']))
# data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], config['img_slice'], train=True,
#                               batch_size=config['batch_size'])
#   好神奇的可迭代对象
data = config['data']
energy = config['energy']
mask = np.load(f'generated_data/{data}/mask.npy').astype(np.float32)
test_data_np = (np.load(f'generated_data/{data}/CT_{energy}_loader.npy')[499,:,:] - 0.025)* mask * 50
# image = (np.load(f'generated_data/{data}/recon_E1_T1.npy') * mask - 0.025) * mask * 50  # 加载预训练数据
# image[image < 0] = 0  ##数据预处理，把CT图中负值置为零
# image = image / np.max(image)  ##### 数据归一化，非常重要!极大影响性能。
# for it, (grid, image) in enumerate(data_loader):
# Input coordinates (x,y) grid and target image
grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=config['img_size']),
                                 torch.linspace(0, 1, steps=config['img_size'])])
grid = torch.stack([grid_y, grid_x], dim=-1)
grid = grid.cuda()  # [bs, h, w, 2], [0, 1]
# Data loading
# Change training inputs for downsampling image
test_data = torch.from_numpy(test_data_np).cuda().float()
train_data = torch.from_numpy(test_data_np).cuda().float()
grid_embedding = encoder.embedding(grid)
mask_t = torch.from_numpy(mask).cuda().float()
#   不需要save训练数据
# torchvision.utils.save_image(test_data[1].cpu().permute(0, 3, 1, 2).data, os.path.join(image_directory, "test.png"))
# torchvision.utils.save_image(train_data[1].cpu().permute(0, 3, 1, 2).data,
#                                  os.path.join(image_directory, "train.png"))

# Train model
loss_curve = []
for iterations in range(max_iter):
    model.train()
    optim.zero_grad()

    # [B, H, W, embedding*2]，对坐标进行编码
    train_output = model(grid_embedding)[:, :, 0] * mask_t  # [B, H, W, 3]
    # train_data[1] = train_data[1].float()
    train_loss = 0.5 * loss_fn(train_output, train_data)

    train_loss.backward()
    optim.step()
    loss_curve.append(train_loss.item())
    # Compute training psnr
    if (iterations + 1) % config['log_iter'] == 0:
        train_nrmse = torch.sqrt(2 * train_loss) / torch.norm(train_data) * 256
        train_loss = train_loss.item()

        train_writer.add_scalar('train_loss', train_loss, iterations + 1)
        train_writer.add_scalar('train_NRMSE', train_nrmse, iterations + 1)
        print("[Iteration: {}/{}] Train loss: {:.4g} | Train NRMSE: {:.4g}".format(iterations + 1, max_iter,
                                                                                  train_loss, train_nrmse))

    # Compute testing psnr
    if (iterations + 1) % config['val_iter'] == 0:
        model.eval()
        with torch.no_grad():
            test_output = model(grid_embedding)[:, :, 0] * mask_t
            test_loss = 0.5 * loss_fn(test_output , test_data)
            test_nrmse = torch.sqrt(2 * test_loss) / torch.norm(test_data) * 256
            test_loss = test_loss.item()

        train_writer.add_scalar('test_loss', test_loss, iterations + 1)
        train_writer.add_scalar('test_nrmse', test_nrmse, iterations + 1)
        # Must transfer to .cpu() tensor firstly for saving images
        # torchvision.utils.save_image(test_output[:, :, 0], os.path.join(image_directory,
        #                                                                 "recon_{}.png".format(
        #                                                                     iterations + 1)))
        # plt.imshow((test_output * scaler).detach().cpu().numpy(), cmap='gray', vmin=0, vmax=80)
        # plt.colorbar()
        # plt.title('iter:{}'.format(iterations + 1))
        # plt.show()
        print("[Validation Iteration: {}/{}] Test loss: {:.4g} | Test NRMSE: {:.4g}".format(iterations + 1, max_iter,
                                                                                            test_loss, test_nrmse))

    # Save final model
model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (iterations + 1))
torch.save(dict(net=model.state_dict(), enc=encoder.B, opt=optim.state_dict()), model_name)
np.save(os.path.join(image_directory, 'final_image.npy' ),test_output.detach().cpu().numpy())
# np.save('paper_plot/fig16/loss_pretrain3000.npy', loss_curve)
# img1 = test_output[:, :, 0]
# img1 = img1.cpu()
# img1 = img1.numpy()
# img2 = image
# img2 = img2.cpu().numpy()
# fig, axes = plt.subplots(1, 2)
# ax1, ax2 = axes.flatten()
# ax1.imshow(img2)
# ax1.set_title('FBPfromSinogram')
# ax2.imshow(img1)
# ax2.set_title('outputfromMLP')
# plt.show()
# plt.imshow(img1-img2,vmin=(img1-img2).min(),vmax=(img1-img2).max())
# plt.colorbar()
# plt.title('error')
# plt.show()


# grid, image = ds[0]
# grid = grid.unsqueeze(0).to(device)
# image = image.unsqueeze(0).to(device)

# downsample_ratio = 2
# test_data = (grid, image)  # [1, 512, 512, 2], [1, 512, 512, 3]
# train_data = (grid[:, ::downsample_ratio, ::downsample_ratio, :], image[:, ::downsample_ratio, ::downsample_ratio, :])  # [1, 256, 256, 2], [1, 256, 256, 3]
# torchvision.utils.save_image(train_data[1].cpu().permute(0, 3, 1, 2).data, f"outputs/phantom/train.jpeg")
# torchvision.utils.save_image(test_data[1].cpu().permute(0, 3, 1, 2).data, f"outputs/phantom/test.jpeg")

# # Downsample
# # exp = "downsample"
# # downsample_ratio = 2
# # train_data = (grid[:, ::downsample_ratio, ::downsample_ratio, :], image[:, ::downsample_ratio, ::downsample_ratio, :])  # [1, 256, 256, 2], [1, 256, 256, 3]

# # Randomsample
# exp = "randomsample"
# index_x = np.sort(np.random.permutation(np.arange(img_size))[:(img_size//2)])
# index_y = np.sort(np.random.permutation(np.arange(img_size))[:(img_size//2)])
# # train_data = (grid[:, :, index_x, :], image[:, :, index_x, :])
# # print(train_data[1].shape)
# # train_data = (grid[:, index_y, :, :], image[:, index_y, :, :])
# # print(train_data[1].shape)
# train_data = (grid[:, index_y, :, :][:, :, index_x, :], image[:, index_y, :, :][:, :, index_x, :])
# print(train_data[1].shape)
