import os

import matplotlib.pyplot as plt
import numpy as np
from unet_models import *
from skimage.metrics import normalized_root_mse as nrmse
import torch
# import torch.optim
from utils_unet.denoising_utils import *

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from leaptorch import Projector
from leaptorch import FBP
from specTool import DEsino_Syn, DECT_Syn
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
numAngles = 1000
numRows = 1
numCols = 384
pixelSize = 1
proj_FP = Projector(forward_project=True, use_static=True, use_gpu=True, gpu_device=torch.device('cuda:0'),
                    batch_size=2)
proj_FP.leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5 * (numRows - 1),
                                0.5 * (numCols - 1), proj_FP.leapct.setAngleArray(numAngles, 180.0))
proj_FP.leapct.set_volume(256, 256, 1, pixelSize * 384.0 / 256.0, pixelSize)
proj_FP.allocate_batch_data()

proj_FBP = FBP(forward_FBP=True, use_static=True, use_gpu=True, gpu_device=torch.device('cuda:0'),batch_size=1)
proj_FBP.leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5 * (numRows - 1),
                                0.5 * (numCols - 1), proj_FBP.leapct.setAngleArray(numAngles, 180.0))
proj_FBP.leapct.set_volume(256, 256, 1, pixelSize * 384.0 / 256.0, pixelSize)
proj_FBP.allocate_batch_data()
spec_low = torch.from_numpy(np.load('raw_data/spec_data/spec_low.npy')[25:81]).float().cuda()  # 80kV
spec_high = torch.from_numpy(np.load('raw_data/spec_data/spec_high.npy')[25:141]).float().cuda()  # 140kV
atten_iodine = torch.from_numpy(np.load('raw_data/spec_data/atten_iodine.npy')).float().cuda()
atten_water = torch.from_numpy(np.load('raw_data/spec_data/atten_water.npy')).float().cuda()

mask = np.load('generated_data/spec351/mask.npy')

water_ID = np.load('generated_data/spec351/water_FBP.npy') * 1000
iodine_ID = np.load('generated_data/spec351/iodine_FBP.npy') * mask* 1000
# E1_recon1500 = np.load(
#         'generated_data/spec35/recon_E1_T1.npy')* mask
# E2_recon1500 = np.load(
#     'generated_data/spec35/recon_E2_T2.npy') * mask
E1_recon1500 = np.load(
        'outputs/train_byStep_spec/spec351E1/time_num5_img256_proj1000_SIREN_1024_256_8_L2_lr1e-10_encoder_gauss_scale4_size512/images/extrapolation1500.npy')* mask
E2_recon1500 = np.load(
    'outputs/train_byStep_spec/spec351E2/time_num5_img256_proj1000_SIREN_1024_256_8_L2_lr1e-10_encoder_gauss_scale4_size512/images/extrapolation1500.npy') * mask

FBP_E1_mu = (np.load('generated_data/spec35/recon_E1_T1.npy'))* mask
FBP_E2_mu = (np.load('generated_data/spec35/recon_E2_T2.npy'))* mask

decom_matrix = np.load('generated_data/spec351/dec_IDlsq.npy') # different methods to calculate the decomposition matrix
# decom_matrix = np.load('generated_data/spec351/dec_ID.npy')
# decom_matrix = np.load('generated_data/spec351/dec_final.npy')

E1_recon1500_mu = (E1_recon1500/1000*0.03127156+0.03127156)* mask
E2_recon1500_mu = (E2_recon1500/1000*0.02727755625413329+0.02727755625413329)* mask

water_FBP = (decom_matrix[0][0] * FBP_E1_mu  + decom_matrix[0][1] * FBP_E2_mu ) * mask* 1000
iodine_FBP = (decom_matrix[1][0] * FBP_E1_mu  + decom_matrix[1][1] * FBP_E2_mu ) * mask* 1000

water_INR = (decom_matrix[0][0] * FBP_E1_mu  + decom_matrix[0][1] * FBP_E2_mu ) * mask* 1000
iodine_INR = (decom_matrix[1][0] * FBP_E1_mu  + decom_matrix[1][1] * FBP_E2_mu ) * mask* 1000

# water_INR = np.load('generated_data/spec351/water_ID.npy') * mask * 1000
# iodine_INR = np.load('generated_data/spec351/iodine_ID.npy') * mask * 1000

water_truth = np.load('generated_data/spec351/T1500_water.npy')
iodine_truth = np.load('generated_data/spec351/T1500_iodine.npy')

mask_t = torch.from_numpy(mask).cuda().type(dtype)
CT_E1truth = DECT_Syn(torch.from_numpy(water_truth).float().cuda(), torch.from_numpy(iodine_truth).float().cuda(), spec_low, atten_water[:56], atten_iodine[:56], proj_FP,proj_FBP).clone() * mask_t
CT_E2truth = DECT_Syn(torch.from_numpy(water_truth).float().cuda(), torch.from_numpy(iodine_truth).float().cuda(), spec_high, atten_water, atten_iodine, proj_FP,proj_FBP) * mask_t

print(nrmse(E1_recon1500_mu, CT_E1truth.cpu().detach().numpy()))
print(nrmse(E2_recon1500_mu, CT_E2truth.cpu().detach().numpy()))

# input_image = torch.from_numpy(np.array([E1_recon1500_mu, E2_recon1500_mu]))
# input_image = torch.from_numpy(np.array([CT_E1truth.cpu().detach().numpy(), CT_E2truth.cpu().detach().numpy()]))
# sino_E1 = proj_FP(torch.from_numpy(E1_recon1500).float().cuda().unsqueeze(0).unsqueeze(0))[0,:,0,:].clone()
# sino_E2 = proj_FP(torch.from_numpy(E2_recon1500).float().cuda().unsqueeze(0).unsqueeze(0))[0,:,0,:].clone()
# train_data = torch.stack((torch.from_numpy(E1_recon1500_mu), torch.from_numpy(E2_recon1500_mu)),dim=0).type(dtype).cuda()
train_data = torch.stack((torch.from_numpy(FBP_E1_mu).cuda(), torch.from_numpy(FBP_E2_mu).cuda()),dim=0).type(dtype)

# INPUT = 'noise'  # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

OPTIMIZER = 'adam'  # 'LBFGS'
save_every = 50
show_every = 200
exp_weight = 0.99

num_iter = 3000
input_depth = 2  # 输入channel为2
output_depth = 2  # 输出channel为2
train_psnr_plot = []
net = skip(
    input_depth, output_depth,
    num_channels_down=[16, 32, 64, 128, 128],
    num_channels_up=[16, 32, 64, 128, 128],
    num_channels_skip=[4, 4, 4, 4, 4],
    filter_size_down=3,
    filter_size_up=3,
    filter_skip_size=3,
    upsample_mode='bilinear',
    need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU')

net = net.type(dtype)

net_input = get_noise(input_depth, 'noise', (256,256)).type(dtype).detach()
#
# net_input = torch.reshape(input_image, (1, 2, 256, 256)).type(dtype).cuda()

# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()])
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)
LR = 0.00001
# alpha = 0.000000001
out_avg = None
last_net = None
psrn_noisy_last = 0

lamda = 1.3797849707218919
MDimgs_list = []
loss_curve = []
parameters = get_params(OPT_OVER, net, net_input)
optimizer = torch.optim.Adam(parameters, lr=LR)
nrmse_list = []
# spec_low.requires_grad_(True)
# spec_high.requires_grad_(True)
# atten_water.requires_grad_(True)
# atten_iodine.requires_grad_(True)
for i in range(num_iter):
    net.train()
    optimizer.zero_grad()
    proj_FP.train()
    proj_FBP.train()
    output = net(net_input)[0, :, :, :] * mask_t  # (1,2,512,512)
    CT_E1t = DECT_Syn((output[0] * 20 + 1000)*mask_t , output[1], spec_low, atten_water[:56], atten_iodine[:56], proj_FP,proj_FBP).clone()
    CT_E2t = DECT_Syn((output[0] * 20 + 1000)*mask_t , output[1], spec_high, atten_water, atten_iodine, proj_FP,proj_FBP).clone()
    total_loss = mse(CT_E1t, train_data[0]) + lamda * mse(CT_E2t, train_data[1])
    loss_curve.append(total_loss.item())
    total_loss.backward()
    if i % save_every == 0:
        net.eval()
        proj_FP.eval()
        proj_FBP.eval()
        out = output.detach().cpu().numpy()
        print('Iter:{} NRMSE:water:{:.4g} | iodine:{:.4g}'.format(i, nrmse(water_truth, (out[0, :, :] * 20 + 1000)* mask),
                                                                  nrmse(iodine_truth, out[1])))
        nrmse_list.append(np.array([nrmse(water_truth, (out[0, :, :] * 20 + 1000)* mask),
                                                                  nrmse(iodine_truth, out[1])]))
        # print('loss: E1:{:.4g} | E2:{:.4g}'.format(mse(CT_E1, train_data[0]).item(), mse(CT_E2, train_data[1]).item()))
        MDimgs_list.append(out)
    if i % show_every == 0:
        net.eval()
        proj_FP.eval()
        proj_FBP.eval()
        out = output.detach().cpu().numpy()
        plt.figure(figsize=(24, 12))
        plt.subplot(2, 4, 1)
        plt.imshow(water_truth, cmap='gray', vmin=950, vmax=1100)
        plt.axis('off')
        plt.subplot(2, 4, 2)
        plt.imshow(water_ID, cmap='gray', vmin=950, vmax=1100)
        plt.axis('off')
        plt.subplot(2, 4, 3)
        plt.imshow(water_FBP, cmap='gray', vmin=950, vmax=1100)
        plt.axis('off')
        plt.subplot(2, 4, 4)
        plt.imshow((out[0] * 20 + 1000)* mask, cmap='gray', vmin=950, vmax=1100)
        plt.axis('off')
        plt.subplot(2, 4, 5)
        plt.imshow(iodine_truth, cmap='hot', vmin=0, vmax=3)
        plt.axis('off')
        plt.subplot(2, 4, 6)
        plt.imshow((iodine_ID+0.9)*mask, cmap='hot', vmin=0, vmax=3)
        plt.axis('off')
        plt.subplot(2, 4, 7)
        plt.imshow((iodine_FBP)*mask, cmap='hot', vmin=0, vmax=3)
        plt.axis('off')
        plt.subplot(2, 4, 8)
        plt.imshow((out[1])*mask, cmap='hot', vmin=0, vmax=3)
        plt.axis('off')
        plt.title('Iter: {}'.format(i))
        plt.tight_layout()
        plt.show()
        # plt.imshow(out[1]-iodine_truth,cmap='RdBu',vmin=-1,vmax=1)
        # plt.title('error Iodine Iter: {:.4g}'.format(i))
        # plt.show()
        #
        # plt.imshow((CT_E1t.detach().cpu().numpy() - 0.03127156) / 0.03127156 * 1000 * mask, cmap='gray', vmin=0, vmax=200)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        # plt.imshow((CT_E2t.detach().cpu().numpy() - 0.02727755625413329) / 0.02727755625413329 * 1000 * mask, cmap='gray', vmin=0,
        #            vmax=200)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        #
        # plt.imshow((CT_E1t.detach().cpu().numpy() - CT_E1truth.cpu().detach().numpy()) / 0.03127156 * 1000 * mask, cmap='RdBu',
        #            vmin=-10, vmax=10)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        # plt.imshow((CT_E2t.detach().cpu().numpy() - CT_E2truth.cpu().detach().numpy()) / 0.02727755625413329 * 1000 * mask,
        #            cmap='RdBu', vmin=-10, vmax=10)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        print('Iter:{} NRMSE:water:{:.4g} | iodine:{:.4g}'.format(i, nrmse(water_truth, (out[0, :, :] * 20 + 1000)* mask),
                                                                  nrmse(iodine_truth, out[1])))
        print('Total_loss:{:.4g}'.format(total_loss.item()))
        # print('loss: E1:{:.4g} | E2:{:.4g}'.format(mse(CT_E1, train_data[0]).item(), mse(CT_E2, train_data[1]).item()))
        # MDimgs_list.append(out)
        # out_np = torch_to_np(out)
        # plot_image_grid([np.clip(out_np, 0, 1),
        #                  np.clip(torch_to_np(out_avg), 0, 1)], factor=4, nrow=1)

    # Backtracking
    # if i % show_every:
    #     if psrn_noisy - psrn_noisy_last < -5:
    #         print('Falling back to previous checkpoint.')
    #
    #         for new_param, net_param in zip(last_net, net.parameters()):
    #             net_param.data.copy_(new_param.cuda())
    #
    #         return total_loss * 0
    #     else:
    #         last_net = [x.detach().cpu() for x in net.parameters()]
    #         psrn_noisy_last = psrn_noisy    # psnr下降过多就回溯

    i = i + 1
    optimizer.step()
loss_curve = np.array(loss_curve)
plt.plot(np.arange(num_iter),np.log(loss_curve))
plt.show()

# np.save('paper_plot/fig11/spec351/water_dip.npy',(MDimgs_list[40][0] * 20 + 1000) * mask)
# np.save('paper_plot/fig11/spec351/iodine_dip.npy',MDimgs_list[40][1]* mask)
# np.save('paper_plot/fig16/loss_md20000.npy', loss_curve)

# plt.plot(list(range(0, num_iter, 1)), loss_plot)
# plt.xlabel('Iteration')
# plt.ylabel('train_loss')
# plt.title('Loss Curve')
# plt.show()
# plt.plot(list(range(0, num_iter, show_every)), gt0_psnr_plot, color='red', label='gt0_psnr') # 绘制第一条线，指定颜色为红色，标签为 line1
# plt.plot(list(range(0, num_iter, show_every)), gt1_psnr_plot, color='blue', label='gt1_psnr') # 绘制第二条线，指定颜色为蓝色，标签为 line2
# plt.title('psnr_gt')
# plt.xlabel('Iteration')
# plt.ylabel('psnr_value')
# plt.legend() # 添加图例
# plt.show()
# with open('/data/liqiaoxin/test/CT/Deep_image_prior/deep-image-prior-master/outputs/plot_loss/train_loss_LR{}_iter{}.pkl'.format(LR,num_iter), 'wb') as f:
#     pickle.dump(loss_plot, f)
