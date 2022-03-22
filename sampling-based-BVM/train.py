import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os, argparse
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Generator, FCDiscriminator
from data import get_loader
from utils import adjust_lr, AvgMeter
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation
from tools import *
from visualisation import *


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--lr_dis', type=float, default=1e-5, help='learning rate for discriminator')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=480, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--modal_loss', type=float, default=0.5, help='weight of the fusion modal')
parser.add_argument('--dis_start_epoch', type=int, default=5, help='start epoch of discriminator')

reg_weight=1e-4
latent_weight=10.0
vae_loss_weight=0.4
sampling_iter=10
opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = Generator(channel=opt.feat_channel,latent_dim=8)
generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

predictive_net = FCDiscriminator()
predictive_net.cuda()
predictive_net_params = predictive_net.parameters()
predictive_net_optimizer = torch.optim.Adam(predictive_net_params, opt.lr_dis)

## synthetic ##


##SMOKE5K ##
image_root='/dataset/img/'
gt_root='/dataset/img/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [1]  # multi-scale training


def structure_loss(pred, mask, weight=None):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts
    if weight == None:
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    else:
        weit = 1 + 5 * weight

    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()



print("Let's go!")
for epoch in range(1, (opt.epoch+1)):
    # scheduler.step()
    generator.train()
    predictive_net.train()
    loss_record = AvgMeter()
    loss_record_dis = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    print('Discriminator Learning Rate: {}'.format(var_approx_net_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            var_approx_net_optimizer.zero_grad()
            images, gts = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()
            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            pred_prior_init, pred_prior_ref, kl_latent_loss = generator(images,gts)

            # l2 regularizer the inference model
            reg_loss =  l2_regularisation(generator.x_encoder)+ l2_regularisation(generator.sal_endecoder)
            reg_loss = reg_weight * reg_loss
            # kl divergence
            kl_latent_loss = latent_weight * kl_latent_loss
            
            # sturcture loss: wbce+wiou
           
            sal_loss = 0.5 * (structure_loss(pred_prior_init, gts) + structure_loss(pred_prior_ref, gts))
            """1: gen_loss_cvae=structure_loss+kl divergence"""
            gen_loss_cvae = sal_loss + kl_latent_loss
            gen_loss_cvae = vae_loss_weight * gen_loss_cvae
            """3.total loss"""
            # gen_loss_total = gen_loss_gsnn
            # print('LOSS:',gen_loss_total)
            gen_loss_total = gen_loss_cvae + reg_loss
            sal_preds = torch.sigmoid(pred_prior_ref).clone()

            """sampling using priornet k+1 times"""
            with torch.no_grad():
                for kk in range(sampling_iter):
                    _,pred_prior_ref, _ = generator(images, gts)
                    # (batch,samples,w,h)(5,11,352,352)
                    sal_preds = torch.cat((sal_preds, torch.sigmoid(pred_prior_ref)),1)

            var_map = torch.var(sal_preds, 1, keepdim=True)
            var_map = (var_map - var_map.min()) / (var_map.max() - var_map.min() + 1e-8)
            mean_map = torch.mean(sal_preds, 1, keepdim=True)  # (5,1,352,352)
            gen_loss_total.backward()
            generator_optimizer.step()
            """train var approx net"""
            var_map_real=var_map.detach()
            mean_map1=mean_map.detach()
            
           
            var_map_approx=predictive_net(images,mean_map1)
            var_map_approx=F.upsample(var_map_approx, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            consist_loss = mse_loss(torch.sigmoid(var_map_approx), var_map_real)
            consist_loss.backward()
            var_approx_net_optimizer.step()
            

            folder_path = 'visual_map_final/vis_e' + str(epoch) + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if i == 1:  # only save the first batch for each epoch
                # visualize_prediction_sample(torch.sigmoid(init_pred),folder_path)
                visualize_prediction_var(torch.sigmoid(var_map), folder_path)
                visualize_prediction_init(torch.sigmoid(mean_map), folder_path)
                visualize_dis_out(torch.sigmoid(var_map_approx), folder_path)
                visualize_gt(gts, folder_path)
                visualize_original_img(images, folder_path)

            if rate == 1:
                loss_record.update(gen_loss_total.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Gen Loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,loss_record.show()))

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = 'models_final/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 5 == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
        torch.save(predictive_net.state_dict(),save_path+'Model' + '_%d' % epoch + '_dis.pth')




