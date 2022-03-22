import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse

from scipy import misc
from model.ResNet_models import Generator,FCDiscriminator
from data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from visualisation import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=480, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
opt = parser.parse_args()
###synth
dataset_path = '/data/img/'

generator = Generator(channel=opt.feat_channel,latent_dim=8)
generator.load_state_dict(torch.load('./models_final/Model_50_gen.pth'))

generator.cuda()
generator.eval()

dis_model=FCDiscriminator(ndf=64)
dis_model.load_state_dict(torch.load('models_final/Model_50_dis.pth'))
dis_model.cuda()
dis_model.eval()

test_datasets = ['DS01','DS02','DS03']
# test_datasets=['']
pre_root='./results_final/'
for dataset in test_datasets:
    save_path =  pre_root+ dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/image/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in tqdm(range(test_loader.size)):
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        sal_pred=[]
        #save samples
        save_path_sal = save_path + 'sal/'
        if not os.path.exists(save_path_sal):
            os.makedirs(save_path_sal)
        for k in range(6):
            res=generator(image,training=False)
            sal_pred.append(res.detach())
            res=F.upsample(res,size=[WW,HH],mode='bilinear',align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            # res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
            res[res >= 0.5] = 1
            res[res < 0.5] = 0
            pred = 255 * res
        cv2.imwrite(save_path_sal + name[:-4] + '_' + str(0) + '.png', pred)
        #save variance map
        sal_preds = torch.sigmoid(sal_pred[0]).clone()
        for iter in range(1, 6):
            sal_preds = torch.cat((sal_preds, torch.sigmoid(sal_pred[iter])), 1)
        # print(sal_preds.size())
        var_map = torch.var(sal_preds, 1, keepdim=True)
        var_map = (var_map - var_map.min()) / (var_map.max() - var_map.min() + 1e-8)
        var_map = F.upsample(var_map, size=[WW, HH], mode='bilinear', align_corners=False)
        res = var_map.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        save_path_var =pre_root+ 'var_maps_real/'
        if not os.path.exists(save_path_var):
            os.makedirs(save_path_var)
        name = '{:02d}_sample.png'.format(i)
        fig = plt.figure()
        heatmap = plt.imshow(res, cmap='viridis')
        fig.colorbar(heatmap)
        fig.savefig(save_path_var + name)
        plt.close()
        # get dis result
        mean_map = torch.mean(sal_preds, 1, keepdim=True)
        """get estimated variance"""
        Dis_output=dis_model(image,mean_map)
        Dis_output = F.upsample(Dis_output, size=[WW, HH], mode='bilinear', align_corners=False)
        res = Dis_output
        res = res.sigmoid().data.cpu().numpy().squeeze()
        es_var = (res - res.min()) / (res.max() - res.min() + 1e-8)
        save_path_dis = pre_root + 'estimated_var/'
       
        if not os.path.exists(save_path_dis):
            os.makedirs(save_path_dis)
        name = '{:02d}_sample_.png'.format(i)
        # cv2.imwrite(save_path + name, res)
        fig = plt.figure()
        heatmap = plt.imshow(es_var, cmap='viridis')
        fig.colorbar(heatmap)
        fig.savefig(save_path_dis + name)
        plt.close()


