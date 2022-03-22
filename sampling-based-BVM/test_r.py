import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc
from model.ResNet_models import Generator, FCDiscriminator
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

###our test set
dataset_path = 'dataset/img/'

generator = Generator(channel=opt.feat_channel, latent_dim=8)
generator.load_state_dict(torch.load('./models_final/Model_50_gen.pth'))

generator.cuda()
generator.eval()

dis_model = FCDiscriminator(ndf=64)
dis_model.load_state_dict(torch.load('models_final/Model_50_dis.pth'))
dis_model.cuda()
dis_model.eval()

test_datasets = ['']
pre_root = './results_final_pred_out1/'
for dataset in test_datasets:
    save_path = pre_root + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset
    test_loader = test_dataset(image_root,gt_root, opt.testsize)
    for i in tqdm(range(test_loader.size)):
        image, gt, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        gt=gt.cuda()
        sal_pred = []
        """save samples"""
        save_path_sal = save_path + 'sal/'
        if not os.path.exists(save_path_sal):
            os.makedirs(save_path_sal)

        save_path_mean = save_path + 'sal_mean/'
        if not os.path.exists(save_path_mean):
            os.makedirs(save_path_mean)

        """get one prediction"""
        pred = generator(image, training=False)
        """get estimated variance"""
        Dis_output = dis_model(image, torch.sigmoid(pred))
        Dis_output = F.upsample(Dis_output, size=[WW, HH], mode='bilinear', align_corners=False)

        Dis_output = Dis_output.sigmoid().data.cpu().numpy().squeeze()
        es_var = 255*(Dis_output - Dis_output.min()) / (Dis_output.max() - Dis_output.min() + 1e-8)
        save_path_dis = pre_root + 'estimated_var/'
        if not os.path.exists(save_path_dis):
            os.makedirs(save_path_dis)

        fig = plt.figure()
        heatmap = plt.imshow(es_var, cmap='viridis')
        fig.colorbar(heatmap)
        fig.savefig(save_path_dis + name)
        plt.close()
        """MC-sampling"""
        loss_op=1000000
        aleatoric_op=0
        for k in range(50):
            res = generator(image, training=False)
            sal_pred.append(res.detach())
            
            # temprature scaling
            res = res.sigmoid()
            aleatoric_temp=-(res*torch.log(res+1e-8))
            check_loss=F.binary_cross_entropy_with_logits(aleatoric_temp,gt.sigmoid() , reduce='none')
            aleatoric_value=aleatoric_temp.sum(dim=1).mean()
            if check_loss<loss_op:
                loss_op=check_loss
                aleatoric_op=aleatoric_temp
                res_op=res
        res_op = F.upsample(res_op, size=[WW, HH], mode='bilinear', align_corners=False)
        res_op = res_op.sigmoid().data.cpu().numpy().squeeze()
        res_op = 255 * (res_op - res_op.min()) / (res_op.max() - res_op.min() + 1e-8)
        cv2.imwrite(save_path_sal + name, res_op)
        #aleatoric_op is the aleatoric uncertainty.
        aleatoric_map=aleatoric_op
        aleatoric_map = F.upsample(aleatoric_map, size=[WW, HH], mode='bilinear', align_corners=False)
        aleatoric_map = aleatoric_map.sigmoid().data.cpu().numpy().squeeze()
        aleatoric_map= (aleatoric_map - aleatoric_map.min()) / (aleatoric_map.max() - aleatoric_map.min() + 1e-8)
        save_path_var = pre_root + 'var_maps_aleatoric/'
        if not os.path.exists(save_path_var):
            os.makedirs(save_path_var)

        fig = plt.figure()
        heatmap = plt.imshow(aleatoric_map, cmap='viridis')
        fig.colorbar(heatmap)
        fig.savefig(save_path_var + name)
        plt.close()
        sal_preds = torch.sigmoid(sal_pred[0]).clone()
        for iter in range(1, 50):
            sal_preds = torch.cat((sal_preds, torch.sigmoid(sal_pred[iter])), 1)
        mean_map = torch.mean(sal_preds, 1, keepdim=True)

        save_map=mean_map
        save_map = F.upsample(save_map, size=[WW, HH], mode='bilinear', align_corners=False)
        save_map = save_map.sigmoid().data.cpu().numpy().squeeze()
        save_map = 255 * (save_map - save_map.min()) / (save_map.max() - save_map.min() + 1e-8)
        cv2.imwrite(save_path_mean + name[:-4] + '_' + str(0) + '.png', save_map)

        """total ucnertainty"""
        total_uncertainty_temp=-(mean_map*torch.log(mean_map+ 1e-8))
        total_uncertainty=total_uncertainty_temp
        total_uncertainty = F.upsample(total_uncertainty, size=[WW, HH], mode='bilinear', align_corners=False)
        total_uncertainty = total_uncertainty.sigmoid().data.cpu().numpy().squeeze()
        total_uncertainty = (total_uncertainty - total_uncertainty.min()) / (
                    total_uncertainty.max() - total_uncertainty.min() + 1e-8)
        save_path_var = pre_root + 'var_maps_total/'
        if not os.path.exists(save_path_var):
            os.makedirs(save_path_var)

        fig = plt.figure()
        heatmap = plt.imshow(total_uncertainty, cmap='viridis')
        fig.colorbar(heatmap)
        fig.savefig(save_path_var + name)
        plt.close()
        """epistemic uncertainty"""
        temp=total_uncertainty-aleatoric_map
        episilon=abs(np.min(temp[temp<0]))
        temp[temp<0]=0

        epistemic_uncertainty = temp
        save_path_var = pre_root + 'var_maps_epistemic/'
        if not os.path.exists(save_path_var):
            os.makedirs(save_path_var)

        fig = plt.figure()
        heatmap = plt.imshow(epistemic_uncertainty, cmap='viridis')
        fig.colorbar(heatmap)
        fig.savefig(save_path_var + name)
        plt.close()
