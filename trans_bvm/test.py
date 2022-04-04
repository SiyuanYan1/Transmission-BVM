import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc
from model.ResNet_models import Generator
from data import test_dataset
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--langevin_step_num_des', type=int, default=10, help='number of langevin steps for ebm')
parser.add_argument('-langevin_step_size_des', type=float, default=0.026,help='step size of EBM langevin')
parser.add_argument('--energy_form', default='identity', help='tanh | sigmoid | identity | softplus')
parser.add_argument('--latent_dim', type=int, default=3, help='latent dim')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
opt = parser.parse_args()

# dataset_path = '/home/jing-zhang/jing_file/ICCV2021/COD_RGBD/Dataset/Test/'
dataset_path = '/students/u7050317/AAAI/dataset/SMOKE5K/test/img/'
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.load_state_dict(torch.load('./models/ucnet_trans3/Model_50_gen.pth'))

generator.cuda()
generator.eval()

# test_datasets = ['CAMO','CHAMELEON','COD10K']
test_datasets = ['']

def compute_energy(disc_score):
    if opt.energy_form == 'tanh':
        energy = torch.tanh(-disc_score.squeeze())
    elif opt.energy_form == 'sigmoid':
        energy = F.sigmoid(-disc_score.squeeze())
    elif opt.energy_form == 'identity':
        energy = -disc_score.squeeze()
    elif opt.energy_form == 'softplus':
        energy = F.softplus(-disc_score.squeeze())
    return energy

for dataset in test_datasets:
    save_path = './results/ucnet/' + dataset+'/'
    #save_path = './results/ResNet50/holo/train/left/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset
    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print (i)
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        generator_pred = generator.forward(image, training=False)
        res = generator_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res)
