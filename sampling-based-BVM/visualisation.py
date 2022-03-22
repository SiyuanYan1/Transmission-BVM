import cv2
import os
import numpy as np
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
def visualize_prediction_init(pred,folder_path):
    folder_path = folder_path + 'temp/'
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        name = '{:02d}_init.png'.format(kk)
        cv2.imwrite(folder_path + name, pred_edge_kk)

def visualize_prediction_sample(preds,folder_path):
    folder_path = folder_path + 'samples/'
    for kk in range(preds.shape[1]):# for each batch
        for i in range(preds.shape[0]):  # for each sample
            pred_edge_kk = preds[i,kk,:,:,:]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            name = '{:02d}_{:02d}_sample.png'.format(kk,i)
            cv2.imwrite(folder_path + name, pred_edge_kk)

def visualize_prediction_var(vars,folder_path):
    vars=torch.squeeze(vars,1)#(b,w,h)
    folder_path = folder_path + 'var1/'
    for i in range(vars.shape[0]):
        var=vars[i,:,:]
        var=var.detach().cpu().numpy()
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        name = '{:02d}_sample.png'.format(i)
        fig=plt.figure()
        heatmap=plt.imshow(var,cmap='viridis')
        fig.colorbar(heatmap)
        fig.savefig(folder_path+name)
        plt.close()


def visualize_dis_out(pred,folder_path):
    vars=torch.squeeze(pred,1)#(b,w,h)
    folder_path = folder_path + 'var2/'
    for i in range(vars.shape[0]):
        var=vars[i,:,:]
        var=var.detach().cpu().numpy()
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        name = '{:02d}_sample.png'.format(i)
        fig=plt.figure()
        heatmap=plt.imshow(var,cmap='viridis')
        fig.colorbar(heatmap)
        fig.savefig(folder_path+name)
        plt.close()
    # folder_path = folder_path + 'temp/'
    # for kk in range(pred.shape[0]):
    #     pred_edge_kk = pred[kk,:,:,:]
    #     pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
    #     pred_edge_kk *= 255.0
    #     pred_edge_kk = pred_edge_kk.astype(np.uint8)
    #
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #     name = '{:02d}_dis_output.png'.format(kk)
    #     cv2.imwrite(folder_path + name, pred_edge_kk)

def visualize_dis_target(pred,folder_path):
    folder_path = folder_path + 'temp/'
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        name = '{:02d}_dis_target.png'.format(kk)
        cv2.imwrite(folder_path + name, pred_edge_kk)

def visualize_prediction_ref(pred,folder_path):
    folder_path = folder_path + 'temp/'
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        name = '{:02d}_ref.png'.format(kk)
        cv2.imwrite(folder_path + name, pred_edge_kk)

def visualize_gt(var_map,folder_path):
    folder_path = folder_path + 'temp/'
    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        name = '{:02d}_gt.png'.format(kk)
        #print(folder_path+name)
        cv2.imwrite(folder_path + name, pred_edge_kk)

def visualize_original_img(rec_img,folder_path):
    folder_path = folder_path + 'temp/'
    img_transform = transforms.Compose([
        transforms.Normalize(mean = [-0.4850/.229, -0.456/0.224, -0.406/0.225], std =[1/0.229, 1/0.224, 1/0.225])])
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk,:,:,:]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        name = '{:02d}_img.png'.format(kk)
        current_img = current_img.transpose((1,2,0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(folder_path+name, new_img)
