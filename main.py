import argparse
import model
import torch
import pdb
import functions
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import skimage
import numpy
import scipy.io as scio
from thop import profile
from time import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir', required=True)
    parser.add_argument('--input_ms', help='training lrms image name', required=True)
    parser.add_argument('--input_pan', help='training pan image name', required=True)
    parser.add_argument('--input_gt', help='training hrms image name', required=True)
    parser.add_argument('--channels', help='numble of image channel', default=4)
    parser.add_argument('--sr_factor', help='super resolution factor', type=float, default=4)
    parser.add_argument('--device', default=torch.device('cuda:1'))
    parser.add_argument('--epoch', default=20000)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='D’s learning rate')
    parser.add_argument('--gamma', type=float, default=0.01, help='scheduler gamma')
    parser.add_argument('--noise_amp', type=float, help='addative noise cont weight', default=0)  #0.001
    parser.add_argument('--batchSize', type=float, default=4)
    opt = parser.parse_args()

    netG = model.Generator(opt).to(opt.device)
    netG.apply(model.weights_init)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)

    gt_data= sio.loadmat('%s/%s' % (opt.input_dir, opt.input_gt))['gt']
    gt_data=functions.matRead(gt_data,opt)
    ms_data=sio.loadmat('%s/%s' % (opt.input_dir, opt.input_ms))['ms']
    ms_data=functions.matRead(ms_data,opt)
    # ms_data = torch.nn.functional.interpolate(ms_data, size=(gt_data.shape[2],gt_data.shape[3]), mode='bilinear')
    pan_data=sio.loadmat('%s/%s' % (opt.input_dir, opt.input_pan))['pan']
    pan_data = pan_data[:, :, :, None]
    pan_data=functions.matRead(pan_data,opt)

    loss = torch.nn.MSELoss()

    print('start train:')
    start_time=time()
    for i in range(opt.epoch):
        print('epoch:[%d/%d]' % (i, opt.epoch))
        start_time_epoch=time()
        for j in range(100):
            ms_image, pan_image, gt_image = functions.getBatch(ms_data, pan_data, gt_data, opt.batchSize)
            netG.zero_grad()
            fake_image = netG(ms_image,pan_image)
            rec_loss = loss(fake_image, gt_image)
            rec_loss.backward(retain_graph=True)
            optimizerG.step()
        end_time_epoch = time()
        print(end_time_epoch-start_time_epoch)

        schedulerG.step()
        torch.save(netG.state_dict(), os.path.join('Output/G_epoch_{}.pth'.format(i)))
    end_time = time()
    print(end_time-start_time)
