import argparse
import model
import torch
import pdb
import functions
import matplotlib.pyplot as plt
import math
from time import *
import pylab as pl
import os
import skimage.io
import numpy
import scipy.io as sio
from time import *
from thop import profile

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--test_ms', help='test lrms image name', required=True)
    parser.add_argument('--test_pan', help='test hrpan image name', required=True)
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--device',default=torch.device('cuda:1'))
    parser.add_argument('--channels', default=4)
    opt=parser.parse_args()

    start_time=time()
    ms_data = skimage.io.imread('%s' % (opt.test_ms))
    ms_data=functions.test_matRead(ms_data,opt)
    pan_data = skimage.io.imread('%s' % (opt.test_pan))
    pan_data = pan_data[ :, :, None]
    pan_data=functions.test_matRead(pan_data,opt)
    net = model.Generator(opt).to(opt.device)
    input1=(torch.randn(1,opt.channels,64,64)).to(opt.device).type(torch.cuda.FloatTensor)
    input2 = (torch.randn(1, 1, 256, 256)).to(opt.device).type(torch.cuda.FloatTensor)
    flops,params=profile(net,(input1,input2))
    print('flops:',flops,'params:',params)

    start_time=time()
    net.load_state_dict(torch.load('Output/G_epoch_{}.pth'), strict=False)
    in_s = net(ms_data, pan_data)
    skimage.io.imsave('result/01.tif', functions.convert_image_np((in_s.detach())).astype(numpy.uint16))
    # end_time = time()
    end_time=time()
    print(end_time-start_time)

if __name__ == '__main__':
    main()


