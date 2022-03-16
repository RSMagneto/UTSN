from skimage import io as skimage
import torch
import pdb
import numpy
import math

def matRead(data,opt):
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.to(opt.device).type(torch.cuda.FloatTensor)
    data=torch.cat([data,data,data],dim=0)
    data=data[:1800,:,:,:]
    return data

def convert_image_mat(inp,opt):
    inp=inp[-1,:,:,:]
    inp = inp.to(torch.device('cpu'))
    inp = inp.numpy().transpose((1,2,0))
    inp=inp*2047.
    return inp

def test_matRead(data,opt):
    data=data[None, :, :, :]
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.type(torch.cuda.FloatTensor)
    data = data.to(opt.device)
    return data

def test_matRead1(data,opt):
    data=data[None, :, :, :]
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.type(torch.FloatTensor)
    data = data.to(opt.device)
    return data

def denorm(x):
    out=(x+1)/2
    return out.clamp(0,1)

def convert_image_np(inp):
    inp=inp[-1,:,:,:]
    inp = inp.to(torch.device('cpu'))
    inp = inp.numpy().transpose((1,2,0))
    inp = numpy.clip(inp,0,1)
    inp=inp*2047.
    return inp

def getBatch(ms_data,pan_data,gt_data, bs):
    N = gt_data.shape[0]
    batchIndex = numpy.random.randint(0, N, size=bs)
    msBatch = ms_data[batchIndex, :, :, :]
    panBatch = pan_data[batchIndex, :, :, :]
    gtBatch = gt_data[batchIndex, :, :, :]
    return msBatch,panBatch,gtBatch

def getBatch2se(gtt_data,gt_data, bs):
    batchIndex = numpy.random.randint(0, 201, size=bs)
    gttBatch = gtt_data[batchIndex, :, :, :]
    gtBatch = gt_data[batchIndex, :, :, :]
    return gttBatch, gtBatch

def getBatch2seik(gtt_data,gt_data, bs):
    gtt_data = gtt_data[None, :, :, :]
    gt_data = gt_data[None, :, :, :]
    batchIndex = 0
    gttBatch = gtt_data[batchIndex, :, :, :]

    gtBatch = gt_data[batchIndex, :, :, :]
    return gttBatch, gtBatch
