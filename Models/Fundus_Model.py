import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from Utils.my_pooling import my_MaxPool2d, my_AvgPool2d
from torchvision import models
import math
import torch.nn as nn
import torch
from torchvision import models
import torch.backends.cudnn as cudnn
import random

def Mask(nb_batch, channels,gnum):
    half=channels//2
    foo = [1] * half + [0] * (channels-half)
    bar = []
    for i in range(gnum):
        random.shuffle(foo)
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch, gnum * channels, 1, 1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar

def super_loss(x, cnum, gnum,mask=None):
    branch = x
    branch_1 = my_AvgPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch)

    if (mask.shape != branch_1.shape):
        mask.repeat(branch_1.shape[0], branch_1.shape[1], 1, 1)

    # 尝试去平均后再SoftMax
    branch_mean=torch.mean(branch_1, dim=[2, 3], keepdim=True)
    M = (branch_1 > branch_mean).float()
    if(mask==None):
        mask=1

    branch_1 = branch_1*M*mask
    # print("stage1",branch_1.shape)
    branch = branch_1.reshape(branch_1.size(0), branch_1.size(1), branch_1.size(2) * branch_1.size(3))
    branch = F.softmax(branch, 2)
    branch = branch.reshape(branch.size(0), branch.size(1), x.size(2), x.size(2))

    branch = my_MaxPool2d(kernel_size=(1, gnum), stride=(1, gnum))(branch)

    branch = branch.reshape(branch.size(0), branch.size(1), branch.size(2) * branch.size(3))
    loss_2 = 1.0 - 0.5*torch.mean(torch.sum(branch, 2))  # set margin = 3.0
    # loss_2 = lb - torch.mean(torch.sum(branch, 2))  # set margin = 3.0
    # print(loss_2)
    # if tk:
    #     mask = attMask(x,0.2)
    #     branch_1 = branch_1 * mask

    # attr_branch = my_AvgPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch_1)

    return branch_1 , F.relu(loss_2)

def attMask(attr_branch, scale=0.2):
    channels=8
    nb_batch=attr_branch.shape[0]
    drop_num = int(np.round(channels*scale))

    foo = [1] * drop_num + [0] * (channels-drop_num)
    bar = []

    random.shuffle(foo)
    bar += foo

    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch, channels, 1, 1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar

class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        bias = self.bias
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))

        with torch.no_grad():
            torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
            torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
            eps_in = func(self.epsilon_input)
            eps_out = func(self.epsilon_output)
            noise_v = torch.mul(eps_in, eps_out).detach()
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)

class FundusGen_pre(nn.Module):
    def __init__(self, attr_num=16, cot=512):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.attr_num = attr_num
        self.cot_num = cot // attr_num
        resnet_layer = nn.Sequential(*list(self.model.children())[:-2])
        self.resnet = resnet_layer

        self.compressor = nn.Sequential(
            *[nn.Conv2d(in_channels=512, out_channels=cot, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
              nn.Conv2d(in_channels=512, out_channels=cot, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)])
        self.attr_fc = nn.Conv2d(in_channels=self.attr_num, out_channels=5, kernel_size=(16, 16), stride=(16, 16), padding=0,
                                 bias=False)
        self.ffc = nn.Linear(512, 5)
        self.rand_origin = nn.Linear(512, self.attr_num)
        self.rand_group = NoisyFactorizedLinear(self.attr_num, self.attr_num)

        mask = torch.zeros((1, 1, 16, 16))
        mask[0, 0, 2:-2, 2:-2] = 1
        self.mask = mask.cuda()
    def forward(self, x):
        fm = self.resnet(x)
        # 取CAM图
        vec = nn.AdaptiveAvgPool2d((1, 1))(fm)
        fm_vec=F.softmax(vec,1)
        cam=(fm*fm_vec).sum(1, keepdim=True)
        vec = vec.view(vec.size(0), -1)
        x = self.compressor(fm.detach())
        x = F.relu(x)

        # x=x*self.mask
        attr_branch, loss_2 = super_loss(x, self.cot_num, self.attr_num, self.mask)
        attr_score = self.attr_fc(attr_branch)
        attr_score = attr_score.squeeze()

        score = self.ffc(vec)+attr_score
        return score, attr_branch, loss_2, cam