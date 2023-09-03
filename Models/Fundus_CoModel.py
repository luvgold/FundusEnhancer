import numpy as np
import torch.nn.functional as F
from Utils.my_pooling import my_MaxPool2d, my_AvgPool2d
import math
import torch.nn as nn
import torch
from torchvision import models

class NoisyFactorizedLinear(nn.Linear):
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
            bias = bias + self.sigma_bias * (eps_out.t()*self.training)

        return F.linear(input, self.weight + self.sigma_weight * (noise_v*self.training), bias)

def super_loss(x, cnum, gnum):
    branch = x
    branch_1 = my_AvgPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch)
    # 尝试去平均后再SoftMax
    branch_mean=torch.mean(branch_1, dim=[2, 3], keepdim=True)
    M = (branch_1 > branch_mean).float()
    branch_1 = branch_1*M
    # print("stage1",branch_1.shape)
    branch = branch_1.reshape(branch_1.size(0), branch_1.size(1), branch_1.size(2) * branch_1.size(3))
    branch = F.softmax(branch, 2)
    branch = branch.reshape(branch.size(0), branch.size(1), x.size(2), x.size(2))

    branch = my_MaxPool2d(kernel_size=(1, gnum), stride=(1, gnum))(branch)

    branch = branch.reshape(branch.size(0), branch.size(1), branch.size(2) * branch.size(3))
    loss_2 = 1.0 - 0.5 * torch.mean(torch.sum(branch, 2))  # set margin = 3.0
    return branch_1 , loss_2*0.1


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
        self.attr_fc = nn.Conv2d(in_channels=self.attr_num, out_channels=5, kernel_size=(16, 16), stride=(16, 16),
                                 padding=0,
                                 bias=False)
        self.ffc = nn.Linear(512, 5)
        # self.rand_origin = nn.Linear(512, self.attr_num)
        self.rand_group = NoisyFactorizedLinear(self.attr_num, self.attr_num)

    def forward(self, x):
        fm = self.resnet(x)
        fm=fm.detach()
        vec = nn.AdaptiveAvgPool2d((1, 1))(fm)
        fm_vec = F.softmax(vec, 1)
        cam = (fm * fm_vec).sum(1, keepdim=True)
        vec = vec.view(vec.size(0), -1)
        x = self.compressor(fm.detach())
        x = F.relu(x)
        attr_branch, loss_2 = super_loss(x, self.cot_num, self.attr_num)
        attr_score = self.attr_fc(attr_branch)
        attr_score = attr_score.squeeze()
        score = self.ffc(vec) + attr_score

        attr_vec = nn.AdaptiveAvgPool2d((1, 1))(attr_branch)
        attr_vec=attr_vec.view(attr_vec.shape[0], -1)
        attr_vec=attr_vec.detach()

        attr_vec=self.rand_group(attr_vec).view(attr_vec.shape[0],attr_vec.shape[1],1,1)

        att_map = (attr_vec*attr_branch).sum(1, keepdim=True)

        return score, attr_branch, loss_2, cam.detach(), att_map


def makeGaussian(size, fwhm = 3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
class Saliency_Sampler_eye(nn.Module):
    def __init__(self, task_network, task_input_size, saliency_input_size):
        super(Saliency_Sampler_eye, self).__init__()
        self.pre = FundusGen_pre()
        for p in self.pre.named_parameters():
            if("rand_group" not in p[0]):
                p[1].requires_grad = False

        self.classfier = task_network
        self.grid_size = 31
        self.padding_size = 30
        self.global_size = self.grid_size + 2 * self.padding_size
        self.input_size = saliency_input_size
        self.input_size_net = task_input_size
        self.conv_last = nn.Conv2d(256, 1, kernel_size=1, padding=0, stride=1)
        gaussian_weights = torch.FloatTensor(makeGaussian(2 * self.padding_size + 1, fwhm=13))
        # Spatial transformer localization-network
        self.filter = nn.Conv2d(1, 1, kernel_size=(2 * self.padding_size + 1, 2 * self.padding_size + 1), bias=False)
        self.filter.weight[0].data[:, :, :] = gaussian_weights
        for p in self.filter.parameters():
            p.requires_grad = False
        self.P_basis = torch.zeros(2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size)
        for k in range(2):
            for i in range(self.global_size):
                for j in range(self.global_size):
                    self.P_basis[k, i, j] = k * (i - self.padding_size) / (self.grid_size - 1.0) + (1.0 - k) * (
                                j - self.padding_size) / (self.grid_size - 1.0)

        mask = torch.zeros((1,1,16, 16))
        mask[0, 0, 2:-2, 2:-2] = 1
        self.mask=mask.cuda()


    def create_grid(self, x):
        P = torch.autograd.Variable(
            torch.zeros(1, 2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size).cuda(),
            requires_grad=False)
        P[0, :, :, :] = self.P_basis
        P = P.expand(x.size(0), 2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size)

        x_cat = torch.cat((x, x), 1)
        p_filter = self.filter(x)
        x_mul = torch.mul(P, x_cat).view(-1, 1, self.global_size, self.global_size)
        all_filter = self.filter(x_mul).view(-1, 2, self.grid_size, self.grid_size)

        x_filter = all_filter[:, 0, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)
        y_filter = all_filter[:, 1, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)

        x_filter = x_filter / p_filter
        y_filter = y_filter / p_filter

        xgrids = x_filter * 2 - 1
        ygrids = y_filter * 2 - 1
        xgrids = torch.clamp(xgrids, min=-1, max=1)
        ygrids = torch.clamp(ygrids, min=-1, max=1)
        xgrids = xgrids.view(-1, 1, self.grid_size, self.grid_size)
        ygrids = ygrids.view(-1, 1, self.grid_size, self.grid_size)
        grid = torch.cat((xgrids, ygrids), 1)
        grid = nn.Upsample(size=(self.input_size_net, self.input_size_net), mode='bilinear')(grid)
        grid = torch.transpose(grid, 1, 2)
        grid = torch.transpose(grid, 2, 3)
        return grid

    def forward(self, xh):
        x=F.interpolate(xh,(512,512),mode='bilinear')
        score_pre, attr_branch, loss_2, cam, att_map = self.pre(x)
        if(self.mask.shape!=att_map.shape):
            self.mask.repeat(att_map.shape[0],1,1,1)
        att_map=att_map*self.mask
        cmap=F.relu(att_map)
        cmap=cmap.detach()
        cam=cam.detach()
        adp_val=F.adaptive_avg_pool2d(cmap,(1,1))
        adp_val = torch.clamp(adp_val*20,1e-8, 1.0)
        xs = nn.Upsample(size=(self.grid_size, self.grid_size), mode='bilinear')(cmap)
        xc = nn.Upsample(size=(self.grid_size, self.grid_size), mode='bilinear')(cam)
        xs=xs/adp_val+0.1  + xc

        xs_hm = nn.ReplicationPad2d(self.padding_size)(xs)
        grid = self.create_grid(xs_hm)

        x_sampled = F.grid_sample(xh, grid, align_corners=False)
        score = self.classfier(x_sampled)
        return attr_branch, xs, x_sampled, score
