import torch
from skimage import measure
import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import cohen_kappa_score
from torch.optim import *
import os
import cv2
import logging
from torch.nn.parameter import Parameter
import torch.nn as nn
def adjust_learning_rate(optimizer, epoch, init_lr,stride=30,x=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // stride))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)

def max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    return in_.div(max_+1e-8)


def AOLM(A):
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    M = (A > a).float()
    coordinates = []
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(15, 15)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))


        intersection = (component_labels==(max_idx+1)).astype(int)
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 14, 14]
            print('there is one img no intersection')
        else:
            bbox = prop[0].bbox


        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def mean_and_std(train_dataset, batch_size, num_workers):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    num_samples = 0.
    channel_mean = torch.Tensor([0., 0., 0.])
    channel_std = torch.Tensor([0., 0., 0.])
    for samples in tqdm(loader):
        X, _ = samples
        channel_mean += X.mean((2, 3)).sum(0)
        num_samples += X.size(0)
    channel_mean /= num_samples

    for samples in tqdm(loader):
        X, _ = samples
        batch_samples = X.size(0)
        X = X.permute(0, 2, 3, 1).reshape(-1, 3)
        channel_std += ((X - channel_mean) ** 2).mean(0) * batch_samples
    channel_std = torch.sqrt(channel_std / num_samples)

    return channel_mean.tolist(), channel_std.tolist()

def quadratic_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')

def inverse_gumbel_cdf(y, mu, beta):
    return mu - beta * np.log(-np.log(y))

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def gumbel_softmax_sampling(h, mu=0, beta=1, tau=0.5):
    """
    h : (N x K) tensor. Assume we need to sample a NxK tensor, each row is an independent r.v.
    """
    shape_h = h.shape
    p = F.softmax(h, dim=1)
    y = torch.rand(shape_h) + 1e-25  # ensure all y is positive.
    g = inverse_gumbel_cdf(y, mu, beta)
    g=g.cuda()
    x = torch.log(p) + g  # samples follow Gumbel distribution.
    # using softmax to generate one_hot vector:
    x = x/tau
    x = F.softmax(x, dim=1)  # now, the x approximates a one_hot vector.
    return x


def submean(dir,save):
    #
    image_path =dir
    file_names = os.listdir(image_path)
    count = 0
    save_path=save
    for file in file_names:

        mean = np.zeros(3, np.int64)
        imgs=os.listdir(image_path + '/' + file)
        for imgn in tqdm(imgs):
            # print(imgn)
            pre=image_path + '/' + file + '/' + imgn
            post=save_path + '/' + file + '/' + imgn
            img = cv2.imread(pre)
            count += 1
            mean += np.sum(img, axis=(0, 1)).astype(int)
            h, w = img.shape[0:-1]
            mean = mean / (1.0 * h * w )
            # print(h, w, count, mean)
            img=img-mean
            cv2.imwrite(post,img)


class Focal_Loss():
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        eps = 1e-7
        y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)

        target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)

        ce = -1 * torch.log(y_pred + eps) * target
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)

def adjust_learning_rate_warm(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=20):
    warmup_epoch = warmup
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    elif current_epoch < max_epoch:
        lr = lr_min + (lr_max - lr_min) * (
                    1 + np.cos(np.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + np.cos(np.pi * (current_epoch-max_epoch) / (max_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class WarmupCosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warm_up=0, T_max=10, start_ratio=0.1) -> object:
        """
        Description:
            - get warmup consine lr scheduler

        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_min: (float), minimum learning rate
            - lr_max: (float), maximum learning rate
            - warm_up: (int),  warm_up epoch or iteration
            - T_max: (int), maximum epoch or iteration
            - start_ratio: (float), to control epoch 0 lr, if ratio=0, then epoch 0 lr is lr_min

        Example:
            <<< epochs = 100
            <<< warm_up = 5
            <<< cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
            <<< lrs = []
            <<< for epoch in range(epochs):
            <<<     optimizer.step()
            <<<     lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            <<<     cosine_lr.step()
            <<< plt.plot(lrs, color='r')
            <<< plt.show()

        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.T_max = T_max
        self.start_ratio = start_ratio
        self.cur = 0  # current epoch or iteration

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warm_up == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warm_up != 0) & (self.cur <= self.warm_up):
            if self.cur == 0:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur + self.start_ratio) / self.warm_up
            else:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur) / self.warm_up
                # print(f'{self.cur} -> {lr}')
        else:
            # this works fine
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * \
                 (np.cos((self.cur - self.warm_up) / (self.T_max - self.warm_up) * np.pi) + 1)

        self.cur += 1

        return [lr for base_lr in self.base_lrs]


def to_one_hot(inp, num_classes, cuda=False):
    print(inp.shape)
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    if cuda:
        return Variable(y_onehot.cuda(), requires_grad=False)
    else:
        return Variable(y_onehot , requires_grad=False)

def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out * lam + out[indices] * (1 - lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)

    # t1 = target.data.cpu().numpy()
    # t2 = target[indices].data.cpu().numpy()
    # print (np.sum(t1==t2))
    return out, target_reweighted


def mixup_data(x, y, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

def per_image_standardization(x):
    y = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
    mean = y.mean(dim=1, keepdim = True).expand_as(y)
    std = y.std(dim=1, keepdim = True).expand_as(y)
    adjusted_std = torch.max(std, 1.0/torch.sqrt(torch.cuda.FloatTensor([x.shape[1]*x.shape[2]*x.shape[3]])))
    y = (y- mean)/ adjusted_std
    standarized_input =  y.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
    return standarized_input

def create_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(output_dir)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger



class GeM(torch.nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def gem(x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def AUC1(labels, pred_prob):
    auc1 = 0.
    num=len(labels)
    roc_point = []
    for i in range(num):
        i = pred_prob[i]
        TP = 0  # 真阳样本数
        FP = 0  # 假阳样本数
        TP_rate = 0.  # 真阳率
        FP_rate = 0.  # 假阳率
        pos_num = 0  # 预测真样本数

        # 计数过程
        for ind, prob in enumerate(pred_prob):
            if prob > i:
                pos_num += 1
            if prob > i and labels[ind] > 0.5:
                TP += 1
            elif prob > i and labels[ind] < 0.5:
                FP += 1
        if pos_num != 0:
            TP_rate = TP / sum(labels)
            FP_rate = FP / (num - sum(labels))
        roc_point.append([FP_rate, TP_rate])  # 记录ROC中的点
    # 画出ROC曲线
    # roc_point.sort(key=lambda x: x[0])
    # plt.plot(np.array(roc_point)[1:, 0], np.array(roc_point)[1:, 1])
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    # plt.show()

    # 计算每个小长方形的面积，求和即为auc
    lastx = 0.
    for x, y in roc_point:
        auc1 += (x - lastx) * y  # 底乘高
        lastx = x
    return auc1

def AUC2(labels, pred_prob):
    P_ind = []  # 正样本下标
    F_ind = []  # 负样本下标
    P_F = 0  # 正样本分数高于负样本的数量
    F_P = 0  # 负样本分数高于正样本的数量

    #  计数过程
    for ind, val in enumerate(labels):
        if val > 0.5:
            P_ind.append(ind)
        else:
            F_ind.append(ind)
    for Pi in P_ind:
        for Fi in F_ind:
            if pred_prob[Pi] > pred_prob[Fi]:
                P_F += 1
            else:
                F_P += 1
    auc2 = P_F / (len(P_ind) * len(F_ind))
    return auc2

def AUC3(labels, pred_prob):
    P_ind = []  # 正样本下标
    F_ind = []  # 负样本下标

    for ind, val in enumerate(labels):
        if val > 0.5:
            P_ind.append(ind)
        else:
            F_ind.append(ind)

    new_data = [[p, l] for p, l in zip(pred_prob, labels)]
    new_data.sort(key=lambda x: x[0])

    # 求正样本rank之和
    rank_sum = 0
    for ind, [prob, label] in enumerate(new_data):
        if label > 0.5:
            rank_sum += ind
    auc3 = (rank_sum - len(P_ind) * (1 + len(P_ind)) / 2) / (len(P_ind) * len(F_ind))
    return auc3