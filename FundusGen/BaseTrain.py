import sys
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from utils import *
import numpy as np
from Utils.fast import DataLoaderX, img_data_prefetcher
from timm.models.swin_transformer import swin_base_patch4_window12_384
import random
import os
from Models.Fundus_CoModel import Saliency_Sampler_eye
##### Param ###############
batch_size = 8
img_size=800
# DDR
# alr=0.001

# EyePacs
alr = 3e-5

num_workers = 8
print("batch_size", batch_size)
print("num_workers", num_workers)

max_epoch=128
# alr=0.01
momentum=0.9
# 5e-4
weight_decay=1e-4

##### Random seed setting
seed=1234
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

########################配置项######################################
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
data_path = '../Dataset/PAC800'
traindir = os.path.join(data_path, 'train')
valdir = os.path.join(data_path, 'valid')
###################################################################

def label_smooth(y_true, y_pred):
    y_true=((1-0.1)*y_true+0.05)
    return nn.CrossEntropyLoss()(y_true, y_pred)

def train(**kwargs):
    mode_path = "./SaveModels/"
    if not os.path.exists(mode_path):
        os.makedirs(mode_path)
    gen_mode_path = "..."
    save_mode_path = "..."
    res = swin_base_patch4_window12_384(pretrained=True,num_classes=5)
    best = 0.0
    model = Saliency_Sampler_eye(res, 384, 384)
    model.pre.load_state_dict(torch.load(gen_mode_path), strict=False)
    model.cuda()
    train_data = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
        ]))
    val_data = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]))
    dataset = train_data
    # dataset = semi_data
    print(len(train_data))
    print(len(val_data))

    train_dataloader = DataLoaderX(dataset, batch_size,
                                  shuffle=True, num_workers=num_workers,pin_memory=True)
    val_dataloader = DataLoaderX(val_data, batch_size,
                                shuffle=False, num_workers=num_workers,pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    # EyePacs
    optimizer = torch.optim.AdamW(model.parameters(), lr=alr, weight_decay=weight_decay)
    # DDR
    # optimizer = torch.optim.SGD(model.parameters(), lr=alr,
    #                              momentum=momentum,
    #                              weight_decay=weight_decay)
    cosine_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-8)

    tbar = int(len(dataset) / batch_size)
    vbar = int(len(val_data) / batch_size)
    for epoch in range(1, max_epoch+1):
        model.train()
        prefetcher = img_data_prefetcher(train_dataloader)
        data, label = prefetcher.next()
        iter_id = 0
        with tqdm(total=tbar) as pbar:
            while data is not None:
                iter_id += 1
                pbar.update()
                input = data.cuda()
                target = label.cuda()
                raw_input=torch.nn.functional.interpolate(input,(384,384))
                raw_score = model.classfier(raw_input)
                attr_branch, cmap, x_sampled, score = model(input)

                optimizer.zero_grad()
                cls = criterion(score, target)
                raw_cls = criterion(raw_score, target)
                loss = cls + raw_cls
                loss.backward()
                optimizer.step()
                cosine_lr.step()
                data, label = prefetcher.next()

        # valide
        model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        label_all = []
        pred_all = []

        with torch.no_grad():
            prefetcher = img_data_prefetcher(val_dataloader)
            data, label = prefetcher.next()
            iter_id = 0
            with tqdm(total=vbar) as pbar:
                while data is not None:
                    iter_id += 1
                    pbar.update()
                    input, target = data.cuda(), label.cuda()
                    attr_branch, cmap, x_sampled, score = model(input)
                    cls = criterion(score, target)
                    loss = cls
                    prec1, prec5 = accuracy(score, target, topk=(1, 5))
                    losses.update(loss.data, input.size(0))
                    top1.update(prec1, input.size(0))
                    pred = score.argmax(dim=1).detach().cpu().numpy()
                    label_all.extend(label.cpu().numpy())
                    pred_all.extend(pred)
                    data, label = prefetcher.next()
            qwk= quadratic_kappa(np.array(label_all), np.array(pred_all))
            # cm = confusion_matrix(np.array(label_all), np.array(pred_all), normalize='true')

        if qwk > best:
            best = qwk
            torch.save(model.state_dict(), save_mode_path)

def get_one_hot_np(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    train()