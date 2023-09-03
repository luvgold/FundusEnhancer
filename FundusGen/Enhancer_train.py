import sys
from random import random

sys.path.append("/home/user6/Work/CANet")
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from utils import *
from Models.Fundus_Model import FundusGen_pre
from Utils.fast import DataLoaderX, img_data_prefetcher
import os
##### Param ###############
load_mode_path="..."
save_mode_path = "..."

if not os.path.exists(mode_path):
    os.makedirs(mode_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data_path = "..."

batch_size = 16
img_size = 512
alr=0.001
num_workers = 16
print("batch_size", batch_size)
print("num_workers", num_workers)
traindir = os.path.join(data_path, 'train')
valdir = os.path.join(data_path, 'test')
max_epoch=120
momentum=0.9
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

def label_smooth(y_true, y_pred):
    y_true=((1-0.1)*y_true+0.05)
    return nn.CrossEntropyLoss()(y_true, y_pred)

criterion = nn.CrossEntropyLoss()

def train():
    best = 0.0
    best_avg = 0.0
    model = FundusGen_pre(16)
    model.model.load_state_dict(torch.load(load_mode_path), strict=False)
    model.cuda()
    dataset = datasets.ImageFolder(
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

    print(len(val_data))

    train_data = dataset

    train_dataloader = DataLoaderX(train_data, batch_size,
                                   shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoaderX(val_data, batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)
    optimizer2 = torch.optim.SGD(model.parameters(), lr=alr,
                                 momentum=momentum,
                                 weight_decay=weight_decay)
    cudnn.benchmark = True

    tbar = int(len(train_data) / batch_size)
    vbar = int(len(val_data) / batch_size)
    cosine_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=120, eta_min=1e-8)
    for epoch in range(max_epoch):
        model.train()
        prefetcher = img_data_prefetcher(train_dataloader)
        data, label = prefetcher.next()
        iter_id = 0
        # adjust_learning_rate(optimizer2,epoch,alr)
        with tqdm(total=tbar) as pbar:
            while data is not None:
                iter_id += 1
                pbar.update()
                input = data.cuda()
                target = label.cuda()
                score, attr_branch, loss_2, simg = model(input)
                cls = criterion(score, target)
                loss = cls+loss_2
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
                cosine_lr.step()
                data, label = prefetcher.next()

        # valide

        losses = AverageMeter()
        top1 = AverageMeter()
        label_all = []
        pred_all = []
        model.eval()

        with torch.no_grad():
            prefetcher = img_data_prefetcher(val_dataloader)
            data, label = prefetcher.next()
            iter_id = 0
            with tqdm(total=vbar) as pbar:
                while data is not None:
                    iter_id += 1
                    pbar.update()
                    input, target = data.cuda(), label.cuda()
                    # score, cimg = t_model(input)
                    score, attr_branch, loss_2, simg = model(input)
                    loss = criterion(score, target)
                    prec1, prec5 = accuracy(score, target, topk=(1, 5))
                    losses.update(loss.data, input.size(0))
                    top1.update(prec1, input.size(0))
                    pred = score.argmax(dim=1).detach().cpu().numpy()
                    label_all.extend(label.cpu().numpy())
                    pred_all.extend(pred)
                    data, label = prefetcher.next()

            qwk = quadratic_kappa(np.array(label_all), np.array(pred_all))

            print("epoch:{} ---- kappa:{}".format(epoch,qwk))
            print("losses:{}   top_1_pre:{} ".format(losses.avg, top1.avg))
            if qwk > best:
                best = qwk
                torch.save(model.state_dict(), os.path.join(mode_path, 'kappa_resgen_nums16_woml' + '.pth'))
            elif top1.avg > best_avg:
                best_avg = top1.avg
                torch.save(model.state_dict(), os.path.join(mode_path, 'acc_resgen_nums32_woml' + '.pth'))

def accuracy(output, target, topk=(1,)):
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