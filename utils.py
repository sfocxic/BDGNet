import datetime
import os
import sys
import json
import pickle
import random
from collections import defaultdict, deque
import time
import torch.nn.functional as F

import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import cv2
import torch.distributed as dist
import matplotlib.pyplot as plt



def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)#12, 6, 256, 256
        target = self._one_hot_encoder(target)#[12, 6, 256, 256]
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

#Get edge
def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k) #-1,0,1
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D

def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D

def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3  # actual size of the directional kernel
    # increase for a while to avoid interpolation when rotating
    k_increased = k_thin + 2

    # get 0° angle directional kernel
    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

    # rotate the 0° angle directional kernel to get the other ones
    thin_kernels = []
    for angle in range(start, end, step):
        (h, w) = thin_kernel_0.shape
        # get the center to not rotate around the (0, 0) coord point
        center = (w // 2, h // 2)
        # apply rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

        # get the k=3 kerne
        kernel_angle = kernel_angle_increased[1:-1, 1:-1]
        is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
        kernel_angle = kernel_angle * is_diag  # because of the interpolation
        thin_kernels.append(kernel_angle)
    return thin_kernels

class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 device=None,
                 use_cuda=True):
        super(CannyFilter, self).__init__()
        # device
        self.device = device if use_cuda else 'cpu'

        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        with torch.no_grad():
            self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)


        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        with torch.no_grad():
            self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)

        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        with torch.no_grad():
            self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)

        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        with torch.no_grad():
            self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        with torch.no_grad():
            self.hysteresis.weight[:] = torch.from_numpy(hysteresis)

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):


        # set the setps tensors
        B, C, H, W = img.shape

        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            blurred[:, c:c + 1] = self.gaussian_filter(img[:, c:c + 1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c + 1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c + 1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180  # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges
        #print(type(grad_magnitude))
        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()

        #non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1

        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges

class MaskBoundary(nn.Module):
    def __init__(self):
        super(MaskBoundary, self).__init__()
    def forward(self,input):
        x=input
        t=x.clone()
        x1 = torch.roll(x, shifts=(-1), dims=(-1))  # 向左滚动
        num1 = 1-x.eq_(x1)
        num1[:, :, :, -1] = 0
        x = t.clone()
        x2 = torch.roll(x, shifts=(1), dims=(-1))  # 向右滚动
        num2 = 1-x.eq_(x2)
        num2[:, :, :, 0] = 0
        x = t.clone()
        x3 = torch.roll(x, shifts=(-1), dims=(-2))  # 向上滚动
        num3 = 1-x.eq_(x3)
        num3[:, :, -1, :] = 0
        x = t.clone()
        x4 = torch.roll(x, shifts=(1), dims=(-2))  # 向下滚动
        num4 = 1-x.eq_(x4)
        num4[:, :, 0, :] = 0
        x = t.clone()
        x5 = torch.roll(x, shifts=(-1, 1), dims=(-1, -2))  # 向左下滚动
        num5 = 1-x.eq_(x5)
        num5[:, :, 0, :] = 0
        num5[:, :, :, -1] = 0
        x = t.clone()
        x6 = torch.roll(x, shifts=(-1, -1), dims=(-1, -2))  # 向左上滚动
        num6 = 1-x.eq_(x6)
        num6[:, :, -1, :] = 0
        num6[:, :, :, -1] = 0
        x = t.clone()
        x7 = torch.roll(x, shifts=(1, 1), dims=(-1, -2))  # 向右下滚动
        num7 = 1-x.eq_(x7)
        num7[:, :, 0, :] = 0
        num7[:, :, :, 0] = 0
        x = t.clone()
        x8 = torch.roll(x, shifts=(1, -1), dims=(-1, -2))  # 向右上滚动
        num8 = 1-x.eq_(x8)
        num8[:, :, -1, :] = 0
        num8[:, :, :, 0] = 0

        num1 = torch.tensor(num1, dtype=torch.int32)
        num2 = torch.tensor(num2, dtype=torch.int32)
        num3 = torch.tensor(num3, dtype=torch.int32)
        num4 = torch.tensor(num4, dtype=torch.int32)
        num5 = torch.tensor(num5, dtype=torch.int32)
        num6 = torch.tensor(num6, dtype=torch.int32)
        num7 = torch.tensor(num7, dtype=torch.int32)
        num8 = torch.tensor(num8, dtype=torch.int32)

        ans = num1 | num2 | num3 | num4 | num5 | num6 | num7 | num8
        ans=torch.tensor(ans,dtype=torch.float32)

        return ans



class BoundaryLoss(nn.Module):
    def __init__(self,device, n_classes=6):
        super(BoundaryLoss, self).__init__()
        self.n_classes=n_classes
        self.conny = CannyFilter(device=device)
        self.maskboundary=MaskBoundary()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def get_loss(self, score, target,eps=1e-8):
        # target = target.float()
        intersect = 2*torch.sum(score * target)+eps
        union=torch.sum(target * target)+torch.sum(score * score)+eps
        loss = 1 - intersect/union
        return loss

    def forward(self, input, orig_x,target,eps=1e-8):
        one_hot_target = self._one_hot_encoder(target)

        target=target.unsqueeze(1)

        _,_,_,grad,_,conny=self.conny(orig_x,low_threshold=0.5,high_threshold=0.9, hysteresis=True)

        target = torch.tensor(target, dtype=torch.float)
        #conny_mask=self.maskboundary(conny)
        boundary_mask=conny
        target_mask = self.maskboundary(target)
        target_mask=self.maskboundary(target_mask)
        focus_point = boundary_mask * target_mask

        conny = target_mask + focus_point

        one_hot_target=(1-target_mask)*one_hot_target
        mid_input=(1-target_mask)*input
        #计算内部损失
        in_loss = 0.0
        for i in range(0, self.n_classes):
            mid_loss = self.get_loss(mid_input[:, i], one_hot_target[:, i])
            in_loss += mid_loss
        in_loss = in_loss / self.n_classes

        B,_,H,W=conny.shape
        input=input.argmax(dim=1).unsqueeze(1)
        input=torch.tensor(input,dtype=torch.float)

        input=self.maskboundary(input)
        input = self.maskboundary(input)

        loss_boundary=self.get_loss(input,conny)
        loss=loss_boundary+in_loss

        return loss

class Comp_Boundary_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_boundary=MaskBoundary()

    def forward(self, input, target):
        target = target.unsqueeze(1)
        target=self.mask_boundary(target)
        target=target.squeeze(1)
        target = torch.tensor(target, dtype=torch.long)

        return nn.functional.cross_entropy(input, target,ignore_index=255)




class Detail_local_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, input, target):
        input=self.up(input)

        return nn.functional.cross_entropy(input, target, ignore_index=255)

class Global_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, input, target):
        input=self.up(input)

        return nn.functional.cross_entropy(input, target, ignore_index=255)


def criterion(inputs, target, device, orig_x,epoch,n_classs=6):
    loss_weight = torch.as_tensor([1.0, 1., 1., 1., 1., 1.], device=device)
    losses = {}
    diceloss=DiceLoss(n_classes=n_classs)
    focalloss=FocalLoss(weight=loss_weight)
    edgeloss=BoundaryLoss(device=device,n_classes=n_classs).to(device)

    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        t=x
        eloss=edgeloss(t,orig_x,target)#gpu
        losses[name] = nn.functional.cross_entropy(x, target, loss_weight, ignore_index=255)+\
            diceloss(x,target,loss_weight)+(eloss if epoch>20 else 0)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def train_one_epoch_seg(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)

            loss = criterion(output, target,device,image,epoch)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别的召回率
        rec=torch.diag(h)/h.sum(0)
        #计算F1值
        f1= 2*acc*rec/(acc+rec)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu, rec, f1

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu, rec, f1 = self.compute()
        mean_5=0
        for i in range(len(iu)):
            if i==4:
                continue
            mean_5=mean_5+iu[i]*100
        mean_5=mean_5/5
        return (
            'global correct: {:.2f}\n'
            'precision: {}\n'
            'recall: {}\n'
            'f1_score: {}\n'
            'mean f1: {:.2f}\n'
            'IoU: {}\n'
            'mean IoU: {:.2f}\n'
            'mean_5 IoU: {:.2f}').format(
                acc_global.item() * 100,
                ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.2f}'.format(i) for i in (rec * 100).tolist()],
                ['{:.2f}'.format(i) for i in (f1 * 100).tolist()],
                f1.mean().item()*100,
                ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100,
                mean_5
            )

@torch.no_grad()
def evaluate_seg(model, data_loader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']
            output=output.to(device)

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)