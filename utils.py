import logging
import torch
import torch.nn as nn
import os
import numpy as np


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.cls = classes
        self.smoothing = smoothing
        self.dim = dim
        self.confidence = 1.0 - smoothing


    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        # print(pred)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        #     print(true_dist)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))



def create_logging(log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    handlers = []
    stream_handler = logging.StreamHandler()
    handlers.append(stream_handler)
    file_handler = logging.FileHandler(log_file, file_mode)
    handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    for handler in handlers:
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger



def save_checkpoint(args, epoch_idx, net, optimizer, save_name='latest'):
    state_dict = {
        'epoch': epoch_idx,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state_dict, os.path.join(args.path_log, '%s.pth' % save_name))



def adjust_learning_rate(optimizer, idx, epoch, minibatch_count, args):
    # epoch >= 1
    if epoch <= args.warmup:
        lr = args.start_lr * ((epoch - 1) / args.warmup + idx / (args.warmup * minibatch_count))
    else:
        decay_rate = 0.5 * (1 + np.cos((epoch - 1) * np.pi / args.end_epoch))
        lr = args.start_lr * decay_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr




def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct_temp = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct_temp.reshape(-1).float().sum(0, keepdim=True)
        res = float(correct.mul_(100.0/batch_size))
        return res




def accuracy_single(output, target, tp_single, pre_single, sum_single):

    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct_temp = pred.eq(target.view(1, -1).expand_as(pred))
        for i in range(correct_temp.size(1)):
            correct_leibie = target.cpu().numpy()
            correct_leibie = correct_leibie[i]
            sum_single[correct_leibie] += 1
            predict_leibie = pred.cpu().numpy()[0][i]
            pre_single[predict_leibie] += 1
            if correct_temp[0][i]:
                tp_single[correct_leibie] += 1
        return tp_single, sum_single, pre_single


