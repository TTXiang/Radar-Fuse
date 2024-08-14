import argparse
import random
import time

import utils

import dataset
from models.sequence_embed import Track_predict
import torch.optim as optim
from utils import *
import datetime
from torch.utils.tensorboard import SummaryWriter
# tensorboard的保存路径
tb = SummaryWriter('./logs/pic/')


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-random_seed', default=1, type=int)

    # data
    parser.add_argument('-data_root', default='./data', type=str)  # data路径,需要修改
    parser.add_argument('-save_root', default='logs', type=str)  # 日志路径,默认为logs
    parser.add_argument('-split_k', default=0.7, type=float)  # 划分数据集时候用, 此处与网络无关
    parser.add_argument('-batch_size', default=16, type=int)   # 可修改
    parser.add_argument('-num_workers', default=16, type=int)  # 可修改
    parser.add_argument('-max_length', default=128, type=int)  # 这里代表输入的航迹点数量上限，不足的数据处理时候会补零，可修改
    parser.add_argument('-min_length', default=5, type=int)  # 最少是6个点，可修改

    #  model
    parser.add_argument('-attn_heads', default=2, type=int)  # 可修改multi-attens参数
    parser.add_argument('-trans_layers', default=2, type=int)  # 可修改multi-attens参数
    parser.add_argument('-embed_size', default=256, type=int)  # 无需修改
    parser.add_argument('-trans_hidden', default=256*2, type=int)  # 无需修改
    parser.add_argument('-dropout', default=0.1, type=float)  # 无需修改
    parser.add_argument('-resume_best', default='', type=str, help="Path of the best weight")
    parser.add_argument('-resume_latest', default='', type=str, help="Path of the latest weight")
    parser.add_argument('-num_classes', default=4, type=int, help="num_classes")  # 可修改类别
    parser.add_argument('-model_name', default='Modelxyz', type=str, help="model name")  # 模型的名称可以修改
    parser.add_argument('-use_trans', default=True, type=bool, help="use_trans")  # 默认使用Transformer结构


    #  train
    parser.add_argument('-use_cuda', default=True, type=bool)
    parser.add_argument('-gpu_ids', nargs='+', type=str, default=['0', '1'])
    parser.add_argument('-optimizer_type', default='SGD', type=str, help="SGD or Adam")  # 提供了两种优化器选择，默认为SGD
    parser.add_argument('-start_lr', default=0.0001, type=float, help="learning rate of optimizer")  # 初始学习率
    parser.add_argument('-adam_weight_decay', default=0.01, type=float, help="weight_decay of Adam")
    parser.add_argument('-adam_beta1', default=0.9, type=float, help="adam first beta value")
    parser.add_argument('-adam_beta2', default=0.999, type=float, help="adam first beta value")
    parser.add_argument('-warmup', default=20, type=int, help="warmup of lr")
    parser.add_argument('-end_epoch', default=100, type=int, help="end_epoch")  # 训练的epoch


    args = parser.parse_args()
    return args


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True





def train(train_loader, device, net, criterion, optimizer, epoch, logger, args):
    net.train()
    minibatch_count = len(train_loader)
    t_start = time.time()
    loss_sum = 0.0
    acc_sum = 0.0
    for i, datas in enumerate(train_loader):
        leibie, img_list, track_content, feats = datas
        optimizer.zero_grad()
        learning_rate = utils.adjust_learning_rate(optimizer, i, epoch, minibatch_count, args)
        t_data = time.time() - t_start
        # optimizer.zero_grad()

        leibie = leibie.cuda(non_blocking=True, device=device)

        img_list = img_list.cuda(non_blocking=True, device=device)
        track_content = track_content.cuda(non_blocking=True, device=device)
        feats = feats.cuda(non_blocking=True, device=device)

        # 计算输出更新参数
        output = net(img_list, track_content, feats)
        loss = criterion(output, leibie)

        acc = utils.accuracy(output, leibie)
        loss.backward()
        optimizer.step()

        tend = time.time()
        ttrain = tend - t_start
        t_start = tend
        args.time_sec_tot += ttrain
        time_sec_avg = args.time_sec_tot / ((epoch - args.start_epoch) * minibatch_count + i + 1)
        eta_sec = time_sec_avg * ((args.end_epoch + 1 - epoch) * minibatch_count - i - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

        outputs = [
            "e: {}/{},{}/{}".format(epoch, args.end_epoch, i, minibatch_count),
            "{:.2f} mb/s".format(1. / ttrain),
            'eta: {}'.format(eta_str),
            'time: {:.3f}'.format(ttrain),
            'data_time: {:.3f}'.format(t_data),
            'lr: {:.4f}'.format(learning_rate),
            'acc: {:.4f}'.format(acc),
            'loss: {:.4f}'.format(loss.item()),
        ]

        if t_data / ttrain > .05:
            outputs += [
                "dp/tot: {:.4f}".format(t_data / ttrain),
            ]

        if i % 10 == 0:
            logger.info('\t'.join(outputs))
        tags = ["train_loss", "accuracy", "learning_rate"]
        tb.add_scalar(tags[0], loss, i)
        tb.add_scalar(tags[1], acc, i)
        tb.add_scalar(tags[2], learning_rate, i)
        loss_sum += loss
        acc_sum += acc



        # tb.add_histogram('conv31.bias', net.conv31.bias, i)
        # tb.add_histogram('conv31.weight', net.conv31.weight, i)
        # tb.add_histogram('conv31.weight.grad', net.conv31.weight.grad, i)
        # tb.add_histogram('conv32.bias', net.conv32.bias, i)
        # tb.add_histogram('conv32.weight', net.conv32.weight, i)
        # tb.add_histogram('conv32.weight.grad', net.conv32.weight.grad, i)
        # for name, param in net.named_parameters():
        #    tb.add_histogram(name + '_grad', param.grad, i)
        #    tb.add_histogram(name + '_data', param, i)
    loss_sum = loss_sum / len(train_loader)
    acc_sum = acc_sum / len(train_loader)
    tb.add_scalar('acc_train', acc_sum, epoch)
    tb.add_scalar('loss_train', loss_sum, epoch)

    tb.close()


def validate(val_loader, device, net, criterion, epoch, logger):
    logger.info('eval epoch {}'.format(epoch))
    net.eval()
    acc_sum = 0.0
    loss = 0.0
    valdation_num = 0

    for i, datas in enumerate(val_loader):
        leibie, img_list, track_content, feats = datas
        leibie = leibie.cuda(non_blocking=True, device=device)

        img_list = img_list.cuda(non_blocking=True, device=device)
        track_content = track_content.cuda(non_blocking=True, device=device)
        feats = feats.cuda(non_blocking=True, device=device)

        # 计算输出更新参数
        with torch.no_grad():
            output = net(img_list, track_content, feats)
            loss_temp = criterion(output, leibie)
        loss = loss + loss_temp
        valdation_num += leibie.size(0)
        acc_temp = utils.accuracy(output, leibie)
        acc_sum += acc_temp * leibie.size(0)


        output = [
            'acc: {:.4f}'.format(acc_temp),
            'loss: {:.4f}'.format(loss_temp),
            'iter {}/{}'.format(i, len(val_loader)),
        ]
        if i % 10 == 0:
            logger.info('\t'.join(output))


    loss = loss / len(val_loader)
    acc = acc_sum / valdation_num

    outputs = [
        "val e: {}".format(epoch),
        'acc: {:.4f}'.format(acc),
        'loss: {:.4f}'.format(loss),
    ]
    tb.add_scalar('acc_val', acc, epoch)
    tb.add_scalar('loss_val', loss, epoch)

    return acc, outputs




if __name__ == '__main__':
    #  parameters_read
    args = get_argparse()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #  random_seed set
    random_seed(args.random_seed)

    #  train with gpus or cpu
    ids = ','.join(args.gpu_ids)
    print(torch.cuda.is_available())
    if args.use_cuda:
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print("Using %d GPUS for Model" % torch.cuda.device_count())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        print("Using CPU for Model")
        device = torch.device("cpu")
    print('device is', device)


    #  creat logging
    os.makedirs(args.save_root, exist_ok=True)
    log_file_path = os.path.join(args.save_root, '%s_train.log' % args.model_name)
    logger = utils.create_logging(log_file_path)

    #  dataloader, train_data and val_data
    print("Creating Train and Val Dataloader")
    train_loader, val_loader = dataset.run(args)


    #  build model and net
    print("Building Model")
    model = Track_predict(args, device=device)
    net = model.to(device)
    net.cuda()


    #  set optimizer
    optim_SGD = optim.SGD(net.parameters(), lr=args.start_lr, momentum=0.9,
                          weight_decay=args.adam_weight_decay)
    optim_Adam = optim.Adam(net.parameters(), lr=args.start_lr,
                            betas=(args.adam_beta1, args.adam_beta2),
                            weight_decay=args.adam_weight_decay)
    optimizer = optim_SGD if args.optimizer_type == 'SGD' else optim_Adam

    #  set loss
    criterion = utils.LabelSmoothingLoss(classes=args.num_classes, smoothing=0.1).to(device)


    args.path_log = os.path.join('./logs', args.model_name)
    os.makedirs(args.path_log, exist_ok=True)
    args.time_sec_tot = 0.0

    #  print args
    for param in sorted(vars(args).keys()):
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    start_epoch = 1
    args.resume_best = os.path.join(args.path_log, 'best.pth')
    args.resume_latest = os.path.join(args.path_log, 'latest.pth')
    if args.resume_best:
        print(args.resume_best)
        if os.path.isfile(args.resume_best):
            logger.info("=> loading checkpoint '{}'".format(args.resume_best))
            state_dict = torch.load(args.resume_best)
            if 'model' in state_dict:
                start_epoch = state_dict['epoch'] + 1
                net.load_state_dict(state_dict['model'])
                optimizer.load_state_dict(state_dict['optimizer'])
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume_best, state_dict['epoch']))
            else:
                net.load_state_dict(state_dict)
                logger.info("=> loaded checkpoint '{}'".format(args.resume_best))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume_best))



    args.start_epoch = start_epoch
    best_acc = 0.0
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        train(train_loader, device, net, criterion, optimizer, epoch, logger, args)
        utils.save_checkpoint(args, epoch, net, criterion, save_name='latest')
        acc, outputs = validate(val_loader, device, net, criterion, epoch, logger)
        if acc > best_acc:
            print('weight has updated')
            best_acc = acc
            utils.save_checkpoint(args, epoch, net, optimizer, save_name='best')
        outputs += ['best_acc: {:.4f}'.format(best_acc)]
        logger.info('\t'.join(outputs))
        logger.info('Exp path: %s' % args.path_log)


