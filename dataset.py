# import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset
import random
import pylab as plt
import os
from datetime import datetime
import time
from shutil import copy
import datetime
from PIL import Image
import shutil
import cv2
import math

class sort_multi:
    def __init__(self, data, group, pic, cropped):
        self.group = group
        self.pic = pic
        self.cropped = cropped
        self.data = data
    def __repr__(self):
        return repr((self.data, self.group, self.pic, self.cropped))




#  划分数据集三个文件夹
def remove_file(old_path, new_path, reserve=True):
    filelist = os.listdir(old_path)  # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    os.makedirs(new_path)
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        print('src:', src)
        print('dst:', dst)
        copy(src, dst)
    if not reserve:
        shutil.rmtree(old_path)


def train_div_single(data_path, k, train_path, val_path):
    # 按照k的比例划分训练集和验证集
    # 按照文件夹放置

    # 统计样本总数
    files = os.listdir(data_path)
    file_counts = len(files)
    train_num = int(k * file_counts)
    # val_num = file_counts - train_num
    train_index = random.sample(files, k=train_num)

    for index, file in enumerate(files):
        if file in train_index:
            old_path = os.path.join(data_path, file)
            new_path = os.path.join(train_path, file)
            remove_file(old_path, new_path)
        else:
            old_path = os.path.join(data_path, file)
            new_path = os.path.join(val_path, file)
            remove_file(old_path, new_path)


def train_div(data_path, k, train_path, val_path):
    for leibie in os.listdir(data_path):
        data_path_single = os.path.join(data_path, leibie)
        TRAIN_PATH_single = os.path.join(train_path, leibie)
        VAL_PATH_single = os.path.join(val_path, leibie)

        if not os.path.exists(TRAIN_PATH_single):
            os.makedirs(TRAIN_PATH_single)
        if not os.path.exists(VAL_PATH_single):
            os.makedirs(VAL_PATH_single)

        train_div_single(data_path_single, k, TRAIN_PATH_single, VAL_PATH_single)
        print(leibie, '_data_divide has done!')



#  速度计算有关的两个文件
def speed_cal(frame1, frame2, box1, box2):
    if frame1 == 0:
        return(str(frame2)+','+str(box2[0])+','+str(box2[1])
            +','+str(box2[2])+','+str(box2[3])+','+str(0))
    else:
        center1 = np.array([box1[0]+box1[2]/2, box1[1]+box1[3]/2])
        center2 = np.array([box2[0]+box2[2]/2, box2[1]+box2[3]/2])
        speed = round(np.linalg.norm(center1-center2)/(frame2-frame1), 4)
        return (str(frame2)+','+str(box2[0])+','+str(box2[1])
                +','+str(box2[2])+','+str(box2[3])+','+str(speed))


def speed_trans(data_path='./data'):
    PIC_PATH = os.path.join(data_path, 'pic')
    for leibie in os.listdir(PIC_PATH):
        leibie_path = os.path.join(PIC_PATH, leibie)
        for track in os.listdir(leibie_path):
            track_path = os.path.join(leibie_path, track)
            query_txt(track_path, track, data_path)




def query_txt(track_path, track, data_path='./data'):
    track_index = track.split('_')[-1]


    sum_name = track.split('_')[0] + '_sum.txt'
    sum_path = os.path.join(data_path, sum_name)
    sum_txt = np.loadtxt(sum_path, delimiter=',', dtype=int)

    query = sum_txt[sum_txt[:, 0] == int(track_index), :][:, -4:]
    file_track = os.path.join(track_path, track + '.txt')
    with open(file_track, 'w') as f:
        box1 = []
        frame_temp = 0
        for index, crop in enumerate(os.listdir(track_path)):
            # print(crop)
            if crop[-4:] == '.jpg':
                frame = 6 * int(crop.split('_')[1]) + int(crop.split('_')[2]) - 6
                box2 = query[index-1, :]
                f.write(speed_cal(frame_temp, frame, box1, box2)+'\n')
                # print(speed_cal(frame_temp, frame, box1, box2))
                box1 = box2
                frame_temp = frame
        f.close()





def date_trans(date):
    #  date日期变成-1, 1的数

    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    total_days = 365
    date_time = datetime.datetime.strptime(date[1:11], '%Y-%m-%d')
    year = date_time.year
    month = date_time.month
    day = date_time.day

    if year % 4 != 0 or (year % 100 == 0 and year % 400 != 0):
        days[1] += 1
        total_days += 1

    assert day <= days[month - 1]
    sum_days = sum(days[:month - 1]) + day

    assert sum_days > 0 and sum_days <= total_days

    hour_time = datetime.datetime.strptime(date[12:], '%H:%M:%S')
    hour = hour_time.hour
    minute = hour_time.minute

    return round((sum_days / total_days) * 2 - 1, 4), round((hour * 30 + minute) / (24 * 30) - 1, 4)


def encode_loc_time(loc_time):
    feats = np.concatenate((np.sin(math.pi * loc_time), np.cos(math.pi * loc_time)))
    return feats




class Data_divi(Dataset):
    def __init__(self, data_path, task='train', max_len=128, min_len=5, device=None):
        self.data_path = data_path
        self.samples = []
        self.max_len = max_len
        self.min_len = min_len

        # 给出读取后的txt格式，写入矩阵！！！！
        # 形成样本

        pic_root = os.path.join(data_path, task)


        for leibie in os.listdir(pic_root):
            leibie_path = os.path.join(pic_root, leibie)
            for track in os.listdir(leibie_path):
                track_path = os.path.join(leibie_path, track)
                track_content_path = os.path.join(track_path, track + '.txt')
                track_content = np.loadtxt(track_content_path, delimiter=',')


                radar_name = track.split('_')[0] + '_radar_sum.txt'
                radar_path = os.path.join(data_path, radar_name)
                f = open(radar_path, encoding='utf-8')
                line = f.readlines()[0].split('\n')[0].split(',')
                time_encode = date_trans(line[1])
                lon = float(line[2]) / 180
                lat = float(line[3]) / 90
                extra2 = {'day': time_encode[0], 'minute': time_encode[1],
                          'lon': lon, 'lat': lat}

                crop_path = []
                crop_path_list = []
                group_num = []
                pic_num = []
                cropped_num = []
                for crop in os.listdir(track_path):
                    if crop[-4:] == '.jpg':
                        crop = crop.split('.')[0]
                        group_num.append(int(crop.split('_')[1]))
                        pic_num.append(int(crop.split('_')[2]))
                        cropped_num.append(int(crop.split('_')[-1]))

                        crop_path_list.append(sort_multi(crop, int(crop.split('_')[1]), int(crop.split('_')[2]), int(crop.split('_')[-1])))


                crop_path_list.sort(key=lambda x: (x.group, x.pic, x.cropped))
                for i in range(len(crop_path_list)):
                    crop_path.append(os.path.join(track_path, crop_path_list[i].data+'.jpg'))


                len_crop = len(crop_path)
                
                if len_crop > self.min_len:
                    self.samples.append((leibie, crop_path, track_content, extra2))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, item):
        #  返回单条track的信息
        leibie, crop_path, track_content, extra2 = self.samples[item]
        leibie = int(leibie)

        day = extra2['day']  # 拍摄时间
        minute = extra2['minute']  # 拍摄时间
        lat = extra2['lat']
        lon = extra2['lon']
        extra2 = np.array([day, minute, lat, lon])
        feats = encode_loc_time(extra2)


        img_list = []
        len_crop = len(crop_path)
        for crop in crop_path:
            # img = (cv2.imread(crop, cv2.IMREAD_GRAYSCALE)).reshape(-1)
            img = cv2.imread(crop, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (45, 120)).reshape(-1)
            #   delete !!!!!
            # img = np.ones((45*120), dtype='uint8')
            img_list.append(img)
        if len_crop >= self.max_len:
            img_list = img_list[:self.max_len]
        if len_crop < self.max_len:
            for i in range(self.max_len-len_crop):
                zeros_temp = np.zeros((45*120, ), dtype='uint8')
                img_list.append(zeros_temp)


        track_content = track_content
        temp = track_content[0, :3]
        track_content[:, :3] = track_content[:, :3] - temp
        if track_content.shape[0] >= self.max_len:
            track_content = track_content[:self.max_len, :]
        if track_content.shape[0] < self.max_len:
            pad = np.zeros((self.max_len - track_content.shape[0], 6)).astype(np.float64)
            track_content = np.append(track_content, pad, axis=0)


        img_list = np.array(img_list).astype('float32')
        img_list = torch.from_numpy(img_list)
        track_content = np.array(track_content).astype('float32')
        track_content = torch.from_numpy(track_content)

        return leibie, img_list, track_content, feats






def load_train_dataset(args, device=None):
    dataset = Data_divi(data_path=args.data_root, max_len=args.max_length, min_len=args.min_length, task='train', device=device)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


def load_val_dataset(args, device=None):
    dataset = Data_divi(data_path=args.data_root, max_len=args.max_length, min_len=args.min_length, task='val', device=device)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return val_loader


def run(args, device=None):
    # speed_trans(args.data_root)
    PIC_PATH = os.path.join(args.data_root, 'pic')
    TRAIN_PATH = os.path.join(args.data_root, 'train')
    VAL_PATH = os.path.join(args.data_root, 'val')

    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    if not os.path.exists(VAL_PATH):
        os.makedirs(VAL_PATH)

    # train_div(PIC_PATH, args.split_k, TRAIN_PATH, VAL_PATH)
    train_loader = load_train_dataset(args, device=None)
    print("train_dataset has done!")
    val_loader = load_val_dataset(args, device=None)
    return train_loader, val_loader


