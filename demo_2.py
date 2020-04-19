import sys
import os
import cv2
import numpy as np
import random
import math
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
import pickle
import time
import argparse
from PIL import Image, ImageFont, ImageDraw

from baseline.model.DeepMAR import DeepMAR_ResNet50
from baseline.utils.utils import str2bool
from baseline.utils.utils import save_ckpt, load_ckpt
from baseline.utils.utils import load_state_dict 
from baseline.utils.utils import set_devices
from baseline.utils.utils import set_seed


class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
        parser.add_argument('--set_seed', type=str2bool, default=False)
        # model
        parser.add_argument('--resize', type=eval, default=(224, 224))
        parser.add_argument('--last_conv_stride', type=int, default=2, choices=[1,2])
        # demo image
        parser.add_argument('--demo_image', type=str, default=None)
        parser.add_argument('--demo_folder', type=str, default=None)
        parser.add_argument('--npy', type=str, default=None)
        parser.add_argument('--read_csv', type=str, default=None)
        parser.add_argument('--outpath', type=str, default='./')
        ## dataset parameter
        parser.add_argument('--dataset', type=str, default='peta',
                choices=['peta','rap', 'pa100k'])
        # utils
        parser.add_argument('--load_model_weight', type=str2bool, default=True)
        parser.add_argument('--model_weight_file', type=str, default='./exp/deepmar_resnet50/peta/partition0/run1/model/ckpt_epoch150.pth')

        parser.add_argument('--show', type=int, default=0)
        parser.add_argument('--write', type=int, default=0)

        args = parser.parse_args()
        
        # gpu ids
        self.sys_device_ids = args.sys_device_ids

        # random
        self.set_seed = args.set_seed
        if self.set_seed:
            self.rand_seed = 0
        else: 
            self.rand_seed = None
        self.resize = args.resize
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # utils
        self.load_model_weight = args.load_model_weight
        self.model_weight_file = args.model_weight_file
        if self.load_model_weight:
            if self.model_weight_file == '':
                print('Please input the model_weight_file if you want to load model weight')
                raise ValueError
        # dataset 
        datasets = dict()
        datasets['peta'] = './dataset/peta/peta_dataset.pkl'
        datasets['rap'] = './dataset/rap/rap_dataset.pkl'
        datasets['pa100k'] = './dataset/pa100k/pa100k_dataset.pkl'

        # if args.dataset in datasets:
        #     dataset = pickle.load(open(datasets[args.dataset]))
        # else:
        #     print '%s does not exist.'%(args.dataset)
        #     raise ValueError
        self.att_list = ['05', '15', '25', '35', '45', '55', '65', '75']
        
        # demo image
        if args.demo_image:
            self.demo_image = '/content/pedestrian-attribute-recognition-pytorch/dataset/demo/' + args.demo_image
        else:
            self.demo_image = None
        self.image_name = args.demo_image
        self.npy = args.npy

        self.demo_folder = args.demo_folder
        self.read_csv = args.read_csv

        self.outpath = args.outpath

        self.show = args.show
        self.write = args.write

        # model
        model_kwargs = dict()
        model_kwargs['num_att'] = len(self.att_list) + 1
        model_kwargs['last_conv_stride'] = args.last_conv_stride
        self.model_kwargs = model_kwargs

### main function ###
cfg = Config()

# dump the configuration to log.
print('-' * 60)
# print('cfg.__dict__')
# pprint.pprint(cfg.__dict__)
print('-' * 60)


# set the random seed
if cfg.set_seed:
    set_seed( cfg.rand_seed )
# init the gpu ids
set_devices(cfg.sys_device_ids)

# dataset 
normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
test_transform = transforms.Compose([
        transforms.Resize(cfg.resize),
        transforms.ToTensor(),
        normalize,])

### Att model ###
model = DeepMAR_ResNet50(**cfg.model_kwargs)
model.classifier.out_features=9


# load model weight if necessary
if cfg.load_model_weight:
    map_location = (lambda storage, loc:storage)
    ckpt = torch.load(cfg.model_weight_file, map_location=map_location)
    new_state_dict = OrderedDict()
    # for k, v in ckpt['state_dicts'][0].items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    model.load_state_dict(ckpt['state_dicts'][0])

model.cuda()
model.eval()
if cfg.demo_image:
    # load one image 
    img = Image.open(cfg.demo_image)
    img_trans = test_transform( img ) 
    img_trans = torch.unsqueeze(img_trans, dim=0)
    img_var = Variable(img_trans).cuda()
    score = model(img_var).data.cpu().numpy()

    print(score)

    # show the score in command line
    for idx in range(len(cfg.att_list)):
        if score[0, idx] >= 0:
            print('%s: %.2f'%(cfg.att_list[idx], score[0, idx]))

    # show the score in the image
    img = img.resize(size=(256, 512), resample=Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    positive_cnt = 0
    for idx in range(len(cfg.att_list)):
        if score[0, idx] >= 0:
            txt = '%s: %.2f'%(cfg.att_list[idx], score[0, idx])
            draw.text((10, 10 + 10*positive_cnt), txt, (255, 0, 0))
            positive_cnt += 1
    img.save('./dataset/demo/output/' + cfg.image_name + '_result.png')

if cfg.demo_folder:
    files = [os.path.join(cfg.demo_folder, x) for x in os.listdir(cfg.demo_folder)]
    outpath = cfg.outpath
    age_count = [0]*8
    gender_count = [0]*2
    label_count = [0]*16
    age_box = ['05', '15', '25', '35', '45', '55', '65', '75']
    count = 0
    if cfg.show:
        with open('/content/pedestrian-attribute-recognition-pytorch/out_16.txt') as f:
            labels = f.readlines()
        gender_acc = 0
        age_acc = {}
        age_avg = {}
        age_len = {}
        for i in age_box:
            age_len[i] = 0.0
            age_acc[i] = 0.0
            age_avg[i] = 0.0
    else:
        labels = range(len(files))
    print(len(files))
    for file, file_name, label in zip(files, os.listdir(cfg.demo_folder), labels):
        if count % 100 == 0:
            print(count)
        if not os.path.isfile(file):
            continue
        img = Image.open(file)
        img_trans = test_transform( img ) 
        img_trans = torch.unsqueeze(img_trans, dim=0)
        img_var = Variable(img_trans).cuda()
        score = model(img_var).data.cpu().numpy()

        #print(score)

        # show the score in command line
        # for i, idx in range(len(cfg.att_list)):
        #     if score[0, idx] >= 0:
        #         print '%s: %.2f'%(cfg.att_list[idx], score[0, idx])

        # show the score in the image
        img = img.resize(size=(256, 512), resample=Image.BILINEAR)
        draw = ImageDraw.Draw(img)
        positive_cnt = 0
        ages = np.asarray([score[0, x] for x in range(1, 9)])
        age_index = ages.argmax()
        age = age_box[age_index]

        gender = 'Male' if score[0, 0]<=0 else 'Female'
        if cfg.show:
            true_age = age_box[int(label)%8]
            age_len[true_age] += 1.0
            age_avg[true_age] += float(age)
            if true_age == age:
                age_acc[true_age] += 1
            if (gender == 'Male' and int(label) > 7) or (gender=='Female' and int(label) <= 7):
                gender_acc += 1



        if gender == 'Male':
            gender_count[0] += 1
        else:
            gender_count[1] += 1

        age_count[age_index] += 1

        if gender == 'Male':
            label = age_index + 8
        else:
            label = age_index  

        label_count[label] += 1      

        # draw.text((10, 10), 'age : {}, gender : {}'.format(age, gender), (100, 100, 0))
        # for idx in range(9):
        #     if idx != 0:
        #         draw.text((10, 30 + 10*positive_cnt), str(age_box[idx-1])+' : '+str(round(score[0, idx], 2)), (150, 0, 150))
        #     positive_cnt += 1
        # draw.text((10, 20), 'FeMale : '+str(round(score[0, 0], 2)), (100, 100, 0))

        if cfg.write:
            if not os.path.exists(cfg.outpath + 'Male'):
                for i in age_box:
                    os.makedirs(cfg.outpath+'Male/' + i)
                    os.makedirs(cfg.outpath+'Female/'+i)
            img.save('{}{}/{}/{}_result.png'.format(cfg.outpath, gender, age, file_name[:-4]))

        count += 1
    if cfg.show:
        accuracy = float(gender_acc) / float(len(labels))
        print('gender_acc :', accuracy)
        for i in age_box:
            if age_len[i] == 0:
                continue
            print(i, 'pred_avg {} :'.format(age_avg[i]/age_len[i]), 'age_acc  {} :'.format(age_acc[i]/age_len[i]))

    if cfg.show:
        print(len(labels))
        t_age_count = [0]*8
        t_gender_count = [0]*2
        t_label_count = [0]*16
        for i, l in enumerate(labels):
            label = int(l)
            t_label_count[label] += 1
            if label > 7:
                t_gender_count[0] += 1
            else:
                t_gender_count[1] += 1
            age = label % 8
            t_age_count[age] += 1

        plt.subplot(2, 2, 1)
        plt.bar([x for x in range(2)], gender_count, tick_label=['Male', 'FeMale'])
        plt.bar([x+0.2 for x in range(2)], t_gender_count, width=0.4)
        plt.suptitle(str(cfg.model_weight_file), fontsize=10)
        plt.subplot(2, 2, 2)
        plt.bar([x for x in range(8)], age_count, tick_label=age_box)
        plt.bar([x+0.2 for x in range(8)], t_age_count, width=0.4)
        plt.subplot(2, 1, 2)
        plt.bar([x for x in range(16)], label_count, tick_label=[x for x in range(16)])
        plt.bar([x+0.2 for x in range(16)], t_label_count, width=0.4)
        plt.legend()
        plt.savefig('pred_' + str(datetime.now())[-6:])


if cfg.read_csv:
    num_files = np.load(cfg.read_csv[:-4]+'.npy', allow_pickle=True)
    atts = []
    # if not os.path.exists(cfg.outpath):
    #     os.mkdir(cfg.outpath)
    for i, file in enumerate(num_files):
        if file is None:
            atts.append([None, None])
            continue
        file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(np.uint8(file))
        img_trans = test_transform( img ) 
        img_trans = torch.unsqueeze(img_trans, dim=0)
        img_var = Variable(img_trans).cuda()
        score = model(img_var).data.cpu().numpy()
        img = img.resize(size=(256, 512), resample=Image.BILINEAR)
        draw = ImageDraw.Draw(img)
        ages = np.asarray([score[0, x] for x in range(1, 9)])
        age_index = ages.argmax()
        age_box = ['05', '15', '25', '35', '45', '55', '65', '75']
        age = age_box[age_index]

        gender = 'Male' if score[0, 0]<=0 else 'Female'
        if cfg.write:
            img_path = os.path.join(cfg.outpath, gender+'_'+age+'_'+str(i)+'_result.png')
            img.save(img_path)
        atts.append([gender, age])


    df_att = pd.DataFrame(atts)
    columns = ['gender', 'age']
    df_att.columns = columns
    df = pd.read_pickle(cfg.read_csv)
    df_attributes = pd.concat([df, df_att], axis=1)
    file_name = cfg.read_csv[-32:-10]
    df_attributes.to_csv('../csv_files/{}df_attributes.csv'.format(file_name), index=False, encoding='utf-8')






