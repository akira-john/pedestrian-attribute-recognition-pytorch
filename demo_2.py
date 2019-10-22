import sys
import os
import numpy as np
import random
import math
from collections import OrderedDict
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
import cPickle as pickle
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
        parser.add_argument('--outpath', type=str, default='/content/pedestrian-attribute-recognition-pytorch/dataset/demo/output/')
        ## dataset parameter
        parser.add_argument('--dataset', type=str, default='peta',
                choices=['peta','rap', 'pa100k'])
        # utils
        parser.add_argument('--load_model_weight', type=str2bool, default=True)
        parser.add_argument('--model_weight_file', type=str, default='./exp/deepmar_resnet50/peta/partition0/run1/model/ckpt_epoch150.pth')

        parser.add_argument('--show', type=int, default=0)

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
                print 'Please input the model_weight_file if you want to load model weight'
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

        self.demo_folder = args.demo_folder

        self.outpath = args.outpath

        self.show = args.show

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
            print '%s: %.2f'%(cfg.att_list[idx], score[0, idx])

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
    label = 0
    label_count = [0]*16
    count = 0
    print(len(files))
    for file, file_name in zip(files, os.listdir(cfg.demo_folder)):
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
        age_box = ['05', '15', '25', '35', '45', '55', '65', '75']
        age = age_box[age_index]

        gender = 'Male' if score[0, 0]<=0 else 'Female'

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

        if not cfg.show:
            if not os.path.exists(cfg.outpath + 'Male'):
                for i in age_box:
                    os.makedirs(cfg.outpath+'Male/' + i)
                    os.makedirs(cfg.outpath+'Female/'+i)
            img.save(cfg.outpath + gender + '/' + age + '/' + file_name[:-4] + '_result.png')

        count += 1

    if cfg.show:

        with open('/content/pedestrian-attribute-recognition-pytorch/labels_street10.txt') as f:
            labels = f.readlines()
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


