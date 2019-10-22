import os
import numpy as np
import random
#import codecs
import cPickle as pickle
from scipy.io import loadmat
import torchvision

np.random.seed(0)
random.seed(0)

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def eval_weights(labels, num_att):
    counts = [0]*num_att
    for l in labels:
        l = int(l)
        if l <= 7:
            counts[0] += 1
        age = l % 8
        counts[age+1] += 1
    weights = [float(x)/len(labels) for x in counts]
    return weights


def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    dataset = dict()
    dataset['description'] = 'trans'
    dataset['root'] = '/content/drive/My Drive/PET/'
    dataset['image'] = []
    dataset['att'] = []
    dataset['att_name'] = []
    dataset['selected_attribute'] = [x for x in range(9)]
    # load PETA.MAT


    with open('/content/pedestrian-attribute-recognition-pytorch/labels_2700.txt') as f:
        labels = f.readlines()

    for i in ['gender', '05', '15', '25', '35', '45', '55', '65', '75']:
        dataset['att_name'].append(i)

    for i, l in enumerate(labels):
        label_lis = [0]*9
        label = int(l)
        if label > 7:
            label_lis[0] = 0
        else:
            label_lis[0] = 1
        age = label % 8
        # if not age in [0, 1, 7]:
        #     label_lis[age] = 1
        #     label_lis[age+1] = 1
        #     label_lis[age+2] = 1
        # elif age == 0:
        #     label_lis[age+1] = 1
        # elif age == 1:
        #     label_lis[age+1] = 1
        #     label_lis[age+2] = 1
        # elif age == 7:
        #     label_lis[age] = 1
        #     label_lis[age+1] = 1
        label_lis[age+1] += 1
        dataset['image'].append('%05d.png'%(i+1))
        dataset['att'].append(label_lis)
        print(label_lis)
    with open(os.path.join(save_dir, 'trans_dataset.pkl'), 'w+') as f:
        pickle.dump(dataset, f)

def create_trainvaltest_split(traintest_split_file):
    """
    create a dataset split file, which consists of index of the train/val/test splits
    """
    partition = dict()
    partition['trainval'] = []
    partition['train'] = []
    partition['val'] = []
    partition['test'] = []
    partition['weight_trainval'] = []
    partition['weight_train'] = []
    with open('/content/pedestrian-attribute-recognition-pytorch/labels_2700.txt') as f:
        labels = f.readlines()
    # load PETA.MAT
    # data = loadmat(open('./dataset/peta/PETA.mat', 'r'))
    items = [x for x in range(len(labels))]
    for idx in range(2):
        random.shuffle(items)
        s1 = len(items)*2//3
        s2 = len(items)*5//6
        train = items[:s1]
        val = items[s1:s2]
        test = items[s2:]
        print('---train---')
        print(len(train))
        print(len(test))
        trainval = train + val
        partition['train'].append(train)
        partition['val'].append(val)
        partition['trainval'].append(trainval)
        partition['test'].append(test)
        l_trainval = [labels[x] for x in trainval]
        l_train = [labels[x] for x in train]
        weight_trainval = eval_weights(l_trainval, 9)
        weight_train = eval_weights(l_train, 9)
        print(weight_trainval)
        partition['weight_trainval'].append(weight_trainval)
        partition['weight_train'].append(weight_train)
    with open(traintest_split_file, 'w+') as f:
        pickle.dump(partition, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="peta dataset")
    make_dir('./dataset/trans/')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./dataset/trans/')
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default="./dataset/trans/trans_partition.pkl")
    args = parser.parse_args()
    save_dir = args.save_dir
    traintest_split_file = args.traintest_split_file

    generate_data_description(save_dir)
    create_trainvaltest_split(traintest_split_file)
