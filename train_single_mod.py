## To train single modality model
## Setting the different paths to distinguish the lesion MRI and the lesion MRI contained by larger cube. 
import torch
import time
import os
import sys

import torch
import argparse
from utils.dataset_1m import dataset
#from utils.dataset_full import dataset_no_ml
from utils.util import write_num_ls, calc_metrics
from net.clf_1m import be_ma_net

from sklearn.metrics import roc_curve
from sklearn.metrics import auc


import random
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import ml_collections
from termcolor import colored 

parser = argparse.ArgumentParser()
parser.add_argument('-root', type=str, default='')
parser.add_argument('-train_path', type=str, default='')
parser.add_argument('-vali_path', type=str, default='')
parser.add_argument('-mt', type=str, default='')
parser.add_argument('-model_save', type=str, )
parser.add_argument('-seq', type=int, default=1)
parser.add_argument('-cube', type=str, default=False)

args = parser.parse_args()


def train_net_config():

    config = ml_collections.ConfigDict()
    
    # config.mri_type=args.mt
    # config.seq=args.seq
    
    config.mri_type=args.mt
    config.seq=args.seq
    config.root=args.root
    
    if args.cube=='yes':
        config.cube=True
    else:
        config.cube=False

    config.train_path=args.train_path
    config.vali_path=args.vali_path

    config_model_save=args.model_save


    if args.cube:
        config.log_prefix=config.mri_type+'_cube_exp'+str(config.seq)
    else:
        config.log_prefix=config.mri_type+'_no_cube_exp'+str(config.seq)

    config.model_save=config_model_save+config.log_prefix+'/'
    
    config.log_path=config.model_save
    config.bs=16
    config.lr=0.01
    config.gpu='0'
    config.seed=1234
    config.min_epoch=1
    config.max_epoch=200
    config.log_name='log_from_epoch_{}_to_{}.txt'.format(config.min_epoch,config.max_epoch)
    config.deterministic=True

    return config

config=train_net_config()
#config=train_no_cube_config()
os.makedirs(config.model_save,exist_ok=True)

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

res_log = config.log_prefix+'_res_log.txt'  ## loss logs
loss_log = config.log_prefix+'_loss_ilog.txt'
loss_log_e =config.log_prefix+'_loss_elog.txt'


res_fd= open(config.log_path+res_log, 'w')
loss_fd_i = open(config.log_path+loss_log, 'w')
loss_fd_e = open(config.log_path+loss_log_e, 'w')



use_model = False
use_ml = False


if config.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


def accuary(outputs, label):
    _, predicted = torch.max(outputs.data, 1)
 
    sum = 0
    for i in range(predicted.shape[0]):
        if predicted[i] == label[i]:
            sum += 1
    return float(sum / len(predicted))


def acc_count(outputs, label):
    _, predicted = torch.max(outputs.data, 1)
    sum = 0
    for i in range(predicted.shape[0]):
        if predicted[i] == label[i]:
            sum += 1
    return sum


def vali(net, valiloader, length):
    net.eval()
    predict = []
    label = []
    pred_proba = []
    vali_acc_sum = 0

    for idx, (volume_batch,label_batch,) in enumerate(valiloader):
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        outputs = net(volume_batch)
        # print(outputs)
        acc = acc_count(outputs, label_batch)
        vali_acc_sum += acc

        _, predict_batch = torch.max(outputs.data, 1)
        softmax_out = F.softmax(outputs, dim=1).cpu()
        proba = np.squeeze(softmax_out.data.numpy())
        label_v = label_batch[0].item()
        pred_v = predict_batch[0].item()

        predict.append(pred_v)
        label.append(label_v)
        pred_proba.append(proba)

    vali_acc = vali_acc_sum / length
    #acc_auc_fd.writelines('iteration: {}, vali acc: {}.\n'.format(iter_num, vali_acc))
    return vali_acc, label, predict, pred_proba

if __name__ == '__main__':


    net = be_ma_net(1).cuda()
    
    print(res_log)
    print(loss_log)
    print(loss_log_e)


    print('train_root is {}, train_path is {}'.format(config.root,config.train_path))
    print('vali_root is {}, vali_path is {}'.format(config.root,config.vali_path))
    print('batch_size is: {}, init_lr is: {}'.format(config.bs,config.lr))
    print("save_path is: ",config.model_save)
    print("log save_path is: ",config.log_path)
    print("################# Loading Data ###########################\n")


    train_data = dataset(config.root, config.train_path)
    trainloader = DataLoader(train_data, batch_size=config.bs, shuffle=True, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)                         
    train_len = len(train_data)

    vali_data = dataset(config.root, config.vali_path)
    valiloader = DataLoader(vali_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    vali_len = len(vali_data)
    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
  
    lr_ = config.lr
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    # net.train()
    print("the number of train samples.", len(trainloader))
    print("#################### Train Start ####################")

    date = time.strftime("%Y_%m_%d_%H_%M")
    save_path = config.model_save

    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)



    for iter_num in range(1,config.max_epoch):
        net.train()
        loss_sum = 0
        time1 = time.time()

        for idx, (volume_batch,label_batch,) in enumerate(trainloader):
            # print('fetch data cost {}'.format(time2-time1))
            # print("label value....",label_batch)

            # print("volume_batch....", volume_batch.shape)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = net(volume_batch)
            # outputs = net(volume_batch)

            # print(outputs.shape)
            loss = criterion(outputs, label_batch)
            loss_iter_v=loss.item()
            #print(loss_iter_v)
            loss_fd_i.write(str(loss_iter_v) + ' ')
            ## acc=accuary(outputs,label_batch)
            loss_sum += loss_iter_v
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        time2 = time.time()
        loss_v = loss_sum / len(trainloader)
        print('iteration:{}, loss value:{}, time:{}.'.format(iter_num, loss_v, time2 - time1))
        res_fd.writelines('iteration:{}, loss value:{}, time:{}.\n'.format(iter_num, loss_v, time2 - time1))
        loss_fd_e.writelines('iteration:{}, loss value:{}.\n'.format(iter_num, loss_v))
       
     

        if iter_num %10==0:
            train_acc, _, _, _ = vali(net, trainloader, train_len)
            vali_acc, label, predict, pred_proba = vali(net, valiloader, vali_len)

            print('train acc: {} and vali_acc: {}'.format(train_acc,vali_acc))
            res_fd.writelines('train acc: {} and vali_acc: {}\n'.format(train_acc,vali_acc))
            pred_proba = np.array(pred_proba)[:, -1].tolist()
         
            auc_v,acc,sensitivity,specificity,npv,ppv,=calc_metrics(label, predict, pred_proba, False)
            print(colored('auc_v {},acc {},sensitivity {},specificity {},npv {},ppv {}'.format(auc_v,acc,sensitivity,specificity,npv,ppv),'green'))
            res_fd.writelines("auc_v {},acc {},sensitivity {},specificity {},npv {},ppv {}\n".format(auc_v,acc,sensitivity,specificity,npv,ppv))
            model_name='iter_' + str(iter_num + 1)+'_'+str(auc_v)+'_'+str(acc)+'.pth'
            mode_path = os.path.join(save_path, model_name)
            torch.save(net.state_dict(), mode_path)
