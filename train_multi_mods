import torch
import time
import os
import sys

import torch
import argparse

#from utils.dataset_mms_cat import dataset_mms
from utils.util import write_num_ls, calc_metrics



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
from net.clf_mm import be_ma_net
import ml_collections
from termcolor import colored 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

parser = argparse.ArgumentParser()

parser.add_argument('-d1_root', type=str,default='')
parser.add_argument('-d2_root', type=str,default='')
parser.add_argument('-t2_root', type=str,default='')
parser.add_argument('-nf_root', type=str,default='')

parser.add_argument('-d1_sr', type=str,default='')
parser.add_argument('-d2_sr', type=str,default='')
parser.add_argument('-t2_sr', type=str,default='')
parser.add_argument('-nf_sr', type=str,default='')

parser.add_argument('-train_path', type=str,default='')
parser.add_argument('-vali_path', type=str,default='')

parser.add_argument('-model_save', type=str,default='')

parser.add_argument('-use_sobel', type=str,default='yes')

parser.add_argument('-seq', type=int)
args = parser.parse_args()



def train_mms_config():



    config = ml_collections.ConfigDict()
    

    
    # config.seq=args.seq
    # config.use_sobel=args.us
    
    config.seq=args.seq
    if args.use_sobel=='yes':
        config.use_sobel=True
    else:
        config.use_sobel=False
 
    config.d1_root=args.d1_root    
    config.d2_root=args.d2_root
    config.t2_root=args.t2_root
    config.nonfs_root=args.nf_root

    config.d1_sr=args.d1_sr
    config.d2_sr=args.d2_sr
    config.t2_sr=args.t2_sr
    config.nf_sr=args.nf_sr
    
    config.train_path=args.train_path
    config.vali_path=args.vali_path
    
    config_model_save=args.model_save
    
   
    if config.use_sobel:
        config.log_prefix='four_mods_exp_'+str(config.seq)
    else:
        config.log_prefix='four_mods_nosobel_exp_'+str(config.seq)

    config.model_save=config_model_save+config.log_prefix+'/'
    
    config.log_path=config.model_save
    config.bs=16
    config.lr=0.01
    config.gpu='0'
    config.seed=1234

    config.min_epoch=1
    config.max_epoch=200


    config.log_name='log_from_epoch_{}_to_{}.txt'.format(config.min_epoch,config.max_epoch)
    config.use_8bit=False
    config.deterministic=True

    return config

config=train_mms_config()

os.makedirs(config.model_save,exist_ok=True)

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

res_log = config.log_prefix+'_res_log.txt'  
loss_log = config.log_prefix+'_loss_ilog.txt'
loss_log_e =config.log_prefix+'_loss_elog.txt'


res_fd= open(config.log_path+res_log, 'w')
loss_fd_i = open(config.log_path+loss_log, 'w')
loss_fd_e = open(config.log_path+loss_log_e, 'w')




if config.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


def get_net_trainer():
    if config.use_sobel:  ##8 channels
        
        print("I am 8 channels")
        from utils.dataset_8m import dataset
        ## d1_root, d2_root, t2_root, nf_root
        train_data = dataset(config.d1_root,config.d1_sr,config.d2_root,config.d2_sr,
                config.t2_root,config.t2_sr,config.nonfs_root,config.nf_sr,config.train_path)
        trainloader = DataLoader(train_data, batch_size=config.bs, shuffle=True, num_workers=0, pin_memory=True,
                                 worker_init_fn=worker_init_fn)
      
        vali_data = dataset(config.d1_root,config.d1_sr,config.d2_root,config.d2_sr,
                config.t2_root,config.t2_sr,config.nonfs_root,config.nf_sr,config.vali_path)
        valiloader = DataLoader(vali_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True,
                                worker_init_fn=worker_init_fn)
        net = be_ma_net(8)
        net.cuda()
        return net,trainloader,valiloader,len(train_data),len(vali_data)
    
    else:  ## 4 channels
        from utils.dataset_4m import dataset

        print(colored("I am 4 channels!",'green'))
        train_data = dataset(config.d1_root,config.d2_root, config.t2_root,config.nonfs_root,config.train_path)
        trainloader = DataLoader(train_data, batch_size=config.bs, shuffle=True, num_workers=0, pin_memory=True,
                                 worker_init_fn=worker_init_fn)
      
        vali_data = dataset(config.d1_root,config.d2_root, config.t2_root,config.nonfs_root,config.vali_path)
        valiloader = DataLoader(vali_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True,
                                worker_init_fn=worker_init_fn)
                               
        net = be_ma_net(4)
        net.cuda()
        return net,trainloader,valiloader,len(train_data),len(vali_data)
        
def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def accuary(outputs, label):
    _, predicted = torch.max(outputs.data, 1)
    # print(predicted)
    # print(predicted.data)
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

    for idx, (volume_bt, label_bt,) in enumerate(valiloader):

        volume_bt, label_bt = volume_bt.cuda(),label_bt.cuda()
        outputs = net(volume_bt)
        acc = acc_count(outputs, label_bt)
        vali_acc_sum += acc

        _, predict_batch = torch.max(outputs.data, 1)
        softmax_out = F.softmax(outputs, dim=1).cpu()
        proba = np.squeeze(softmax_out.data.numpy())
        label_v = label_bt[0].item()
        pred_v = predict_batch[0].item()

        predict.append(pred_v)
        label.append(label_v)
        pred_proba.append(proba)

    vali_acc = vali_acc_sum / length

    return vali_acc, label, predict, pred_proba

if __name__ == '__main__':
    
    print(res_log)
    print(loss_log)
    print(loss_log_e)
    print('train_root is {} {} {} {}, train_path is {}'.format(config.d1_root,config.d1_root,config.t2_root,config.nonfs_root,config.train_path))
    print('vali_path is {}'.format(config.vali_path))
    print('batch_size is: {}, init_lr is: {}'.format(config.bs,config.lr))
    print("save_path is: ",config.model_save)
    print("log save_path is: ",config.log_path)
    print("################# Loading Data ###########################\n")
    
    net,trainloader,valiloader,train_len,vali_len=get_net_trainer()
 
    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0001)
    
    lr_ = config.lr
    
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    print("the number of train samples.", len(trainloader))
    print("#################### Train Start ####################")


    save_path = config.model_save
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for iter_num in range(1,config.max_epoch):
        net.train()
        loss_sum = 0
        time1 = time.time()
        for idx, (volume_batch,label_batch,) in enumerate(trainloader):
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
               
        if iter_num%10==0:
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
