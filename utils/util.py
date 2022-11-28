import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np


def read_sta(sta_txt):
    sta = open(sta_txt, 'r')
    labels = []
    predicts = []
    predict_scores = []

    cnt = 0
    for line in sta:
        line=line.replace('\n','')
        se_line=line.split(' ')
        if cnt % 3 == 0: #label
            labels.append(se_line)
        if cnt % 3 ==1: # predict
            predicts.append(se_line)
        if cnt % 3 ==2: # score
            predict_scores.append(se_line)

    return  labels,predicts,predict_scores


def write_num_ls(fd,num_ls):
    new_str = ''
    fe_list = list(map(str, num_ls))
    for value in fe_list[0:-1]:
        new_str += value
        new_str += ','
    new_str += str(fe_list[-1])
    new_str += '\n'
    # print(new_str,end=' ')
    fd.writelines(new_str)

def calc_metrics(label,predict,score,show_flag):
    tp_sum = 0
    tn_sum = 0
    fp_sum = 0
    fn_sum = 0

    ep=0

    for i in range(len(label)):
        la = label[i]
        pre = predict[i]
        if la == 1 and pre == la:
            tp_sum += 1
        if la == 1 and pre != la:  
            fn_sum += 1

        if la == 0 and pre == la:
            tn_sum += 1

        if la == 0 and pre != la:  
            fp_sum += 1

    sensitivity = float(tp_sum / (tp_sum + fn_sum))
    specificity = float(tn_sum / (tn_sum + fp_sum))  

    acc = float((tn_sum + tp_sum) / len(label))
    npv = float(tn_sum / (tn_sum + fn_sum+ep))  
    ppv = float(tp_sum / (tp_sum + fp_sum+ep))  

    fpr, tpr, thresholds = roc_curve(label, score, pos_label=1)
    auc_v = auc(fpr, tpr)

    if show_flag:
        print('AUC:', auc_v,'准确度', acc,'敏感性', sensitivity,'特异性', specificity,'阴性预测值', npv,'阳性预测值', ppv)
    return [auc_v,acc,sensitivity,specificity,npv,ppv,]


## calc auc and roc
def auc_roc(label,predict_score):
    fpr, tpr, thresholds = roc_curve(label, predict_score, pos_label=1)
    auc_v = auc(fpr, tpr)
    print("AUC : ", auc_v)
    return auc_v


## draw roc curve

def draw_multi_roc(labels,predict_scores,save_name):
    length=labels.shape[0]
    for i in range(length):
        fpr, tpr, thresholds = roc_curve(labels[i], predict_scores[i], pos_label=1)
        auc_v = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='experiment_{},(area = {:.3f})'.format(i+1,auc_v))

    plt.xlabel('Specificity rate')
    plt.ylabel('Sensitivity rate')
    plt.title('roc curves')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()  
    plt.savefig(save_name)
