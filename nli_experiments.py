from bert_nli import BertNLIModel
import pandas as pd
import csv
import numpy as np
import time
import math
import heapq
import nltk


df = pd.read_csv('sentence-bert_sim.csv',encoding='utf-8')
bert_type = 'bert-large'
model = BertNLIModel(model_path='/output/nli_bert-large-2022-09-08_13-38-01/nli_model_acc0.884513884723805.state_dict',bert_type=bert_type)


seikai=0
huseikai=0
leaner=[]
correct=[]
labels=[]
for e,c,sen in zip(df['学習者訳'],df['正解文'],df["label"]):
    if sen=="不正解文":
        huseikai+=1
        leaner.append(e)
        correct.append(c)
        labels.append(0)
    else:
        if seikai<112:
            seikai+=1
            leaner.append(e)
            correct.append(c)
            labels.append(2)


tp_s=0
tp_f=0
fp_s=0
fp_f=0
hu_sim_ave=0
se_sim_ave=0
print("正解数",seikai)
print("不正回数",huseikai)
for k,s,sen in zip(leaner,correct,labels):
    sent_pairs=[(k,s)]
    results1= model(sent_pairs)
    labels1,probs1=results1[0],results1[1]
    results2=model([(s,k)])
    labels2,probs2=results2[0],results2[1]
    if (labels1[0]!="contradiction" and labels2[0]!="contradiction"):
        if sen==2:
            tp_s+=1
        else:
            fp_s+=1
    else:
        if sen==2:
            tp_f+=1
        else:
            fp_f+=1

print("正解検知",tp_s)
print("不正解検知",fp_f)
def calc_precicsion(tp=0, fp=0):
    pre = tp / (tp + fp)
    return pre

# 再現率の算出
def calc_recall(tp=0, fn=0):
    rec = tp / (tp + fn)
    return rec

# F値の算出
def calc_f(pre=0, rec=0):
    f = (2 * pre * rec) / (pre + rec)
    return f


pre=calc_precicsion(tp_s,fp_s)
rec = calc_recall(tp_s,tp_f)
f=calc_f(pre,rec)
print("正解文",'適合率',pre,'再現率',rec,'f値',f)

pre=calc_precicsion(fp_f,tp_f)
rec = calc_recall(fp_f,fp_s)
f=calc_f(pre,rec)
print("不正解文",'適合率',pre,'再現率',rec,'f値',f)

print(tp_s+fp_s)
print(tp_f+fp_f)