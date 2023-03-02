from bert_nli import BertNLIModel
import pandas as pd
import csv
from sentence_transformers import SentenceTransformer,util
import numpy as np
import time
import math
import heapq
import nltk

df = pd.read_csv('実験用問題セット.csv',encoding='utf-8')
#df = pd.read_csv('/home/matsui/zemi/bert_nli/sentence-bert_sim.csv',encoding='utf-8')
bert_type = 'bert-large'
model = BertNLIModel(model_path='output/nli_bert-large-2022-09-08_13-38-01/nli_model_acc0.884513884723805.state_dict',bert_type=bert_type)
sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
#sbert_model = SentenceTransformer('all-mpnet-base-v2')
#print(df_s.head())
seikai=0
huseikai=0
for sen in df["label"]:
    if sen==0 or sen==1:
        huseikai+=1
    else:
        seikai+=1

def words_to_freqdict(words):
    freqdict = {}
    for word in words:
        if word in freqdict:
            freqdict[word] = freqdict[word] + 1
        else:
            freqdict[word] = 1
    return freqdict

def cos_sim(dictA, dictB):
    dictA= words_to_freqdict(dictA)
    dictB= words_to_freqdict(dictB)
    # 文書Aのベクトル長を計算
    lengthA = 0.0
    for key,value in dictA.items():
        lengthA = lengthA + value*value
    lengthA = math.sqrt(lengthA)

    # 文書Bのベクトル長を計算
    lengthB = 0.0
    for key,value in dictB.items():
        lengthB = lengthB + value*value
    lengthB = math.sqrt(lengthB)

    # AとBの内積を計算
    dotProduct = 0.0
    for keyA,valueA in dictA.items():
        for keyB,valueB in dictB.items():
            if keyA==keyB:
                dotProduct = dotProduct + valueA*valueB
    # cos類似度を計算
    cos = dotProduct / (lengthA*lengthB)
    return cos

tp_s=0  #正解の中で正解と判定
tp_f=0  #正解の中で不正解と判定
fp_s=0  #不正解の中で不正解と判定
fp_f=0  #不正解の中で不正解と判定
print("正解数",seikai)
print("不正回数",huseikai)
for leaner,cor,tra,sen in zip(df['学習者訳'],df['正解文'],df["翻訳文"],df["label"]):
    # print("学習者訳",k)
    # print("正解文",s)
    new_cor=""
    cos1=cos_sim(leaner,cor)
    cos2=cos_sim(leaner,tra)
    if cos1>=cos2:
        new_cor=cor
    else:
        new_cor=tra
    sent_pairs=[(leaner,new_cor)]
    sentences=[leaner]+[new_cor]
    results1= model(sent_pairs)
    labels1,probs1=results1[0],results1[1]
    results2=model([(new_cor,leaner)])
    labels2,probs2=results2[0],results2[1]
    embeddings = sbert_model.encode(sentences)
    sim = util.pytorch_cos_sim(embeddings[0],embeddings[1])
    sim=sim.item()
    # print("ラベル：",sen)
    # print(sent_pairs)
    # print(sim)
    # if (labels1[0]=="entail" and labels2[0]=="entail") and sim>0.87:
    #     if sen==2:
    #         tp_s+=1
    #     else:
    #         fp_s+=1
    # else:
    #     if sen==2:
    #         tp_f+=1
    #     else:
    #         fp_f+=1

    # if  sim>0.90:
    #     if sen==2:
    #         tp_s+=1
    #     else:
    #         fp_s+=1
    # else:
    #     if sen==2:
    #         tp_f+=1
    #     else:
    #         fp_f+=1

    # if (labels1[0]=="entail" and labels2[0]=="entail"):
    #     if sen==2:
    #         tp_s+=1
    #     else:
    #         fp_s+=1
    # else:
    #     if sen==2:
    #         tp_f+=1
    #     else:
    #         fp_f+=1

    if (labels1[0]!="contradiction" and labels2[0]!="contradiction"):
        if sim>=0.80:
            if sen==2:
                tp_s+=1
            else:
                fp_s+=1
        else:
            if sen==2:
                tp_f+=1
            else:
                fp_f+=1          
    else:
        if sim>=0.83:
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


# fp_s-=9
# tp_s+=9

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