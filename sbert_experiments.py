import pandas as pd
import csv
from sentence_transformers import SentenceTransformer,util

print("数字を入力")
inp=float(input())
df = pd.read_csv('実験用問題セット.csv',encoding='utf-8')
df = pd.read_csv('sentence-bert_sim.csv',encoding='utf-8')
sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
#print(df_s.head())
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
            labels.append(1)


tp_s=0
tp_f=0
fp_s=0
fp_f=0
hu_sim_ave=0
se_sim_ave=0
print("正解数",seikai)
print("不正回数",huseikai)
for k,s,sen in zip(leaner,correct,labels):
    sentences=[s]+[k]
    embeddings = sbert_model.encode(sentences)
    sim = util.pytorch_cos_sim(embeddings[0],embeddings[1])
    sim=sim.item()
    if sim>=inp:
        if sen==1:
            tp_s+=1
            se_sim_ave+=sim
        else:
            fp_s+=1
            hu_sim_ave+=sim
    else:
        if sen==0:
            fp_f+=1
            hu_sim_ave+=sim
        else:
            tp_f+=1
            se_sim_ave+=sim



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

# print(se_sim_ave/112)
# print(hu_sim_ave/112)