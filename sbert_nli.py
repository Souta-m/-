from sentence_transformers import SentenceTransformer,util
from bert_nli import BertNLIModel
import pandas as pd
import csv
# 閾値を格納する配列
th_input_list = []
th_input = 0
while th_input != '':
    th_input = input("閾値を入力してください：")
    if th_input == '':
        break
    th_input_list.append(float(th_input))

df = pd.read_csv('/datasets/base.csv',encoding='utf-8')
df_s=pd.read_csv("/amano/sentencebert/アウトソーシング検証用データ.csv")
bert_type="bert-large"
sbert_model = SentenceTransformer('all-mpnet-base-v2')
nli_model = BertNLIModel(model_path='/output/nli_bert-large-2022-09-08_13-38-01/nli_model_acc0.884513884723805.state_dict',bert_type=bert_type)
src='they finally acknowledged it as true .'
trg='they admit that it is true .'
sentences=[src]+[trg]
#Sentences are encoded by calling model.encode()
embeddings = sbert_model.encode(sentences)

#Print the embeddings
print(src,':',trg)
sim = util.pytorch_cos_sim(embeddings[0],embeddings[1])
print(sim.item())

sent_pairs =[(src,trg)]
sent_pairs1=[(sent_pairs[0][1],sent_pairs[0][0])]
results= nli_model(sent_pairs)
labels,probs=results[0],results[1]
print(labels,probs)

for k,s,sen in zip(df['解答文'],df['正答文'],df_s["010のみ正解とする"]):
    print(s,':',k)
    sent_pairs=[(s,k)]
    sentences=[s]+[k]
    results1= nli_model(sent_pairs)
    labels1,probs1=results1[0],results1[1]
    results2=nli_model([(k,s)])
    labels2,probs2=results2[0],results2[1]
    print(labels1,probs1)
    embeddings = sbert_model.encode(sentences)
    sim = util.pytorch_cos_sim(embeddings[0],embeddings[1])
    print(sim.item())