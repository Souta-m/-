import pandas as pd
from bert_nli import BertNLIModel
import csv
from sentence_transformers import SentenceTransformer,util
df=pd.read_csv("/home/matsui/zemi/bert_nli/s-nli.csv")
print(df.head())
bert_type = 'bert-large'
model = BertNLIModel(model_path='/home/matsui/zemi/bert_nli/output/nli_bert-large-2022-09-08_13-38-01/nli_model_acc0.884513884723805.state_dict',bert_type=bert_type)
sbert_model = SentenceTransformer('all-mpnet-base-v2')

with open('/home/matsui/zemi/bert_nli/正解.csv', mode='w') as fs, open('/home/matsui/zemi/bert_nli/不正解.csv', mode='w') as ff:
    writer_s = csv.writer(fs)
    writer_f = csv.writer(ff)
    writer_s.writerow(["mt","answer"])
    writer_f.writerow(["mt","answer"])
    for k,s in zip(df['mt'],df['answer']):
        # print("学習者訳",k)
        # print("正解文",s)
        sent_pairs=[(s,k)]
        sentences=[s]+[k]
        results1= model(sent_pairs)
        labels1,probs1=results1[0],results1[1]
        results2=model([(k,s)])
        labels2,probs2=results2[0],results2[1]
        embeddings = sbert_model.encode(sentences)
        sim = util.pytorch_cos_sim(embeddings[0],embeddings[1])
        sim=sim.item()
        if (labels1[0]=="enatail" and labels2[0]=="entail"):
            writer_s.writerow([k,s])
        elif sim>0.90:
            writer_s.writerow([k,s])
        elif sim<0.76:
            writer_f.writerow([k,s])
        elif(labels1[0]!="enatail" and labels2[0]!="entail"):
            writer_f.writerow([k,s])
        elif(labels1[0]=="entail" and labels2[0]=="contradiction"):
            writer_f.writerow([k,s])
        elif(labels1[0]=="contradiction" and labels2[0]=="entail"):
            writer_f.writerow([k,s])
        else:
            writer_s.writerow([k,s])
