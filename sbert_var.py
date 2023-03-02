import pandas as pd
import csv
from sentence_transformers import SentenceTransformer,util

df=pd.read_csv("正解文翻訳_cos類似度.csv")
sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

with open("正解文_sbert.csv","w")as f:
    writer = csv.writer(f)
    writer.writerow(["出題文","学習者","英語答え","翻訳","cos類似度","学習者訳長さ","正解文長さ"])
    for syu,gak,ans,tra,label in zip(df["出題文"],df["学習者"],df["英語答え"],df["翻訳"],df["ラベル"]):
        if label==0:
            sentences=[str(gak)]+[str(ans)]
            embeddings = sbert_model.encode(sentences)
            sim = util.pytorch_cos_sim(embeddings[0],embeddings[1])
            sim=sim.item()
            result1 = len(str(gak).split())
            result2 = len(str(ans).split())
        else:
            sentences=[str(gak)]+[str(tra)]
            embeddings = sbert_model.encode(sentences)
            sim = util.pytorch_cos_sim(embeddings[0],embeddings[1])
            sim=sim.item()
            result1 = len(str(gak).split())
            result2 = len(str(tra).split())
        writer.writerow([syu,gak,ans,tra,sim,result1,result2])