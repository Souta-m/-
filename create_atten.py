from bert_nli import BertNLIModel
import csv
import pandas as pd
import torch
from torch import nn
import os
import numpy as np
from tqdm import tqdm
from transformers import *
from utils.utils import build_batch

if __name__ == '__main__':
    bert_type = 'bert-large'
    model = BertNLIModel(model_path='output/nli_bert-large-2022-06-23_17-42-07/nli_model_acc0.8831943861332694.state_dict',bert_type=bert_type)
    tokenizer = BertTokenizer.from_pretrained('{}-uncased'.format(bert_type))
    df=pd.read_csv("/home/matsui/zemi/phrase-bert/outsourcing-grammarly.csv")
    with open("contra_attention(large).csv","w")as f:
        writer = csv.writer(f)
        for jp,src,trg in zip(df["Japanese"],df["Correct"],df["English"]):
            sent_pairs1 = [(src,trg)]
            sent_pairs2=[(trg,src)]
            text1 = "[CLS] {} [SEP] {} [SEP]".format(src,trg)
            text2="[CLS] {} [SEP] {} [SEP]".format(trg,src)
            labels1, probs1,attens1 = model(sent_pairs1)
            labels2,probs2,attens2=model(sent_pairs2)
            tokenized_text1=tokenizer.tokenize(text1)
            tokenized_text2=tokenizer.tokenize(text2)
            if labels1[0]=="contradiction" or labels2[0]=="contradiction":
                print(tokenized_text1)
                print(attens1)
                print(tokenized_text2)
                print(attens2)
                writer.writerow([jp])
                writer.writerow(tokenized_text1)
                writer.writerow(attens1.tolist())
                writer.writerow(tokenized_text2)
                writer.writerow(attens2.tolist())

