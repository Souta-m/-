from bert_nli import BertNLIModel
import pandas as pd
import csv

df = pd.read_csv('/bert_nli/datasets/base.csv',encoding='utf-8')
bert_type = 'bert-large'
model = BertNLIModel(model_path='/bert_nli/output/nli_bert-large-2022-06-23_17-42-07/nli_model_acc0.8831943861332694.state_dict',bert_type=bert_type)


with open('datasets/解答文←正答文.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['解答文','正答文','ラベル','ラベルの値'])
    for k,s in zip(df['解答文'],df['正答文']):
        sent_pairs=[(s,k)]
        results1= model(sent_pairs)
        labels1,probs1=results1[0],results1[1]
        #print(labels,probs)
        writer.writerow([k,s,labels[0],max(probs[0])])
