from bert_nli import BertNLIModel
import pandas as pd

def load_dataset(filepath, encoding='utf-8'):
    df = pd.read_csv(filepath, encoding=encoding)
    df['t1']=df['t1'].replace("\'s", '')
    df['t2']=df['t2'].replace("\'s", '')
    a=[]
    b=[]
    c=[]
    for d1 in df['t1']:
        a.append(d1)
    for d2 in df['t2']:
        b.append(d2)
    for l in df['label']:
        c.append(l)
    #for l in df['label']:
        #if l=='not_entailment':
        #    ll='contradiction'
        #elif l=='entailment':
        #    ll='entail'
        #c.append(ll)

    return a,b,c
sent1,sent2,labels=load_dataset('/home/matsui/zemi/データ整形/ge_test.csv')

acc=len(sent1)
pre=0
#bert_type = 'bert-base'

model = BertNLIModel('/home/matsui/bert_nli/output/nli_bert-large-2020-10-07_19-29-08/nli_model_acc0.8768068134109038.state_dict')
for s1,s2,l in zip(sent1,sent2,labels):
    fla=''
    if l in 'entail':
        fla='contradiction'
    else:
        fla='entail'
    sent_pairs = [(s1,s2)]
    #print(sent_pairs)
    label,prob= model(sent_pairs)
    print('正解：',l,'予測',label[0])
    print('前提：',s1)
    print('過程：',s2)
    print(' ')
    if l==label[0] or label[0] in 'neutral':
        pre+=1


print('正解率',pre/acc)
