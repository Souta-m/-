from bert_nli import BertNLIModel

if __name__ == '__main__':
    bert_type = 'bert-large'
    model = BertNLIModel('/home/matsui/bert_nli/output/nli_bert-large-2020-11-07_17-23-12/nli_model_acc0.8831943861332694.state_dict')
    sent_pairs = [('You often eat food which is bad for our health without knowing it.','He often unknowingly eat foods that are harmful to his health.')]
    sent_pairs1=[(sent_pairs[0][1],sent_pairs[0][0])]
    labels, probs = model(sent_pairs)
    labels1,probs1=model(sent_pairs1)
    print(labels,probs)
    print(labels1,probs1)
