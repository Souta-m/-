from sentence_transformers import SentenceTransformer,util


model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

src="Tom insisted that everythig would be all right, but I couldn't help feeling worried."
trg='Tom everything will go well but I worried about that too much.'
sentences=[src]+[trg]
#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
print(src,':',trg)
sim = util.pytorch_cos_sim(embeddings[0],embeddings[1])
print(sim.item())