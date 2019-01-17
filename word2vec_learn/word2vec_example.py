"""
A example of word2vec based on pytorch
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import torch.utils.data as data
import torch.nn.functional as F

#hyper parameter
CONTENT_SIZE = 2
EMBEDDING_DIM = 10
EPOCH = 10
LR = 0.001

#data preparation
test_scentence = """Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say
that they were perfectly normal, thank you very much. They were the last
people you'd expect to bs=20, out_features=128, bias=True)
 involved in anything strange or mysterious,
because they just didn'ts=20, out_features=128, bias=True)
  (linear2): Linear(in_features=128, out_features=160, bias= hold with such nonsense.
Mr. Dursley was the dires=20, out_features=128, bias=True)
  (linear2): Linear(in_features=128, out_features=160, bias=ctor of a firm called Grunnings, which made
drills. He was a big, bes=20, out_features=128, bias=True)
  (linear2): Linear(in_features=128, out_features=160, bias=efy man with hardly any neck, although he did
have a very large mustacs=20, out_features=128, bias=True)
  (linear2): Linear(in_features=128, out_features=160, bias=he. Mrs. Dursley was thin and blonde and had
nearly twice the usual as=20, out_features=128, bias=True)
  (linear2): Linear(in_features=128, out_features=160, bias=mount of neck, which came in very useful as she
spent so much of her tims=20, out_features=128, bias=True)
  (linear2): Linear(in_features=128, out_features=160, bias=e craning over garden fences, spying on the
neighbors. The Dursleys had a small son called Dudley and in their
opinion there was no finer boy anywhere.
The Dursleys had everything they wanted, but they also had a secret, and
their greatest fear was ths=20, out_features=128, bias=True)
  (linear2): Linear(in_features=128, out_features=160, bias=t somebody would discover it. They didn't
think they could bear it is=20, out_features=128, bias=True)
  (linear2): Linear(in_features=128, out_features=160, bias= anyone found out about the Potters. Mrs.
Potter was Mrs. Dursley's s=20, out_features=128, bias=True)
  (linear2): Linear(in_features=128, out_features=160, bias=ister, but they hadn't met for several years;
in fact, Mrs. Dursley prets=20, out_features=128, bias=True)
  (linear2): Linear(in_features=128, out_features=160, bias=nded she didn't have a sister, because her
sister and her good for nothing husband were as unDursleyish as it was
possible to be. The Dursleys shuddered to think what the neighbors would
say if the Potters arrived in the street. The Dursleys knew that the
Potters had a small son, too, but they had never even seen him. This boy
was another good reason for keeping the Potters away; they didn't want
Dudley mixing with a child like that.""".split()

# print(test_scentence)
trigrams = [([test_scentence[i], test_scentence[i + 1]], test_scentence[i + 2]) for i in range(len(test_scentence) - 2)]
# print(trigrams)
#built word index map
word_dict = set(test_scentence)
# print(len(word_dict))
word_index = {word: i for i, word in enumerate(word_dict)}
# print(word_index)

#network
class W2v(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(W2v, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size*embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs).view((1, -1)) #what is view here
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out) #softmax is?
        return log_probs

#train
model = W2v(len(word_dict), EMBEDDING_DIM, CONTENT_SIZE)
optimizer = torch.optim.SGD(model.parameters(), LR)
loss_func = nn.NLLLoss()
losses = []
for epoch in range(EPOCH):
	total_loss = torch.Tensor([0])
	for context, content in trigrams:
		# print(map(lambda w: word_dict[w], context))
		context_idxs = list(map(lambda w: word_index[w], context))
		context_var = torch.autograd.Variable(torch.LongTensor(context_idxs))
		model.zero_grad()
		output = model(context_var)
		loss = loss_func(output, torch.autograd.Variable(torch.LongTensor(word_index[content])))
		loss.backward()
		optimizer.step()

#visualization
