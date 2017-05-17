import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from CRFTrain import CRFTrain
from gensim import models
import math
import torch
import time
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.transforms as T
import gensim
import numpy as np

from torch.autograd import Variable
import torchvision.transforms as T

dim = 25

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.dec = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=10,
                        bias=True,
                        batch_first=False, dropout=True, bidirectional=False)
        self.fc1 = nn.Linear(dim,7)

    def forward(self, x, hidden):
        out, hidden = self.dec(x, hidden)
        y = self.fc1(out.view(-1, out.size(2)))
        return F.softmax(y)

classDictionary = {
    'N' : [1,0,0,0,0,0,0],
    'EB' : [0,1,0,0,0,0,0],
    'EI': [0, 0, 1, 0, 0, 0, 0],
    'PB': [0, 0, 0, 1, 0, 0, 0],
    'PI': [0, 0, 0, 0, 1, 0, 0],
    'TB': [0, 0, 0, 0, 0, 1, 0],
    'TI': [0, 0, 0, 0, 0, 0, 1],
}

idxToValue = {
    0: 'N',
    1: 'EB',
    2: 'EI',
    3: 'PB',
    4: 'PI',
    5: 'TB',
    6: 'TI'
}

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reads annotations and automatically tags them.')
    parser.add_argument('--folder', type=str, nargs='?', help="Annotations folder path")
    parser.add_argument('--pickle', type=str, nargs='?', help="Load saved pickle")
    args = parser.parse_args()
    if args.pickle is not None:
        crftrain = CRFTrain(pickle = args.pickle)
    elif args.folder is not None:
        crftrain = CRFTrain(folder = args.folder)
        crftrain.save_tags()
    else:
        parser.print_help()
        sys.exit(-1)
    f1scores = []
    f1score = crftrain.train(test_size=0.2, max_iterations=100)
    #plt.plot(range(20,200), f1scores)
    #plt.savefig("F1Scores20.200Type.png")

    #print("F1 Score: " + str(crftrain.test_set_test()))d

    crftrain.get_annotations_for_sentence("When was Superman born")
    crftrain.get_annotations_for_sentence("When was the Battle of Gettysburg held")
    crftrain.get_annotations_for_sentence("When was the lord of the rings written")
"""

def getAnnotations(sentence, model, word2VecModel):
    print(sentence)
    all_input = []
    wordDict = {}
    words = []
    for word in sentence:
        words.append(word.lower())
    sentence = words
    for word in sentence:
        if word not in word2VecModel.vocab:
            print("Word " + word + " not in dictionary!")
            break
        input_ = Variable(torch.from_numpy(
            np.array(word2VecModel.word_vec(word)))).float().unsqueeze(0)
        all_input.append(input_)
    print(all_input)
    if (len(all_input) == 0):
        return []
    all_input = torch.cat(all_input, 0).unsqueeze(1)
    print(all_input.size())

    hx = Variable(torch.zeros(10 * 1, 1, dim))
    cx = Variable(torch.zeros(10 * 1, 1, dim))

    y = model(all_input, (hx, cx))

    for i in range(len(y.data)):
        arr = y.data[i].numpy()
        idx = arr.argmax()
        print(words[i] + " " + str(idxToValue[idx]) + " " + str(idx))

def test(testSet, model, word2VecModel):
    totalLoss = 0.0
    for annotation in testSet:
        hx = Variable(torch.zeros(10 * 1, 1, dim))
        cx = Variable(torch.zeros(10 * 1, 1, dim))

        all_input = []
        all_target = []
        for word in annotation:
            if word[0] not in word2VecModel.vocab:
                continue
            # print(word[0])
            input_ = Variable(torch.from_numpy(
                np.array(word2VecModel.word_vec(word[0])))).float().unsqueeze(0)
            if (word[1] in classDictionary):
                target = Variable(torch.from_numpy(
                    np.array(classDictionary[word[1]]))).float().unsqueeze(0)
            else:
                all_input = []
                all_target = []
                break
            all_input.append(input_)
            all_target.append(target)
        if (len(all_input) == 0 or len(all_target) == 0 or len(
                all_input) != len(all_target)):
            continue
        all_input = torch.cat(all_input, 0).unsqueeze(1)
        all_target = torch.cat(all_target, 0)
        y = model(all_input, (hx, cx))
        loss = nn.MSELoss()(y, all_target)
        totalLoss = totalLoss + loss.data[0]
        # print (hx.data.numpy())
        print("Test loss:" + str(loss))
        is_best = False
    return totalLoss / len(testSet)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    import shutil
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

emsize = 650
nhid = 650
dropout = 0.5
epochs = 40
nlayers = 2
crftrain = CRFTrain()
annotations = crftrain.get_all_annotations_in_folder("./Adnotari/")





word2VecModel = gensim.models.KeyedVectors.load_word2vec_format('./Models/glove.word2vec', binary=False)

new_anns = []

for annotation in annotations:
    for word, target in annotation:
        if word.lower() not in word2VecModel.vocab:
            annotations.remove(annotation)
            break
    new_anns.append([(word.lower(), target) for (word, target) in annotation])
annotations = new_anns
annNo = len(annotations)
trainNo = int(0.75 * annNo)
validNo = int(annNo - trainNo)

random.shuffle(annotations)

trainSet = annotations[0:trainNo]
validSet = annotations[-validNo:]

print(len(trainSet))

model = Network()
optimizer = optim.Adam(model.parameters())

step = 1
minLoss = 1.0

#checkpoint = torch.load("model_best.pth.tar")
#best_prec1 = checkpoint['min_loss']
#model.load_state_dict(checkpoint['state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer'])

#print(test(testSet, model, word2VecModel))
#getAnnotations(["where", "was", "Justin", "Bieber", "born"], model, word2VecModel)
#getAnnotations(["when", "was", "sherlock", "holmes", "written"], model, word2VecModel)
#getAnnotations(["when", "was", "the", "world", "war"], model, word2VecModel)
#getAnnotations(["what", "language", "does", "indian", "people", "speak"], model, word2VecModel)

print("Loaded model...Starting training")

for epoch in range(100):
    totalLoss = 0.0
    trained = 0

    for annotation in trainSet:
        hx = Variable(torch.zeros(10 * 1,1, dim))
        cx = Variable(torch.zeros(10 * 1,1, dim))
        all_input = []
        all_target = []
        toTrain = True

        if toTrain:
            trained = trained + 1
            for word in annotation:
                # print(word[0])
                input_ = Variable(torch.from_numpy(np.array(word2VecModel.word_vec(word[0].lower())))).float().unsqueeze(0)
                if(word[1] in classDictionary):
                    target = Variable(torch.from_numpy(np.array(classDictionary[word[1]]))).float().unsqueeze(0)
                else:
                    all_input = []
                    all_target = []
                    break
                all_input.append(input_)
                all_target.append(target)
            if(len(all_input) == 0 or len(all_target) == 0 or
                       len(all_input) != len(all_target)):
                continue
            all_input = torch.cat(all_input, 0).unsqueeze(1)
            all_target = torch.cat(all_target, 0)
            optimizer.zero_grad()
            y = model(all_input, (hx, cx))

            loss = nn.MSELoss()(y, all_target)
            # print (hx.data.numpy())
            totalLoss = totalLoss + loss.data[0]

            loss.backward()
            optimizer.step()
    totalLoss = totalLoss / trained

    print("Epoch: " + str(epoch) + " " + "Total train Loss: " + str(totalLoss) + " on " + str(trained) + " sentences ")
    totalLoss = 0.0
    trained = 0
    for annotation in validSet:
        hx = Variable(torch.zeros(10 * 1, 1, dim))
        cx = Variable(torch.zeros(10 * 1, 1, dim))

        all_input = []
        all_target = []
        toTrain = True
        for word, target in annotation:
            if word.lower() not in word2VecModel.vocab:
                toTrain = False
                break
        if toTrain:
            trained = trained + 1
            for word in annotation:
                # print(word[0])
                input_ = Variable(torch.from_numpy(
                    np.array(word2VecModel.word_vec(word[0].lower())))).float().unsqueeze(0)
                if (word[1] in classDictionary):
                    target = Variable(torch.from_numpy(
                        np.array(classDictionary[word[1]]))).float().unsqueeze(0)
                else:
                    all_input = []
                    all_target = []
                    break
                all_input.append(input_)
                all_target.append(target)
            if (len(all_input) == 0 or len(all_target) == 0 or len(
                    all_input) != len(all_target)):
                continue
            all_input = torch.cat(all_input, 0).unsqueeze(1)
            all_target = torch.cat(all_target, 0)
            y = model(all_input, (hx, cx))
            loss = nn.MSELoss()(y, all_target)
            totalLoss = totalLoss + loss.data[0]
            # print (hx.data.numpy())
            is_best = False
            if(loss < minLoss):
                minLoss = loss.data[0]
                is_best = True

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'min_loss': minLoss,
                'optimizer': optimizer.state_dict(),
            }, is_best)
    totalLoss = totalLoss / len(validSet)
    print("Epoch: " + str(epoch) + " " + "Total Validation Loss: " + str(totalLoss) + " on " + str(trained) + " sentences ")



