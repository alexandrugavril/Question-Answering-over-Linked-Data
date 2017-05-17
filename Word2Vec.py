import gensim
import pickle
import operator


def camel(s):
    return (s != s.lower() and s != s.upper())

def isPropsInModel(props, model):
    for prop in props:
        if not prop in model.vocab:
            return False
    return True

def getWords(s):
    import re
    s = s[0].upper() + s[1:]
    wordList = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', s)
    for i in range(len(wordList)):
        wordList[i] = wordList[i].lower()
    return wordList

def loadWords():
    return pickle.load(open("./Models/words.p", "rb"))

def getScore(eProp, prop, model):
    score = 0.0
    for p in prop:
        max = -1000
        for ep in eProp:
            s = model.similarity(p, ep)
            if s > max:
                max = s
        score = score + max
    return score


def getBestPropertyMatch(annotations, model, n=1):
    properties = []
    cProp = []

    for ann in annotations:
        if ann[1] == "PB":
            if len(cProp) != 0:
                if isPropsInModel(cProp, model):
                    properties.append(cProp)
                cProp = []
            cProp.append(ann[0])
        if ann[1] == "PI":
            cProp.append(ann[0])

    if len(cProp) != 0:
        if isPropsInModel(cProp, model):
            properties.append(cProp)

    print(properties)
    if len(properties) == 0:
        return [[(None, 0.0)]]

    properties[0].append(annotations[0][0].lower())
    existingProperties = loadWords()
    print("Loaded existing properties")
    finalScores = []
    scores = {}

    for prop in properties:
        for eProp in existingProperties.keys():
            scores[eProp] = getScore(existingProperties[eProp], prop, model)

        sortedList = sorted(scores.items(), key=operator.itemgetter(1))
        finalScore = sortedList[len(sortedList)-n:]
        finalScores.append(finalScore)

    return finalScores

def generatePropertyModel(model):
    f = open("./Models/agent_properties.txt", "rt")
    properties = f.readlines()
    properties = [p.strip() for p in properties]

    wordDict = {}

    j = 0
    for prop in properties:
        props = getWords(prop)
        if(len(props) > 2):
            continue
        if isPropsInModel(props, model):
            wordDict[prop] = props
            j = j + 1
        if (j == 500):
            break
    import pickle as pkl
    pkl.dump(wordDict, open("./Models/words.p", "wb"))
    print ("Saved model in ./Models/words.p")
    # Load Google's pre-trained Word2Vec model.
    #print("Loading word 2 vec")
    #print("Loaded word 2 vec...")
    #print(model.similarity("born", "birth"))
    #print(model.wv['computer'])
