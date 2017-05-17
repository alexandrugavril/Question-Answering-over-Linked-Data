import argparse
import sys
import numpy as np
import gensim
from Word2Vec import getBestPropertyMatch
from Word2Vec import generatePropertyModel
from CRFTrain import CRFTrain
from sparql import composeSparqlQuery
from sparql import launchSparqlQuery
from sparql import getPropertyResult
from gensim import models
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reads annotations and automatically tags them.')
    parser.add_argument('--folder', type=str, nargs='?', help="Annotations folder path")
    parser.add_argument('--pickle', type=str, nargs='?', help="Load saved pickle")
    args = parser.parse_args()
"""

pickle= "Datasets/save.p"
if pickle is not None:
    crftrain = CRFTrain(pickle = pickle)
#elif args.folder is not None:
#    crftrain = CRFTrain(folder = args.folder)
 #   crftrain.save_tags()
#else:
 #   parser.print_help()
 #   sys.exit(-1)
f1scores = []
f1score = crftrain.train(test_size=0.2, max_iterations=100)
#plt.plot(range(20,200), f1scores)
#plt.savefig("F1Scores20.200Type.png")

#print("F1 Score: " + str(crftrain.test_set_test()))d
print("Loading word model.")
model = gensim.models.KeyedVectors.load_word2vec_format('./Models/GoogleNews-vectors-negative300.bin.gz', binary=True)
print("Loaded Model")
while True:
    try:
        question = str(input("Enter your question: "))
        question = question.strip().replace("?","")

        annotations = crftrain.get_annotations_for_sentence(question)
        print("Annotations: " + str(annotations))
        result = getBestPropertyMatch(annotations, model, n=5)
        if (len(result) > 0):
            print("Resulting properties: " + str(result))
        print("Launching dbpedia sparql queries...\n")
        for properties in result:
            for property in properties:
                query = composeSparqlQuery(annotations, property[0])
                print("Sparql Query: " + str(query))
                data = launchSparqlQuery(query)
                print("Response: " + str(data))
                response = getPropertyResult(data)
                if(len(response)):
                    print(str(property[0]) + ": " + str(response) + " " + str(property[1]))
                print
    except KeyboardInterrupt:
        print("Done.")
        sys.exit(0)

    """
    crftrain = CRFTrain()
    annotations = crftrain.get_all_annotations_in_folder(args.folder)
    sentences = []
    annsForSents = []
    for annSent in annotations:
        cSent = []
        cAnn = []
        for word, ann in annSent:
            cSent.append(word)
            cAnn.append(ann)
        annsForSents.append(cAnn)
        sentences.append(cSent)
    model = models.Word2Vec(sentences, size = 10, window = 5, min_count=5, workers=4)
    modeAnn = models.Word2Vec(annsForSents, size=10, window = 5, min_count = 5, workers=4)
    print(modeAnn.wv['N'])
    """
