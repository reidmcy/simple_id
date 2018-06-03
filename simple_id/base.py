import os
import torch
import time
from .utilities import preprocesing, getTrainTest
from .neuralnet import BiRNN
from .training import trainModel
import gensim
import pandas

def createClassifier(df, nnWidth = 128, nnHeight = 2, stepSize = .001, w2vDim = 200, useFeather = True, saveDF = True, outputsDir = 'models', w2vFname = 'word2vec.bin', pickleFname = 'dfPickles.p', trainFname = 'training.csv', testFname = 'testing.csv', epochSize = 500, numEpochs = 50):
    #TODO: Explain defaults

    os.makedirs(outputsDir, exist_ok = True)

    #TODO: Check df has right column names / allow different names
    #Currently using: 'title', 'abstract' and 'class'

    df, w2v = preprocesing(df, outputsDir, w2vFname, pickleFname, w2vDim, useFeather = useFeather, saveDF = saveDF)

    #Currently assuming postives are rarer than negatives
    dfTrain, dfTest = getTrainTest(df, w2v, splitRatio = .1)

    print("Saving test-train data")
    dfTest[['source', 'title', 'class']].to_csv("{}/{}".format(outputsDir, testFname))
    dfTrain[['source', 'title', 'class']].to_csv("{}/{}".format(outputsDir, trainFname))

    #TODO: Test other NN sizes
    Net = BiRNN(w2vDim, #W2V size
                nnWidth, #Width
                nnHeight, #Height
                stepSize, #eta
                outputsDir #Autosave location
                )

    tstart = time.time()
    e = trainModel(Net, dfTest, dfTrain, epochSize, numEpochs)
    deltaT = time.time() - tstart

    print("Done {} Epochs in {:.0f}m {:.1f}s saving final model".format(Net.epoch, deltaT // 60, deltaT % 60))
    Net.save()

    if e is not None:
        print("Error: {}".format(e))
        raise e
    else:
        print("Done")
    return Net

def loadModel(modelPath):
    with open(modelPath, 'rb') as f:
        N = torch.load(f)
    if torch.cuda.is_available():
        N.cuda()
    return N

def analyseDF(df, model, w2vFname = 'word2vec.bin', outputsDir = 'models'):
    try:
        w2vPath = '{}/{}'.format(outputsDir, w2vFname)
        w2v = gensim.models.Word2Vec.load(w2vPath)
    except FileNotFoundError:
        try:
            w2v = gensim.models.Word2Vec.load(w2vFname)
        except FileNotFoundError:
            raise FileNotFoundError("W2V file missing, try running this from the same directory as the model was generated or give the W2V files path as `w2vFname`")

    results = []
    indices = []
    tstart = time.time()
    for i, (r_index, row) in enumerate(df.iterrows()):
        eta = (time.time() - tstart) / (i + 1) * (len(df) - i)
        print("Analysing {}: {:.1f}%, ETA {:.0f}m {:.1f}s".format(i,
                                                    i / len(df) * 100,
                                                    eta // 60,
                                                    eta % 60).ljust(80)
                                                  , end = '\r')
        indices.append(r_index)
        try:
            results.append(model.predictRow(row, w2v = w2v))
        except AssertionError:
            results.append({'weightP' : 0,
                    'weightN' : 0,
                    'probPos' : 0,
                    'probNeg' : 0,
                    'prediction' : 0,
            })
    deltaT = time.time() - tstart
    print("Done {} rows in {:.0f}m {:.1f}s, {:.1f}s per row".format(len(df),
                                                            deltaT // 60,
                                                            deltaT % 60,
                                                            deltaT / len(df)).ljust(80))
    return pandas.DataFrame(results, index = indices)
