import gensim
import pandas
import nltk
import numpy as np
import torch
import torch.optim
import torch.nn
import torch.autograd
import torch.nn.functional
import multiprocessing
import matplotlib.pyplot as plt
import seaborn

import pickle
import os.path

#For limitation (bug) in pickle
max_bytes = 2**31 - 1

def tokenizer(target):
    try:
        return nltk.word_tokenize(str(target).lower())
    except:
        raise
        #import pdb; pdb.set_trace()

def sentinizer(sent):
    try:
        return [tokenizer(s) for s in nltk.sent_tokenize(sent)]
    except TypeError:
        #Missing abstract
        return []

def genVecSeq(target, model):
    vecs = []
    try:
        if isinstance(target[0], list):
            target = sum(target, [])
    except IndexError:
        pass
    for t in target:
        try:
            vecs.append(model.wv[t])
        except KeyError:
            print("KeyError: {}".format(repr(t)))
            pass
    return vecs

def varsFromRow(row, w2v = None):
    if 'title_tokens' not in row:
        row['title_tokens'] = tokenizer(row['title'])
    if 'abstract_tokens' not in row:
        row['abstract_tokens'] = sentinizer(row['abstract'])
    if w2v is None:
        abVec = torch.autograd.Variable(torch.from_numpy(np.stack(row['abstract_vecs'])).unsqueeze(0))

        tiVec = torch.autograd.Variable(torch.from_numpy(np.stack(row['title_vecs'])).unsqueeze(0))

        yVec = torch.autograd.Variable(torch.from_numpy(np.array([row['class']])))
    else:
        try:
            abVec = torch.autograd.Variable(torch.from_numpy(np.stack(genVecSeq(row['abstract_tokens'], w2v))).unsqueeze(0))
        except ValueError:
            abVec = torch.autograd.Variable(torch.from_numpy(np.zeros([1, 200])).unsqueeze(0))

        try:
            tiVec = torch.autograd.Variable(torch.from_numpy(np.stack(genVecSeq(row['title_tokens'], w2v))).unsqueeze(0))
        except ValueError:
            tiVec = torch.autograd.Variable(torch.from_numpy(np.zeros([1, 200])).unsqueeze(0))
        yVec = torch.autograd.Variable(torch.from_numpy(np.array([-1])))

    if torch.cuda.is_available():
        abVec = abVec.cuda()
        tiVec = tiVec.cuda()
        yVec = yVec.cuda()

    return abVec, tiVec, yVec

def genWord2Vec(df, w2vDim):
    vocab = list(df['title_tokens'])
    vocab += df['abstract_tokens'].sum()

    model = gensim.models.Word2Vec(vocab,
        hs = 1, #Hierarchical softmax is slower, but better for infrequent words
        size = w2vDim, #Dim
        window = 5, #Might want to increase this
        min_count = 0,
        max_vocab_size = None,
        workers = multiprocessing.cpu_count(), #All the cores
        )
    return model

def abSplitter(ab):
    try:
        return [s.split(' ') for s in ab.split('|')]
    except AttributeError:
        return []

def saveFeather(df, savePath):
    df = df.copy()
    try:
        df = df.reset_index()
    except ValueError:
        pass
    df['title_tokens'] = df['title_tokens'].apply(lambda x: ' '.join(x))
    df['abstract_tokens'] = df['abstract_tokens'].apply(lambda x: '|'.join([' '.join(s) for s in x]))
    print("Saving DF with feather")
    df.to_feather(savePath)

def readFeather(targetPath):
    df = pandas.read_feather(targetPath)
    df['title_tokens'] = df['title_tokens'].apply(lambda x : x.split(' '))
    df['abstract_tokens'] = df['abstract_tokens'].apply(abSplitter)
    return df

def loadPickleDF(target):
    bytes_in = bytearray(0)
    input_size = os.path.getsize(target)
    with open(target, 'rb') as f:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f.read(max_bytes)
    return pickle.loads(bytes_in)

def preprocesing(df, outputsDir, w2vFname, saveFname, w2vDim, useFeather = False, saveDF = True):

    picklePath = '{}/{}'.format(outputsDir, saveFname)
    w2vPath = '{}/{}'.format(outputsDir, w2vFname)
    os.makedirs(outputsDir, exist_ok = True)

    if os.path.isfile(picklePath):
        print("Loading tokenized DF")
        if useFeather:
            df = readFeather(picklePath)
        else:
            bytes_in = bytearray(0)
            input_size = os.path.getsize(picklePath)
            with open(picklePath, 'rb') as f:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += f.read(max_bytes)
            df = pickle.loads(bytes_in)
    else:
        print("Tokenizing Titles")
        df['title_tokens'] = df['title'].apply(tokenizer)

        print("Tokenizing Abstracts")
        df['abstract_tokens'] = df['abstract'].apply(sentinizer)
        if useFeather and saveDF:
            print("Prepping DF for feather")
            saveFeather(df, picklePath)
        elif saveDF:
            print("Saving DF as pickle")
            bytes_out = pickle.dumps(df)
            n_bytes = len(bytes_out)
            with open(picklePath, 'wb') as f:
                for idx in range(0, n_bytes, max_bytes):
                    f.write(bytes_out[idx:idx+max_bytes])

    if os.path.isfile(w2vPath):
        print("Loading W2V model")
        w2v = gensim.models.Word2Vec.load(w2vPath)
    else:
        print("Generating Word2Vec model, with {} rows of data".format(len(df)))
        w2v = genWord2Vec(df, w2vDim)
        w2v.save(w2vPath)
    return df, w2v

def genWordVecs(df, w2v):
    df['title_vecs'] = df['title_tokens'].apply(lambda x : genVecSeq(x, w2v))
    df['abstract_vecs'] = df['abstract_tokens'].apply(lambda x : genVecSeq(x, w2v))

    return df

def getTrainTest(df, w2v, splitRatio = .1):
    print("Generating training and testing sets")

    dfClassified = df[df['class'] == 1]
    dfClassified = dfClassified.append(df[df['class'] == 0].sample(min([int(len(dfClassified) * 2), len(df[df['class'] == 0])])))
    dfClassified.index = range(len(dfClassified))

    print("Generating word vectors")
    dfClassified['title_vecs'] = dfClassified['title_tokens'].apply(lambda x : genVecSeq(x, w2v))
    dfClassified['abstract_vecs'] = dfClassified['abstract_tokens'].apply(lambda x : genVecSeq(x, w2v))

    dfTest = dfClassified.sample(frac = splitRatio)
    dfTrain = dfClassified.loc[set(dfClassified.index) - set(dfTest.index)]

    dfTrain.index = range(len(dfTrain))
    dfTest.index = range(len(dfTest))

    return dfTrain.copy(), dfTest.copy()

def compareRows(rows, N, w2v = None, useTitle = True, maxLen = 30):
    fig, axes = plt.subplots(figsize = (20,15),
                             nrows = len(rows) + 1 if len(rows) > 1 else 1,
                             gridspec_kw = {'height_ratios': [5] * len(rows) + ([1] if len(rows) > 1 else []) })
    aLst = []
    for i, row in enumerate(rows):
        abVec, tiVec, yVec = varsFromRow(row, w2v = w2v)
        if useTitle:
            outLSTM, (h_n, c_n) = N.lstmTi(tiVec)
            s = row['title']
        else:
            outLSTM, (h_n, c_n) = N.lstmAb(abVec)
            s = row['abstract']
        out = N(abVec, tiVec)
        probNeg = np.exp(out.data[0][0])
        probPos = np.exp(out.data[0][1])
        probNeg = probNeg / (probNeg + probPos)
        probPos = probPos / (probNeg + probPos)

        a = np.array(outLSTM.data.tolist())
        aLst.append(a[0, -1:, :])
        a = a[:,:maxLen,:]
        df = pandas.DataFrame(a[0, :, :])
        df.index = nltk.word_tokenize(s)[:a.shape[1]]
        ax = axes[i] if len(rows) > 1 else axes
        seaborn.heatmap(df, ax = ax)
        ax.set_title("Article Title: '{}'\n$P_{{negative}} = {:.4f}, P_{{positive}} = {:.4f}$".format(row['title'], probNeg, probPos))
        ax.set_xticklabels([])
        ax.set_yticklabels(ax.get_yticklabels(),rotation = 35, fontsize = 20)

    if len(rows)  > 1:
        dfDiff = pandas.DataFrame([aLst[0][0], (aLst[0] - aLst[1])[0], aLst[1][0]], index= ['top', 'diff', 'bottom'])
        seaborn.heatmap(dfDiff, ax = axes[-1], xticklabels = [i if i in np.linspace(0, aLst[0].shape[1] - 1, num = 10, dtype='int') else '' for i in range(aLst[0].shape[1])])
        axes[-1].set_title('Difference in Final Output Vectors')

    return fig, axes
