from .utilities import tokenizer, sentinizer

import zipfile
import tempfile
import io

import gensim
import torch
import numpy as np

def makeVarArray(title, abstract, w2vPath, modelPath):
    Net = torch.load(modelPath).cuda()
    row_dict = {
        'title_tokens' : tokenizer(title),
        'abstract_tokens' : sentinizer(abstract),
    }
    row_dict['title_vecs'] = genVecSeqWithZip(row_dict['title_tokens'], w2vPath)
    row_dict['abstract_vecs'] = genVecSeqWithZip(row_dict['abstract_tokens'], w2vPath)
    preds = []
    for i in range(len(row_dict['title_vecs'])):
        predT = []
        for j in range(len(row_dict['abstract_vecs'])):
            newDict = {
                'abstract_tokens' : row_dict['abstract_tokens'][:j+1],
                'title_tokens' : row_dict['title_tokens'][:i + 1],
                'title_vecs' : row_dict['title_vecs'][:i + 1],
                'abstract_vecs' : row_dict['abstract_vecs'][:j+1],
                }
            pred = Net.predictRow(newDict)
            predT.append(float(pred['probPos']))
        preds.append(predT)
    return row_dict, preds

def w2vToZip(w2v, zipName):
    n = len(w2v.wv.vocab)
    with zipfile.ZipFile(zipName, 'w') as myzip:
        for i, word in enumerate(w2v.wv.vocab):
            with tempfile.NamedTemporaryFile() as tmp:
                np.save(tmp, w2v.wv[word])
                tmp.seek(0)
                try:
                    myzip.write(
                        filename = tmp.name,
                        arcname = word,
                    )
                except:
                    print(word)
                    pass
            if i % 1000:
                print(f"{i} out of {n}, {i / n * 100 :.3f}%")

def w2vZipLookup(words, zipLoc):
    words = set(words)
    retMapping = {}
    with zipfile.ZipFile('w2v.zip', 'r') as myzip:
        for word in words:
            with myzip.open(word.lower()) as myfile:
                f = io.BytesIO(myfile.read())
                a = np.load(f)
                retMapping[word] = a
    return retMapping

def genVecSeqWithZip(target, zipPath):
    vecs = []
    try:
        if isinstance(target[0], list):
            target = sum(target, [])
    except IndexError:
        pass
    wordsMap = w2vZipLookup(target, zipPath)
    for t in target:
        try:
            vecs.append(wordsMap[t])
        except KeyError:
            pass
    return vecs
