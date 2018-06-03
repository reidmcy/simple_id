import torch
import torch.cuda
import torch.optim
import torch.nn
import torch.autograd
import torch.nn.functional
import numpy as np
import time

from .utilities import varsFromRow

def trainModel(N, dfTest, dfTrain, epochSize, numEpochs, nTest = None):

    trainNp = len([c for c in dfTrain['class'] if c == 1])
    trainNn = len([c for c in dfTrain['class'] if c == 0])
    testNp = len([c for c in dfTest['class'] if c == 1])
    testNn = len([c for c in dfTest['class'] if c == 0])
    print("Training: {} postive, {} negative, {:.3f} percent".format(trainNp, trainNn, trainNp / (trainNn + trainNp)))
    print("Testing : {} postive, {} negative, {:.3f} percent".format(testNp, testNn, testNp / (testNn + testNp)))

    if torch.cuda.is_available():
        N.cuda()

    optimizer = torch.optim.Adam(N.parameters(), lr=N.eta)

    if nTest is None:
        nTest = len(dfTest)

    try:
        for i in range(numEpochs):
            tstart = time.time()
            tEpoch = tstart
            for j in range(epochSize):
                if j % (epochSize // 20) == 0:
                    eta = (time.time() - tstart) / (j + 1) * (epochSize - j)
                    print("Training {}: {:.1f}%, ETA {:.0f}m {:.1f}s".format(j,
                                                    j / epochSize * 100,
                                                    eta // 60,
                                                    eta % 60).ljust(80)
                                                   ,end = '\r')
                row = dfTrain.sample(n = 1).iloc[0]
                try:
                    abVec, tiVec, yVec = varsFromRow(row)
                except:
                    raise
                    #TODO: Remove this or make it nicer
                    print()
                    print("Error encountered entering debugger")
                    print("Typing 'N.save()' will save the current model")
                    import pdb; pdb.set_trace()
                optimizer.zero_grad()
                #import pdb; pdb.set_trace()
                outputs = N(abVec, tiVec)
                #TODO: Test other losses
                loss = torch.nn.functional.cross_entropy(outputs, yVec)
                loss.backward()
                optimizer.step()

            losses = []
            errs = []
            detectionRate = []
            falsePositiveRate = []

            tstart = time.time()
            for j in range(nTest):
                if j % (nTest // 20) == 0:
                    eta = (time.time() - tstart) / (j + 1) * (nTest - j)
                    print("Testing {}: {:.1f}%, ETA {:.0f}m {:.1f}s".format(j,
                                                    j / nTest * 100,
                                                    eta // 60,
                                                    eta % 60).ljust(80)
                                                   ,end = '\r')

                row = dfTest.sample(n = 1).iloc[0]

                abVec, tiVec, yVec = varsFromRow(row)

                outputs = N(abVec, tiVec)

                loss = torch.nn.functional.cross_entropy(outputs, yVec)
                losses.append(loss.data[0])
                pred = outputs.data.max(1)[1]

                if torch.cuda.is_available():
                    errs.append(1 - pred.eq(yVec.data)[0][0])
                    if dfTest['class'].iloc[j] == 1:
                        detectionRate.append(pred.eq(yVec.data)[0][0])
                    else:
                        falsePositiveRate.append(1 - pred.eq(yVec.data)[0][0])
                else:
                    errs.append(1 - pred.eq(yVec.data)[0])
                    if dfTest['class'].iloc[j] == 1:
                        detectionRate.append(pred.eq(yVec.data)[0])
                    else:
                        falsePositiveRate.append(1 - pred.eq(yVec.data)[0])

            delta = time.time() - tEpoch
            print("Epoch {}, loss {:.3f}, error {:.3f}, detectionRate {:.3f}, falseP {:.3f}, in {:.0f}m {:.0f}s".format(i + 1, np.mean(losses), np.mean(errs), np.mean(detectionRate),  np.mean(falsePositiveRate), delta // 60, delta % 60).ljust(80))

            N.epoch += 1
            N.save()

    except KeyboardInterrupt as e:
        print("Exiting")
        return e
