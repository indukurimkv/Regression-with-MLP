import numpy as np
import dummyData as dD
import skeletonNeuralNet as sNN

def sigmoid(x):
    return 1/(1+np.e**-x)
def dSigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def loss(out, expected):
    return np.square(np.subtract(expected, out))

def forward(inp, weights):
    layerOuts = []
    previousOut = inp
    for i in weights:
        previousOut = sigmoid(np.dot(previousOut, i[:-1]))
        layerOuts.append(previousOut)
    return layerOuts, previousOut
        


dataGen = dD.DummyData(10, 10)
weights = sNN.skeletonNet(3,2,2).skeleton
learnR = 0.1
for _ in range(110):
    dataSet, observedSet = dataGen.getData()
    losses = []
    predictionSet = []
    layerOutputSet = []
    for i, data in enumerate(dataSet):
        prediction = []
        layerOuts = []
        for j in data:
            _layerOut, _pred = forward(j, weights)
            prediction.append(_pred)
            layerOuts.append(_layerOut)
        lossList = [loss(prediction[ind], observedSet[ind]) for ind in range(len(prediction))]
        losses.append(np.average(lossList))
        predictionSet.append(prediction)
        layerOutputSet.append(layerOuts)
    inverseSig = [dSigmoid(i) for i in list(zip(*layerOuts))[-1]]
    stage1 = np.subtract(observedSet, predictionSet)*inverseSig
    weights[-1] -= np.average(stage1) * learnR
    print(losses)
        # weights[-1] -= prediction[-1] * dSigmoid(weights[::-1][1])
        

