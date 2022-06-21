import hashlib
import numpy as np

# seed = hashlib.sha256(b"Ignore mimes").hexdigest()
# np.random.seed(int(seed, 36) & 0XFFFFFFFF)

class DummyData:
    def __init__(self, batchSize, epochs, numDataPoints = 100000, radius = 3):
        self.radius = 3
        self.batchSize = batchSize
        self.numDPoints = numDataPoints
        self.epochs = epochs
        
        self.randArray = np.random.uniform(0,1,(self.numDPoints,2))
        self.polarPoints = [(2*np.pi*i[0], self.radius*np.sqrt(i[1])) for i in self.randArray]
        self.cartInp = [(i[1]*np.cos(i[0]), i[1]*np.sin(i[0])) for i in self.polarPoints]
        
        
    def getData(self):
        datOut = []
        expOut = []
        for i in range(self.epochs):
            indeces = np.random.randint(0,self.numDPoints, (self.batchSize, ))
            dataPoints = [self.cartInp[i] for i in indeces]
            datOut.append(dataPoints)
            expOut.append([1 if ((i[0]**2+i[1]**2-9<0) and ((i[0]**2-i[1]<0))) else 0 for i in dataPoints])
            
        self.currData = datOut
        self.expected = expOut
        
        return self.currData, self.expected
