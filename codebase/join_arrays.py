import numpy as np 

with open('carFiles.txt') as r:
	carFiles = [c.strip() for c in r.readlines()] 
	joinedCarArr = np.concatenate([np.load(npyFile) for npyFile in carFiles])

with open('useFiles.txt') as r:
	useFiles = [c.strip() for c in r.readlines()] 
	joinedUseArr = np.concatenate([np.load(npyFile) for npyFile in useFiles])

assert joinedUseArr.shape[0] == joinedCarArr.shape[0]

num = joinedUseArr.shape[0]

# Train, val, test split
train, val, test = 0.8, 0.1, 0.1 

indices = np.array(list(range(num)))
numTrain, numVal, numTest = int(train * num), int(val * num), int(test * num)


# Choose random indices to make up our train, val, test sets, without replacement
trainIndices = np.random.choice(indices, numTrain, replace = False)
indices = np.array([i for i in indices if i not in trainIndices])

valIndices = np.random.choice(indices, numVal, replace = False)
indices = np.array([i for i in indices if i not in valIndices])

testIndices = indices

print(joinedUseArr.shape, joinedCarArr.shape)

# Pull out the subsets and save them
trainUse, trainCar = np.take(joinedUseArr, trainIndices, 0), np.take(joinedCarArr, trainIndices, 0)
valUse, valCar = np.take(joinedUseArr, valIndices, 0), np.take(joinedCarArr, valIndices, 0)
testUse, testCar = np.take(joinedUseArr, testIndices, 0), np.take(joinedCarArr, testIndices, 0)

np.savetxt('data/split/train/use.csv', trainUse, delimiter = ',')
np.savetxt('data/split/train/car.csv', trainCar, delimiter = ',')

np.savetxt('data/split/val/use.csv', valUse, delimiter = ',')
np.savetxt('data/split/val/car.csv', valCar, delimiter = ',')

np.savetxt('data/split/test/use.csv', testUse, delimiter = ',')
np.savetxt('data/split/test/car.csv', testCar, delimiter = ',')
