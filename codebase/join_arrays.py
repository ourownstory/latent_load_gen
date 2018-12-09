import numpy as np 
import create_full_csv 



with open('carFiles.txt') as r:
	carFiles = [c.strip() for c in r.readlines()] 
	joinedCarArr = np.concatenate([np.load(npyFile) for npyFile in carFiles])

with open('useFiles.txt') as r:
	useFiles = [c.strip() for c in r.readlines()] 
	joinedUseArr = np.concatenate([np.load(npyFile) for npyFile in useFiles])

with open('metaFiles.txt') as r:
	metaFiles = [c.strip() for c in r.readlines()]
	joinedMetaArr = np.concatenate([np.load(npyFile) for npyFile in metaFiles])



print(len(carFiles), len(useFiles), len(metaFiles))


assert joinedUseArr.shape[0] == joinedCarArr.shape[0] == joinedMetaArr.shape[0]

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

print(joinedUseArr.shape, joinedCarArr.shape, joinedMetaArr.shape)

# Pull out the subsets and save them
trainUse, trainCar, trainMeta = np.take(joinedUseArr, trainIndices, 0), np.take(joinedCarArr, trainIndices, 0), np.take(joinedMetaArr, trainIndices, 0)
valUse, valCar, valMeta = np.take(joinedUseArr, valIndices, 0), np.take(joinedCarArr, valIndices, 0), np.take(joinedMetaArr, valIndices, 0)
testUse, testCar, testMeta = np.take(joinedUseArr, testIndices, 0), np.take(joinedCarArr, testIndices, 0), np.take(joinedMetaArr, testIndices, 0)

np.savetxt('data/split/train/use.csv', trainUse, delimiter = ',')
np.savetxt('data/split/train/car.csv', trainCar, delimiter = ',')
np.savetxt('data/split/train/meta.csv', trainMeta, delimiter = ',')

np.savetxt('data/split/val/use.csv', valUse, delimiter = ',')
np.savetxt('data/split/val/car.csv', valCar, delimiter = ',')
np.savetxt('data/split/val/meta.csv', valMeta, delimiter = ',')

np.savetxt('data/split/test/use.csv', testUse, delimiter = ',')
np.savetxt('data/split/test/car.csv', testCar, delimiter = ',')
np.savetxt('data/split/test/meta.csv', testMeta, delimiter = ',')



# Replicate the same logic as above in the event that we are using loess smoothed data as well
if create_full_csv.INCLUDE_LOESS:
	with open('loessUseFiles.txt') as r:
		loessUseFiles = [c.strip() for c in r.readlines()]

		print('lenloess', len(loessUseFiles))
		joinedLoessUseArr = np.concatenate([np.load(npyFile) for npyFile in loessUseFiles])
	with open('loessCarFiles.txt') as r:
		loessCarFiles = [c.strip() for c in r.readlines()]
		print('lenloess', len(loessUseFiles))

		joinedLoessCarArr = np.concatenate([np.load(npyFile) for npyFile in loessCarFiles])

	assert joinedLoessUseArr.shape[0] == joinedLoessCarArr.shape[0]

	loessTrainUse, loessTrainCar = np.take(joinedLoessUseArr, trainIndices, 0), np.take(joinedLoessCarArr, trainIndices, 0)
	loessValUse, loessValCar = np.take(joinedLoessUseArr, valIndices, 0), np.take(joinedLoessCarArr, valIndices, 0)
	loessTestUse, loessTestCar = np.take(joinedLoessUseArr, testIndices, 0), np.take(joinedLoessCarArr, testIndices, 0)

	np.savetxt('data/split/train/loess_use.csv', loessTrainUse, delimiter = ',')
	np.savetxt('data/split/train/loess_car.csv', loessTrainCar, delimiter = ',')

	np.savetxt('data/split/val/loess_use.csv', loessValUse, delimiter = ',')
	np.savetxt('data/split/val/loess_car.csv', loessValCar, delimiter = ',')

	np.savetxt('data/split/test/loess_use.csv', loessTestUse, delimiter = ',')
	np.savetxt('data/split/test/loess_car.csv', loessTestCar, delimiter = ',')

