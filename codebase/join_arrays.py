import numpy as np 
import create_full_csv 


with open('carFiles.txt') as r:
	carFiles = [c.strip() for c in r.readlines()]
	joinedCarArr = np.concatenate([np.load(npyFile) for npyFile in carFiles])

with open('otherFiles.txt') as r:
	otherFiles = [c.strip() for c in r.readlines()] 
	joinedOtherArr = np.concatenate([np.load(npyFile) for npyFile in otherFiles])

houseIdsLi = []
with open('metaFiles.txt') as r:
    metaFiles = [c.strip() for c in r.readlines()]
    houseIds = [x.split('_')[3].split('.')[0] for x in metaFiles]

    lengths = [np.load(npyFile).shape[0] for npyFile in metaFiles]
    for i, houseId in enumerate(houseIds):
        for _ in range(lengths[i]):
            houseIdsLi.append(houseId)
    joinedMetaArr = np.concatenate([np.load(npyFile) for npyFile in metaFiles])
print(len(carFiles), len(otherFiles), len(metaFiles), len(houseIdsLi))


assert joinedOtherArr.shape[0] == joinedCarArr.shape[0] == joinedMetaArr.shape[0]

num = joinedOtherArr.shape[0]
print('num', num)

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

print(joinedOtherArr.shape, joinedCarArr.shape, joinedMetaArr.shape)

# Pull out the subsets and save them
print(len(trainIndices), len(valIndices), len(testIndices))
trainOther, trainCar, trainMeta = np.take(joinedOtherArr, trainIndices, 0), np.take(joinedCarArr, trainIndices, 0), np.take(joinedMetaArr, trainIndices, 0)
valOther, valCar, valMeta = np.take(joinedOtherArr, valIndices, 0), np.take(joinedCarArr, valIndices, 0), np.take(joinedMetaArr, valIndices, 0)
testOther, testCar, testMeta = np.take(joinedOtherArr, testIndices, 0), np.take(joinedCarArr, testIndices, 0), np.take(joinedMetaArr, testIndices, 0)

trainHouseIds = [houseIdsLi[i] for i in trainIndices]
valHouseIds = [houseIdsLi[i] for i in valIndices]
testHouseIds = [houseIdsLi[i] for i in testIndices]


with open('houseIdsTrain.txt', 'w+') as w:
    s = '\n'.join(trainHouseIds)
    w.write(s)

with open('houseIdsVal.txt', 'w+') as w:
    s = '\n'.join(valHouseIds)
    w.write(s)

with open('houseIdsTest.txt', 'w+') as w:
    s = '\n'.join(testHouseIds)
    w.write(s)


np.savetxt('data/split/train/other.csv', trainOther, delimiter = ',')
np.savetxt('data/split/train/car.csv', trainCar, delimiter = ',')
np.savetxt('data/split/train/meta.csv', trainMeta, delimiter = ',')

np.savetxt('data/split/val/other.csv', valOther, delimiter = ',')
np.savetxt('data/split/val/car.csv', valCar, delimiter = ',')
np.savetxt('data/split/val/meta.csv', valMeta, delimiter = ',')

np.savetxt('data/split/test/other.csv', testOther, delimiter = ',')
np.savetxt('data/split/test/car.csv', testCar, delimiter = ',')
np.savetxt('data/split/test/meta.csv', testMeta, delimiter = ',')



# Replicate the same logic as above in the event that we are using loess smoothed data as well
if create_full_csv.INCLUDE_LOESS:
	with open('loessOtherFiles.txt') as r:
		loessOtherFiles = [c.strip() for c in r.readlines()]
		print('lenloess', len(loessOtherFiles))
		joinedLoessOtherArr = np.concatenate([np.load(npyFile) for npyFile in loessOtherFiles])
	with open('loessCarFiles.txt') as r:
		loessCarFiles = [c.strip() for c in r.readlines()]
		print('lenloess', len(loessCarFiles))
		joinedLoessCarArr = np.concatenate([np.load(npyFile) for npyFile in loessCarFiles])

	assert joinedLoessOtherArr.shape[0] == joinedLoessCarArr.shape[0]

	loessTrainOther, loessTrainCar = np.take(joinedLoessOtherArr, trainIndices, 0), np.take(joinedLoessCarArr, trainIndices, 0)
	loessValOther, loessValCar = np.take(joinedLoessOtherArr, valIndices, 0), np.take(joinedLoessCarArr, valIndices, 0)
	loessTestOther, loessTestCar = np.take(joinedLoessOtherArr, testIndices, 0), np.take(joinedLoessCarArr, testIndices, 0)

	np.savetxt('data/split/train/loess_other.csv', loessTrainOther, delimiter = ',')
	np.savetxt('data/split/train/loess_car.csv', loessTrainCar, delimiter = ',')

	np.savetxt('data/split/val/loess_other.csv', loessValOther, delimiter = ',')
	np.savetxt('data/split/val/loess_car.csv', loessValCar, delimiter = ',')

	np.savetxt('data/split/test/loess_other.csv', loessTestOther, delimiter = ',')
	np.savetxt('data/split/test/loess_car.csv', loessTestCar, delimiter = ',')

