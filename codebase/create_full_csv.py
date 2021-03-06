import numpy as np
import os
from tqdm import tqdm
import re
from collections import defaultdict
import pandas as pd # For importing metadata
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from statsmodels.nonparametric.smoothers_lowess import lowess



INCLUDE_LOESS = True

# (3) Make hourly data in same format
# (4) Make weekly dataset
# (5) Mean and stddev scaling


def getDayOfWeek(date):

	month, day, year = date.split('/')
	month, day, year = int(month), int(day), int(year)
	return datetime(year, month, day).weekday()


def aggregateWeatherMetadata():

	wf1 = '/Users/willlauer/Desktop/latent_load_gen/data/weather_2015_2017.csv'
	wf2 = '/Users/willlauer/Desktop/latent_load_gen/data/weather_2017_2018.csv' # Has labels
	
	weather2 = pd.read_csv(wf2) 
	weather1 = pd.read_csv(wf1, names = weather2.columns.values)

	# Filter out duplicate 2017 values. 
	weather1 = weather1[~weather1['localhour'].str.contains('/17 ')] # space is crucial

	weatherFull = weather2.append(weather1)

	# For each day, average each metric
	weather = weatherFull[['localhour', 'temperature', 'pecip_intensity']]

	di = defaultdict(int)
	days = [x.split()[0] for x in weather['localhour']]
	for d in days:
		di[d] += 1
	
	minTemps, maxTemps, sumPrecip, dayOfWeek = [], [], [], []
	days = list(set(days)) # Filter out duplicate days
	for d in tqdm(days):

		subset = weather[weather['localhour'].str.contains(d)]
		minTemps.append(subset['temperature'].min())
		maxTemps.append(subset['temperature'].max())
		sumPrecip.append(subset['pecip_intensity'].sum())
		dayOfWeek.append(getDayOfWeek(d))


	# charge home / charge home lv2 (slow / fast charging)
	dfDict = {'days': days, 'minTemps': minTemps, 'maxTemps': maxTemps, 'sumPrecip': sumPrecip, 'dayOfWeek': dayOfWeek}
	weatherFinal = pd.DataFrame(dfDict, columns = ['days', 'minTemps', 'maxTemps', 'sumPrecip', 'dayOfWeek'])

	weatherFinal.to_csv('/Users/willlauer/Desktop/latent_load_gen/data/weather_metadata_parsed.csv', mode = 'w+')

##############################################################################
# Return the length of the longest consecutive sequence of 0's, and the total 
# number of 0's
##############################################################################
def longestNeg1Subsequence(li):
	cur = 0 # Track the longest subsequence we've seen 
	longest = 0 # Track the next longest 
	totZero = 0
	for tm, x in li:
		if x[0] == -1.0:
			cur += 1 
			totZero += 1
		else:
			longest = max(cur, longest)								
			cur = 0
	return longest, totZero	




def createFullCSV(includeMetadata = None):
	# If loess == true, then add additional use_loess, test_loess, val_loess files with the loess-smoothed entries
	# Store the years to access, and the months for which we have data in each of those years
	years = ['2015', '2016', '2017', '2018']
	allMonths = {'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06', 'july': '07',
				 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'
				 }

	monthsDi = {
		'2015': allMonths,#{'january': '01', 'february': '02', 'march': '03'},
		'2016': allMonths,
		'2017': allMonths, #{'december': '12'},
		'2018': {'january': '01', 'february': '02', 'march': '03', 'april': '04', 
				 'may': '05', 'june': '06', 'july': '07', 'august': '08'
				 }
	}


	metadata = pd.read_csv('/Users/willlauer/Desktop/latent_load_gen/data/weather_metadata_parsed.csv')

	daysInMonths = [(1, 31), (2, 28), (3, 31), (4, 30), (5, 31), (6, 30), (7, 31), (8, 30), (9, 31), (10, 31), (11, 30), (12, 31)]

	EMPTY = 0.0

	minuteResolution = [(x, y) for x in range(24) for y in range(0, 50, 15)]
	assert len(minuteResolution) == 96

	times = []
	for hr in range(24):
		for minute in range(0, 46, 15):
			hrStr = str(hr) if hr >= 10 else '0' + str(hr)
			minStr = str(minute) if minute >= 10 else '0' + str(minute)
			times.append(hrStr + ':' + minStr + ':' + '00')

	savedOtherFiles, savedLoessOtherFiles, savedCarFiles, savedMetaFiles, savedLoessCarFiles = [], [], [], [], []

	#########
	# Stats #
	#########

	numMissing = 0
	numTotalHouses = 0

	for year in years:
		months = monthsDi[year]
		for month in months:
			print(year, month)
			with open('/Users/willlauer/Documents/data_15_minute/' + month + '_' + year + '.txt') as f:
				
				# Reduce excess tabs, remove '"', and split on tabs for each line
				# Also remove the newline character
				
				lines = [x.strip('\n') for x in f.readlines()]
				lines = [[y.replace('\"', '') for y in x.split('\t')] for x in lines]
				lines = [[elem for elem in line if elem != ''] for line in lines]
				lines = [line for line in lines if len(line) == 4]
			
				day2sec = 24 * 60 * 60
				hr2sec = 60 * 60
				min2sec = 60

				def toSeconds(dateStr):
					a, b = dateStr.split(' ') 
					y, m, d = a.split('-')
					hr, mn, se = b.split(':')
					return int(d) * day2sec + int(hr) * hr2sec + int(mn) * min2sec
				
				houseIDs = list(set([x[0] for x in lines]))

				lookingFor = [
					year + '-' + months[month] + '-' + (str(day) if day >= 10 else ('0' + str(day))) + ' ' + tm 
						for day in range(1, daysInMonths[int(months[month])-1][1]+1)
						for tm in times
				]

				# houseID : map to rest of line
				di = defaultdict(list)
				for line in lines:
					di[line[0]].append(line)


				counter = 0
				for houseID in tqdm(houseIDs):

					houseDict = {s: (-1.0, -1.0) for s in lookingFor}
					
					otherArr = np.empty((0, 96))
					carArr = np.empty((0, 96))
					metadataArr = np.empty((0, 4)) # Store max temp, min temp, sum precip, day of week 

					houseData = di[houseID]
					
					for i, e in enumerate(houseData):
						date = e[1]
						assert(len(e) == 4)
						if date in lookingFor:
							houseDict[date] = (float(e[2]), float(e[3]))

					# Load back in and sort in terms of ascending date
					houseData = sorted(list(houseDict.items()), key = lambda x: toSeconds(x[0]))
					
					##################################################
					# Skip all households have too many missing values
					##################################################
					consecZerosThreshold = 10		# 2.5 hours of missing data
					totZeroThreshold = 24			# If a quarter of the day is missing overall 

					# List of tuples
					# Array of lists of length 96
					dayData1 = [houseData[i:i+96] for i in range(0, len(houseData), 96)]
					zeros = [longestNeg1Subsequence(day) for day in dayData1] 

					# [
					#	[entries for day] = [(time, (x, y))]
					# ]
					# Filter out days
					dayData = [dayData1[i] for i in range(len(dayData1)) 
									if zeros[i][0] <= consecZerosThreshold and zeros[i][1] <= totZeroThreshold]

					############

					full = [] # Where we fill in all of the data
					have, haveNot = [], [] # Indices where we have and do not have data (get the pun? haha)
					
					# For each day
					for day in dayData:
						have.append([])
						haveNot.append([])
						#print(day)
						for i, x in enumerate(day):
							if x[1][0] == -1.0: # Both elements of the house data tuples are are set to -1 by default
								haveNot[-1].append(i)
							else:
								have[-1].append(i)
						
						assert(len(have[-1]) + len(haveNot[-1]) == 96)

					# Other option is to fill in the entire day
					if len(have) == 0:
						continue

					for i in range(len(dayData)):
						fI = [(-1.0, -1.0)] * 96 # full i
						for j in have[i]:
							fI[j] = dayData[i][j][1] # ignore time entry
						full.append(fI)
					
					# Fill in the missing entries
	
					for j, day in enumerate(haveNot):
						for i in day:
							next = [x for x in have[j] if x > i]
							if len(next) == 0:
								full[j][i] = full[j][i-1] # The last element is missing. Just reuse one
							else:
								next = next[0]
								distToNext = next - i + 1 # added the 1 to get the diff between filled entries
								assert(distToNext != 0)
								
								a = max(full[j][i-1][0], 0) # use
								b = max(full[j][i-1][1], 0) # car

								interpolatedUse = a + (full[j][next][0] - a) / distToNext
								interpolatedCar = b + (full[j][next][1] - b) / distToNext
								full[j][i] = (interpolatedUse, interpolatedCar)

					dayData = full # Complete with filled in entries and such

					for d, dayDatum in enumerate(dayData):
						dayStr = str(int(months[month])) + '/' + str(d+1) + '/' + str(int(year))[2:]
						
						# Pull the last three columns and use .values to convert to npy. has shape (1,3)
						metadataSubarr = metadata[metadata['days'].str.match(dayStr)].iloc[:,-4:].values 
						arr = np.array(dayDatum) 

						# other = use - car
						#print(arr)
						otherSubarr = np.clip(np.expand_dims(arr[:,0] - arr[:,1], 0), 0, 30)
						#print('-', otherSubarr)
						carSubarr = np.clip(np.expand_dims(arr[:,1], 0), 0, 30)

						if otherSubarr.shape[1] != 96 or otherSubarr.shape[1] != 96: 
							# some entries are 1-length remainders, from wraparound to the next day
							pass  

						else:
							otherArr = np.concatenate((otherArr, otherSubarr)) 
							carArr = np.concatenate((carArr, carSubarr))
							metadataArr = np.concatenate((metadataArr, metadataSubarr))



					# If we are applying loess smoothing, then do one more pass through all of the data and apply loess 
					# smoothing to each of the entries. Save these as separate files
					# Use default parameters at the moment
				
					
					if INCLUDE_LOESS:

						flattenedOther = otherArr.flatten()
						flattenedCar = carArr.flatten()

						frac = 6 / flattenedOther.shape[0]
						loessOther = lowess(flattenedOther, list(range(flattenedOther.shape[0])), return_sorted=False, frac=frac)
						loessCar = lowess(flattenedCar, list(range(flattenedCar.shape[0])), return_sorted=False, frac=frac)
						loessOtherDaySplit = np.array([loessOther[i:i+96] for i in range(0, flattenedOther.shape[0], 96)])
						loessCarDaySplit = np.array([loessCar[i:i+96] for i in range(0, flattenedCar.shape[0], 96)])

						savedLoessOtherFiles.append('loess_other_' + year + '_' + month + '_' + houseID + '.npy')
						savedLoessCarFiles.append('loess_car_' + year + '_' + month + '_' + houseID + '.npy')
						
						np.save('loess_other_' + year + '_' + month + '_' + houseID + '.npy', loessOtherDaySplit)
						np.save('loess_car_' + year + '_' + month + '_' + houseID + '.npy', loessCarDaySplit)

					np.save('other_' + year + '_' + month + '_' + houseID + '.npy', otherArr)
					np.save('car_' + year + '_' + month + '_' + houseID + '.npy', carArr)
					np.save('meta_' + year + '_' + month + '_' + houseID + '.npy', metadataArr)
					savedOtherFiles.append('other_' + year + '_' + month + '_' + houseID + '.npy')
					savedCarFiles.append('car_' + year + '_' + month + '_' + houseID + '.npy')
					savedMetaFiles.append('meta_' + year + '_' + month + '_' + houseID + '.npy')

	otherFiles = '\n'.join(savedOtherFiles)
	carFiles = '\n'.join(savedCarFiles)
	metaFiles = '\n'.join(savedMetaFiles)

	with open('otherFiles.txt', 'w+') as w:
		w.write(otherFiles)
	with open('carFiles.txt', 'w+') as w:
		w.write(carFiles)
	with open('metaFiles.txt', 'w+') as w:
		w.write(metaFiles)

	if INCLUDE_LOESS:
		loessOtherFiles = '\n'.join(savedLoessOtherFiles)
		loessCarFiles = '\n'.join(savedLoessCarFiles)
		with open('loessOtherFiles.txt', 'w+') as w:
			w.write(loessOtherFiles)
		with open('loessCarFiles.txt', 'w+') as w:
			w.write(loessCarFiles)
	


def visualizeData():

	root = '/Users/willlauer/Desktop/latent_load_gen/data/split'
	trainOther, trainCar, trainMeta = pd.read_csv(root + '/train/other.csv'), pd.read_csv(root + '/train/car.csv'), pd.read_csv(root + '/train/meta.csv')
	valOther, valCar, valMeta = pd.read_csv(root + '/val/other.csv'), pd.read_csv(root + '/val/car.csv'), pd.read_csv(root + '/val/meta.csv')
	testOther, testCar, testMeta = pd.read_csv(root + '/test/other.csv'), pd.read_csv(root + '/test/car.csv'), pd.read_csv(root + '/test/meta.csv')

	xAxis = list(range(96))
	numCurves = 10

	plt.figure(1, figsize=(9, 4))
	
	plt.subplot(131)
	for i in range(numCurves):
		plt.plot(xAxis, trainOther.loc[i].values)
	'''
	plt.subplot(132)
	for i in range(numCurves):
		plt.plot(xAxis, trainL)
	'''
	plt.subplot(133)
	for i in range(numCurves):
		plt.plot(xAxis, trainCar.loc[i].values)



	plt.show()


def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--visualize', type=int, default=0)
	parser.add_argument('--aggregate_metadata', type=int, default=0)
	args = parser.parse_args()

	if args.visualize == 1:
		visualizeData()
	else:
		if args.aggregate_metadata == 1:
			aggregateWeatherMetadata()
		createFullCSV(True)

if __name__=='__main__':
	main()
