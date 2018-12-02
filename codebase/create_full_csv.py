import numpy as np
import os
from tqdm import tqdm
import re
from collections import defaultdict
import pandas as pd # For importing metadata
import matplotlib.pyplot as plt

# Needed for data: 
# (2) Add metadata: day of week, weather (total rain and min+max temperature)
# 		Link house IDs with EV to two metadata columns - vehicle_chargn_home, vehicle_chargn_lv2
# (3) Make hourly data in same format
# (4) Make weekly dataset
# (5) Mean and stddev scaling

def displayDayData(f):
	'''
	Takes in npy filename and prints it out
	'''
	arr = np.load(f)
	plt.plot(arr)

	plt.show()


def aggregateWeatherMetadata():

	wf1 = '/Users/willlauer/Desktop/latent_load_gen/data/weather_2015_2017.csv'
	wf2 = '/Users/willlauer/Desktop/latent_load_gen/data/weather_2017_2018.csv' # Has labels
	
	weather2 = pd.read_csv(wf2) 
	weather1 = pd.read_csv(wf1, names = weather2.columns.values)

	# Filter out duplicate 2017 values
	weather1 = weather1[~weather1['localhour'].str.contains('/17 ')] # space is crucial

	weatherFull = weather2.append(weather1)

	# For each day, average each metric
	weather = weatherFull[['localhour', 'temperature', 'pecip_intensity']]

	di = defaultdict(int)
	days = [x.split()[0] for x in weather['localhour']]
	for d in days:
		di[d] += 1
	
	minTemps, maxTemps, sumPrecip = [], [], []
	days = list(set(days)) # Filter out duplicate days
	for d in tqdm(days):
		subset = weather[weather['localhour'].str.contains(d)]
		minTemps.append(subset['temperature'].min())
		maxTemps.append(subset['temperature'].max())
		sumPrecip.append(subset['pecip_intensity'].sum())


	dfDict = {'days': days, 'minTemps': minTemps, 'maxTemps': maxTemps, 'sumPrecip': sumPrecip}
	weatherFinal = pd.DataFrame(dfDict, columns = ['days', 'minTemps', 'maxTemps', 'sumPrecip'])

	weatherFinal.to_csv('/Users/willlauer/Desktop/latent_load_gen/data/weather_metadata_parsed.csv', mode = 'w+')


	#meanWeather = weather.mean(['temperature', 'pecip_intensity'], axis = 0)

##############################################################################
# Return the length of the longest consecutive sequence of 0's, and the total 
# number of 0's
##############################################################################
def longestNeg1Subsequence(li):
	cur = 0 # Track the longest subsequence we've seen 
	longest = 0 # Track the next longest 
	totZero = 0
	for i in range(len(li)):
		if i == -1.0:
			cur += 1 
			totZero += 1
		else:
			cur = max(cur, longest)								
			nextLongest = 0
	return longest, totZero	


def createFullCSV(includeMetadata = None):
	# Store the years to access, and the months for which we have data in each of those years
	years = ['2015', '2016', '2017']

	allMonths = {'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06', 'july': '07',
				 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'
				 }

	monthsDi = {
		'2015': allMonths,
		'2016': allMonths,
		'2017': allMonths,
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

	savedUseFiles, savedCarFiles = [], []

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

				#for x in di:
				#	print(x, '\t', len(di[x]))

				counter = 0
				for houseID in tqdm(houseIDs):

					houseDict = {s: (-1.0, -1.0) for s in lookingFor}
					
					useArr = np.empty((0, 96))
					carArr = np.empty((0, 96))
					metadataArr = np.empty((0, 3))

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

					times = [x[0] for x in houseData]
					houseData = [x[1] for x in houseData]  # get all times seen for this house, sorted

					full = [] # Where we fill in all of the data
					have, haveNot = [], [] # Indices where we have and do not have data (get the pun? haha)
					for i, x in enumerate(houseData):
						if x[0] == -1.0: # Both elements of the house data tuples are are set to -1 by default
							haveNot.append(i)
						else:
							have.append(i)


					# Fill in the entries we currently have
					full = [(-1.0, -1.0)] * len(houseData)
					for i in have:
						full[i] = houseData[i]

					# Fill in the missing entries
					for i in haveNot:
						next = [x for x in have if x > i]
						if len(next) == 0:
							full[i] = full[i-1] # The last element is missing. Just reuse one
						else:
							next = next[0]
							distToNext = next - i + 1 # added the 1 to get the diff between filled entries
							assert(distToNext != 0)

							interpolatedUse = full[i-1][0] + (houseData[next][0] - full[i-1][0]) / distToNext
							interpolatedCar = full[i-1][1] + (houseData[next][1] - full[i-1][1]) / distToNext
							full[i] = (interpolatedUse, interpolatedCar)

					houseData = full # Complete with filled in entries and such

					assert(len([x for x in houseData if x[0] == -1.0]) == 0)

					# Each entry should contain 1 day, with 2 elements
					dayData = [houseData[i:i+96] for i in range(0, len(houseData), 96)]

					zeros = [longestNeg1Subsequence(day) for day in dayData]
					dayData = [(times[i], dayData[i]) for i in range(len(dayData)) 
									if zeros[i][0] <= consecZerosThreshold and zeros[i][1] <= totZeroThreshold]


					for tm, dayDatum in dayData:

						dayStr = tm.split()[0]
						uds = dayStr.split('-') # In form year-month-day
						dayStr = '/'.join([uds[1], uds[2], uds[0]]) # Format as the month/day/year, as in metadata
						dayMetadata = metadata[metadata['days'].str.contains(dayStr)]
						print(dayMetadata.shape)

						arr = np.array(dayDatum) 
						useSubarr = np.expand_dims(arr[:,0], 0) 
						carSubarr = np.expand_dims(arr[:,1], 0)					

						if useSubarr.shape[1] != 96 or carSubarr.shape[1] != 96:
							pass  # some entries are 1-length remainders, from wraparound to the next day
						else:
							useArr = np.concatenate((useArr, carSubarr))
							carArr = np.concatenate((carArr, useSubarr))



					np.save('use_' + year + '_' + month + '_' + houseID + '.npy', useArr)
					np.save('car_' + year + '_' + month + '_' + houseID + '.npy', carArr)
					savedUseFiles.append('use_' + year + '_' + month + '_' + houseID + '.npy')
					savedCarFiles.append('car_' + year + '_' + month + '_' + houseID + '.npy')

				
	useFiles = '\n'.join(savedUseFiles)
	carFiles = '\n'.join(savedCarFiles)

	with open('useFiles.txt', 'w+') as w:
		w.write(useFiles)
	with open('carFiles.txt', 'w+') as w:
		w.write(carFiles)


def main():
	#displayDayData('use_2017_october_9934.npy')
	#aggregateWeatherMetadata()
	createFullCSV(True)

if __name__=='__main__':
	main()
