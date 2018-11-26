import numpy as np
import os
from tqdm import tqdm
import re
from collections import defaultdict


#months = {'january': '01', 'february': '02', 'march': '03', 'may': '05'}  # missing april
#years = ['2015']


#april_2015.txt		august_2016.txt		february_2017.txt	july_2016.txt		may_2016.txt		october_2015.txt	september_2015.txt
#august_2015.txt		december_2016.txt	january_2017.txt	june_2015.txt	november_2016.txt	october_2016.txt	september_2016.txt

# Store the years to access, and the months for which we have data in each of those years
years = ['2015', '2016', '2017']
monthsDi = {
	'2015': {'april': '04', 'august': '08', 'june': '06', 'october': '10', 'september': '09'},
	'2016': {'august': '08', 'december': '12', 'july': '07', 'may': '05', 'october': '10', 'september': '09', 'november': '11'},
	'2017': {'january': '01', 'february': '02'}
}


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

for year in years:
	months = monthsDi[year]
	for month in months:
		with open('/Users/willlauer/Downloads/' + month + '_' + year + '.txt') as f:
			
			# Reduce excess tabs, remove '"', and split on tabs for each line
			# Also remove the newline character
			
			lines = [x.strip('\n') for x in f.readlines()]
			lines = [[y.replace('\"', '') for y in x.split('\t')] for x in lines]
			
			day2sec = 24 * 60 * 60
			hr2sec = 60 * 60
			min2sec = 60

			def toSeconds(dateStr):
				#print(dateStr)
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

			di = defaultdict(list)
			for line in lines:
				di[line[0]].append(line)



			counter = 0
			for houseID in tqdm(houseIDs):

				houseDict = {s: (0.0, 0.0) for s in lookingFor}
				
				useArr = np.empty((0, 96))
				carArr = np.empty((0, 96))

				houseData = di[houseID]

				# Fill in houseDict
				for i, e in enumerate(houseData):
					date = e[1]
					e = [x for x in e if x != '']
					if len(e) == 3: # We have use data but no car data
						houseDict[date] = (float(e[2]), 0.0)
					elif len(e) == 4:
						houseDict[date] = (float(e[2]), float(e[3]))


				# Load back in and sort in terms of ascending date
				houseData = sorted(list(houseDict.items()), key = lambda x: toSeconds(x[0]))
				
				houseData = [x[1] for x in houseData]

				# Each entry should contain 1 day, with 2 elements
				dayData = np.array([houseData[i:i+96] for i in range(0, len(houseData), 96)])

				for d in dayData:
					arr = np.array(d) # Cut off extra entry
					useSubarr = np.expand_dims(arr[:,1], 0) # Confused about this ordering
					carSubarr = np.expand_dims(arr[:,0], 0)

					if useSubarr.shape[1] != 96 or carSubarr.shape[1] != 96:
						pass  # many entries are 1-length remainders, from wraparound to the next day
					else:
						#print('not')
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