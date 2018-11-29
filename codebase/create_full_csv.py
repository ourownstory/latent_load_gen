
'''
To keep in numerical form:
- dataid (categorical, but we need to preserve it so it can be matched with the other tables)
- resident_*
- ac_compressor_num
- total_annual_income
- temp_setting_* (need to replace On/off)
- thermostat_is_programmable
- electronic_device_* (need to replace None with 0 and replace 5 or more with 5)
- vehicle_charge* (remove %)


Other modifications:
- water_leak_caused_financial_cost (deleted)
- collect_water_rain_way (change to indicator variable)
'''

#######################################################################################
# Handle all preprocessing of the metadata file. Mostly encocing categorical variables,
# and some formatting
#######################################################################################

'''
metadata2017 = pd.read_csv('/Users/willlauer/Downloads/pecan_surveys_2017.csv', encoding = "ISO-8859-1")

replacementDict = {
	'Off, or do not have heating system': 0,
	'None': 0,
	'5 or more': 5
}

toKeepNumerical = [
	'dataid',
	'residents_under_5', 'residents_6_to_12',
	'residents_13_to_18', 'residents_19_to_24',
	'residents_25_to_34', 'residents_35_to_49',	
	'residents_50_to_64', 'residents_65_and_older',
	'ac_compressor_num',
	'total_annual_income',
	'temp_setting_weekend_daylight_wintr', 'thermostat_brand',
	'thermostat_is_programmable', 'temp_setting_weekday_workday_sumr',
	'temp_setting_weekday_morning_sumr', 'temp_setting_weekday_evening_sumr',
	'temp_setting_weekday_sleeping_hr_sumr', 'temp_setting_weekend_daylight_sumr',
	'temp_setting_weekday_workday_wintr', 'temp_setting_weekday_morning_wintr',
		#'temp_setting_sleeping_hr_wintr',
	'electronic_device_cable_box_number',
	'electronic_device_dvr_number',
	'electronic_device_router_number',
	'electronic_device_gaming_sys_number'
]
# Fill in all missing with 0
metadata2017 = metadata2017.fillna(0)


# Remove percent symbols in vehicle_charge columns
vehicleChargeHeaders = ['vehicle_chargn_home', 'vehicle_chargn_level2ev', 'vehicle_chargn_public', 'vehicle_chargn_work']
for vch in vehicleChargeHeaders:
	metadata2017[vch] = metadata2017[vch].map(lambda x: int(str(x).replace('%', '')))

# Use lower end of income range rather than full range 
# Parse and convert to int
def setIncomeAsLowerBound(x):
	if x == 0 or x[0] == 'L': return 0
	else: return int(x.split('-')[0].strip().replace(',', '')[1:])
metadata2017['total_annual_income'] = metadata2017['total_annual_income'].map(setIncomeAsLowerBound)


# Do some other replacements
metadata2017 = metadata2017.replace(replacementDict)


# Convert all categorical column values into numerical indices
for header in list(metadata2017):
	if header not in toKeepNumerical: 
		di = dict([e[:: -1] for e in enumerate(metadata2017[header].unique())])
		metadata2017[header] = metadata2017[header].map(di)

# metadata2017.to_csv('metadata_parsed.csv')

print(metadata2017.loc[[0]])

dataIds = list(enumerate(metadata2017['dataid']))
metadataDict = {dataId[1]: metadata2017.loc[[dataId[0]]][1:] for dataId in dataIds}
for x in metadataDict.items():
	print(x)
exit(0)
'''

import numpy as np
import os
from tqdm import tqdm
import re
from collections import defaultdict
import pandas as pd # For importing metadata

# Store the years to access, and the months for which we have data in each of those years
years = ['2015', '2016', '2017']
monthsDi = {
	'2015': {'april': '04', 'august': '08', 'december': '12',
			 'june': '06', 'october': '10', 'september': '09',
			 'july': '07', 'november': '11'
			 },
	'2016': {'january': '01', 'february': '02', 
			 'march': '03', 'april': '04', 'may': '05', 
			 'june': '06', 'july': '07', 'august': '08', 
			 'december': '12', 'july': '07', 'may': '05', 
			 'october': '10', 'september': '09', 'november': '11'
			 },
	'2017': {'january': '01', 'february': '02',
			 'october': '10', 'november': '11', 
			 'december': '12'
			 },
	'2018': {'january': '01', 'february': '03', 'march': '03',
			 'april': '04', 'may': '05', 'june': '06'
			 }
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

#########
# Stats #
#########

numMissing = 0
numTotalHouses = 0

for year in years:
	months = monthsDi[year]
	for month in months:
		print(year, month)
		with open('/Users/willlauer/Downloads/data_15_minute/' + month + '_' + year + '.txt') as f:
			
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
			#print('lookingFor', len(lookingFor))


			# houseID : map to rest of line
			di = defaultdict(list)
			for line in lines:
				di[line[0]].append(line)


			counter = 0
			for houseID in tqdm(houseIDs):

				houseDict = {s: (0.0, 0.0) for s in lookingFor}
				
				useArr = np.empty((0, 96))
				carArr = np.empty((0, 96))

				houseData = di[houseID]

				##############################################################################
				# Return the length of the longest consecutive sequence of 0's, and the total 
				# number of 0's
				##############################################################################
				def longest0Subsequence(li):
					cur = 0 # Track the longest subsequence we've seen 
					longest = 0 # Track the next longest 
					totZero = 0
					for i in range(len(li)):
						if i == 0:
							cur += 1 
							totZero += 1
						else:
							cur = max(cur, longest)								
							nextLongest = 0

					return longest, totZero			

				
				
				for i, e in enumerate(houseData):
					date = e[1]
					assert(len(e) == 4)
					if date in lookingFor:
						houseDict[date] = (float(e[2]), float(e[3]))
				#print(len(list(houseDict.items())))
				#################################################
				# Skip all households that do not contain EV data
				#################################################
				#if not containsEV:
				#	continue


				# Load back in and sort in terms of ascending date
				houseData = sorted(list(houseDict.items()), key = lambda x: toSeconds(x[0]))
				#for l in houseData:
				#	print('-', l)
				useVals = [x[0] for x in houseData]

				##################################################
				# Skip all households have too many missing values
				##################################################
				consecZerosThreshold = 10		# 2.5 hours of missing data
				totZeroThreshold = 24			# If a quarter of the day is missing overall
				#consecZeros, totZero = longest0Subsequence(useVals)
				#if consecZeros > consecZerosThreshold or totZero > totZeroThreshold:
				#	continue


				houseData = [x[1] for x in houseData]  # get all times seen for this house


				# Each entry should contain 1 day, with 2 elements
				dayData = [houseData[i:i+96] for i in range(0, len(houseData), 96)]

				zeros = [longest0Subsequence(day) for day in dayData]
				dayData = [dayData[i] for i in range(len(dayData)) 
								if zeros[i][0] <= consecZerosThreshold and zeros[i][1] <= totZeroThreshold]

				for d in dayData:

					arr = np.array(d) 
					useSubarr = np.expand_dims(arr[:,0], 0) # Confused about this ordering
					carSubarr = np.expand_dims(arr[:,1], 0)
					#print('use', useSubarr)
					#print('car', carSubarr)

					if useSubarr.shape[1] != 96 or carSubarr.shape[1] != 96:
						pass  # many entries are 1-length remainders, from wraparound to the next day
					else:
						useArr = np.concatenate((useArr, carSubarr))
						carArr = np.concatenate((carArr, useSubarr))



				np.save('use_' + year + '_' + month + '_' + houseID + '.npy', useArr)
				np.save('car_' + year + '_' + month + '_' + houseID + '.npy', carArr)
				savedUseFiles.append('use_' + year + '_' + month + '_' + houseID + '.npy')
				savedCarFiles.append('car_' + year + '_' + month + '_' + houseID + '.npy')

		#print('NumMissing: ', str(numMissing), 'numTotalHouses', str(numTotalHouses))
			
useFiles = '\n'.join(savedUseFiles)
carFiles = '\n'.join(savedCarFiles)

with open('useFiles.txt', 'w+') as w:
	w.write(useFiles)
with open('carFiles.txt', 'w+') as w:
	w.write(carFiles)