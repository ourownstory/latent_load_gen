import numpy as np
import os

# months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
# years = ['2015', '2016', '2017']

months = ['january', 'february', 'march', 'may']  # missing april
years = ['2015']
repo = '/Users/willlauer/latent_load_gen/data2'


daysInMonths = [(1, 31), (2, 28), (3, 31), (4, 30), (5, 31), (6, 30), (7, 31), (8, 30), (9, 31), (10, 31), (11, 30), (12, 31)]
priorDays = [sum([x[1] for x in daysInMonths[:i]]) for i in range(len(daysInMonths))]


def dayIndex(year, m, d):
    if year == 2016 and m > 2:
        return d + priorDays[m-1] + 1 # Because of leap year
    else:
        return d + priorDays[m-1] 


EMPTY = -1.0

for year in years:
    for month in months: # These are the months and years we are parsing
        with open('/Users/willlauer/Documents/' + month + '_' + year + '.txt') as f:
            lines = [[y.replace('\"', '') for y in x.split()] for x in f.readlines()]
            for l in lines[:10]:
                print(l)
            houseIDs = list(set([x[0] for x in lines]))
    
            # For each iteration, save |month| csvs for this house
            for houseID in houseIDs:

                houseData = [x for x in lines if x[0] == houseID]

                uniqueDates = list(set([x[1] for x in houseData]))

                # Save 1 csv
                for date in uniqueDates:
                    dayData = [x for x in houseData if x[1] == date]
                    dayArr = np.empty((0, 2))
                    for entry in dayData:

                        if len(entry) == 4:
                            # No ev
                            dayArr = np.concatenate((dayArr, np.array([[float(entry[3]), float(EMPTY)]])))
                        elif len(entry) == 5:
                            # 1 ev 
                            dayArr = np.concatenate((dayArr, np.array([[float(entry[3]), float(entry[4])]])))
                        else:
                            pass
                    
                    if not os.path.isdir(repo + '/' + houseID):
                        os.system('mkdir ' + repo + '/' + houseID)
                    
                    y, m, d = date.split('-')
                    m, d, y = int(m), int(d), int(y)
                    np.save(repo + '/' + houseID + '/' + str(year) + '_' + str(dayIndex(y, m, d)) + '.npy', dayArr)



