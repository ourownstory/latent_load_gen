import selenium
import time
from selenium.webdriver.common.keys import Keys

import threading

from collections import defaultdict

# For selecting the last downloaded file
import glob 
import os

# For reducing the data resolution
import numpy as np

# For dealing with the 0th column of the localminute field
from datetime import datetime

from tqdm import tqdm

import json

def main():
   
    # Used to mark missing entries 
    EMPTY = -1.0

   
    # Download a single day and variable
    def downloadDate(dateFrom, dateTo, dataStreamsToDownload):
        url = "https://www.dataport.cloud/"   
        usr = "oskartriebe@stanford.edu"
        pwd = "GansterWill"
        print('downloadData', dateFrom, dateTo, dataStreamsToDownload)
        while True: 
            try:
                # Launch the webdriver
                options = selenium.webdriver.chrome.options.Options()
                options.add_argument("headless")
                driver = selenium.webdriver.Chrome(options = options)
                
                # Navigate to dataport.cloud
                driver.get(url)

                # Open the login window
                driver.find_element_by_css_selector('img#dropdown.downarrow').click()
                driver.find_element_by_xpath('//a[@href="/#login"]').click()


                # Enter username and password and agree to terms and conditions
                username = driver.find_element_by_css_selector('input#user_login1.user_forms')
                password = driver.find_element_by_css_selector('input#user_pass.user_forms')
                ldl = driver.find_element_by_name('login_data_license')
                submit = driver.find_element_by_name('submit')
                
                time.sleep(2)
                
                username.send_keys(usr)
                password.send_keys(pwd)
                ldl.click()
                submit.click()

        
                time.sleep(2)
                driver.get('https://www.dataport.cloud/data/interactive')
               
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
                time.sleep(2)

                dfrom = driver.find_element_by_name('date_from')
                dto = driver.find_element_by_name('date_to')

                time.sleep(2)


                dfrom.send_keys(dateFrom)
                dto.send_keys(dateTo)

                driver.find_element_by_xpath("//select[@name='table']/option[text()='Electric Data - 1 Minute Resolution']").click()
                time.sleep(2)
               

                # Should correspond to the checkbox outputs 
                checkboxes = driver.find_elements_by_xpath("//input[@name='tbl_group[]']")
                time.sleep(2)
                for checkbox in checkboxes:
                    # Won't work at the moment if we need to scroll to find the 
                    # checkbox element to be clicked
                    if not checkbox.is_selected() \
                    and checkbox.is_displayed() \
                    and checkbox.get_attribute('value') in dataStreamsToDownload:
                        checkbox.click()
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 4/5);")

                time.sleep(2)
                selectionLink = driver.find_element_by_id('quick_electric')
                selectionLink.click()
            

                # Slightly longer sleep time here to give the
                # 'Updating data IDs' message enough time to clear
                time.sleep(5)

                exportLink = driver.find_element_by_id('export_data')
                exportLink.click()
                
                # Want to make sure it has enough time to finish downloading the file
                # before exiting and closing the driver
                time.sleep(30)
                driver.close()
                break

            #except selenium.common.exceptions.ElementNotVisibleException as e:
            except Exception as e:
                # If we hit an exception, close the driver and try again
                print('exception')
                driver.close()


    def reduceData(date, var, fname, variableMap):
        '''
        Method: reduceData
        Converts our 1-minute resolution data into 15 minute resolution data, 
        joins the most recently downloaded file
        to some aggregate file, then deletes the old file
        '''
    
        print('reduceData')
        # Reduce after each 4 days are done
 
         #print('reduce data for', var)
        month, day, year = date.split('/')
        listOfFiles = glob.glob('/Users/willlauer/Downloads/*.csv') 
        mostRecentDownload = max(listOfFiles, key = os.path.getctime)

        # Sort in terms of decending time
        #mostRecentNDownloads = sorted(listOfFiles, key = -os.path.getctime)[:len(varsToDownload)]
        
        # This will vary depending on what variables we extract
        headers = ['localminute', 'dataid', var]

        def dateTimeParser(x): 
            # Since we're downloading a single day at a time, we can just encode this
            # as the number of minutes in a day
            if isinstance(x, (bytes, bytearray)):
                x = str(x, 'utf-8')
                if x in headers: 
                    # We need to know which variable this is that we're downloading, 
                    # since order is no longer guaranteed 
                    #return headers.index(x)
                    return 0
            d, t = x.split(' ')
            year, month, day = d.split('-')
            hour, minute, _ = t.split(':')
            
            return int(hour) * 60 + int(minute)

        # Read into np array from the downloaded csv format
        data = np.genfromtxt(mostRecentDownload, delimiter = ',', 
                    converters = {0: dateTimeParser}, dtype = np.float) 
        # print('loaded data as npy')
        data = np.array([[data[i][0], data[i][1], data[i][2]] for i in range(data.shape[0])])

        # var = headers[data[0,2]] # Extract the string name of our variable
        data = data[1:] # Skip the line with the field names
    
        
        # Remove the downloaded csv's
        #for x in mostRecentNDownloads:
        #   os.system('rm ' + x)
        
        interval = 15
    
        # fullDayAggregate = np.empty((0, 3))
     
        uniqueHouseholds = np.unique(data[:,1])

        # Get the data for a household
        # Locate missing time values
        for household in tqdm(uniqueHouseholds):
           
            if household not in variableMap:
                # Then we haven't created a directory for this household yet
                os.system('mkdir ' + str(household))
                variableMap[household] = defaultdict(list)

            if date not in variableMap[household]:
                # Then we need to create a new file for this date
                newEmpty = np.empty((97, 0))  
                np.save(str(household) + '/' + fname + '.npy', newEmpty)

            # Select the data corresponding to this household
            # Assume data within household sorted by time
            householdData = data[np.where(data[:,1] == household)]
          
            # Fill in the values that we do have
            # Empty ones are marked with EMPTY
            fullTimeCol = np.full((1441), EMPTY)  # The correct number of minutes
            fullTimeCol[householdData[:,0].astype(np.int)] = householdData[:,2] # Set the minute entries

            nonEmpty = np.where(fullTimeCol != EMPTY)
            empty = np.where(fullTimeCol == EMPTY)
            mean = np.mean(fullTimeCol[nonEmpty])
            fullTimeCol[empty] = mean  # Fill in all empty values with the mean of the nonempty
          
            # Sum each 15 values
            # Reduced is a column vector
            reduced = np.expand_dims(np.array([np.sum(fullTimeCol[i:i+interval]) for i in range(0, 1441, interval)]), axis = 1)

            # Load the file corresponding to this household
            a = np.load(str(household) + '/' + fname + '.npy')
            a = np.concatenate((a, reduced), axis = 1) # Add as a new column corresponding to this variable
            np.save(str(household) + '/' + fname + '.npy', a)

            # Now add this variable to the list for this household
            n = len(variableMap[household])

            # Update the variable list for this household and date
            variableMap[household][date].append((n, var)) 
        
        # Wipe all the downloaded csvs from this round
        os.system('rm /Users/willlauer/Downloads/*.csv')        


        
    # Get missing entries
    # Mean and std dev over different values - over variables

    def columnInit():
        '''
        Return a column vector that has num rows equal to the number of minutes in a day
        '''
        numMinutes = 24 * 60
        arr = np.full((numMinutes), EMPTY)
        return arr



    def launch():
        
        variableMap = {} # Map from household to tuples of (variable, column), so we know what is stored where

        daysInMonths = [(1, 31), (2, 28), (3, 31), (4, 30), (5, 31), (6, 30), (7, 31), (8, 30), (9, 31), (10, 31), (11, 30), (12, 31)] 
        priorDays = [sum([x[1] for x in daysInMonths[:i]]) for i in range(len(daysInMonths))] 
        #yearStr = ['2015', '2016', '2017']
        years = ['2015', '2016', '2017']
       
        #dataStreams = ['use', 'air1', 'air2', 'air3', 'bathroom1', 'bathroom2', 'bedroom1', \
        #            'bedroom2', 'bedroom3', 'bedroom4', 'bedroom5', 'car1', 'clotheswasher1', 'dishwasher1']
        #dataStreams = ['use', 'air1', 'air2', 'air3', 'bathroom1', 'bathroom2', 'bed]
        dataStreams = ['use']
       
        for yearStr in years:
            for month, days in daysInMonths:
                # Pass the date in as month/day/year so that we can use it in selecting
                # the date to download from dataport. After that, we'll go to the year_dayIndex form
                monthStr = '0' + str(month) if month < 10 else str(month)
                for day in range(1, days+1):
                    for dataStream in dataStreams:
                        # For each day in this month
                            
                        # Get the date 
                        dayStr = '0' + str(day) if day < 10 else str(day)
                        date = monthStr + '/' + dayStr + '/' + yearStr
                        dayIndex = priorDays[month-1] + day - 1 # Decrement month by 1 cause it's 1-indexed originally

                        # Download that date. Handles all variables for this particular date
                        # parallelDownload(date, date, dataStreams)
                        downloadDate(date, date, [dataStream])
                      
                        # Now only reducing the datastreams should need to be done sequentially
                        for dataStream in dataStreams:
                            reduceData(date, dataStream, yearStr + '_' + str(dayIndex), variableMap)

    
    # Launch everything
    launch() 

    # Store the dict of our households and dates
    with open('variableMap.json', 'w+') as f:
        json.dump(variableMap, f)



if __name__=='__main__':
    main()

