import selenium
import time
from selenium.webdriver.common.keys import Keys

# For selecting the last downloaded file
import glob 
import os

# For reducing the data resolution
import numpy as np

# For dealing with the 0th column of the localminute field
from datetime import datetime


def main():
    
    def downloadDate(dateFrom, dateTo, dataStreamsToDownload):
        url = "https://www.dataport.cloud/"   
        usr = "oskartriebe@stanford.edu"
        pwd = "GansterWill"

        while True: 
            try:
                # Launch the webdriver
                driver = selenium.webdriver.Chrome()
                
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
                time.sleep(1)
                for checkbox in checkboxes:
                    # Won't work at the moment if we need to scroll to find the 
                    # checkbox element to be clicked
                    if not checkbox.is_selected() \
                    and checkbox.is_displayed() \
                    and checkbox.get_attribute('value') in dataStreamsToDownload:
                        checkbox.click()
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 4/5);")

                selectionLink = driver.find_element_by_id('quick_electric')
                selectionLink.click()
            

                # Slightly longer sleep time here to give the
                # 'Updating data IDs' message enough time to clear
                time.sleep(5)

                exportLink = driver.find_element_by_id('export_data')
                exportLink.click()
                
                # Want to make sure it has enough time to finish downloading the file
                # before exiting and closing the driver
                time.sleep(60)
                driver.close()
                break

            except selenium.common.exceptions.ElementNotVisibleException as e:
                # If we hit an exception, close the driver and try again
                driver.close()

  

    def reduceData(date):
        '''
        Method: reduceData
        Converts our 1-minute resolution data into 15 minute resolution data, 
        joins the most recently downloaded file
        to some aggregate file, then deletes the old file
        '''
        listOfFiles = glob.glob('/Users/willlauer/Downloads/*.csv') 
        mostRecentDownload = max(listOfFiles, key = os.path.getctime)
        
        # This will vary depending on what variables we extract
        headers = ['localminute', 'dataid', 'use']

        def dateTimeParser(x): 
            # Not sure how we want to deal with the date encoding, but since
            # we need it as an int to be in the numpy array, I just sort of encoded
            # it as the concatenation of the individual strings. 
            if isinstance(x, (bytes, bytearray)):
                x = str(x, 'utf-8')
                if x in headers: 
                    return 0
            d, t = x.split(' ')
            year, month, day = d.split('-')
            hour, minute, _ = t.split(':')
            
            return int(year + month + day + str(int(hour) * 60 + int(minute)))


        data = np.genfromtxt(mostRecentDownload, delimiter = ',', 
                    converters = {0: dateTimeParser}, dtype = np.float) 

        data = np.array([[data[i][0], data[i][1], data[i][2]] for i in range(data.shape[0])])
        data = data[1:] # Skip the line with the field names
        # os.system('rm ' + mostRecentDownload)

        # data = np.sort(data, order = ['dataid', 'localminute'])
        data = data[data[:,1].argsort()] # Sorts so all the data ids are together

        interval = 15
        idx = 0
    
        print('pre aggregate', data.shape)
        fullDayAggregate = np.empty((0, 3))
        
        while idx < data.shape[0]:
            window = data[idx:idx+interval,:]
            
            # Since the number of minutes in the day is a multiple of four, so long as we have
            # an equal number for each ID we should be good to go with a simple approach
            # Otherwise may have to add a few extra lines to determine where the house ID changes

            # Sum the energy entries here and add it to the output array
            # le travail te fait libre
            window = np.expand_dims(
                np.array([window[0,0], window[0,1], np.sum(window[:,2])]),
                axis = 0
            )
            fullDayAggregate = np.append(fullDayAggregate, window, axis = 0)
            idx += interval


            '''
            brokeEarly = False
            diffIndex = -1
            if window[0,1] != window[interval-1,1]:
                brokeEarly = True
                for i in range(1, interval):
                    if window[i,1] != window[0,1]:
                        # Then aggregate everything before this and start fresh
                        # at the next index
                        window = window[:i,:]
                        break
            # Sum the entries in the window and scale by the number of entries compared to the
            # full interval size, in the event that we broke early
            '''
        print('post aggregate', fullDayAggregate.shape)

        '''
        We now have the reduced day
        Store as its own data file
        At the end, we'll add them all together
        Modify the date string so it's not interpreted as a filepath because of the backslash
        '''
        date = date.replace('/', '_')
        print('saving ', date, ' with shape ', fullDayAggregate.shape)
        np.save(date, fullDayAggregate)


    def run():
        # daysInMonths = [(1, 31), (2, 30), (3, 31), (4, 30), (5, 31), (6, 30), (7, 31), (8, 30), (9, 31), (10, 31), (11, 30), (12, 31)] 
        daysInMonths = [(1, 31)] 
        yearStr = '2017'
        dataStream = ['use']


        for month, days in daysInMonths:
            monthStr = '0' + str(month) if month < 10 else str(month)
            for day in range(1, days+1):
                dayStr = '0' + str(day) if day < 10 else str(day)
                print('downloading ', monthStr, dayStr)
                date = monthStr + '/' + dayStr + '/' + yearStr
                downloadDate(date, date, dataStream)
                print('download complete')
                reduceData(date)
                print('reduce complete')


    run()



if __name__=='__main__':
    main()
