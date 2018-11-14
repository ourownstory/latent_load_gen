import selenium
import time
from selenium.webdriver.common.keys import Keys


def main():
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

    
            print('aayy')
            time.sleep(2)
            driver.get('https://www.dataport.cloud/data/interactive')
           
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
            time.sleep(2)

            dateFrom = '01/01/2017'
            dateTo = '01/02/2017'
            dfrom = driver.find_element_by_name('date_from')
            dto = driver.find_element_by_name('date_to')

            time.sleep(2)


            dfrom.send_keys(dateFrom)
            dto.send_keys(dateTo)

            print('yaa')
            driver.find_element_by_xpath("//select[@name='table']/option[text()='Electric Data - 1 Minute Resolution']").click()
            time.sleep(2)
            print('past')
           

            dataStreamsToDownload = ['use']
            # Should correspond to the checkbox outputs 
            checkboxes = driver.find_elements_by_xpath("//input[@name='tbl_group[]']")
            print(len(checkboxes))
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
        
            time.sleep(1)

            exportLink = driver.find_element_by_id('export_data')
            exportLink.click()
            time.sleep(90)
            print('finished')


            break

        except selenium.common.exceptions.ElementNotVisibleException as e:
            print(str(e) + '\n\n\n')
            driver.close()


    time.sleep(5)


if __name__=='__main__':
    main()
