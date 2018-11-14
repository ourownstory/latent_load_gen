import requests
import mechanicalsoup 
from urllib import request

"""
<form name="interactive_data_form" id="interactive_data_form" action="/data/interactive" method="post" _lpchecked=1>

(?) input id = hidden download?

To select data:
<div class="data_steps">
    <div class="data_steps_left">
        "From:"
            <input class="update_calc update_step1and2 hasDatepicker" placeholder="MM/DD/YYYY" id="date_from" name="date_from" type="text">
    </div>
    <div class="data_steps_right">
        "To:"
            <input class="update_calc update_step1and2 hasDatepicker" placeholder="MM/DD/YYYY" id="date_to" name="date_to" type="text">
    </div>

To select resolution:
<select class="dbtable" name="table">
    <option value="eg1m">Electric Data - 1 Minute Resolution</option>


To choose all house IDs with electric data:
<table class="table table-striped">
<li class="quick_select ms-hover" style="cursor: pointer;" 
    id="quick_electric">DataIDs with Electric Data</li>


<input id="export_data" value="Export Selected Data Type" type="button">
"""

def main():
    url = "https://www.dataport.cloud/"   
    # url = 'http://www.theonion.com/'
    username = "oskartriebe@stanford.edu"
    password = "GansterWill"

    
    browser = mechanicalsoup.StatefulBrowser()
    cert = '/usr/local/lib/python3.7/site-packages/certifi/cacert.pem'
    res = browser.open(url, verify=False)
    print(res)

    browser.follow_link("login")
    browser.select_form()
    browser['user'] = username
    browser['pass'] = password
    browser['login_data_license'] = True
    browser.launch_browser()

    resp = browser.submit_selected()

    print('reached')
    browser.launch_browser()

    #/usr/local/lib/python3.7/site-packages/certifi/        
    # div.sign-in-container
    # div#login_div
    # form action = '/'
    
    # input#user_login1.user_forms
    # input#user_pass.user_forms

    #browser.select_form('form[action="/post"]')
    #browser.get_current_form().print_summary()
    """ 
    form = {
            "date_from": "01/01/2017",
            "date_to": "01/02/2017",
            " 
            # Pick the checkbox by value. Should get the checkbox
            # corresponding to the datafield we're interested in, and set it to "on"
            webpage.find(string=dataField): "on" 

           }

    r = requests.post("https://dataport.cloud/data/interactive"
    """ 
if __name__=="__main__":
    main()
