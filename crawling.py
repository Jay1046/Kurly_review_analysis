from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.chrome.options import Options
import time
import warnings
import chromedriver_autoinstaller
import re
import json
from selenium.webdriver import ChromeOptions
import pandas as pd
warnings.filterwarnings(action='ignore')
# chromedriver
chrome_Ver = chromedriver_autoinstaller.get_chrome_version()
chromedriver_autoinstaller.install(True)
chromedriver_path = f'./{chrome_Ver.split(".")[0]}/chromedriver.exe'

# options
options = ChromeOptions()
options.add_argument('--headless')
options.add_argument('window-size=1024,25000')


url = "https://www.kurly.com/collections/market-best"
driver = webdriver.Chrome(options=options)
driver.get(url)
driver.implicitly_wait(5)

full_df = pd.DataFrame(columns=['item','review','date'])

try:
    for i in range(1, 96):
        
        
        driver.find_element(By.XPATH, f'//*[@id="container"]/div/div[2]/div[2]/a[{i}]').click()
        time.sleep(2)
        
        # Get item name
        item = driver.find_element(By.CLASS_NAME, 'css-1bhm8h2.ezpe9l12').text       
        print(item)
        
        
        item_info = {}
        item_list = []
        review_list = []        
        date_list = []
        
        for _ in range(10):
            
            time.sleep(0.5)
            # Get reviews
            reviews = driver.find_elements(By.CLASS_NAME, 'css-i69j0n.e36z05c5')
            
            for review in reviews:
                review_list.append(review.text)
                driver.implicitly_wait(2)
                item_list.append(item)

            driver.implicitly_wait(5)
            
            # Get dates
            dates = driver.find_elements(By.CLASS_NAME, 'css-14kcwq8.e1bup33p0')
            
            for _date in dates:
                date_list.append(_date.text)
                driver.implicitly_wait(2)
           
            
            driver.implicitly_wait(5)
            try:
                driver.find_element(By.XPATH, '//*[@id="review"]/section/div[2]/div[15]/button[2]').click()
                time.sleep(1)
            except:
                continue
        
        item_info['item'] = item_list
        item_info['review'] = review_list
        item_info['date'] = date_list
        
        temp_df = pd.DataFrame(item_info)
        full_df = full_df.append(temp_df, ignore_index=True)
        
        driver.find_element(By.XPATH, '//*[@id="header"]/div/ul/li[2]/span').click()
     
        
except Exception as e:
    print("에러가 발생했습니다!!! : ", e)
    
finally:
    full_df.to_csv('./items.csv')
