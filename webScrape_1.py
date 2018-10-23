#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 17:15:14 2018

@author: Richi
"""

# import serve per caricare la pagina web
import requests
page = requests.get("https://stackoverflow.com/")
page
page.status_code
# status code con 200 generalmente significa ok
#page.content mostra il contenuto dell'intera pagina

# import BS e creo un'istanza per la classe BS
from bs4 import BeautifulSoup
soup = BeautifulSoup(page.content, 'html.parser')

# cerchiamo tutti gli elementi "a" con una specifica classe "post-tag"
# questa è facile da trovare usando ispeziona html

p = soup.find_all('a', class_='post-tag')

tag_list = []
for element in p:
    tag = element.get_text() # estrae il testo contenuto nei tag selezionati
    tag_list.append(tag)

# count single tags
tag_dict = {}
for row in tag_list:
    if (row in tag_dict):
        tag_dict[row] += 1
    else:
        tag_dict[row] = 1
tag_dict

# trasform in DataFrame
import pandas as pd
tag_df = pd.DataFrame.from_dict(tag_dict, orient = "index")
tag_df.reset_index(level=0, inplace=True)
tag_df.columns = ["tag", "count"]

df_total = pd.DataFrame(columns=["tag", "count"])

df_total = df_total.append(tag_df)


# db save
import sqlite3
import os

path = "/Users/Richi/PythonProjects/sql_tutorial/"
os.chdir(path)

conn = sqlite3.connect("stack_scrape.db") # creates db if not exist
cur = conn.cursor()

def create_table_tags():
    cur.execute('CREATE TABLE IF NOT EXISTS tags(tag_name TEXT, count INTEGER)')
    
create_table_tags()

def insert_scraped(df_scrape):
    
    for index, row in df_scrape.iterrows():
        tag_scraped = row['tag']
        count_scraped = row['count']
        
        cur.execute('INSERT INTO tags (tag_name, count) VALUES (?,?)',
                    (tag_scraped, count_scraped))
        conn.commit()
        
insert_scraped(tag_df)
cur.close()
conn.close()

# saving output
fname = "~/PythonProjects/WebCrawler/stack_tags.csv"

df_update = pd.read_csv(fname, index_col = False)

df_update = df_update.append(df_total) # update

df_update.to_csv("~/PythonProjects/WebCrawler/stack_tags.csv", index = False)

# check file


# tag_df.head

# --- schedule scraping

import requests
from bs4 import BeautifulSoup
import pandas as pd

import schedule
import time

def job():
    print("I'm working...")
    page = requests.get("https://stackoverflow.com/")
    soup = BeautifulSoup(page.content, 'html.parser')
    p = soup.find_all('a', class_='post-tag')
    tag_list = []
    for element in p:
        tag = element.get_text() # estrae il testo contenuto nei tag selezionati
        tag_list.append(tag)
        # count single tags
    tag_dict = {}
    for row in tag_list:
        if (row in tag_dict):
            tag_dict[row] += 1
        else:
            tag_dict[row] = 1
    tag_df = pd.DataFrame.from_dict(tag_dict, orient = "index")
    tag_df.reset_index(level=0, inplace=True)
    tag_df.columns = ["tag", "count"]
    # append to total
    df_total = pd.DataFrame(columns=["tag", "count"])
    df_total = df_total.append(tag_df)
    # saving output
    fname = "~/PythonProjects/WebCrawler/stack_tags.csv"
    df_update = pd.read_csv(fname, index_col = False)
    df_update = df_update.append(df_total) # update
    df_update.to_csv("~/PythonProjects/WebCrawler/stack_tags.csv", index = False)
    
schedule.every(1).minute.do(job)


while 1:
    schedule.run_pending()
    time.sleep(1)
