#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 11:12:25 2018

@author: Richi
"""

# NewsScraper ---
import requests
import pandas as pd
from bs4 import BeautifulSoup
import sqlite3
import os


# 1 Scrape from html page ---
page = requests.get("https://########/")
page
page.status_code

soup = BeautifulSoup(page.content, 'html.parser')

p = soup.find_all('h3', class_='title_art')

tag_list = []
for element in p:
    tag = element.get_text() # estrae il testo contenuto nei tag selezionati
    tag_list.append(tag)

#tag_list
scraped_titles = pd.DataFrame(tag_list)
scraped_titles["newspaper_id"] = 1
scraped_titles["newspaper_name"] = "corriere"
scraped_titles.columns = ["title", "newspaper_id", "newspaper_name"]
new_order = [1,2,0]
scraped_titles = scraped_titles[scraped_titles.columns[new_order]]

# 2 Save article titles to SQL DB ---
path = "/Users/Richi/PythonProjects/WebCrawler/NewScraper/" # file path
os.chdir(path)

conn = sqlite3.connect("news_scraped.db") # creates db if not exist
cur = conn.cursor()

# skip if not first time
def create_table_tags():
    cur.execute(
            'CREATE TABLE IF NOT EXISTS day_title(newspaper_id INTEGER, newspaper_name TEXT, title TEXT)')
create_table_tags()

#DB: table: day_title; (newspaper_id integer PRIMARY KEY, title text)

# row insert in db
def insert_scraped(df_scrape):
    
    for index, row in df_scrape.iterrows():
        id_scraped = row['newspaper_id']
        name_scraped = row['newspaper_name']
        title_scraped = row['title']
        
        cur.execute('INSERT INTO day_title (newspaper_id, newspaper_name, title) VALUES (?,?,?)',
                    (id_scraped, name_scraped, title_scraped))
        conn.commit()
        
insert_scraped(scraped_titles) # call to insert
cur.close()
conn.close()



