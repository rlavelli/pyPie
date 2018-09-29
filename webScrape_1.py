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
# tag_df.head