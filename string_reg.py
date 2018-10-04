# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:53:43 2018

@author: riccardo.lavelli
"""

# regex example:

# match the first three characters in string (no white spaces)
import re

#1 first three on a string with 3 or more characters:

s = "A string example"
re.findall("\w{3}", s) #['str', 'ing', 'exa', 'mpl'] # I'd take the first [0]

s = "Astring"
re.findall("\w{3}", s) #['Ast', 'rin'] # I'd take the first [0]

s = "A S t ring example"
re.findall("\w{3}", s) #['rin', 'exa', 'mpl'] # I'd take the first [0]

s = "1 A string"
re.findall("\w{3}", s) #['str', 'ing'] # I'd take the first [0]

# ---
#2 first any three characters, just no spaces
s = "A string example"
re.findall("\w{3}", s.replace(" ", "")) #['Ast', 'rin', 'gex', 'amp'] # I'd take the first [0]

s = "A S t ring example"
re.findall("\w{3}", s.replace(" ", "")) #['ASt', 'rin', 'gex', 'amp'] # I'd take the first [0]

s = "1 A string"
re.findall("\w{3}", s.replace(" ", "")) #['1As', 'tri'] # I'd take the first [0]