# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 08:19:45 2024

@author: anike
"""

from bs4 import BeautifulSoup  as bs

import requests
link="https://m.imdb.com/title/tt0068646/reviews?ref_=tt_urv"
page=requests.get(link)
page
page.content
soup=bs(page.content,'html.parser')
print(soup.prettify())

##########################################################

title=soup.find_all('a',class_='title')
title
review_title=[]

for i in range(0,len(title)):
    review_title.append(title[i].get_text())
review_title
review_title[:]=[title.strip('\n') for title in review_title]
review_title
len(review_title)

###########################################################
#scrap rating

rating=soup.find_all('snap',class_="point-scale")
rating
rate=[]

for i in range(0,len(rating)):
    rate.append(rating[i].get_text())
rate

rate[:]=[r.strip('/') for r in rate]
rate
print(len(rate))
rate.append('')
rate.append('')
print(len(rate))

######################################################

#review body


review=soup.find_all('div',class_='text')
review
review_body=[]
for i in range(0,len(review)):
    review_body.append(review[i].get_text())
review_body

len(review_body)


############################################
import pandas as pd 
df=pd.DataFrame()
df["Review_Title"]=review_title
df['Rate']=rate
df["Review"]=review_body
df

df.to_csv("C:\9-web_crapping/godfather.csv")

################################################

import pandas as pd
from textblob import TextBlob
sent="This is very excellent garden"
pol=TextBlob(sent).sentiment.polarity
pol
df=pd.read_csv("C:\9-web_crapping\godfather.csv")
df.head()

df['polarity']=df["Review"].apply(lambda x: TextBlob(str(x)).sentences)

df['polarity']
