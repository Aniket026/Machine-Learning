# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:02:12 2024

@author: anike
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 08:48:59 2024

@author: aniket

"""


from bs4 import BeautifulSoup

soup=BeautifulSoup(open("C:/9-web_crapping/sample_doc.html"),'html.parser')

print(soup)



soup.text

soup.find('address')

soup.find_all('address')

soup.find_all('q')

soup.find_all('b')
soup.find_all('b')

table=soup.find('table')
table

for row in table.find_all('tr'):
    columns=row.find_all('td')
    print(columns)
    

#it will show all the rows except first row
#now we wnt to display M.tech which is located in third row
#i need to give [3][2]
    table.find_all('tr')[3].find_all('td')[2]
    
    
from bs4 import BeautifulSoup as bs
import requests

link="https://sanjivanicoe.org.in/index.php/contact"
page=requests.get(link)
page
#resopnce[200]

page.content
#all html code in crowdy text

soup=bs(page.content,'html.parser')
soup

print(soup.prettify())
# the text is neat and clean

list=(soup.children)
#findall content using tab

soup.find_all('p')
#suppose yoe want to extract contents from first row















