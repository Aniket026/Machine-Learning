# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 07:49:29 2024

@author: aniket 

"""
##

a=input("Enter the value")
print(len(a))
print((a))

print("Aniket"[3])
=> k

print(23_45_880)
=> 2345880

-------------------------------------------------------
a="aniket"
print(type(a))

=><class 'str'>

-------------------------------------------------------


a="Aniket"
b=78
b=str(b)
print(a+b)

=>Aniket78

-------------------------------------------------------
#round 

print(round(45.980,2))
=>45.98
-------------------------------------------------------

print(9/2)
=>4.5

print(9//2)
=>4

-------------------------------------------------------

a=10
print(f"my age is {a}")

=>my age is 10

-------------------------------------------------------

#calculate week reaming in life

age=int(input("Enter your age"))
years=90-age
week=52*years
print(week)

=>3588

-------------------------------------------------------


a=2
b=8
if(a>3 and b<2):
    print("true")
else:
    print("false")
    
    
-----------------------------------------------------------
#Randomization

import random

print(random.randint(1,10))

-----------------------------------------------------------

#list

list1=["Aniket","rohit","ram"]
print(list1[1])


list1.append("Yashraj")
print(list1)
=>['Aniket', 'rohit', 'ram', 'Yashraj', 'Yashraj']


items=["rajiv","pawan"]
list1.extend(items)
print(list1)

=>['Aniket','rohit', 'ram', 'Yashraj', 'Yashraj', 'rajiv', 'pawan']

print(list1[-1])
=>pawan


list1.pop(1)
#it pop the element and return it
=>'rohit'


list2=[2,3,4,5,32,5,7,8]

list2.reverse()
print(list2)
=>[8, 7, 5, 32, 5, 4, 3, 2]




x=slice(1,4)
print(list2[x])

=>[7, 5, 32]

list2.insert(0,"Anni")
print(list2)


#nested list

a=[1,23,4]
b=[5,6,7]
c=[a,b]
print(c)

=>[[1, 23, 4], [5, 6, 7]]

print(c[1][1])
=>6

c[1][2]="hii"  #update
print(c)

for i,j,k in c:
    print(i,j,k)
    
=>1 23 4
  5 6 hii
---------------------------------------------------
q=1
b=5
for number in range(b):
    print(number)

=>0
1
2
3
4
    
for number in range(q,b):
    print(number)
    
=>
1
2
3
4

for number in range(q,b,2):
    print(number)
    
---------------------------------------------------

def myfun(name):
    print(f"my name is {name}")
myfun("aniket")

import math

#math.ceil()

a=1.01
print(math.ceil(a))

=>2

---------------------------------------------


x=int(input("Enter the number "))

def check(x):
    for i in range(2,x):
        if(x%i==0):
            return "not prime"
            break
        else:
            return "prime"
        break
check(17)

---------------------------------------------
#dictionary

dict1={
       "Name":"Aniket"
       }
print(dict1)

dict1["Age"]=21

print(dict1)

for i,j in dict1.items():
    print(i,j)
    
    
st1="my name is aniket"

print(st1.title())


from itertools import *

print(list(repeat("python",5)))


a=[2,3,4]
b=[5,6,7,8]
h=[45,8,9]

for i in zip_longest(a,b,h):
    print(i)
    





import pandas as pd
import numpy as np
dic1={"name":"aniket",
      "age":23}
list1=[1,2,3,"ani"]
df=pd.Series(list1)
df=df.astype(int)
df=pd.to_numeric(df,errors="coerce")
print(df.dtypes)
print(df)


np.arange(0, 10, 2)  # array from 0 to 10, step size of 2

arr=np.array([1,2,3,4,5,3])
arr.reshape(2, 3)  # reshape 1D array into 2x2 matrix



import re 

pattren=r'[a-zA-Z0-9]+@[a-zA-Z.]+\.[a-zA-Z]{2,}'
ttern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
text="my email is aniket07@gmail.com.com"


find=re.findall(pattren,text)
print(find)
