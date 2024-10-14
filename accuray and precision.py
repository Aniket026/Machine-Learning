# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:07:12 2024

@author: aniket

"""

import numpy as np
import pandas as pd

true_value=50
'''
Accurate but not Precies:these value are centered around the truevalue(50)

but there is some spread (random variance).
this simulation measurements that are accurate (close to the value) but not precies out
'''

accurate_measurements = np.random.normal(loc=true_value, scale=5, size=10)
precise_measurements = np.random.normal(loc=true_value, scale=5, size=10)

'''precise but not Accurate :
    this value is tightly clustered around 60,
    not near the true value is higher the accuracy
    the acuuracy formula is 1-(diffrences/true_value)
    
    this give a number between 0 to 1
    '''
def calculate_accuracy(measurements,true_value):
    average_measurement=np.mean(measurements)
    accuarcy=1-(average_measurement-true_value)/true_value
    return accuarcy


#function to calculate precision

'''
precision is determine by std of measurement .std measure
how spread out measurements are. if the std is small *measurement are close together)
    precision will be high .we use 1/std_dev to represent precision 
    so samller spread gives a higher value for precision
    '''
    
def calculate_precision(measurements):
        #precision : how close the measurements are to each other
        #(low std means high precision)
        precision=1/np.std(measurements)
        return precision
    
    
''' accurate measurment :we calculate the accuracy and precision of measurement that are close 
to true value but spread out
'''

accuracy_of_accurate=calculate_accuracy(accurate_measurements,true_value)
precision_of_accurate=calculate_precision(accurate_measurements)


accuracy_of_precise=calculate_accuracy(accurate_measurements,true_value)
precision_of_precise=calculate_precision(accurate_measurements)


print("Accurate but not precise measurement : ")
print(f'measurement:{accurate_measurements}')
print(f'Accuracy:{accuracy_of_accurate:2f}')
print(f'Precision:{precision_of_accurate:2f}')


print(" precise  but not Accuracy  measurement : ")
print(f'measurement:{precise_measurements}')
print(f'Accuracy:{accuracy_of_precise:2f}')
print(f'Precision:{precision_of_precise:2f}')


