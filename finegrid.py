"""
Written March 2015
@author: Josh T Fuchs
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from intmodels import models #This interpolates the models to a small grid


'''
:DESCRIPTION: Takes the best Teff and log(g) from the fitting to the coarse grid. Sets up ranges and values to interpolate the grid to smaller Teff and log(g) spacing. Calls intmodels.py that does the actual interpolation.
    
:INPUTS: 
    
bestT: integer, best-fitting Teff from the coarse grid
    
bestg: integer, best-fitting log(g) from the coarse grid. In format: log(g) = 8.0 as bestg = 800
 
'''
script, midt, midg = sys.argv

midt = int(midt)#12500
midg = int(midg)#800

#Choose coarse models to import for interpolation. This is for getting finer log(g)
#For model spacings of 250 K and .25 logg
#testt = [midt-500,midt-250,midt,midt+250,midt+500]
#testg = [midg-50,midg-25,midg,midg+25,midg+50]

#Define coarse model spacings of 100 K and .1 logg
testt, testg = [], []
for n in range(51):
    testt.append(midt-2500+100*n)    
for m in range(11):
    testg.append(midg-50+10*m)
#print testt
#print testg
#exit()
#Set up finer grid for log(g)

#Set final number of models you want
numberg = range(21) 
gridg = np.empty(len(numberg))
for n in numberg:
    gridg[n] =  (midg/100.-0.50 + 0.05*n) #############(midg/100.-0.5 + 0.005*n)   (midg/100.-0.25+0.05*n)

#Begin iterating over different Teffs to get finer log(g)'s
for i in testt:
    print ''
    print 'Now starting with Teffs of ',i
    filenames = ['da' + str(i) + '_' + str(x) + '.dk' for x in testg]
    grid = gridg
    case = 0 # Use 0 for log(g) interp. and 1 for Teff interp. Just a binary switch.
    models(filenames,grid,case,midt,midg)
    print 'Made it back!'


print 'Done with all Teffs.'
    
#Now we want to create our finer grid of Teff. We need to read in our interpolated models in logg
#gridg is our set of logg's
#Set up new grid for new Teffs
numbert = range(501)
gridt = np.empty(len(numbert))
for n in numbert:
    gridt[n] = midt-2500.+10.*n #########midt-250.+10*n

#Begin iterating over different logg's to get finer Teffs
for i in gridg:
    print ''
    print 'Now starting with log(g) of ',i
    intlogg = str(i * 1000.)
    intlogg = intlogg[:-2]
    #filenames = ['da' + str(testt[0]) + '_' + intlogg + '.jf','da' + str(testt[1]) + '_' + intlogg + '.jf','da' + str(testt[2]) + '_' + intlogg + '.jf','da' + str(testt[3]) + '_' + intlogg + '.jf','da' + str(testt[4]) + '_' + intlogg + '.jf']
    filenames = ['da' + str(x) + '_' + intlogg + '.jf' for x in testt]
    grid = gridt
    case = 1 # Use 0 for log(g) interp. and 1 for Teff interp. Just a binary switch.
    models(filenames,grid,case,midt,midg)
    print 'Made it back!'

print 'Done with all the log(g)s.'

'''
#Save file names to interpolated_names.txt

print 'Saving file names to interpolated_names.txt.'
lowt = midt - 1500
lowg = midg*10 - 250
ranget = 10*np.arange(301)#steps of 5 in Teff
rangeg = 50*np.arange(11)#steps of 0.005 in log(g)
f = open('interpolated_names.txt','a')
for y in ranget:
    teffwrite = lowt + y
    for x in rangeg:
        loggwrite = lowg + x
        file =  'da' + str(teffwrite) + '_' + str(loggwrite) + '.jf'
        f.write(file + '\n')
f.close()
'''
print 'File saved.'
print 'The finer grid is complete!'
