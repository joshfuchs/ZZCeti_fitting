"""
Written March 2015
@author: Josh T Fuchs
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from intmodels import models #This interpolates the models to a small grid

def makefinegrid(bestT,bestg):

    '''
    :DESCRIPTION: Takes the best Teff and log(g) from the fitting to the coarse grid. Sets up ranges and values to interpolate the grid to smaller Teff and log(g) spacing. Calls intmodels.py that does the actual interpolation.
    
    :INPUTS: 
    
       bestT: integer, best-fitting Teff from the coarse grid
    
       bestg: integer, best-fitting log(g) from the coarse grid. In format: log(g) = 8.0 as bestg = 800
 
    '''

    firstt = bestT#12500
    firstg = bestg#800

#Choose coarse models to import for interpolation. This is for getting finer log(g)
    testt = [firstt-500,firstt-250,firstt,firstt+250,firstt+500]
    testg = [firstg-50,firstg-25,firstg,firstg+25,firstg+50]
    
#Set up finer grid for log(g)
#grid should go from best log(g) -.5 to best log(g) +.5 in steps of 0.05
# firstg/100.-0.5 + 0.05*n
    numberg = range(21) #Normally want this to be 201##############
    #gridg = np.array([7.75,7.80,7.85,7.90,7.95,8.00,8.05,8.10,8.15,8.20,8.25])
    #gridg = np.array([7.84,7.85,7.86,7.87,7.88,7.89,7.90,7.91,7.92])
    gridg = np.empty(len(numberg))
    for n in numberg:
        gridg[n] =  (firstg/100.-0.5 + 0.05*n) #############(firstg/100.-0.5 + 0.005*n)   (firstg/100.-0.25+0.05*n)

#Begin iterating over different Teffs to get finer log(g)'s
    for i in testt:
        print ''
        print 'Now starting with Teffs of ',i
        filenames = ['da' + str(i) + '_' + str(testg[0]) + '.dk','da' + str(i) + '_' + str(testg[1]) + '.dk','da' + str(i) + '_' + str(testg[2]) + '.dk','da' + str(i) + '_' + str(testg[3]) + '.dk','da' + str(i) + '_' + str(testg[4]) + '.dk']
        grid = gridg
        case = 0 # Use 0 for log(g) interp. and 1 for Teff interp. Just a binary switch.
        models(filenames,grid,case,firstt,firstg)
        print 'Made it back!'


    print 'Done with all Teffs.'
    
#Now we want to create our finer grid of Teff. We need to read in our interpolated models in logg
#gridg is our set of logg's
#Set up new grid for new Teffs
    numbert = range(101)
    #gridt = np.array([11750.,11800.,11850.,11900.,11950.,12000.,12050.,12100.,12150.,12200.,12250.])
    #gridt = np.array([12250.,12300.,12350.,12400.,12450.,12500.,12550.,12600.,12650.,12700.,12750.])
    #gridt = np.array([12500.,12510.,12520.,12530.,12540.,12550.,12560.,12570.,12580.,12590.,12600.,12610.,12620.,12630.,12640.,12650.,12660.,12670.,12680.,12690.,12700.])
    gridt = np.empty(len(numbert))
    for n in numbert:
        gridt[n] = firstt-500.+10.*n #########firstt-250.+10*n

#Begin iterating over different logg's to get finer Teffs
    for i in gridg:
        print ''
        print 'Now starting with log(g) of ',i
        intlogg = str(i * 1000.)
        intlogg = intlogg[:-2]
        filenames = ['da' + str(testt[0]) + '_' + intlogg + '.jf','da' + str(testt[1]) + '_' + intlogg + '.jf','da' + str(testt[2]) + '_' + intlogg + '.jf','da' + str(testt[3]) + '_' + intlogg + '.jf','da' + str(testt[4]) + '_' + intlogg + '.jf']
    
        grid = gridt
        case = 1 # Use 0 for log(g) interp. and 1 for Teff interp. Just a binary switch.
        models(filenames,grid,case,firstt,firstg)
        print 'Made it back!'

    print 'Done with all the log(g)s.'
    #Save file names to interpolated_names.txt
    
    print 'Saving file names to interpolated_names.txt.'
    lowt = firstt - 500
    lowg = firstg*10 - 500
    ranget = 10*np.arange(101)#steps of 5 in Teff
    rangeg = 50*np.arange(21)#steps of 0.005 in log(g)
    f = open('interpolated_names.txt','a')
    for y in ranget:
        teffwrite = lowt + y
        for x in rangeg:
            loggwrite = lowg + x
            file =  'da' + str(teffwrite) + '_' + str(loggwrite) + '.jf'
            f.write(file + '\n')
    f.close()
    
    print 'File saved.'
    print 'The finer grid is complete!'
