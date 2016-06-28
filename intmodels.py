# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:03:22 2015

@author: joshfuchs
"""

#This program interpolates Koester's DA models

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import RectBivariateSpline
import os
import datetime

def models(filenames,grid,case,bestt,bestg):
    print 'Starting to run intmodels.py'
    #Read in model. There are 33 lines of header information before the data starts
    # We do not have to integrate over mu because these files already have
    tuse = bestt-500
    guse = bestg-50
    newpath = '/srv/two/jtfuchs/Interpolated_Models/1000K_1g/bottom' + str(tuse) + '_' + str(guse) 
    if case == 0:
        os.chdir('/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/modelfitting/DA_models/')
    if case == 1:
        #newpath = '/srv/two/jtfuchs/Interpolated_Models/10teff05logg/center' + str(bestt) + '_' + str(bestg) 
        os.chdir(newpath)
        #os.chdir('/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/modelfitting/DA_models/Interpolated_Models/')

    filename1 = filenames[0]
    filename2 = filenames[1]
    filename3 = filenames[2]
    filename4 = filenames[3]
    filename5 = filenames[4]

    if case == 0:
        lambdas1, inten1 = np.genfromtxt(filename1,skip_header=33,unpack=True)
        lambdas2, inten2 = np.genfromtxt(filename2,skip_header=33,unpack=True)
        lambdas3, inten3 = np.genfromtxt(filename3,skip_header=33,unpack=True)
        lambdas4, inten4 = np.genfromtxt(filename4,skip_header=33,unpack=True)
        lambdas5, inten5 = np.genfromtxt(filename5,skip_header=33,unpack=True)
        logg1 = float(filename1[8:11])/100.
        logg2 = float(filename2[8:11])/100.
        logg3 = float(filename3[8:11])/100.
        logg4 = float(filename4[8:11])/100.
        logg5 = float(filename5[8:11])/100.
        teff = str(filename1[2:7])

    if case ==1:
        lambdas1, inten1 = np.genfromtxt(filename1,unpack=True) #These are the 
        lambdas2, inten2 = np.genfromtxt(filename2,unpack=True) #models we have
        lambdas3, inten3 = np.genfromtxt(filename3,unpack=True) #interpolated.
        lambdas4, inten4 = np.genfromtxt(filename4,unpack=True)
        lambdas5, inten5 = np.genfromtxt(filename5,unpack=True)
        teff1 = float(filename1[2:7])
        teff2 = float(filename2[2:7])
        teff3 = float(filename3[2:7])
        teff4 = float(filename4[2:7])             
        teff5 = float(filename5[2:7]) 
        logg = str(filename1[8:12])              

    plt.clf()


#Do a cubic spline interpolation to interpolate the model to even wavelength
#points at 0.1A intervals so the model can be convolved.

#Low and high wavelengths need to be 18A more than desired range
#Set lambda range from ~3650 to 6770
#The Balmer jump in the models makes the spline bulge to its left
#So start to the right of it

    intlambda = np.divide(range(31100),10.) + 3660.0
#print intlambda[1]

#Interpolate the model spectrum at 0.1 A intervals. But interp1d does not have
#a range option, so we will only feed in the portion of the spectrum we want
#to interpolate to speed things up. Again we will go a little beyond the region
#we care about to minimize edge effects of the interpolation. Will use ~3600
#to 6760
    

    if case == 0:
        shortlambdas1 = lambdas1[600:4300]
        shortinten1 = inten1[600:4300]
        shortlambdas2 = lambdas2[600:4300]
        shortinten2 = inten2[600:4300]
        shortlambdas3 = lambdas3[600:4300]
        shortinten3 = inten3[600:4300]
        shortlambdas4 = lambdas4[600:4300]
        shortinten4 = inten4[600:4300]
        shortlambdas5 = lambdas5[600:4300] 
        shortinten5 = inten5[600:4300]

    if case == 1:
        shortlambdas1 = lambdas1
        shortinten1 = inten1
        shortlambdas2 = lambdas2
        shortinten2 = inten2
        shortlambdas3 = lambdas3
        shortinten3 = inten3
        shortlambdas4 = lambdas4
        shortinten4 = inten4
        shortlambdas5 = lambdas5 
        shortinten5 = inten5

    #print 'shortlambdas1 run from ',shortlambdas1[0], '',shortlambdas1[-1]
    #print 'shortlambdas2 run from ',shortlambdas2[0], '',shortlambdas2[-1]
    #print 'shortlambdas3 run from ',shortlambdas3[0], '',shortlambdas3[-1]
    #print 'shortlambdas4 run from ',shortlambdas4[0], '',shortlambdas4[-1]
    #print 'shortlambdas5 run from ',shortlambdas5[0], '',shortlambdas5[-1]

    print 'Starting the interpolation'
    interp = InterpolatedUnivariateSpline(shortlambdas1,shortinten1,k=1)
    intflux = interp(intlambda)
    interp2 = InterpolatedUnivariateSpline(shortlambdas2,shortinten2,k=1)
    intflux2 = interp2(intlambda)
    interp3 = InterpolatedUnivariateSpline(shortlambdas3,shortinten3,k=1)
    intflux3 = interp3(intlambda)
    interp4 = InterpolatedUnivariateSpline(shortlambdas4,shortinten4,k=1)
    intflux4 = interp4(intlambda)
    interp5 = InterpolatedUnivariateSpline(shortlambdas5,shortinten5,k=1)
    intflux5 = interp5(intlambda)
    print 'Done with the interpolation'

    #################################3
    #Now Convolve the models to a FWHM of 4.4
    #print 'Begin convolution. '
    #FWHM = 4.4
    #sig = FWHM / (2. * np.sqrt(2.*np.log(2.)))
    #gx = np.divide(range(360),10.)
    #gauss = (1./(sig * np.sqrt(2. * np.pi))) * np.exp(-(gx-18.)**2./(2.*sig**2))
    #gf1 = np.divide(np.outer(intflux,gauss),10.)
    #gf2 = np.divide(np.outer(intflux2,gauss),10.)
    #gf3 = np.divide(np.outer(intflux3,gauss),10.)
    #gf4 = np.divide(np.outer(intflux4,gauss),10.)
    #gf5 = np.divide(np.outer(intflux5,gauss),10.)

    #length = len(intflux) - 360
    #cflux = range(length)
    #cflux2 = range(length)
    #cflux3 = range(length)
    #cflux4 = range(length)
    #cflux5 = range(length)
    #clambda = intlambda[180:len(intlambda)-180]
    #x  = 0
    #while x < length:
    #    cflux[x] = np.sum(np.diagonal(gf1,x,axis1=1,axis2=0),dtype='d')
    #    cflux2[x] = np.sum(np.diagonal(gf2,x,axis1=1,axis2=0),dtype='d')
    #    cflux3[x] = np.sum(np.diagonal(gf3,x,axis1=1,axis2=0),dtype='d')
    #    cflux4[x] = np.sum(np.diagonal(gf4,x,axis1=1,axis2=0),dtype='d')
    #    cflux5[x] = np.sum(np.diagonal(gf5,x,axis1=1,axis2=0),dtype='d')

     #   x += 1
    
    #intflux = cflux
    #intflux2 = cflux2
    #intflux3 = cflux3
    #intflux4 = cflux4
    #intflux5 = cflux5
    #############################

#plot the interpolated spectra
    #plt.plot(intlambda,intflux,'ro',label='7.5')
    #plt.plot(intlambda,intflux2,'b^',label='7.75')
    #plt.plot(intlambda,intflux3,'m*',label='8.0')
    #plt.plot(intlambda,intflux4,'gs',label='8.25')
    #plt.plot(intlambda,intflux5,'cp',label='8.5')


#Now do the 2D interpolation
    if case == 0:
        xval = np.array([logg1,logg2,logg3,logg4,logg5]) #This is our x
    if case == 1:
        xval = np.array([teff1,teff2,teff3,teff4,teff5]) #This is our x                  
#intlambda is our y
    fluxes = np.array([intflux,intflux2,intflux3,intflux4,intflux5]) #This is our z
#print type(gravs)
#print type(intlambda)

    print 'Starting 2D interpolation'
    out = RectBivariateSpline(xval,intlambda,fluxes,kx=1,ky=1,s=0)
    print 'Done with the 2D interpolation. Starting to read off new values.'
#Iterate over the output to build an array with the new fluxes
#Need to set up an array that is of size grid by intlambda
    intfluxes = []
    #grid = [7.80,7.85,7.90,7.95,8.05,8.10,8.15,8.20]
    #print grid
    for x in grid:
        #print x
        for i in intlambda:
            new = float(out([x],[i]))
            if i == 3660:
                newflux = [new]
            else:
                newflux.append(new)
        intfluxes.append(newflux)
     #   if x == 12250:
     #       plt.plot(intlambda,newflux,'r^',label='Interp')
     #       plt.show()
     #       plt.clf()
     #       plt.plot(lambdas2, inten2,'bs',label='Model')
     #       plt.legend()
     #       plt.show()
    #print np.shape(intfluxes)

    #plt.plot(intlambda,newflux7,'k+',label='Interp - 8.2')
    #plt.plot(intlambda,newflux1,'kx',label='Interp - 7.85')
    #plt.legend()
    #plt.show()

    #Write and save files
    print 'Starting to save files.'
    #os.chdir('/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/modelfitting/DA_models/Interpolated_Models/') #Save all interpolated models here.
    os.chdir(newpath)
    if case == 0:
        newg = np.multiply(grid,1000.)
        n = 0
        for x in newg:
            thisg = str(x)
            thisg = thisg[:-2]
            newfile = 'da' + str(teff) + '_' + thisg + '.jf'
            np.savetxt(newfile,np.transpose([intlambda,intfluxes[n]]))
            #Write out last file saved and time
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
            f = open('lastsaved.txt','a')
            f.write(newfile + ',' + now + '\n')
            f.close()
            n += 1
        
    if case ==1:
        n = 0
        for x in grid:
            thist = str(x)
            thist = thist[:-2]
            newfile = 'da' + thist + '_' + logg + '.jf'  
            np.savetxt(newfile,np.transpose([intlambda,intfluxes[n]])) 
            #Write out last file saved and time
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
            f = open('lastsaved.txt','a')
            f.write(newfile + ',' + now + '\n')
            f.close()
            n += 1          
