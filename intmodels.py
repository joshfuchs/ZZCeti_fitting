# -*- coding: utf-8 -*-
"""
Created March 2015 by JT Fuchs, UNC.

This program interpolates grids of models. It is called by finegrid.py, which contains all the necessary options. Unless you want to change which wavelengths to keep, you shouldn't need to change things here. But keep in mind that interpolation is tricky, you should look at the results carefully.
"""


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
    tuse = bestt-2500
    guse = bestg-25
    #newpath = '/srv/two/jtfuchs/Interpolated_Models/Koester_ML2alpha08/bottom' + str(tuse) + '_' + str(guse)
    newpath = '/srv/two/jtfuchs/Interpolated_Models/Koester_ML2alpha08/bottom10000_700'
    if case == 0:
        os.chdir('/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/modelfitting/Koester_08/')
    if case == 1:
        #newpath = '/srv/two/jtfuchs/Interpolated_Models/10teff05logg/center' + str(bestt) + '_' + str(bestg) 
        os.chdir(newpath)
        #os.chdir('/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/modelfitting/DA_models/Interpolated_Models/')


    #Do a cubic spline interpolation to interpolate the model to even wavelength
    #points at 0.1A intervals so the model can be convolved.
    
    #Low and high wavelengths need to be 18A more than desired range
    #Set lambda range from ~3650 to 6770
    #The Balmer jump in the models makes the spline bulge to its left
    #So start to the right of it

    #Interpolate the model spectrum at 0.1 A intervals. But interp1d does not have
    #a range option, so we will only feed in the portion of the spectrum we want
    #to interpolate to speed things up. Again we will go a little beyond the region
    #we care about to minimize edge effects of the interpolation. Will use ~3600
    #to 6760
    
    intlambda = np.divide(range(31100),10.) + 3660.0


    
    if case == 0:
        '''
        filename1 = filenames[0]
        filename2 = filenames[1]
        filename3 = filenames[2]
        #filename4 = filenames[3]
        #filename5 = filenames[4]
        lambdas1, inten1 = np.genfromtxt(filename1,skip_header=33,unpack=True)
        lambdas2, inten2 = np.genfromtxt(filename2,skip_header=33,unpack=True)
        lambdas3, inten3 = np.genfromtxt(filename3,skip_header=33,unpack=True)
        #lambdas4, inten4 = np.genfromtxt(filename4,skip_header=33,unpack=True)
        #lambdas5, inten5 = np.genfromtxt(filename5,skip_header=33,unpack=True)
        logg1 = float(filename1[8:11])/100.
        logg2 = float(filename2[8:11])/100.
        logg3 = float(filename3[8:11])/100.
        #logg4 = float(filename4[8:11])/100.
        #logg5 = float(filename5[8:11])/100.
        teff = str(filename1[2:7])
        '''
        lambdas = np.zeros([len(filenames),31100])
        inten = np.zeros([len(filenames),31100])
        fluxes = np.zeros([len(filenames),31100])
        logg = np.zeros(len(filenames))
        teff = str(filenames[0][2:7])
        for n in np.arange(len(filenames)):
            alllambda, allinten = np.genfromtxt(filenames[n],skip_header=33,unpack=True)
            lowlambda = np.min(np.where(alllambda > 3600.))
            highlambda = np.min(np.where(alllambda > 6800.))
            shortlambdas = alllambda[lowlambda:highlambda]
            shortinten = allinten[lowlambda:highlambda]

            interp = InterpolatedUnivariateSpline(shortlambdas,shortinten,k=1)
            fluxes[n,:] = interp(intlambda)

            #lambdas[n,:] , inten[n,:] = np.genfromtxt(filenames[n],unpack=True)
            logg[n] = float(filenames[n][8:11])/100.

    if case ==1:
        '''
        filename1 = filenames[0]
        filename2 = filenames[1]
        filename3 = filenames[2]
        filename4 = filenames[3]
        filename5 = filenames[4]
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
        '''
        lambdas = np.zeros([len(filenames),31100])
        inten = np.zeros([len(filenames),31100])
        teff = np.zeros(len(filenames))
        logg = str(filenames[0][8:12])
        for n in np.arange(len(filenames)):
            lambdas[n,:] , inten[n,:] = np.genfromtxt(filenames[n],unpack=True)
            teff[n] = float(filenames[n][2:7])
                      
        #print teff

    plt.clf()

    
    '''
    if case == 0: #600:4300
        lowlambda1 = np.min(np.where(lambdas1 > 3600.))
        highlambda1 = np.min(np.where(lambdas1 > 6800.))
        shortlambdas1 = lambdas1[lowlambda1:highlambda1]
        shortinten1 = inten1[lowlambda1:highlambda1]
        
        lowlambda2 = np.min(np.where(lambdas2 > 3600.))
        highlambda2 = np.min(np.where(lambdas2 > 6800.))
        shortlambdas2 = lambdas2[lowlambda2:highlambda2]
        shortinten2 = inten2[lowlambda2:highlambda2]

        lowlambda3 = np.min(np.where(lambdas3 > 3600.))
        highlambda3 = np.min(np.where(lambdas3 > 6800.))
        shortlambdas3 = lambdas3[lowlambda3:highlambda3]
        shortinten3 = inten3[lowlambda3:highlambda3]
        
        lowlambda4 = np.min(np.where(lambdas4 > 3600.))
        highlambda4 = np.min(np.where(lambdas4 > 6800.))
        shortlambdas4 = lambdas4[lowlambda4:highlambda4]
        shortinten4 = inten4[lowlambda4:highlambda4]

        lowlambda5 = np.min(np.where(lambdas5 > 3600.))
        highlambda5 = np.min(np.where(lambdas5 > 6800.))
        shortlambdas5 = lambdas5[lowlambda5:highlambda5]
        shortinten5 = inten5[lowlambda5:highlambda5]
    '''


    #print 'shortlambdas1 run from ',shortlambdas1[0], '',shortlambdas1[-1]
    #print 'shortlambdas2 run from ',shortlambdas2[0], '',shortlambdas2[-1]
    #print 'shortlambdas3 run from ',shortlambdas3[0], '',shortlambdas3[-1]
    #print 'shortlambdas4 run from ',shortlambdas4[0], '',shortlambdas4[-1]
    #print 'shortlambdas5 run from ',shortlambdas5[0], '',shortlambdas5[-1]


    if case == 1:
        '''
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
        '''
        fluxes = np.zeros([len(filenames),31100])
        for n in np.arange(len(filenames)):
            interp = InterpolatedUnivariateSpline(lambdas[n,:],inten[n,:],k=1)
            fluxes[n,:] = interp(intlambda)

    #plot the interpolated spectra
    #plt.plot(intlambda,intflux,'ro',label='7.0')
    #plt.plot(intlambda,intflux2,'b^',label='7.25')
    #plt.plot(intlambda,intflux3,'m*',label='7.5')
    #plt.plot(intlambda,intflux4,'gs',label='8.25')
    #plt.plot(intlambda,intflux5,'cp',label='8.5')
    #plt.show()

    #Now do the 2D interpolation
    if case == 0:
        xval = logg
        #xval = np.array([logg1,logg2,logg3,logg4,logg5]) #This is our x
        #fluxes = np.array([intflux,intflux2,intflux3])#This is our z
    if case == 1:
        xval = teff
    #    xval = np.array([teff1,teff2,teff3,teff4,teff5]) #This is our x  
    #    fluxes = np.array([intflux,intflux2,intflux3,intflux4,intflux5]) #This is our z
    #intlambda is our y

    print 'Starting 2D interpolation'
    #print xval
    out = RectBivariateSpline(xval,intlambda,fluxes,kx=1,ky=1,s=0)
    print 'Done with the 2D interpolation. Starting to read off new values.'
    #Iterate over the output to build an array with the new fluxes
    #Need to set up an array that is of size grid by intlambda
    intfluxes = []
    for x in grid:
        #print x
        for i in intlambda:
            new = float(out([x],[i]))
            if i == 3660:
                newflux = [new]
            else:
                newflux.append(new)
        intfluxes.append(newflux)
        #if x == 10410:
        #    plt.clf()
        #    plt.plot(intlambda,newflux,'r^',label='Interp')
        #    #plt.show()
        #    #plt.clf()
        #    plt.plot(lambdas[4,:], inten[4,:],'bs',label='Model')
        #    plt.legend()
        #    plt.show()
    #if case == 1:
    #    plt.clf()
    #    plt.plot(intlambda,intfluxes[39],'b',label='Interp - 8.2')
    #    plt.plot(intlambda,intfluxes[40],'r',label='Interp - 7.85')
    #    plt.plot(intlambda,intfluxes[41],'g')
    #    plt.plot(intlambda,intfluxes[42],'c,')
    #    #plt.legend()
    #    plt.show()
    
    #Write and save files
    print 'Starting to save files.'
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
