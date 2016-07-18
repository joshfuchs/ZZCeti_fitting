# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 20:16:17 2015

@author: joshfuchs, based on some initial work by BH Dunlap

This program is run from fitspec.py. It interpolates and convolves the models, fits them with pseudogaussians, and normalizes. It compared the model lines to the observed line profiles and calculates a chi-square. 


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
import os
import sys
import datetime
import mpfit
#import pyfits as pf # Infierno doesn't support astropy for some reason so using pyfits

#ported to python from IDL by Josh Fuchs
#Based on the IDL routine written by Bart Dunlap

#==================================
#Pseudogaussian plus cubic for continuum
def pseudogausscubic(x,p):
    #The model function with parameters p
    return p[0]*1. + p[1]*x + p[2]*x**2. + p[7]*x**3. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.)*p[5]))**p[6])


def fitpseudogausscubic(p,fjac=None,x=None, y=None, err=None):
    #Parameter values are passed in p
    #fjac=None just means partial derivatives will not be computed.
    model = pseudogausscubic(x,p)
    status = 0
    return([status,(y-model)/err])

#==================================

#Define pseudogauss to fit one spectral line
def pseudogauss(x,p):
    #The model function with parameters p
    return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.)*p[5]))**p[6])


def fitpseudogauss(p,fjac=None,x=None, y=None, err=None):
    #Parameter values are passed in p
    #fjac=None just means partial derivatives will not be computed.
    model = pseudogauss(x,p)
    status = 0
    return([status,(y-model)/err])

#==================================

#Pseudogaussian plus cubic through h11

def gauss11cubic(x,p):
    #The model function with parameters p
    return p[0]*1. + p[1]*x + p[2]*x**2. + p[27]*x**3. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.)*p[5]))**p[6]) + p[7]*np.exp(-(np.abs(x-p[8])/(np.sqrt(2.)*p[9]))**p[10]) + p[11]*np.exp(-(np.abs(x-p[12])/(np.sqrt(2.)*p[13]))**p[14]) + p[15]*np.exp(-(np.abs(x-p[16])/(np.sqrt(2.)*p[17]))**p[18]) + p[19]*np.exp(-(np.abs(x-p[20])/(np.sqrt(2.)*p[21]))**p[22]) + p[23]*np.exp(-(np.abs(x-p[24])/(np.sqrt(2.)*p[25]))**p[26])  #This one includes Hdelta, Hepsilon, H8, H9, H10, and H11

def fitgauss11cubic(p,fjac=None,x=None, y=None, err=None):
    #Parameter values are passed in p
    #fjac=None just means partial derivatives will not be computed.
    model = gauss11cubic(x,p)
    status = 0
    return([status,(y-model)/err])

#==================================

#Pseudogaussian plus parabola through h11

def gauss11parabola(x,p):
    #The model function with parameters p
    return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.)*p[5]))**p[6]) + p[7]*np.exp(-(np.abs(x-p[8])/(np.sqrt(2.)*p[9]))**p[10]) + p[11]*np.exp(-(np.abs(x-p[12])/(np.sqrt(2.)*p[13]))**p[14]) + p[15]*np.exp(-(np.abs(x-p[16])/(np.sqrt(2.)*p[17]))**p[18]) + p[19]*np.exp(-(np.abs(x-p[20])/(np.sqrt(2.)*p[21]))**p[22]) + p[23]*np.exp(-(np.abs(x-p[24])/(np.sqrt(2.)*p[25]))**p[26])  #This one includes Hdelta, Hepsilon, H8, H9, H10, and H11

def fitgauss11parabola(p,fjac=None,x=None, y=None, err=None):
    #Parameter values are passed in p
    #fjac=None just means partial derivatives will not be computed.
    model = gauss11parabola(x,p)
    status = 0
    return([status,(y-model)/err])

#==================================

def multipseudogauss(x,p):
    #return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.*p[5])))**p[6]) + p[7]*np.exp(-(np.abs(x-p[8])/(np.sqrt(2.*p[9])))**p[10]) + p[11]*np.exp(-(np.abs(x-p[12])/(np.sqrt(2.*p[13])))**p[14]) + p[15]*np.exp(-(np.abs(x-p[16])/(np.sqrt(2.*p[17])))**p[18]) #This one includes Hdelta, Hepsilon, H8, and H9
    return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.)*p[5]))**p[6]) + p[7]*np.exp(-(np.abs(x-p[8])/(np.sqrt(2.)*p[9]))**p[10]) + p[11]*np.exp(-(np.abs(x-p[12])/(np.sqrt(2.)*p[13]))**p[14]) + p[15]*np.exp(-(np.abs(x-p[16])/(np.sqrt(2.)*p[17]))**p[18]) + p[19]*np.exp(-(np.abs(x-p[20])/(np.sqrt(2.)*p[21]))**p[22]) #This one includes Hdelta, Hepsilon, H8, H9, and H10

def multifitpseudogauss(p,fjac=None,x=None, y=None, err=None):
    #Parameter values are passed in p
    #fjac=None just means partial derivatives will not be computed.
    model = multipseudogauss(x,p)
    status = 0
    return([status,(y-model)/err])
#==================================

#Case = 0 means using D. Koester's raw models
#Case = 1 means using the interpolation of those models to a smaller grid.

def intspecs(alllambda,allnline,allsigma,lambdaindex,case,filenames,lambdas,zzcetiblue,zzcetired,FWHM,indices,path,marker,redfile):

    '''
    :DESCRIPTION: Interpolates and convolves DA models to match observed spectra. Fits pseudogaussians to DA models and compares to normalized, observed spectra. Save chi-square values.

    :INPUTS
       alllambda: 1D numpy array, wavelengths of normalized Balmer lines, shifted from observed

       allnline: 1D numpy array, observed, normalized Balmer line fluxes

       allsigma: 1D numpy array, sigma values for fluxes

       lambdaindex: 1D numpy array, index value for fitting of the Balmer lines for wavelengths. These indices work only for 'lambdas', i.e. the full blue + red wavelength range

       case: boolean, Case = 0 means using D. Koester's raw models, Case = 1 means using the interpolation of those models to a smaller grid.

       filenames: string, name of the text file that has the list of models to compare the spectrum to

       lambdas: 1D numpy array, all observed wavelength values. Needed for interpolation and convolution of models

       zzcetiblue: string, name of blue spectrum. Used for saving new files

       zzcetired: string, name of red spectrum, Used for saving new files

       FWHM: float, observed FWHM in Angstroms of spectrum. Used for convolving the models

       indices: 1D numpy array, indices to break up alllambda, allnline, and allsigma into individual balmer lines. Starts with highest order line and goes through H alpha.

       path: string, file path path to models

       redfile: True if Halpha included. False if not included.

    '''


    print 'Starting to run intspec.py'
    lambdarange = lambdas #This is the full wavelength range from blue and red setups.
    if case == 0:
        os.chdir(path)
        files = np.genfromtxt(filenames,dtype='str')
        #Create array of all logg and Teff values for computation of errors
        lowestg = float(files[0][8:11]) / 100.
        deltag = (float(files[1][8:11])/100.) - lowestg 
        highestg = float(files[-1][8:11]) / 100.
        numg = round((highestg - lowestg ) /deltag + 1.)
        gridg = np.linspace(lowestg,highestg,num = numg)
        lowestt = float(files[0][2:7])
        deltat = float(files[numg][2:7]) - lowestt
        highestt = float(files[-1][2:7])
        numt = round((highestt - lowestt) / deltat + 1.)
        gridt = np.linspace(lowestt,highestt,num=numt)
    if case == 1:
        os.chdir(path)
        files = np.genfromtxt(filenames,dtype='str')
        #Create array of all logg and Teff values for computation of errors
        lowestg = float(files[0][8:12]) / 1000.
        deltag = (float(files[1][8:12])/1000.) - lowestg
        highestg = float(files[-1][8:12]) / 1000.
        numg = round((highestg - lowestg ) /deltag + 1.) #Use round to avoid weird conversion to integer problems. Ensures this number is correct by correcting for computer representation stuff.
        gridg = np.linspace(lowestg,highestg,num = numg)
        lowestt = float(files[0][2:7])
        deltat = float(files[numg][2:7]) - lowestt 
        highestt = float(files[-1][2:7])
        numt = round((highestt - lowestt) / deltat + 1.)#Use round to avoid weird conversion to integer problems. Ensures this number is correct by correcting for computer representation stuff.
        gridt = np.linspace(lowestt,highestt,num=numt)
    n = 0.
    #Read in model. There are 33 lines of header information before the data starts. This first read in part of the header to get the T_eff and log_g. So reading in the file twice actually.
    # We do not have to integrate over mu because these files already have
    #alphapdf =  PdfPages('Model_Alpha_fit.pdf')
    #betapdf = PdfPages('Model_Beta_fit.pdf')
    #gammapdf = PdfPages('Model_Gamma_fit.pdf')
    #higherpdf = PdfPages('Model_Higher_fit.pdf')
    for dracula in files:    
        filename = dracula
        print ''
        print 'Now checking model ', filename
        teff = float(filename[2:7])
        
        if case == 0:
            logg = float(filename[8:11])
            lambdas, inten = np.genfromtxt(filename,skip_header=33,unpack=True)
        if case == 1:
            logg = float(filename[8:12])
            intlambda, intflux = np.genfromtxt(filename,unpack=True) #We already interpolated the lambda and flux
        print 'Teff and log_g of model are ', teff, 'and ', logg

        #plt.clf()
        #plt.plot(lambdas,inten,'ro',label='Model')
        #plt.show()

        #Do a cubic spline interpolation to interpolate the model to even wavelength
        #points at 0.1A intervals so the model can be convolved.
        #Low and high wavelengths need to be 18A more than desired range
        #Set lambda range from ~3650 to 6760
        #The Balmer jump in the models makes the spline bulge to its left
        #So start to the right of it

        # The interpolated models are already on the correct wavelength spacing
        # So we don't need to interpolate them again
        if case == 0:
            intlambda = np.divide(range(31000),10.) + 3660.0
            #print intlambda[1]

            #Interpolate the model spectrum at 0.1 A intervals. But interp1d does not have
            #a range option, so we will only feed in the portion of the spectrum we want
            #to interpolate to speed things up. Again we will go a little beyond the region
            #we care about to minimize edge effects of the interpolation. Will use ~3600
            #to 6050
            shortlambdas = lambdas[700:4300]
            shortinten = inten[700:4300]

            print 'Starting the 1D interpolation of the model'
            interp = InterpolatedUnivariateSpline(shortlambdas,shortinten,k=1)
            #InterpolatedUnivariateSpline seems to work as well as interp1d
            #and is significantly faster.
            intflux = interp(intlambda)
            print 'Done with the 1D interpolation of the model'
            

        #Plot interpolated spectrum
        #plt.clf()
        #plt.plot(lambdas,inten,'ro',label='Model')
        #plt.plot(intlambda,intflux,'g.',label='Interpolated Model')
        #plt.legend()
        #plt.plot(intlambda,checkint)
        #plt.show()
        #sys.exit()
        #Convolve each point with a Gaussian with FWHM
        #FWHM = 2*sqrt(2*alog(2))*sigma
        #Let the gaussian go out +/- 180 bins = +/- 18A > 9*sig
        #This seems to get all the light numerically

        sig = FWHM / (2. * np.sqrt(2.*np.log(2.)))

        #Gaussian with area normalized to 1
        #gw = 360.
        gx = np.divide(range(360),10.)
        gauss = (1./(sig * np.sqrt(2. * np.pi))) * np.exp(-(gx-18.)**2./(2.*sig**2.))

        #To get an array with 361 columns and N_elements(intflux) rows,
        #we'd multiply gauss * intflux (column by row) and each row would contain the
        #Gaussian multiplied by the value of the flux, then, starting at the top right
        #we'd want to sum the diagonals to get the convolved flux corresponding to the 
        #wavelength at whose row the diagonal crossed the center. The summing will be 
        #easier if we flip the array l-r, which can be done by flipping the gaussian
        #before multiplying. Because the Gaussian is symmetrical and the points are
        #equally spaced around its center, the 2D array will be symmetric, and this
        #won't matter.
        #Use np.outer for this

        #Need to divide by 10, beacuse although the area under the Gaussian is 1, the
        #sum of the points of the Gaussian is 10x that because the points are spaced
        #at 1/10 of an Angstrom.

        gf = np.divide(np.outer(intflux,gauss),10.)

        #Sum the diagonals to get the flux for each wavelength
        # Use np.diagonal. Want each sum to move down the matrix, so 
        #setting axis1 and axis2 so that happens.
        length = len(intflux) - 360.
        cflux = np.zeros(length)
        clambda = intlambda[180:len(intlambda)-180]

        x  = 0
        while x < length:
            cflux[x] = np.sum(np.diagonal(gf,x,axis1=1,axis2=0),dtype='d')
            x += 1
    
        #plt.clf()
        #plt.plot(clambda,cflux,'bs',label='k=1')
        #plt.plot(clambda,cflux2,'r^',label='k=2')
        #plt.legend()
        #plt.show()

        #Now we need to spline interpolate the convolved model and read out points
        #at the corresponding wavelength of the actual spectrum

        print 'Starting the interpolation of the convolved flux'
        interp2 = InterpolatedUnivariateSpline(clambda,cflux,k=1)
        cflux2 = interp2(lambdarange)
        #plt.clf()
        #plt.plot(lambdarange,cflux2,'b+')
        #plt.show()
        #print 'Done with the interpolation of the convolved flux'
        #np.savetxt('da13750_825_interp.txt',np.transpose([lambdarange,cflux2]))
        #sys.exit()
        #Fit a line to the endpoints and normalize
        #Must do this for each line separately
        #First set pixel ranges using the results from our observed spectrum
        

        ####################
        #For alpha through H10
        ####################
        if redfile:
            afitlow = lambdaindex[20]
            afithi = lambdaindex[21]
            alow = lambdaindex[22]
            ahi = lambdaindex[23]
            bfitlow = lambdaindex[16]
            bfithi = lambdaindex[17]
            blow = lambdaindex[18]
            bhi = lambdaindex[19]
            gfitlow = lambdaindex[12]
            gfithi = lambdaindex[13]
            glow = lambdaindex[14]
            ghi = lambdaindex[15]
            hlow = lambdaindex[10]
            hhi = lambdaindex[11]
            dlow = lambdaindex[8]
            dhi = lambdaindex[9]
            elow = lambdaindex[6]
            ehi = lambdaindex[7]
            H8low = lambdaindex[4]
            H8hi = lambdaindex[5]
            H9low = lambdaindex[2]
            H9hi = lambdaindex[3]
            H10low = lambdaindex[0]
            H10hi = lambdaindex[1]
            H11low = np.min(np.where(lambdarange > 3770.))
        else:
            bfitlow = lambdaindex[16]
            bfithi = lambdaindex[17]
            blow = lambdaindex[18]
            bhi = lambdaindex[19]
            gfitlow = lambdaindex[12]
            gfithi = lambdaindex[13]
            glow = lambdaindex[14]
            ghi = lambdaindex[15]
            hlow = lambdaindex[10]
            hhi = lambdaindex[11]
            dlow = lambdaindex[8]
            dhi = lambdaindex[9]
            elow = lambdaindex[6]
            ehi = lambdaindex[7]
            H8low = lambdaindex[4]
            H8hi = lambdaindex[5]
            H9low = lambdaindex[2]
            H9hi = lambdaindex[3]
            H10low = lambdaindex[0]
            H10hi = lambdaindex[1]
            H11low = np.min(np.where(lambdarange > 3770.))


        #=====================================================================================
        # Start section using mpfit to fit pseudogaussians to models to normalize
        #=====================================================================================
        '''
        #First set up the estimates

        #Choose the initial guesses in a smart AND consistent manner
        alambdas = lambdarange[afitlow:afithi+1.]
        alphaval = cflux2[afitlow:afithi+1.]
        asigmas = np.ones(len(alphaval))
        
        aest = np.zeros(8)
        xes = np.array([lambdarange[alow],lambdarange[alow+10],lambdarange[ahi-10],lambdarange[ahi]])
        yes = np.array([cflux2[alow],cflux2[alow+10],cflux2[ahi-10],cflux2[ahi]])
        ap = np.polyfit(xes,yes,3)
        app = np.poly1d(ap)
        aest[0] = ap[3]
        aest[1] = ap[2]
        aest[2] = ap[1]
        aest[7] = ap[0]
        #plt.clf()
        #smooth = app(lambdarange[alow:ahi+1])
        #print len(lambdarange[alow:ahi+1]), len(cflux2[alow:ahi+1])
        #plt.plot(lambdarange[alow:ahi+1],cflux2[alow:ahi+1],'b')
        #plt.plot(lambdarange[alow:ahi+1],smooth)
        #plt.show()
        aest[3] = np.min(cflux2[alow:ahi+1]) - app(6562.79) #depth of line relative to continuum
        aest[4] = 6562.79 #rest wavelength of H alpha
        #aest[5] = 25. #NEED TO CHECK THIS
        ahalfmax = app(6562.79) + aest[3]/3. #going to find the full width at 1/3 max
        adiff = np.abs(alphaval-ahalfmax)
        alowidx = adiff[np.where(alambdas < 6562.79)].argmin() #find index where data value closest to 1/3 max at shorter wavelengths
        ahighidx = adiff[np.where(alambdas > 6562.79)].argmin() + len(adiff[np.where(alambdas < 6562.79)]) #And now at longer wavelengths
        aest[5] = (alambdas[ahighidx] - alambdas[alowidx]) / (2.*np.sqrt(2.*np.log(2.))) #difference is FWHM then convert to sigma
        aest[6] = 1. #how much of a pseudo-gaussian

        #Now beta
        blambdas = lambdarange[bfitlow:bfithi+1.]
        betaval = cflux2[bfitlow:bfithi+1.]
        bsigmas = np.ones(len(betaval))
        best = np.zeros(8)
        xes = np.array([lambdarange[bfitlow],lambdarange[blow],lambdarange[blow+10],lambdarange[bhi-10],lambdarange[bhi],lambdarange[bfithi]])
        yes = np.array([cflux2[bfitlow],cflux2[blow],cflux2[blow+10],cflux2[bhi-10],cflux2[bhi],cflux2[bfithi]])
        bp = np.polyfit(xes,yes,3)
        bpp = np.poly1d(bp)
        best[0] = bp[3]
        best[1] = bp[2]
        best[2] = bp[1]
        best[7] = bp[0]
        best[3] = np.min(cflux2[blow:bhi+1]) - bpp(4862.710) #depth of line relative to continuum
        best[4] = 4862.710 #rest wavelength of H beta
        #best[5] = 34. #NEED TO CHECK THIS
        bhalfmax = bpp(4862.71) + best[3]/2.5
        bdiff = np.abs(betaval-bhalfmax)
        blowidx = bdiff[np.where(blambdas < 4862.71)].argmin()
        bhighidx = bdiff[np.where(blambdas > 4862.71)].argmin() + len(bdiff[np.where(blambdas < 4862.71)])
        best[5] = (blambdas[bhighidx] - blambdas[blowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
        best[6] = 1. 

        #Now gamma
        glambdas = lambdarange[gfitlow:gfithi+1.]
        gamval = cflux2[gfitlow:gfithi+1.]
        gsigmas = np.ones(len(gamval))
        gest = np.zeros(8)
        xes = np.array([lambdarange[gfitlow],lambdarange[gfitlow+10],lambdarange[gfithi-10],lambdarange[gfithi]])
        yes = np.array([cflux2[gfitlow],cflux2[gfitlow+10],cflux2[gfithi-10],cflux2[gfithi]])
        gp = np.polyfit(xes,yes,3)
        gpp = np.poly1d(gp)
        gest[0] = gp[3]
        gest[1] = gp[2]
        gest[2] = gp[1]
        gest[7] = gp[0]
        gest[3] = np.min(cflux2[glow:ghi+1]) - gpp(4341.692) #depth of line relative to continuum
        gest[4] = 4341.692 #rest wavelength of H beta
        #gest[5] = 18. #NEED TO CHECK THIS
        ghalfmax = gpp(4341.69) + gest[3]/3.
        gdiff = np.abs(gamval-ghalfmax)
        glowidx = gdiff[np.where(glambdas < 4341.69)].argmin()
        ghighidx = gdiff[np.where(glambdas > 4341.69)].argmin() + len(bdiff[np.where(glambdas < 4341.69)])
        gest[5] = (glambdas[ghighidx] - glambdas[glowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
        gest[6] = 1. 

        #Now higher order lines
        hlambdas = lambdarange[H11low:hhi+1.] #hlow:hhi+1
        hval = cflux2[H11low:hhi+1.]#hlow:hhi+1.
        dlambdasguess = lambdarange[dlow:dhi+1]
        dval = cflux2[dlow:dhi+1]
        elambdasguess = lambdarange[elow:ehi+1]
        epval = cflux2[elow:ehi+1]
        H8lambdasguess = lambdarange[H8low:H8hi+1]
        H8val = cflux2[H8low:H8hi+1]
        H9lambdasguess = lambdarange[H9low:H9hi+1]
        H9val = cflux2[H9low:H9hi+1]
        H10lambdasguess = lambdarange[H10low:H10hi+1]
        H10val = cflux2[H10low:H10hi+1]
        H11lambdasguess = lambdarange[H11low:H10low+1]
        H11val = cflux2[H11low:H10low+1]
        hest = np.zeros(27)
        #Use points to fit parabola over continuum.
        xes = np.array([lambdarange[H10low],lambdarange[H9low],lambdarange[H8low],lambdarange[elow],lambdarange[dlow],lambdarange[dhi]])
        yes = np.array([cflux2[H10low],cflux2[H9low],cflux2[H8low],cflux2[elow],cflux2[dlow],cflux2[dhi]])
        yes += hval[0]/30. #Trying an offset to make sure the continuum is above the lines
        hp = np.polyfit(xes,yes,2)
        hpp = np.poly1d(hp)
        hest[0] = hp[2]
        hest[1] = hp[1]
        hest[2] = hp[0]
        #plt.clf()
        #plt.plot(alllambda[H10low:dhi+1],cflux2[H10low:dhi+1],'b')
        #plt.plot(alllambda[H10low:dhi+1],x[0]*alllambda[H10low:dhi+1]**2.+x[1]*alllambda[H10low:dhi+1]+x[2],'k+',markersize=5)
        #plt.show()

        #Now delta
        hest[3] = np.min(cflux2[dlow:dhi+1]) - hpp(4102.892) #depth of line relative to continuum
        hest[4] = 4102.83 #rest wavelength of H delta  4102.892
        #hest[5] = 17. #NEED TO CHECK THIS
        dhalfmax = hpp(4102.89) + hest[3]/3.
        ddiff = np.abs(dval-dhalfmax)
        dlowidx = ddiff[np.where(dlambdasguess < 4102.89)].argmin()
        dhighidx = ddiff[np.where(dlambdasguess > 4102.89)].argmin() + len(ddiff[np.where(dlambdasguess < 4102.89)])
        hest[5] = (dlambdasguess[dhighidx] - dlambdasguess[dlowidx]) / (2.*np.sqrt(2.*np.log(2.)))
        hest[6] = 1.2 #how much of a pseudo-gaussian

        #Now epsilon
        hest[7] = np.min(cflux2[elow:ehi+1]) - hpp(3971.198) #depth of line relative to continuum
        hest[8] = 3971.16 #rest wavelength of H epsilon   3971.198
        #hest[9] = 14. #NEED TO CHECK THIS
        ehalfmax = hpp(3971.19) + hest[7]/3.
        ediff = np.abs(epval-ehalfmax)
        elowidx = ediff[np.where(elambdasguess < 3971.19)].argmin()
        ehighidx = ediff[np.where(elambdasguess > 3971.19)].argmin() + len(ediff[np.where(elambdasguess < 3971.19)])
        hest[9] = (elambdasguess[ehighidx] - elambdasguess[elowidx]) / (2.*np.sqrt(2.*np.log(2.)))
        hest[10] = 1.2 #how much of a pseudo-gaussian

        #Now H8
        hest[11] = np.min(cflux2[H8low:H8hi+1]) - hpp(3890.166) #depth of line relative to continuum
        hest[12] = 3890.22 #rest wavelength of H8  3890.166
        #hest[13] = 14. #NEED TO CHECK THIS
        H8halfmax = hpp(3890.16) + hest[11]/3.
        H8diff = np.abs(H8val-H8halfmax)
        H8lowidx = H8diff[np.where(H8lambdasguess < 3890.16)].argmin()
        H8highidx = H8diff[np.where(H8lambdasguess > 3890.16)].argmin() + len(H8diff[np.where(H8lambdasguess < 3890.16)])
        hest[13] = (H8lambdasguess[H8highidx] - H8lambdasguess[H8lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
        hest[14] = 1.2 #how much of a pseudo-gaussian

        #Now H9
        hest[15] = np.min(cflux2[H9low:H9hi+1]) - hpp(3836.485) #depth of line relative to continuum
        hest[16] = 3836.48 #rest wavelength of H9  3836.485 
        #hest[17] = 14. #NEED TO CHECK THIS
        H9halfmax = hpp(3836.48) + hest[15]/3.
        H9diff = np.abs(H9val-H9halfmax)
        H9lowidx = H9diff[np.where(H9lambdasguess < 3836.48)].argmin()
        H9highidx = H9diff[np.where(H9lambdasguess > 3836.48)].argmin() + len(H9diff[np.where(H9lambdasguess < 3836.48)])
        hest[17] = (H9lambdasguess[H9highidx] - H9lambdasguess[H9lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
        hest[18] = 1.2 #how much of a pseudo-gaussian

        #Now H10
        hest[19] = np.min(cflux2[H10low:H10hi+1]) - hpp(3797.909) #depth of line relative to continuum
        hest[20] = 3798.8 #rest wavelength of H10   3797.909
        #hest[21] = 14. #NEED TO CHECK THIS
        H10halfmax = hpp(3798.8) + hest[19]/3.
        H10diff = np.abs(H10val-H10halfmax)
        H10lowidx = H10diff[np.where(H10lambdasguess < 3798.8)].argmin()
        H10highidx = H10diff[np.where(H10lambdasguess > 3798.8)].argmin() + len(H10diff[np.where(H10lambdasguess < 3798.8)])
        hest[21] = (H10lambdasguess[H10highidx] - H10lambdasguess[H10lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
        hest[22] = 1.2 #how much of a pseudo-gaussian

        #now H11
        hest[23] = np.min(cflux2[H11low:H10low+1]) - hpp(3770.63) #depth of line relative to continuum
        hest[24] = 3771.8 #rest wavelength of H10  3770.633
        #hest[25] = 10. #NEED TO CHECK THIS
        H11halfmax = hpp(3771.8) + hest[23]/3.
        H11diff = np.abs(H11val-H11halfmax)
        H11lowidx = H11diff[np.where(H11lambdasguess < 3771.8)].argmin()
        H11highidx = H11diff[np.where(H11lambdasguess > 3771.8)].argmin() + len(H11diff[np.where(H11lambdasguess < 3771.8)])
        hest[25] = 2.* (H11lambdasguess[H11highidx] - H11lambdasguess[H11lowidx]) / (2.*np.sqrt(2.*np.log(2.))) #Multiply by two since we are cutting off H11 in the middle.
        hest[26] = 1.2 #how much of a pseudo-gaussian


        #Fit H alpha
        print 'Now fitting H alpha.'
        #mpfit terminates when the relative error is at most xtol. Since the first parameter on these fits is really large, this has to be a smaller number to get mpfit to actually iterate. 
        #alambdas = lambdarange[afitlow:afithi+1.]
        #alphaval = cflux2[afitlow:afithi+1.]
        #asigmas = np.ones(len(alphaval))
        afa = {'x':alambdas, 'y':alphaval, 'err':asigmas}
        paralpha = [{'step':0.} for i in range(7)]
        #paralpha[0]['step'] = 1e14
        aparams = mpfit.mpfit(fitpseudogausscubic,aest,functkw=afa,maxiter=3000,ftol=1e-12,xtol=1e-11,quiet=True)
        print 'Number of evaluations: ', aparams.niter
        #falpha = open('alphafits.txt','a')
        #alphafitstosave = str(aparams.params)
        #falpha.write(alphafitstosave + '\n')
        #falpha.close()
        alphafit = pseudogausscubic(alambdas,aparams.params)
        acenter = aparams.params[4]
        alphavariation = np.sum((alphafit - alphaval)**2.)
        #Save fit to model to PDF
        alphatitle = 'Model: ' + str(logg) + ' and ' + str(teff)
        #plt.clf()
        #plt.plot(alambdas,alphaval,'b',label='Model')
        #plt.plot(alambdas,alphafit,'g',label='Fit')
        #plt.title(alphatitle)
        #plt.legend(loc=4)
        #alphapdf.savefig()
        #plt.show()


        #Fit H beta
        print 'Now fitting H beta.'
        #blambdas = lambdarange[bfitlow:bfithi+1.]
        #betaval = cflux2[bfitlow:bfithi+1.]
        #bsigmas = np.ones(len(betaval))
        bfa = {'x':blambdas,'y':betaval,'err':bsigmas}
        bparams = mpfit.mpfit(fitpseudogausscubic,best,functkw=bfa,maxiter=3000,ftol=1e-12,xtol=1e-11,quiet=True)
        print 'Number of evaluations: ', bparams.niter
        #fbeta = open('betafits.txt','a')
        #betafitstosave = str(bparams.params)
        #fbeta.write(betafitstosave + '\n')
        #fbeta.close()
        betafit = pseudogausscubic(blambdas,bparams.params)
        bcenter = bparams.params[4]
        betavariation = np.sum((betafit - betaval)**2.)
        #print filename
        #print best[0] - bparams.params[0]
        #print bparams.status
        #plt.clf()
        #plt.plot(blambdas,betaval,'b',label='Model')
        #plt.plot(blambdas,betafit,'g',label='Fit')
        ##plt.plot(blambdas,pseudogauss(blambdas,best),'k')
        #plt.title(alphatitle)
        #plt.legend(loc=4)
        #betapdf.savefig()
        #plt.show()


        #Fit H gamma
        print 'Now fitting H gamma.'
        #glambdas = lambdarange[gfitlow:gfithi+1.]
        #gamval = cflux2[gfitlow:gfithi+1.]
        #gsigmas = np.ones(len(gamval))
        gfa = {'x':glambdas,'y':gamval,'err':gsigmas}
        gparams = mpfit.mpfit(fitpseudogausscubic,gest,functkw=gfa,maxiter=3000,ftol=1e-12,xtol=1e-11,quiet=True)
        print 'Number of evaluations: ', gparams.niter
        #fgamma = open('gammafits.txt','a')
        #gammafitstosave = str(gparams.params)
        #fgamma.write(gammafitstosave + '\n')
        #fgamma.close()
        gamfit = pseudogausscubic(glambdas,gparams.params)
        gcenter = gparams.params[4]
        gammavariation = np.sum((gamfit - gamval)**2.)
        #print filename
        #print gest[0] - gparams.params[0]
        #print gparams.status
        #plt.clf()
        #plt.plot(glambdas,gamval,'b',label='Model')
        #plt.plot(glambdas,gamfit,'g',label='Fit')
        ##plt.plot(glambdas,pseudogauss(glambdas,gest),'k')
        #plt.title(alphatitle)
        #plt.legend(loc=4)
        #gammapdf.savefig()
        #plt.show()

        #Fit higher order lines
        print 'Now fitting the higher order lines.'
        #hlambdas = lambdarange[H11low:hhi+1.] #hlow:hhi+1
        dlambdas = lambdas[dlow:dhi+1]
        elambdas = lambdas[elow:ehi+1]
        H8lambdas = lambdas[H8low:H8hi+1]
        H9lambdas = lambdas[H9low:H9hi+1]
        H10lambdas = lambdas[H10low:H10hi+1]
        #hval = cflux2[H11low:hhi+1.]#hlow:hhi+1.
        hsigmas = np.ones(len(hval))
        hfa = {'x':hlambdas,'y':hval,'err':hsigmas}
        #Limit parameters
        limparams = [{'limits':[0,0],'limited':[0,0]} for i in range(len(hest))] #23 total parameters
        limparams[20]['limited'] = [1,0]
        limparams[20]['limits'] = [3784.,0.]
        limparams[24]['limited'] = [0,1]
        limparams[24]['limits'] = [0,3784.]

        #hparams = mpfit.mpfit(multifitpseudogauss,hest,functkw=hfa,maxiter=3000,ftol=1e-14,parinfo=limparams,quiet=True)
        hparams = mpfit.mpfit(fitgauss11parabola,hest,functkw=hfa,maxiter=2000,ftol=1e-9,xtol=1e-11,quiet=True,parinfo=limparams)
        print 'Number of evaluations: ', hparams.niter
        #fhigh = open('highfits.txt','a')
        #highfitstosave = str(hparams.params)
        #fhigh.write(highfitstosave + '\n')
        #fhigh.close()
        hfit = gauss11parabola(hlambdas,hparams.params)
        dcenter = hparams.params[4]
        ecenter = hparams.params[8]
        H8center = hparams.params[12]
        H9center = hparams.params[16]
        H10center = hparams.params[20]
        highervariation = np.sum((hfit - hval)**2.)
        #hightitle = alphatitle + '   ' + str(np.round(hparams.params[20],decimals=3))
        #plt.clf()
        #plt.plot(hlambdas,hval,'b',label='Model')
        #plt.plot(hlambdas,hfit,'g',label='Fit')
        ##plt.plot(hlambdas,multipseudogauss(hlambdas,hest),'k')
        #plt.title(hightitle)
        #plt.legend(loc=3)
        #higherpdf.savefig()
        #plt.show()
        #sys.exit()


        #Normalize the lines. This  option use the pseudogaussian fits to the models to normalize.
        #First H10
        #First make center of pseudo gauss fit to be at center of model
        hlambdastemp = hlambdas - (H10center-3798.9799)
        #Since we have the center matched up, now set the normalization points and normalize to those
        H10normlow = np.min(np.where(hlambdastemp > 3785.))
        H10normhi = np.min(np.where(hlambdastemp > 3815.))
        H10slope = (hfit[H10normhi] - hfit[H10normlow]) / (hlambdastemp[H10normhi] - hlambdastemp[H10normlow])
        H10nline = H10slope * (hlambdastemp[H10normlow:H10normhi+1] - hlambdastemp[H10normlow]) + hfit[H10normlow]
        H10fluxtemp = cflux2[H11low:hhi+1]
        H10nfluxtemp = H10fluxtemp[H10normlow:H10normhi+1] / H10nline
        #Our wavelengths are now different from those in our observed spectrum since we shifted. So interpolate the normalized line and read out the wavelengths we want.
        interp = InterpolatedUnivariateSpline(hlambdastemp[H10normlow:H10normhi+1],H10nfluxtemp,k=1)
        H10nflux = interp(alllambda[indices[0]:indices[1]+1])


        #Now H9
        hlambdastemp = hlambdas - (H9center- 3836.4726)
        H9normlow = np.min(np.where(hlambdastemp > 3815.))
        H9normhi = np.min(np.where(hlambdastemp > 3855.))
        H9slope = (hfit[H9normhi] - hfit[H9normlow]) / (hlambdastemp[H9normhi] - hlambdastemp[H9normlow])
        H9nline = H9slope * (hlambdastemp[H9normlow:H9normhi+1] - hlambdastemp[H9normlow]) + hfit[H9normlow]
        #H9nflux = cflux2[H9low:H9hi+1] / H9nline
        H9fluxtemp = cflux2[H11low:hhi+1]
        H9nfluxtemp = H9fluxtemp[H9normlow:H9normhi+1] / H9nline
        interp = InterpolatedUnivariateSpline(hlambdastemp[H9normlow:H9normhi+1],H9nfluxtemp,k=1)
        H9nflux = interp(alllambda[indices[2]:indices[3]+1])


        #Then H8
        hlambdastemp = hlambdas - (H8center-3890.1461)
        H8normlow = np.min(np.where(hlambdastemp > 3859.))
        H8normhi = np.min(np.where(hlambdastemp > 3919.))
        H8slope = (hfit[H8normhi] - hfit[H8normlow]) / (hlambdastemp[H8normhi] - hlambdastemp[H8normlow])
        H8nline = H8slope * (hlambdastemp[H8normlow:H8normhi+1] - hlambdastemp[H8normlow]) + hfit[H8normlow]
        #H8nflux = cflux2[H8low:H8hi+1] / H8nline
        H8fluxtemp = cflux2[H11low:hhi+1]
        H8nfluxtemp = H8fluxtemp[H8normlow:H8normhi+1] / H8nline
        interp = InterpolatedUnivariateSpline(hlambdastemp[H8normlow:H8normhi+1],H8nfluxtemp,k=1)
        H8nflux = interp(alllambda[indices[4]:indices[5]+1])


        #Then H epsilon
        hlambdastemp = hlambdas - (ecenter-3971.1751)
        enormhi = np.min(np.where(hlambdastemp > 4015.))
        enormlow = np.min(np.where(hlambdastemp > 3925.))
        eslope = (hfit[enormhi] - hfit[enormlow]) / (hlambdastemp[enormhi] - hlambdastemp[enormlow])
        enline = eslope * (hlambdastemp[enormlow:enormhi+1] - hlambdastemp[enormlow]) + hfit[enormlow]
        #enflux = cflux2[elow:ehi+1] / enline
        efluxtemp = cflux2[H11low:hhi+1]
        enfluxtemp = efluxtemp[enormlow:enormhi+1] / enline
        interp = InterpolatedUnivariateSpline(hlambdastemp[enormlow:enormhi+1],enfluxtemp,k=1)
        enflux = interp(alllambda[indices[6]:indices[7]+1])


        #Then H delta
        hlambdastemp = hlambdas - (dcenter-4102.9071)
        dnormhi = np.min(np.where(hlambdastemp > 4171.))
        dnormlow = np.min(np.where(hlambdastemp > 4031.))
        dslope = (hfit[dnormhi] - hfit[dnormlow]) / (hlambdastemp[dnormhi] - hlambdastemp[dnormlow])
        dnline = dslope * (hlambdastemp[dnormlow:dnormhi+1] - hlambdastemp[dnormlow]) + hfit[dnormlow]
        dfluxtemp = cflux2[H11low:hhi+1]
        #dnflux = cflux2[dlow:dhi+1] / dnline
        dnfluxtemp = dfluxtemp[dnormlow:dnormhi+1] / dnline
        interp = InterpolatedUnivariateSpline(hlambdastemp[dnormlow:dnormhi+1],dnfluxtemp,k=1)
        dnflux = interp(alllambda[indices[8]:indices[9]+1])

        #For H alpha and beta we fit to larger regions than we want to compare. So need to normalize
        #using our narrower region.
        
        #Now H gamma
        glambdas = glambdas - (gcenter-4341.6550)
        gnormlow = np.min(np.where(glambdas > 4220.))
        gnormhi = np.min(np.where(glambdas > 4460.))
        gslope = (gamfit[gnormhi] - gamfit[gnormlow]) / (glambdas[gnormhi] - glambdas[gnormlow])
        gvalnew = gamval[gnormlow:gnormhi+1]
        glambdasnew = glambdas[gnormlow:gnormhi+1]
        gnline = gslope * (glambdasnew - glambdas[gnormlow]) + gamfit[gnormlow]
        gnfluxtemp = gvalnew / gnline
        interp = InterpolatedUnivariateSpline(glambdasnew,gnfluxtemp,k=1)
        gnflux = interp(alllambda[indices[10]:indices[11]+1])

    
        #Now H beta
        blambdas = blambdas - (bcenter-4862.6510)
        bnormlow = np.min(np.where(blambdas > 4721.))
        bnormhi = np.min(np.where(blambdas > 5001.))
        bslope = (betafit[bnormhi] - betafit[bnormlow]) / (blambdas[bnormhi] - blambdas[bnormlow])
        blambdasnew = blambdas[bnormlow:bnormhi+1]
        bvalnew = betaval[bnormlow:bnormhi+1.]
        bnline = bslope * (blambdasnew - blambdas[bnormlow]) + betafit[bnormlow]
        bnfluxtemp = bvalnew / bnline
        interp = InterpolatedUnivariateSpline(blambdasnew,bnfluxtemp,k=1)
        bnflux = interp(alllambda[indices[12]:indices[13]+1])


        #Now H alpha
        alambdas = alambdas - (acenter- 6564.6047)
        #alambdas includes extra fitting, so need to select inner points for normalization
        anormlow = np.min(np.where(alambdas > 6413.))
        anormhi = np.min(np.where(alambdas > 6713.))
        aslope = (alphafit[anormhi] - alphafit[anormlow]) / (alambdas[anormhi] - alambdas[anormlow])
        alambdasnew = alambdas[anormlow:anormhi+1]
        avalnew = alphaval[anormlow:anormhi+1]
        anline = aslope * (alambdasnew - alambdas[anormlow]) + alphafit[anormlow]
        anfluxtemp = avalnew / anline
        interp = InterpolatedUnivariateSpline(alambdasnew,anfluxtemp,k=1)
        anflux = interp(alllambda[indices[14]:+indices[15]+1])
        '''
        #=====================================================================================
        # End  section using mpfit to fit pseudogaussians to models to normalize
        #=====================================================================================

        #=====================================================================================
        # Start section using model points to normalize
        #=====================================================================================
        cflux2 = interp2(alllambda)
        if redfile:
            ahi = indices[15]
            alow = indices[14]
        bhi = indices[13]
        blow = indices[12]
        ghi = indices[11]
        glow = indices[10]
        dhi = indices[9]
        dlow = indices[8]
        ehi = indices[7]
        elow = indices[6]
        H8hi = indices[5]
        H8low = indices[4]
        H9hi = indices[3]
        H9low = indices[2]
        H10hi = indices[1]
        H10low = indices[0]
        
        
        #Now H alpha
        if redfile:
            alambdas = alllambda[alow:ahi+1]
            aslope = (cflux2[ahi] - cflux2[alow]) / (alllambda[ahi] - alllambda[alow])
            anline = aslope * (alllambda[alow:ahi+1.] - alllambda[alow]) + cflux2[alow]
            anflux = cflux2[alow:ahi+1.] / anline
        
        #Now H beta
        blambdas = alllambda[blow:bhi+1.]
        
        bslope = (cflux2[bhi] - cflux2[blow]) / (alllambda[bhi] - alllambda[blow])
        bnline = bslope * (alllambda[blow:bhi+1.] - alllambda[blow]) + cflux2[blow]
        bnflux = cflux2[blow:bhi+1.] / bnline

        #Now H gamma
        glambdas = alllambda[glow:ghi+1.]

        gslope = (cflux2[ghi] - cflux2[glow]) / (alllambda[ghi] - alllambda[glow])
        gnline = gslope * (alllambda[glow:ghi+1.] - alllambda[glow]) + cflux2[glow]
        gnflux = cflux2[glow:ghi+1.] / gnline

        #Now H delta
        dlambdas = alllambda[dlow:dhi+1]
        
        dslope = (cflux2[dhi] - cflux2[dlow]) / (alllambda[dhi] - alllambda[dlow])
        dnline = dslope * (alllambda[dlow:dhi+1.] - alllambda[dlow]) + cflux2[dlow]
        dnflux = cflux2[dlow:dhi+1.] / dnline

        #Now H epsilon
        elambdas = alllambda[elow:ehi+1]

        eslope = (cflux2[ehi] - cflux2[elow]) / (alllambda[ehi] - alllambda[elow])
        enline = eslope * (alllambda[elow:ehi+1.] - alllambda[elow]) + cflux2[elow]
        enflux = cflux2[elow:ehi+1.] / enline

        #Now H8
        H8lambdas = alllambda[H8low:H8hi+1]

        H8slope = (cflux2[H8hi] - cflux2[H8low]) / (alllambda[H8hi] - alllambda[H8low])
        H8nline = H8slope * (alllambda[H8low:H8hi+1.] - alllambda[H8low]) + cflux2[H8low]
        H8nflux = cflux2[H8low:H8hi+1.] / H8nline

        #Now H9
        H9lambdas = alllambda[H9low:H9hi+1]
        
        H9slope = (cflux2[H9hi] - cflux2[H9low]) / (alllambda[H9hi] - alllambda[H9low])
        H9nline = H9slope * (alllambda[H9low:H9hi+1.] - alllambda[H9low]) + cflux2[H9low]
        H9nflux = cflux2[H9low:H9hi+1.] / H9nline

        #Now H10
        H10lambdas = alllambda[H10low:H10hi+1]

        H10slope = (cflux2[H10hi] - cflux2[H10low]) / (alllambda[H10hi] - alllambda[H10low])
        H10nline = H10slope * (alllambda[H10low:H10hi+1.] - alllambda[H10low]) + cflux2[H10low]
        H10nflux = cflux2[H10low:H10hi+1.] / H10nline

        #=====================================================================================
        # End section using model points to normalize
        #=====================================================================================

        #Concatenate into one normalized array.
        if redfile:
            ncflux = np.concatenate((H10nflux,H9nflux,H8nflux,enflux,dnflux,gnflux,bnflux,anflux))
        else:
            ncflux = np.concatenate((H10nflux,H9nflux,H8nflux,enflux,dnflux,gnflux,bnflux))
        #print len(H10nflux),len(H9nflux),len(H8nflux),len(enflux),len(dnflux),len(gnflux),len(bnflux),len(anflux)
        #plt.clf()
        #plt.plot(alllambda,ncflux,'b^')
        #plt.show()

        #newfilename = 'da_norm_' + str(logg) + '_' + str(teff) + '.txt'
        #np.savetxt(newfilename,np.transpose([alllambda,ncflux]))

        #Get the observed fluxes and sigma for each line so that we can compute chi square for each line individually
        if redfile:
            obsalpha = allnline[indices[14]:+indices[15]+1]
            obsalphasig = allsigma[indices[14]:+indices[15]+1]
        
        obsbeta = allnline[indices[12]:indices[13]+1]
        obsgamma = allnline[indices[10]:indices[11]+1]
        obsdelta = allnline[indices[8]:indices[9]+1]
        obsepsilon = allnline[indices[6]:indices[7]+1]
        obs8 = allnline[indices[4]:indices[5]+1]
        obs9 = allnline[indices[2]:indices[3]+1]
        obs10 = allnline[indices[0]:indices[1]+1]

        obsbetasig = allsigma[indices[12]:indices[13]+1]
        obsgammasig = allsigma[indices[10]:indices[11]+1]
        obsdeltasig = allsigma[indices[8]:indices[9]+1]
        obsepsilonsig = allsigma[indices[6]:indices[7]+1]
        obs8sig = allsigma[indices[4]:indices[5]+1]
        obs9sig = allsigma[indices[2]:indices[3]+1]
        obs10sig = allsigma[indices[0]:indices[1]+1]
 
        #Save interpolated and normalized model
        #intmodelname = 'da' + str(teff) + '_' + str(logg) + zzcetiblue[5:zzcetiblue.find('.ms.')] + '_' + str(np.round(FWHM,decimals=2)) + '_norm.txt'
        #np.savetxt(intmodelname,np.transpose([alllambda,ncflux]))

        #Calculate residuals and chi-square
        if n == 0:
            residual = np.empty(len(files))
            chisq = np.empty(len(files))
            allg = np.empty(len(files))
            allt = np.empty(len(files))
            chis = np.empty([numg,numt]) #logg by Teff #######
            if redfile:
                chisalpha = np.empty([numg,numt])
            chisbeta = np.empty([numg,numt])
            chisgamma = np.empty([numg,numt])
            chisdelta = np.empty([numg,numt])
            chisepsilon = np.empty([numg,numt])
            chis8 = np.empty([numg,numt])
            chis9 = np.empty([numg,numt])
            chis10 = np.empty([numg,numt])
            #variation = np.empty([numg,numt])
        residual[n] =  np.sum((allnline - ncflux)**2.,dtype='d')
        chisq[n] = np.sum(((allnline - ncflux) / allsigma)**2.,dtype='d')
        allg[n] = logg
        allt[n] = teff
        if case == 0:
            ng = round(((logg/100.) - lowestg) / deltag) #round to fix float/int issue
            nt = round((teff - lowestt) / deltat)#round to fix float/int issu
        if case == 1:
            ng = round(((logg/1000.) - lowestg) / deltag)#round to fix float/int issu
            nt = round((teff - lowestt) / deltat)#round to fix float/int issu
        chis[ng][nt] = chisq[n] #Save values in a matrix
        if redfile:
            chisalpha[ng][nt] = np.sum(((obsalpha - anflux) / obsalphasig)**2.,dtype='d')
        chisbeta[ng][nt] = np.sum(((obsbeta -bnflux) / obsbetasig)**2.,dtype='d')
        chisgamma[ng][nt] = np.sum(((obsgamma - gnflux) / obsgammasig)**2.,dtype='d')
        chisdelta[ng][nt] = np.sum(((obsdelta - dnflux) / obsdeltasig)**2.,dtype='d')
        chisepsilon[ng][nt] = np.sum(((obsepsilon - enflux) / obsepsilonsig)**2.,dtype='d')
        chis8[ng][nt] = np.sum(((obs8 - H8nflux) / obs8sig)**2.,dtype='d')
        chis9[ng][nt] = np.sum(((obs9 - H9nflux) / obs9sig)**2.,dtype='d')
        chis10[ng][nt] = np.sum(((obs10 - H10nflux) / obs10sig)**2.,dtype='d')
        #variation[ng][nt] = alphavariation + betavariation + gammavariation + highervariation
        print 'Chi-square is ',chisq[n]
        if n == 0:
            bestchi = chisq[n]

        if chisq[n] <= bestchi:
            bestchi=chisq[n]
            bestT = teff
            bestg = logg
            bestmodel = ncflux
        #plt.clf()
        #plt.plot(alllambda,allnline,'bs',label='Normalized data')
        #plt.plot(alllambda,bestmodel,'r^',label='Model')
        #plt.legend()
        #plt.show()
        n += 1.

# Now we want to find the errors on the fit if we are looking at the fine grid


    #First subtract the best chi-squared from all chi-squared values
    #alphapdf.close()
    #betapdf.close()
    #gammapdf.close()
    #higherpdf.close()
    deltachi = np.subtract(chis,bestchi)
    
    #Save information on best fitting model and the convolved model itself
    f = open('fitting_solutions.txt','a') #'a' means solution will be appended to file if it exists, otherwise it will be created.
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    if case == 0:
        bestmodelname = 'da' + str(int(bestT)) + '_' + str(int(bestg)) + '.dk'
    if case == 1:
        bestmodelname = 'da' + str(int(bestT)) + '_' + str(int(bestg)) + '.jf'
    info = zzcetiblue + '\t' + zzcetired + '\t' +  bestmodelname + '\t' + str(bestT) + '\t' + str(bestg) + '\t' + marker + '\t' + str(bestchi) + '\t' + now
    f.write(info + '\n')
    f.close()
    #Now save the best convolved model and delta chi squared surface
    endpoint = '.ms.' #For shorter names, use '_930'
    newmodel = 'model_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_' + now[5:10] + '_' + marker + '.txt' #NEED TO CHECK THIS TO MAKE SURE IT WORKS GENERALLY
    np.savetxt(newmodel,np.transpose([alllambda,bestmodel]))
    chiname = 'chi_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_' + now[5:10] + '_' + marker + '.txt'
    np.savetxt(chiname,chis)
    if redfile:
        alphaname = 'chi_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_alpha_' + now[5:10] + '_' + marker + '.txt'
        np.savetxt(alphaname,chisalpha)
    betaname = 'chi_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_beta_' + now[5:10] + '_' + marker + '.txt'
    np.savetxt(betaname,chisbeta)
    gammaname = 'chi_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_gamma_' + now[5:10] + '_' + marker + '.txt'
    np.savetxt(gammaname,chisgamma)
    deltaname = 'chi_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_delta_' + now[5:10] + '_' + marker + '.txt'
    np.savetxt(deltaname,chisdelta)
    epsilonname = 'chi_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_epsilon_' + now[5:10] + '_' + marker + '.txt'
    np.savetxt(epsilonname,chisepsilon)
    H8name = 'chi_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_H8_' + now[5:10] + '_' + marker + '.txt'
    np.savetxt(H8name,chis8)
    H9name = 'chi_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_H9_' + now[5:10] + '_' + marker + '.txt'
    np.savetxt(H9name,chis9)
    H10name = 'chi_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_H10_' + now[5:10] + '_' + marker + '.txt'
    np.savetxt(H10name,chis10)
    #variationname = 'variation_models_' + zzcetiblue[5:zzcetiblue.find('_930_')] + now[5:10] + '_' + marker + '.txt'
    #np.savetxt(variationname,variation)
    

    print ''
    print 'Best chi-squared is ', bestchi
    if case == 0:
        print 'Best T_eff is ', bestT
        print 'Best log_g is ', bestg/100.
    if case == 1:
        print 'Best T_eff is ', bestT
        print 'Best log_g is ', bestg/1000.
    print 'Done running intspec.py'
    #plt.clf()
    #plt.plot(alllambda,allnline,'bs',label='Normalized data')
    #plt.plot(alllambda,bestmodel,'r^',label='Model')
    #plt.legend()
    #plt.show()
    return ncflux,int(bestT),int(bestg)

