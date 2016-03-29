# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 20:16:17 2015

@author: joshfuchs
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
import os
import sys
import datetime
import mpfit
#import pyfits as pf # Infierno doesn't support astropy for some reason so using pyfits



#ported to python from IDL by Josh Fuchs
#Based on the IDL routine written by Bart Dunlap

######################################################
#You should not run this program by itself. It is called
#at the end of fitspec and depends on the variables created
#in that program.
######################################################

#Define pseudogauss to fit one spectral line
def pseudogauss(x,p):
    #The model function with parameters p
    return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.*p[5])))**p[6])


def fitpseudogauss(p,fjac=None,x=None, y=None, err=None):
    #Parameter values are passed in p
    #fjac=None just means partial derivatives will not be computed.
    model = pseudogauss(x,p)
    status = 0
    return([status,(y-model)/err])

def multipseudogauss(x,p):
    #return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.*p[5])))**p[6]) + p[7]*np.exp(-(np.abs(x-p[8])/(np.sqrt(2.*p[9])))**p[10]) + p[11]*np.exp(-(np.abs(x-p[12])/(np.sqrt(2.*p[13])))**p[14]) + p[15]*np.exp(-(np.abs(x-p[16])/(np.sqrt(2.*p[17])))**p[18]) #This one includes Hdelta, Hepsilon, H8, and H9
    return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.*p[5])))**p[6]) + p[7]*np.exp(-(np.abs(x-p[8])/(np.sqrt(2.*p[9])))**p[10]) + p[11]*np.exp(-(np.abs(x-p[12])/(np.sqrt(2.*p[13])))**p[14]) + p[15]*np.exp(-(np.abs(x-p[16])/(np.sqrt(2.*p[17])))**p[18]) + p[19]*np.exp(-(np.abs(x-p[20])/(np.sqrt(2.*p[21])))**p[22]) #This one includes Hdelta, Hepsilon, H8, H9, and H10

def multifitpseudogauss(p,fjac=None,x=None, y=None, err=None):
    #Parameter values are passed in p
    #fjac=None just means partial derivatives will not be computed.
    model = multipseudogauss(x,p)
    status = 0
    return([status,(y-model)/err])

#Case = 0 means using D. Koester's raw models
#Case = 1 means using the interpolation of those models to a smaller grid.

def intmodel(alllambda,allnline,allsigma,lambdaindex,case,filenames,lambdas,zzcetiblue,zzcetired,FWHM,indices):

    '''
    :DESCRIPTION: Interpolates and convolves DA models to match observed spectra. Fits pseudogaussians to DA models and compares to normalized, observed spectra. Save chi-square values.

    :INPUTS
       alllambda: 1D numpy array, observed wavelengths

       allnline: 1D numpy array, observed, normalized Balmer line fluxes

       allsigma: 1D numpy array, sigma values for fluxes

       lambdaindex: 1D numpy array, index value for fitting of the Balmer lines for wavelengths

       case: boolean, Case = 0 means using D. Koester's raw models, Case = 1 means using the interpolation of those models to a smaller grid.

       filenames: string, name of the text file that has the list of models to compare the spectrum to

       lambdas: 1D numpy array, all observed wavelength values. Needed for interpolation and convolution of models

       zzcetiblue: string, name of blue spectrum. Used for saving new files

       zzcetired: string, name of red spectrum, Used for saving new files

       FWHM: float, observed FWHM of spectrum. Used for convolving the models

       indices: 1D numpy array, indices to break up alllambda, allnline, and allsigma into individual balmer lines. Starts with highest order line and goes through H alpha.

    '''


    print 'Starting to run intspec.py'
    lambdarange = lambdas #This is the full wavelength range from blue and red setups.
    if case == 0:
        os.chdir('/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/modelfitting/DA_models')
        files = np.genfromtxt(filenames,dtype='str')
        #Create array of all logg and Teff values for computation of errors
        lowestg = float(files[0][8:11]) / 100.
        deltag = (float(files[1][8:11])/100.) - lowestg 
        highestg = float(files[-1][8:11]) / 100.
        numg = (highestg - lowestg ) /deltag + 1.
        gridg = np.linspace(lowestg,highestg,num = numg)
        lowestt = float(files[0][2:7])
        deltat = float(files[numg][2:7]) - lowestt
        highestt = float(files[-1][2:7])
        numt = (highestt - lowestt) / deltat + 1.
        gridt = np.linspace(lowestt,highestt,num=numt)
    if case == 1:
        os.chdir('/srv/two/jtfuchs/Interpolated_Models/center12750_800')
        files = np.genfromtxt(filenames,dtype='str')
        #Create array of all logg and Teff values for computation of errors
        lowestg = float(files[0][8:12]) / 1000.
        deltag = (float(files[1][8:12])/1000.) - lowestg 
        highestg = float(files[-1][8:12]) / 1000.
        numg = (highestg - lowestg ) /deltag + 1.
        gridg = np.linspace(lowestg,highestg,num = numg)
        lowestt = float(files[0][2:7])
        deltat = float(files[numg][2:7]) - lowestt
        highestt = float(files[-1][2:7])
        numt = (highestt - lowestt) / deltat + 1.
        gridt = np.linspace(lowestt,highestt,num=numt)
    n = 0.
#Read in model. There are 33 lines of header information before the data starts. This first read in part of the header to get the T_eff and log_g. So reading in the file twice actually.
# We do not have to integrate over mu because these files already have
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
#interp = interp1d(shortlambdas,shortinten,kind='cubic')
            interp = InterpolatedUnivariateSpline(shortlambdas,shortinten,k=1)
            #interp2 = InterpolatedUnivariateSpline(shortlambdas,shortinten,k=2)
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
        gauss = (1./(sig * np.sqrt(2. * np.pi))) * np.exp(-(gx-18.)**2./(2.*sig**2))

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
        length = len(intflux) - 360
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
#interp2 = interp1d(clambda,cflux,kind='cubic')
        interp2 = InterpolatedUnivariateSpline(clambda,cflux,k=1)
        cflux2 = interp2(lambdarange)
        #plt.clf()
        #plt.plot(lambdarange,cflux2,'b+')
        #plt.show()
        print 'Done with the interpolation of the convolved flux'

#Fit a line to the endpoints and normalize
#Must do this for each line separately
#First set pixel ranges
        ##################
        #For beta through H9
        ##################
        #blow = lambdaindex[10]
        #bhi = lambdaindex[11]
        #glow = lambdaindex[8]
        #ghi = lambdaindex[9]
        #dlow = lambdaindex[6]
        #dhi = lambdaindex[7]
        #elow = lambdaindex[4]
        #ehi = lambdaindex[5]
        #H8low = lambdaindex[2]
        #H8hi = lambdaindex[3]
        #H9low = lambdaindex[0]
        #H9hi = lambdaindex[1]

        ####################
        #For beta through H8
        ####################
        #blow = lambdaindex[8]
        #bhi = lambdaindex[9]
        #glow = lambdaindex[6]
        #ghi = lambdaindex[7]
        #dlow = lambdaindex[4]
        #dhi = lambdaindex[5]
        #elow = lambdaindex[2]
        #ehi = lambdaindex[3]
        #H8low = lambdaindex[0]
        #H8hi = lambdaindex[1]

        ####################
        #For alpha through H10
        ####################
        afitlow = lambdaindex[0]
        afithi = lambdaindex[1]
        alow = lambdaindex[2]
        ahi = lambdaindex[3]
        bfitlow = lambdaindex[4]
        bfithi = lambdaindex[5]
        blow = lambdaindex[6]
        bhi = lambdaindex[7]
        gfitlow = lambdaindex[8]
        gfithi = lambdaindex[9]
        glow = lambdaindex[10]
        ghi = lambdaindex[11]
        hlow = lambdaindex[12]
        hhi = lambdaindex[13]
        dlow = lambdaindex[14]
        dhi = lambdaindex[15]
        elow = lambdaindex[16]
        ehi = lambdaindex[17]
        H8low = lambdaindex[18]
        H8hi = lambdaindex[19]
        H9low = lambdaindex[20]
        H9hi = lambdaindex[21]
        H10low = lambdaindex[22]
        H10hi = lambdaindex[23]

        ####################
        #For alpha through H9
        ####################
        #alow = lambdaindex[12]
        #ahi = lambdaindex[13]
        #blow = lambdaindex[10]
        #bhi = lambdaindex[11]
        #glow = lambdaindex[8]
        #ghi = lambdaindex[9]
        #dlow = lambdaindex[6]
        #dhi = lambdaindex[7]
        #elow = lambdaindex[4]
        #ehi = lambdaindex[5]
        #H8low = lambdaindex[2]
        #H8hi = lambdaindex[3]
        #H9low = lambdaindex[0]
        #H9hi = lambdaindex[1]


        ####################
        #For alpha through H8
        ####################
        #alow = lambdaindex[10]
        #ahi = lambdaindex[11]
        #blow = lambdaindex[8]
        #bhi = lambdaindex[9]
        #glow = lambdaindex[6]
        #ghi = lambdaindex[7]
        #dlow = lambdaindex[4]
        #dhi = lambdaindex[5]
        #elow = lambdaindex[2]
        #ehi = lambdaindex[3]
        #H8low = lambdaindex[0]
        #H8hi = lambdaindex[1]

        ##################
        #Use mpfit to fit pseudo-gaussian to models.
        #First set up the estimates
        #aest = [5.4e15,-6.21e11,0.,-6.07e14,6562.8,700.,0.8]
        #best = [1.29e16,-1.96e12,0.,-2.04e15,4.861e3,407.,0.84]
        #gest = [1.35e16,-2.07e12,0.,-3.03e15,4340.9,313.9,0.94]
        #hest = [-9.6e15,3.4e12,0.,-3.9e15,4102.9,835.,0.8,-2.7e15,3971.6,623.,1.01,-2.2e15,3891.5,205.,1.23,-1.54e15,3835.1,100.,1.,-9.13e14,3797.9,80.,1.] #Through H10

        #Choose the initial guesses in a smart AND consistent manner
        aest = np.zeros(7)
        xes = np.array([lambdarange[alow],lambdarange[alow+10],lambdarange[ahi-10],lambdarange[ahi]])
        yes = np.array([cflux2[alow],cflux2[alow+10],cflux2[ahi-10],cflux2[ahi]])
        ap = np.polyfit(xes,yes,2)
        aest[0] = ap[2]
        aest[1] = ap[1]
        aest[2] = ap[0]

        #plt.clf()
        #f = np.poly1d(p)
        #smooth = f(alllambda[alow:ahi+1])
        #plt.plot(alllambda[alow:ahi+1],cflux2[alow:ahi+1],'b')
        #plt.plot(alllambda[alow:ahi+1],smooth)
        #plt.show()

        aest[3] = np.min(cflux2[alow:ahi+1]) - (aest[2]*6562.8**2.+aest[1]*6562.8+aest[0]) #depth of line relative to continuum
        aest[4] = 6562.79 #rest wavelength of H alpha
        aest[5] = 600. #NEED TO CHECK THIS
        aest[6] = 1. #how much of a pseudo-gaussian

        best = np.zeros(7)
        xes = np.array([lambdarange[blow],lambdarange[blow+10],lambdarange[bhi-10],lambdarange[bhi]])
        yes = np.array([cflux2[blow],cflux2[blow+10],cflux2[bhi-10],cflux2[bhi]])
        bp = np.polyfit(xes,yes,2)
        best[0] = bp[2]
        best[1] = bp[1]
        best[2] = bp[0]
        best[3] = np.min(cflux2[blow:bhi+1]) - (best[2]*4862.71**2.+best[1]*4862.71+best[0]) #depth of line relative to continuum
        best[4] = 4862.710 #rest wavelength of H beta
        best[5] = 400. #NEED TO CHECK THIS
        best[6] = 1. 

        gest = np.zeros(7)
        #points are alllambda[glow], cflux2[glow]
        #           alllambda[glow+10], cflux2[glow+10]
        #           alllambda[ghi-10], cflux2[ghi-10]
        #           alllambda[ghi], cflux2[ghi]
        xes = np.array([lambdarange[glow],lambdarange[glow+10],lambdarange[ghi-10],lambdarange[ghi]])
        yes = np.array([cflux2[glow],cflux2[glow+10],cflux2[ghi-10],cflux2[ghi]])
        gp = np.polyfit(xes,yes,2)
        gest[0] = gp[2]
        gest[1] = gp[1]
        gest[2] = gp[0]
        gest[3] = np.min(cflux2[glow:ghi+1]) - (gest[2]*4341.692**2.+gest[1]*4341.692+gest[0]) #depth of line relative to continuum
        gest[4] = 4341.692 #rest wavelength of H beta
        gest[5] = 400. #NEED TO CHECK THIS
        gest[6] = 1. 

        hest = np.zeros(23)
        #Use three points to fit parabola over continuum. The above is just a line, parabola gives more consistent fit.
        #points are alllambda[H10low], cflux2[H10low]
        #           alllambda[elow], cflux2[elow]
        #           alllambda[dhi], cflux2[dhi]
        xes = np.array([lambdarange[H10low],lambdarange[elow],lambdarange[dhi]])
        yes = np.array([cflux2[H10low],cflux2[elow],cflux2[dhi]])
        hp = np.polyfit(xes,yes,2)
        hest[0] = hp[2]
        hest[1] = hp[1]
        hest[2] = hp[0]
        #plt.clf()
        #plt.plot(alllambda[H10low:dhi+1],cflux2[H10low:dhi+1],'b')
        #plt.plot(alllambda[H10low:dhi+1],x[0]*alllambda[H10low:dhi+1]**2.+x[1]*alllambda[H10low:dhi+1]+x[2],'k+',markersize=5)
        #plt.show()
        #Now delta
        hest[3] = np.min(cflux2[dlow:dhi+1]) - (hest[2]*4102.892**2.+hest[1]*4102.892+hest[0]) #depth of line relative to continuum
        hest[4] = 4102.892 #rest wavelength of H delta
        hest[5] = 300. #NEED TO CHECK THIS
        hest[6] = 1. #how much of a pseudo-gaussian
        #Now epsilon
        hest[7] = np.min(cflux2[elow:ehi+1]) - (hest[2]*3971.198**2.+hest[1]*3971.198+hest[0]) #depth of line relative to continuum
        hest[8] = 3971.198 #rest wavelength of H epsilon
        hest[9] = 200. #NEED TO CHECK THIS
        hest[10] = 1. #how much of a pseudo-gaussian
        #Now H8
        hest[11] = np.min(cflux2[H8low:H8hi+1]) - (hest[2]*3890.166**2.+hest[1]*3890.166+hest[0]) #depth of line relative to continuum
        hest[12] = 3890.166 #rest wavelength of H8
        hest[13] = 200. #NEED TO CHECK THIS
        hest[14] = 1. #how much of a pseudo-gaussian
        #Now H9
        hest[15] = np.min(cflux2[H9low:H9hi+1]) - (hest[2]*3836.485**2.+hest[1]*3836.485+hest[0]) #depth of line relative to continuum
        hest[16] = 3836.485 #rest wavelength of H9
        hest[17] = 200. #NEED TO CHECK THIS
        hest[18] = 1. #how much of a pseudo-gaussian
        #Now H10
        hest[19] = np.min(cflux2[H10low:H10hi+1]) - (hest[2]*3797.909**2.+hest[1]*3797.909+hest[0]) #depth of line relative to continuum
        hest[20] = 3797.909 #rest wavelength of H10
        hest[21] = 200. #NEED TO CHECK THIS
        hest[22] = 1. #how much of a pseudo-gaussian


        #Fit H alpha
        #mpfit terminates when the relative error is at most xtol. Since the first parameter on these fits is really large, this has to be a smaller number to get mpfit to actually iterate. 
        alambdas = lambdarange[afitlow:afithi+1.]
        alphaval = cflux2[afitlow:afithi+1.]
        asigmas = np.ones(len(alphaval))
        afa = {'x':alambdas, 'y':alphaval, 'err':asigmas}
        paralpha = [{'step':0.} for i in range(7)]
        paralpha[0]['step'] = 1e14
        aparams = mpfit.mpfit(fitpseudogauss,aest,functkw=afa,maxiter=3000,ftol=1e-16,parinfo=paralpha) #might want to specify xtol=1e-14 or so too
        alphafit = pseudogauss(alambdas,aparams.params)

        #print aest
        #print filename
        #print aest[0] - aparams.params[0]
        #print aparams.status
        #plt.clf()
        #plt.plot(alambdas,alphaval,'b')
        #plt.plot(alambdas,pseudogauss(alambdas,aparams.params),'g')
        #plt.plot(alambdas,pseudogauss(alambdas,aest),'k')
        #plt.show()


        #Fit H beta
        blambdas = lambdarange[bfitlow:bfithi+1.]
        betaval = cflux2[bfitlow:bfithi+1.]
        bsigmas = np.ones(len(betaval))
        bfa = {'x':blambdas,'y':betaval,'err':bsigmas}
        bparams = mpfit.mpfit(fitpseudogauss,best,functkw=bfa,maxiter=3000,ftol=1e-16)
        betafit = pseudogauss(blambdas,bparams.params)
        #print filename
        #print best[0] - bparams.params[0]
        #print bparams.status
        #plt.clf()
        #plt.plot(blambdas,betaval,'b')
        #plt.plot(blambdas,pseudogauss(blambdas,bparams.params),'g')
        #plt.plot(blambdas,pseudogauss(blambdas,best),'k')
        #plt.show()


        #Fit H gamma
        glambdas = lambdarange[gfitlow:gfithi+1.]
        gamval = cflux2[gfitlow:gfithi+1.]
        gsigmas = np.ones(len(gamval))
        gfa = {'x':glambdas,'y':gamval,'err':gsigmas}
        gparams = mpfit.mpfit(fitpseudogauss,gest,functkw=gfa,maxiter=3000,ftol=1e-16)
        gamfit = pseudogauss(glambdas,gparams.params)
        #print filename
        #print gest[0] - gparams.params[0]
        #print gparams.status
        #plt.clf()
        #plt.plot(glambdas,gamval,'b')
        #plt.plot(glambdas,pseudogauss(glambdas,gparams.params),'g')
        #plt.plot(glambdas,pseudogauss(glambdas,gest),'k')
        #plt.show()

        #Fit higher order lines
        hlambdas = lambdarange[H10low:dhi+1.]
        hval = cflux2[H10low:dhi+1.]
        hsigmas = np.ones(len(hval))
        hfa = {'x':hlambdas,'y':hval,'err':hsigmas}
        hparams = mpfit.mpfit(multifitpseudogauss,hest,functkw=hfa,maxiter=3000,ftol=1e-16)
        hfit = multipseudogauss(hlambdas,hparams.params)
        #print filename
        #plt.clf()
        #plt.plot(hlambdas,hval,'b')
        #plt.plot(hlambdas,multipseudogauss(hlambdas,hparams.params),'g')
        #plt.plot(hlambdas,multipseudogauss(hlambdas,hest),'k')
        #plt.show()


        #Normalize the lines. The first option uses the raw models to normalize to itself. The second option use the pseudogaussian fits to the models to normalize.
        #First H10
        H10normlow = np.min(np.where(hlambdas > 3785))
        H10normhi = np.min(np.where(hlambdas > 3815))
        H10slope = (hfit[H10normhi] - hfit[H10normlow]) / (hlambdas[H10normhi] - hlambdas[H10normlow])
        H10nline = H10slope * (hlambdas[H10normlow:H10normhi+1] - hlambdas[H10normlow]) + hfit[H10normlow]
        H10nflux = cflux2[H10low:H10hi+1] / H10nline

        #First H9
        #H9slope = (cflux2[H9hi] - cflux2[H9low]) / (alllambda[H9hi] - alllambda[H9low])
        #H9nline = H9slope * (alllambda[H9low:H9hi+1.] - alllambda[H9low]) + cflux2[H9low]
        #H9nflux = cflux2[H9low:H9hi+1.] / H9nline
        H9normlow = np.min(np.where(hlambdas > 3815))
        H9normhi = np.min(np.where(hlambdas > 3855))
        H9slope = (hfit[H9normhi] - hfit[H9normlow]) / (hlambdas[H9normhi] - hlambdas[H9normlow])
        H9nline = H9slope * (hlambdas[H9normlow:H9normhi+1] - hlambdas[H9normlow]) + hfit[H9normlow]
        H9nflux = cflux2[H9low:H9hi+1] / H9nline

        #Then H8
        #H8slope = (cflux2[H8hi] - cflux2[H8low]) / (alllambda[H8hi] - alllambda[H8low])
        #H8nline = H8slope * (alllambda[H8low:H8hi+1.] - alllambda[H8low]) + cflux2[H8low]
        #H8nflux = cflux2[H8low:H8hi+1.] / H8nline
        H8normlow = np.min(np.where(hlambdas > 3860))
        H8normhi = np.min(np.where(hlambdas > 3930))
        H8slope = (hfit[H8normhi] - hfit[H8normlow]) / (hlambdas[H8normhi] - hlambdas[H8normlow])
        H8nline = H8slope * (hlambdas[H8normlow:H8normhi+1] - hlambdas[H8normlow]) + hfit[H8normlow]
        H8nflux = cflux2[H8low:H8hi+1] / H8nline

        #Then H epsilon
        #eslope = (cflux2[ehi] - cflux2[elow]) / (alllambda[ehi] - alllambda[elow])
        #enline = eslope * (alllambda[elow:ehi+1.] - alllambda[elow]) + cflux2[elow]
        #enflux = cflux2[elow:ehi+1.] / enline
        enormhi = np.min(np.where(hlambdas > 4030))
        enormlow = np.min(np.where(hlambdas > 3930))
        eslope = (hfit[enormhi] - hfit[enormlow]) / (hlambdas[enormhi] - hlambdas[enormlow])
        enline = eslope * (hlambdas[enormlow:enormhi+1] - hlambdas[enormlow]) + hfit[enormlow]
        enflux = cflux2[elow:ehi+1] / enline

        #Then H delta
        #dslope = (cflux2[dhi] - cflux2[dlow]) / (alllambda[dhi] - alllambda[dlow])
        #dnline = dslope * (alllambda[dlow:dhi+1.] - alllambda[dlow]) + cflux2[dlow]
        #dnflux = cflux2[dlow:dhi+1.] / dnline
        dnormhi = np.min(np.where(hlambdas > 4191))
        dnormlow = np.min(np.where(hlambdas > 4040))
        dslope = (hfit[dnormhi] - hfit[dnormlow]) / (hlambdas[dnormhi] - hlambdas[dnormlow])
        dnline = dslope * (hlambdas[dnormlow:dnormhi+1] - hlambdas[dnormlow]) + hfit[dnormlow]
        dnflux = cflux2[dlow:dhi+1] / dnline
        

        #For H alpha and beta we fit to larger regions than we want to compare. So need to normalize
        #using our narrower region.
        
        #Now H gamma
        #gslope = (cflux2[ghi] - cflux2[glow]) / (alllambda[ghi] - alllambda[glow])
        #gnline = gslope * (alllambda[glow:ghi+1.] - alllambda[glow]) + cflux2[glow]
        #gnflux = cflux2[glow:ghi+1.] / gnline
        gnormlow = np.min(np.where(glambdas > 4220.))
        gnormhi = np.min(np.where(glambdas > 4490.))
        gslope = (gamfit[gnormhi] - gamfit[gnormlow]) / (glambdas[gnormhi] - glambdas[gnormlow])
        glambdasnew = glambdas[gnormlow:gnormhi+1]
        gnline = gslope * (glambdasnew - glambdas[gnormlow]) + gamfit[gnormlow]
        gnflux = cflux2[glow:ghi+1.] / gnline
    
        #Now H beta
        #bslope = (cflux2[bhi] - cflux2[blow]) / (alllambda[bhi] - alllambda[blow])
        #bnline = bslope * (alllambda[blow:bhi+1.] - alllambda[blow]) + cflux2[blow]
        #bnflux = cflux2[blow:bhi+1.] / bnline
        bnormlow = np.min(np.where(blambdas > 4710.))
        bnormhi = np.min(np.where(blambdas > 5010.))
        bslope = (betafit[bnormhi] - betafit[bnormlow]) / (blambdas[bnormhi] - blambdas[bnormlow])
        blambdasnew = blambdas[bnormlow:bnormhi+1]
        bnline = bslope * (blambdasnew - blambdas[bnormlow]) + betafit[bnormlow]
        bnflux = cflux2[blow:bhi+1.] / bnline

        #Now H alpha
        #aslope = (cflux2[ahi] - cflux2[alow]) / (alllambda[ahi] - alllambda[alow])
        #anline = aslope * (alllambda[alow:ahi+1.] - alllambda[alow]) + cflux2[alow]
        #anflux = cflux2[alow:ahi+1.] / anline
        anormlow = np.min(np.where(alambdas > 6413.))
        anormhi = np.min(np.where(alambdas > 6713.))
        aslope = (alphafit[anormhi] - alphafit[anormlow]) / (alambdas[anormhi] - alambdas[anormlow])
        alambdasnew = alambdas[anormlow:anormhi+1]
        anline = aslope * (alambdasnew - alambdas[anormlow]) + alphafit[anormlow]
        anflux = cflux2[alow:ahi+1.] / anline
        
        #plt.clf()
        #plt.plot(glambdasnew,gnflux)
        #plt.show()

        #Concatenate into one normalized array. If you want to exclude some regions (e.g. H10) this is where you should do that.
        ###Through H8
        #ncflux = np.concatenate((H8nflux,enflux,dnflux,gnflux,bnflux,anflux))
        #ncflux = np.concatenate((H8nflux,enflux,dnflux,gnflux,bnflux))
        #ncflux = np.concatenate((H9nflux,H8nflux,enflux,dnflux,gnflux,bnflux))
        #ncflux = np.concatenate((H9nflux,H8nflux,enflux,dnflux,gnflux,bnflux,anflux))
        ncflux = np.concatenate((H10nflux,H9nflux,H8nflux,enflux,dnflux,gnflux,bnflux,anflux))
        
        #plt.clf()
        #plt.plot(alllambda,ncflux,'b^')
        #plt.show()

        #newfilename = 'da_norm_' + str(logg) + '_' + str(teff) + '.txt'
        #np.savetxt(newfilename,np.transpose([alllambda,ncflux]))

        #Get the observed fluxes and sigma for each line so that we can compute chi square for each line individually
        obsalpha = allnline[indices[14]:+indices[15]+1]
        obsbeta = allnline[indices[12]:indices[13]+1]
        obsgamma = allnline[indices[10]:indices[11]+1]
        obsdelta = allnline[indices[8]:indices[9]+1]
        obsepsilon = allnline[indices[6]:indices[7]+1]
        obs8 = allnline[indices[4]:indices[5]+1]
        obs9 = allnline[indices[2]:indices[3]+1]
        obs10 = allnline[indices[0]:indices[1]+1]

        obsalphasig = allsigma[indices[14]:+indices[15]+1]
        obsbetasig = allsigma[indices[12]:indices[13]+1]
        obsgammasig = allsigma[indices[10]:indices[11]+1]
        obsdeltasig = allsigma[indices[8]:indices[9]+1]
        obsepsilonsig = allsigma[indices[6]:indices[7]+1]
        obs8sig = allsigma[indices[4]:indices[5]+1]
        obs9sig = allsigma[indices[2]:indices[3]+1]
        obs10sig = allsigma[indices[0]:indices[1]+1]
        '''
        obsalphasig = allsigma[alow:ahi+1]
        obsbetasig = allsigma[blow:bhi+1]
        obsgammasig = allsigma[glow:ghi+1]
        obsdeltasig = allsigma[dlow:dhi+1]
        obsepsilonsig = allsigma[elow:ehi+1]
        obs8sig = allsigma[H8low:H8hi+1]
        obs9sig = allsigma[H9low:H9hi+1]
        obs10sig = allsigma[H10low:H10hi+1]
        '''
        #Save interpolated and normalized model
        #intmodelname = 'da' + str(teff) + '_' + str(logg) + '_norm.txt'
        #np.savetxt(intmodelname,np.transpose([alllambda,ncflux]))

        #Calculate residuals and chi-square
        if n == 0:
            residual = np.empty(len(files))
            chisq = np.empty(len(files))
            allg = np.empty(len(files))
            allt = np.empty(len(files))
            chis = np.empty([numg,numt]) #logg by Teff #######
            chisalpha = np.empty([numg,numt])
            chisbeta = np.empty([numg,numt])
            chisgamma = np.empty([numg,numt])
            chisdelta = np.empty([numg,numt])
            chisepsilon = np.empty([numg,numt])
            chis8 = np.empty([numg,numt])
            chis9 = np.empty([numg,numt])
            chis10 = np.empty([numg,numt])
        residual[n] =  np.sum((allnline - ncflux)**2.,dtype='d')
        chisq[n] = np.sum(((allnline - ncflux) / allsigma)**2.,dtype='d')
        allg[n] = logg
        allt[n] = teff
        if case == 0:
            ng = ((logg/100.) - lowestg) / deltag
            nt = (teff - lowestt) / deltat
        if case == 1:
            ng = ((logg/1000.) - lowestg) / deltag
            nt = (teff - lowestt) / deltat
        chis[ng][nt] = chisq[n] #Save values in a matrix
        chisalpha[ng][nt] = np.sum(((obsalpha - anflux) / obsalphasig)**2.,dtype='d')
        chisbeta[ng][nt] = np.sum(((obsbeta -bnflux) / obsbetasig)**2.,dtype='d')
        chisgamma[ng][nt] = np.sum(((obsgamma - gnflux) / obsgammasig)**2.,dtype='d')
        chisdelta[ng][nt] = np.sum(((obsdelta - dnflux) / obsdeltasig)**2.,dtype='d')
        chisepsilon[ng][nt] = np.sum(((obsepsilon - enflux) / obsepsilonsig)**2.,dtype='d')
        chis8[ng][nt] = np.sum(((obs8 - H8nflux) / obs8sig)**2.,dtype='d')
        chis9[ng][nt] = np.sum(((obs9 - H9nflux) / obs9sig)**2.,dtype='d')
        chis10[ng][nt] = np.sum(((obs10 - H10nflux) / obs10sig)**2.,dtype='d')
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
    deltachi = np.subtract(chis,bestchi)
    plt.clf()
    v = np.array([1.0])
    plt.figure()
    plt.contour(gridt,gridg,deltachi,v)
    CS = plt.contour(gridt,gridg,deltachi,v)
    plt.clabel(CS)#,inline=1,fontsize=10)
    p = CS.collections[0].get_paths()[0]
    v = p.vertices #Get points for delta chi-square =1 surface
    ranget = np.array(v[:,0])
    rangeg = np.array(v[:,1])
    print 'Min and Max Teff are ',np.amin(ranget),' and ',np.amax(ranget)
    print 'Min and Max log(g) are ',np.amin(rangeg),' and ',np.amax(rangeg)
    uppert = np.amax(ranget)
    lowert = np.amin(ranget)
    upperg = np.amax(rangeg)
    lowerg = np.amin(rangeg)
    upperterr = uppert - bestT
    lowerterr = bestT - lowert
    Terr = (upperterr + lowerterr) / 2.
    uppergerr = upperg - (bestg/1000.)
    lowergerr = (bestg/1000.) - lowerg
    gerr = (uppergerr + lowergerr) / 2.
    #plt.show()

    #Save information on best fitting model and the convolved model itself
    f = open('fitting_solutions.txt','a')
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    if case == 0:
        bestmodelname = 'da' + str(int(bestT)) + '_' + str(int(bestg)) + '.dk'
    if case == 1:
        bestmodelname = 'da' + str(int(bestT)) + '_' + str(int(bestg)) + '.jf'
    info = zzcetiblue + '\t' + zzcetired + '\t' +  bestmodelname + '\t' + str(bestT) + '\t' +  str(Terr) + '\t' + str(bestg) + '\t' + str(gerr) + '\t' + str(bestchi) + '\t' + now
    f.write(info + '\n')
    f.close()
    #Now save the best convolved model and delta chi squared surface
    newmodel = 'model_' + zzcetiblue[4:zzcetiblue.find('_930_')] + '.txt' #NEED TO CHECK THIS TO MAKE SURE IT WORKS GENERALLY
    np.savetxt(newmodel,np.transpose([alllambda,bestmodel]))
    deltaname = 'deltachi_' + zzcetiblue[4:zzcetiblue.find('_930_')] + '.txt'
    np.savetxt(deltaname,deltachi)
    alphaname = 'chi_' + zzcetiblue[4:zzcetiblue.find('_930_')] + '_alpha.txt'
    np.savetxt(alphaname,chisalpha)
    betaname = 'chi_' + zzcetiblue[4:zzcetiblue.find('_930_')] + '_beta.txt'
    np.savetxt(betaname,chisbeta)
    gammaname = 'chi_' + zzcetiblue[4:zzcetiblue.find('_930_')] + '_gamma.txt'
    np.savetxt(gammaname,chisgamma)
    deltaname = 'chi_' + zzcetiblue[4:zzcetiblue.find('_930_')] + '_delta.txt'
    np.savetxt(deltaname,chisdelta)
    epsilonname = 'chi_' + zzcetiblue[4:zzcetiblue.find('_930_')] + '_epsilon.txt'
    np.savetxt(epsilonname,chisepsilon)
    H8name = 'chi_' + zzcetiblue[4:zzcetiblue.find('_930_')] + '_H8.txt'
    np.savetxt(H8name,chis8)
    H9name = 'chi_' + zzcetiblue[4:zzcetiblue.find('_930_')] + '_H9.txt'
    np.savetxt(H9name,chis9)
    H10name = 'chi_' + zzcetiblue[4:zzcetiblue.find('_930_')] + '_H10.txt'
    np.savetxt(H10name,chis10)


    print ''
    print 'Best chi-squared is ', bestchi
    if case == 0:
        print 'Best T_eff is ', bestT
        print 'Best log_g is ', bestg/100.
    if case == 1:
        print 'Best T_eff is ', bestT
        print 'Best log_g is ', bestg/1000.
        print 'Error on T_eff is + ',upperterr,' and - ',lowerterr
        print 'Error on log_g is + ',uppergerr,' and - ',lowergerr
    print 'Done running intspec.py'
    plt.clf()
    plt.plot(alllambda,allnline,'bs',label='Normalized data')
    plt.plot(alllambda,bestmodel,'r^',label='Model')
    plt.legend()
    #plt.show()
    return ncflux,int(bestT),int(bestg)





