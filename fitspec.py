# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 14:16:05 2015

@author: joshfuchs

To do:
- Enter spec names and fwhm in command line
- Clean up code and document better

Done:
- Save and plot chi-square surfaces so that we can fit them automatically.
- Save each individual chi-square surface for each line
- Make sure interpolated models go out through H-alpha
- Do we need to fit gamma to larger than the normalization range? Yes.
- need to fit models to region larger than the normalization
- automatically generate pseudo-gauss estimates for observed spectrum
- Read in FWHM so that models are convolved correctly. 
"""
#Based on fitspec.pro written by Bart Dunlap. Ported to python by Josh Fuchs
#At the end of this program, it calls intspec.py to fit to the model. If you move
#this file you need to move that file. Also, you will need to change the path
#listed at the bottom. This program uses MPFIT to fit a pseudogaussian to the balmer lines of interest. Therefore you need to have mpfit.py in the same directory. You want to have the version written by Sergei Koposov as that is the most recent and uses numpy.

import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf # Infierno doesn't support astropy for some reason so using pyfits
#import astropy.io.fits as pf
import mpfit
from intspec import intmodel
from finegrid import makefinegrid
import sys
import os

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

#Define a function that fits higher order lines simultaneously (delta and higher))
def multipseudogauss(x,p):
    #The model function with parameters p
    #return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.*p[5])))**p[6]) + p[7]*np.exp(-(np.abs(x-p[8])/(np.sqrt(2.*p[9])))**p[10]) + p[11]*np.exp(-(np.abs(x-p[12])/(np.sqrt(2.*p[13])))**p[14]) #This one includes Hdelta, Hepsilon, and H8
    #return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.*p[5])))**p[6]) + p[7]*np.exp(-(np.abs(x-p[8])/(np.sqrt(2.*p[9])))**p[10]) + p[11]*np.exp(-(np.abs(x-p[12])/(np.sqrt(2.*p[13])))**p[14]) + p[15]*np.exp(-(np.abs(x-p[16])/(np.sqrt(2.*p[17])))**p[18]) #This one includes Hdelta, Hepsilon, H8, and H9
    return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.*p[5])))**p[6]) + p[7]*np.exp(-(np.abs(x-p[8])/(np.sqrt(2.*p[9])))**p[10]) + p[11]*np.exp(-(np.abs(x-p[12])/(np.sqrt(2.*p[13])))**p[14]) + p[15]*np.exp(-(np.abs(x-p[16])/(np.sqrt(2.*p[17])))**p[18]) + p[19]*np.exp(-(np.abs(x-p[20])/(np.sqrt(2.*p[21])))**p[22]) #This one includes Hdelta, Hepsilon, H8, H9, and H10

def multifitpseudogauss(p,fjac=None,x=None, y=None, err=None):
    #Parameter values are passed in p
    #fjac=None just means partial derivatives will not be computed.
    model = multipseudogauss(x,p)
    status = 0
    return([status,(y-model)/err])

#Now we need to read in actual spectrum. This is for Goodman spectra.
#Eventually make this a command line parameters
#zzcetiblue = 'wnb.WD0122p0030_930_blue_fluxGD50.fits'
#zzcetired = 'wnb.WD0122p0030_930_red_fluxGD50.fits'
#FWHM = 4.4
zzcetiblue = 'wtfb.wd1425-811_930_blue_flux.ms.fits'
zzcetired = 'wtfb.wd1425-811_930_red_flux.ms.fits' 
FWHMpix = 6.1 #The FWHM in pixels

#Read in the blue spectrum
datalistblue = pf.open(zzcetiblue)
datavalblue = datalistblue[0].data[0,0,:] #Reads in the object spectrum,data[0,0,:] is optimally subtracted, data[1,0,:] is raw extraction,  data[2,0,:] is sky, data[3,0,:] is sigma spectrum
sigmavalblue = datalistblue[0].data[3,0,:] #Sigma spectrum
wav0blue = datalistblue[0].header['crval1']
deltawavblue = datalistblue[0].header['cd1_1']
lambdasblue = np.ones(len(datavalblue))
lambdasblue[0] = wav0blue
ivalblue = np.arange(1,len(datavalblue))

for i in ivalblue:
    lambdasblue[i] = lambdasblue[i-1] + deltawavblue

#Read in the red spectrum
datalistred = pf.open(zzcetired)
datavalred = datalistred[0].data[0,0,:] #data[0,0,:] is optimally extracted, data[2,0,:] is sky
sigmavalred = datalistred[0].data[3,0,:] #Sigma spectrum
wav0red = datalistred[0].header['crval1']
deltawavred = datalistred[0].header['cd1_1']
lambdasred = np.ones(len(datavalred))
lambdasred[0] = wav0red
ivalred = np.arange(1,len(datavalred))

for i in ivalred:
    lambdasred[i] = lambdasred[i-1] + deltawavred

FWHM = FWHMpix * deltawavblue #FWHM in Angstroms


#Concatenate both into two arrays
lambdas = np.concatenate((lambdasblue,lambdasred))
dataval = np.concatenate((datavalblue,datavalred))
sigmaval = 2.e-17 * np.ones(len(dataval))#np.concatenate((sigmavalblue,sigmavalred))
'''
#Read in text file
lambdas, dataval = np.genfromtxt('rawmodel.dat',unpack=True) #Files from Bart
lambdas, dataval, junk, junk2 = np.genfromtxt('WD0122+0030_SDSS.csv',skip_header=1,unpack=True,delimiter=",") #For reading in SDSS csv files
sigmaval = np.ones(len(lambdas))
'''

#plot the spectrum
#plt.clf()
#plt.plot(lambdas,sigmaval,label='data')
#plt.show()
#sys.exit()
#print len(dataval)
#print len(wav)

# This sets pixel range.
afitlow = np.min(np.where(lambdas > 6380.))
afithi = np.min(np.where(lambdas > 6760.))
alow = np.min(np.where(lambdas > 6413.))
ahi = np.min(np.where(lambdas > 6713.))

bfitlow = np.min(np.where(lambdas > 4680.))
bfithi = np.min(np.where(lambdas > 5040.))
blow = np.min(np.where(lambdas > 4710.))
bhi = np.min(np.where(lambdas > 5010.))

gfitlow = np.min(np.where(lambdas > 4200.))
gfithi = np.min(np.where(lambdas > 4510.))
glow = np.min(np.where(lambdas > 4220.))
ghi = np.min(np.where(lambdas > 4490.))


#hlow = np.min(np.where(lambdas > 3860.)) #For H8
hlow = np.min(np.where(lambdas > 3782.)) #Includes H10 
hhi = np.min(np.where(lambdas > 4195.)) #4191
dlow = np.min(np.where(lambdas > 4040.))
dhi = np.min(np.where(lambdas > 4191.))
elow = np.min(np.where(lambdas > 3930.))
ehi = np.min(np.where(lambdas > 4030.))
H8low = np.min(np.where(lambdas > 3860.))
H8hi = np.min(np.where(lambdas > 3930.)) #3930
H9low = np.min(np.where(lambdas > 3815.))
H9hi = np.min(np.where(lambdas > 3855.))
H10low = np.min(np.where(lambdas > 3785.))
H10hi = np.min(np.where(lambdas > 3815.))


#Gaussian estimates for parameters for fit
#aest = [9.9e-16,-1.12e-19,0.,-1.40e-16,6562.8,1200.,0.8]
#best = [-7.40e-15,4.23e-18,-5.05e-22,-8.14e-16,4.861e3,1207.1,0.84]#-5.05e-22
#gest = [-6.7e-15,4.3e-18,-5.5e-22,-1e-15,4340.9,713.9,0.94]#-5.5e-22
#hest = [-3.2e-13, 1.6e-16,-1.9e-20, -1.93-15, 4102.9,2835.2, 0.8,-1.2e-15, 3971.6,623.8, 1.01,-7.4e-16, 3891.5,205.4, 1.23] #Through H8
#hest = [-3.2e-13,1.6e-16,-1.5e-20,-1.93e-15,4102.9,2835.2,0.8,-1.2e-15,3971.6,623.8,1.01,-7.4e-16,3891.5,205.4,1.23,-6.0e-16,3835.1,100.,1.2,-3.0e-10,3797.7,80.,1.2] #Through H10 Doesn't fit well.
#hest = [-1.7e-11,8.84e-15,-1.11e-18,4.43e-14,4102.9,399.9,1.1,-1.47e-13,3971.6,199.9,1.11,1.28e-13,3891.5,149.9,1.2,-1.09e-13,3834.9,99.9,1.19,1.77e-13,3797.7,100.0,1.29] #Takes 656 iterations to complete, but fits well.
#hest = [-53.8,0.02,-3.3e-6,-0.7,4102.9,400.,1.1,-0.6,3971.6,200.,1.11,-0.3,3891.5,150.,1.2,-0.2,3835.,100.,1.2,-0.1,3797.7,100.,1.3] #Through H10 #Same as the model spectrum guess from below. Fits in three iterations decently.

#Make the estimates in a smart AND consistent manner. This matches what is done for the model fitting.

aest = np.zeros(7)
xes = np.array([lambdas[alow],lambdas[alow+10],lambdas[ahi-10],lambdas[ahi]])
yes = np.array([dataval[alow],dataval[alow+10],dataval[ahi-10],dataval[ahi]])
ap = np.polyfit(xes,yes,2)
aest[0] = ap[2]
aest[1] = ap[1]
aest[2] = ap[0]
aest[3] = np.min(dataval[alow:ahi+1]) - (aest[2]*6562.8**2.+aest[1]*6562.8+aest[0]) #depth of line relative to continuum
aest[4] = 6562.79 #rest wavelength of H alpha
aest[5] = 1200. #NEED TO CHECK THIS
aest[6] = 1. #how much of a pseudo-gaussian

best = np.zeros(7)
xes = np.array([lambdas[blow],lambdas[blow+10],lambdas[bhi-10],lambdas[bhi]])
yes = np.array([dataval[blow],dataval[blow+10],dataval[bhi-10],dataval[bhi]])
bp = np.polyfit(xes,yes,2)
best[0] = bp[2]
best[1] = bp[1]
best[2] = bp[0]
best[3] = np.min(dataval[blow:bhi+1]) - (best[2]*4862.71**2.+best[1]*4862.71+best[0]) #depth of line relative to continuum
best[4] = 4862.71 #rest wavelength of H beta
best[5] = 1200. #NEED TO CHECK THIS
best[6] = 1. #how much of a pseudo-gaussian

gest = np.zeros(7)
xes = np.array([lambdas[glow],lambdas[glow+10],lambdas[ghi-10],lambdas[ghi]])
yes = np.array([dataval[glow],dataval[glow+10],dataval[ghi-10],dataval[ghi]])
gp = np.polyfit(xes,yes,2)
gest[0] = gp[2]
gest[1] = gp[1]
gest[2] = gp[0]
gest[3] = np.min(dataval[glow:ghi+1]) - (gest[2]*4341.692**2.+gest[1]*4341.692+gest[0]) #depth of line relative to continuum
gest[4] = 4341.692 #rest wavelength of H gamma
gest[5] = 700. #NEED TO CHECK THIS
gest[6] = 1. #how much of a pseudo-gaussian

hest = np.zeros(23)
xes = np.array([lambdas[H10low],lambdas[elow],lambdas[dhi]])
yes = np.array([dataval[H10low],dataval[elow],dataval[dhi]])
hp = np.polyfit(xes,yes,2)
hest[0] = hp[2]
hest[1] = hp[1]
hest[2] = hp[0]
#Now delta
hest[3] = np.min(dataval[dlow:dhi+1]) - (hest[2]*4102.892**2.+hest[1]*4102.892+hest[0]) #depth of line relative to continuum
hest[4] = 4102.892 #rest wavelength of H delta
hest[5] = 400. #NEED TO CHECK THIS
hest[6] = 1.1 #how much of a pseudo-gaussian
#Now epsilon
hest[7] = np.min(dataval[elow:ehi+1]) - (hest[2]*3971.198**2.+hest[1]*3971.198+hest[0]) #depth of line relative to continuum
hest[8] = 3971.198 #rest wavelength of H epsilon
hest[9] = 200. #NEED TO CHECK THIS
hest[10] = 1.1 #how much of a pseudo-gaussian
#Now H8
hest[11] = np.min(dataval[H8low:H8hi+1]) - (hest[2]*3890.166**2.+hest[1]*3890.166+hest[0]) #depth of line relative to continuum
hest[12] = 3890.166 #rest wavelength of H8
hest[13] = 150. #NEED TO CHECK THIS
hest[14] = 1.2 #how much of a pseudo-gaussian
#Now H9
hest[15] = np.min(dataval[H9low:H9hi+1]) - (hest[2]*3836.485**2.+hest[1]*3836.485+hest[0]) #depth of line relative to continuum
hest[16] = 3836.485 #rest wavelength of H9
hest[17] = 100. #NEED TO CHECK THIS
hest[18] = 1.2 #how much of a pseudo-gaussian
#Now H10
hest[19] = np.min(dataval[H10low:H10hi+1]) - (hest[2]*3797.909**2.+hest[1]*3797.909+hest[0]) #depth of line relative to continuum
hest[20] = 3797.909 #rest wavelength of H10
hest[21] = 100. #NEED TO CHECK THIS
hest[22] = 1.3 #how much of a pseudo-gaussian


'''
#These are for sample spectra from Bart.
best = [1.,-0.2,0.,-0.5,4861.,1180.,0.8]
gest = [1.1,-0.2,0.,-0.6,4340.9,700.,0.9]
#hest = [-53.8,0.02,-3.3e-6,-0.7,4102.9,400.,1.1,-0.6,3971.6,200.,1.11,-0.3,3891.5,150.,1.2] #For Hdelta through H8
#hest = [-53.8,0.02,-3.3e-6,-0.7,4102.9,400.,1.1,-0.6,3971.6,200.,1.11,-0.3,3891.5,150.,1.2,-0.2,3835.,100.,1.2] #Through H9 -3.3e-6
hest = [-53.8,0.02,-3.3e-6,-0.7,4102.9,400.,1.1,-0.6,3971.6,200.,1.11,-0.3,3891.5,150.,1.2,-0.2,3835.,100.,1.2,-0.1,3797.7,100.,1.3] #Through H10

#For fitting raw models
aest = [9.9e14,-1.12e12,0.,-1.0e15,6562.8,1200.,0.8]
aest = [1.02e15,1.05e12,-1.4e8,-1.13e15,6562.8,1100.,0.8]
best = [2.17e15,2.7e12,-5.2e8,-3.49e15,4.861e3,1007.1,0.84]
'''
 
#For an SDSS spectrum for testing. Not the flux values are in units of 10^-17 erg
#aest = [-200.,0.069,-85.71e-06,-5.5,6563.3,1454.,0.76]
#best = [-1969.,0.84,-8.8e-05,-21.,4861.3,1837.,0.8]
#gest = [3430.,-1.54,1.7e-4,-20.,4340.9,427.,1.0]
#hest = [-25010.,12.56,-1.5e-3,-467.1,4102.9,1.38e6,0.8,-47.9,3971.6,1892.2,1.01,-25.0,3891.5,398.3,1.14]   # Through H8

#Fit alpha
#To fit line for continuum, fix parameter two to zero.
paralpha = [{'fixed':0} for i in range(7)] #7 total parameters
#paralpha[2]['fixed'] = 1

#alambdas = lambdas[alow:ahi+1]
#asigmas = sigmaval[alow:ahi+1]
#alphaval = dataval[alow:ahi+1]
alambdas = lambdas[afitlow:afithi+1]
asigmas = sigmaval[afitlow:afithi+1]
alphaval = dataval[afitlow:afithi+1]

afa = {'x':alambdas, 'y':alphaval, 'err':asigmas}
aparams = mpfit.mpfit(fitpseudogauss,aest,functkw=afa,maxiter=5000,ftol=1e-16,parinfo=paralpha)

acenter = aparams.params[4]
alphafit = pseudogauss(alambdas,aparams.params)

#plt.plot(alambdas,alphaval,'b')
#plt.plot(alambdas,alphafit,'g')
#plt.plot(alambdas,pseudogauss(alambdas,aest),'k')
#plt.plot(alambdas,aparams.params[0]*1. + aparams.params[1]*alambdas +aparams.params[2]*alambdas**2.)
#plt.show()
#sys.exit()

#Fit beta
#blambdas = lambdas[blow:bhi+1]
#bsigmas = sigmaval[blow:bhi+1]
#betaval = dataval[blow:bhi+1]
blambdas = lambdas[bfitlow:bfithi+1]
bsigmas = sigmaval[bfitlow:bfithi+1]
betaval = dataval[bfitlow:bfithi+1]

bfa = {'x':blambdas, 'y':betaval, 'err':bsigmas}
bparams = mpfit.mpfit(fitpseudogauss,best,functkw=bfa,maxiter=5000,ftol=1e-16)

bcenter = bparams.params[4]
betafit = pseudogauss(blambdas,bparams.params)

#plt.clf()
#plt.plot(blambdas,betaval,label='data')
#plt.plot(blambdas,betafit,label='fit')
#plt.plot(blambdas,bparams.params[0]*1. + bparams.params[1]*blambdas+bparams.params[2]*blambdas**2.)
#plt.show()
#sys.exit()


#Fit gamma
glambdas = lambdas[gfitlow:gfithi+1]
gsigmas = sigmaval[gfitlow:gfithi+1]
gamval = dataval[gfitlow:gfithi+1]
gfa = {'x':glambdas, 'y':gamval, 'err':gsigmas}
gparams = mpfit.mpfit(fitpseudogauss,gest,functkw=gfa,maxiter=5000,ftol=1e-16)

gcenter = gparams.params[4]
gamfit = pseudogauss(glambdas,gparams.params)
#plt.clf()
#plt.plot(glambdas,gamval,label='data')
#plt.plot(glambdas,gamfit,label='fit')
#plt.plot(glambdas,gparams.params[0]*1. + gparams.params[1]*glambdas+gparams.params[2]*glambdas**2.)
#plt.show()
#sys.exit()


#To fit line for continuum, fix parameter two to zero.
parhigh = [{'fixed':0} for i in range(23)] #23 total parameters
#parhigh[2]['fixed'] = 1
#parhigh[1]['fixed'] = 1
#parhigh[0]['fixed'] = 1

#Fit higher order lines
hlambdas = lambdas[hlow:hhi+1]
dlambdas = lambdas[dlow:dhi+1]
elambdas = lambdas[elow:ehi+1]
H8lambdas = lambdas[H8low:H8hi+1]
H9lambdas = lambdas[H9low:H9hi+1]
H10lambdas = lambdas[H10low:H10hi+1]
hsigmas = sigmaval[hlow:hhi+1]
hval = dataval[hlow:hhi+1]
hfa = {'x':hlambdas, 'y':hval, 'err':hsigmas}
hparams = mpfit.mpfit(multifitpseudogauss,hest,functkw=hfa,maxiter=5000,ftol=1e-12,parinfo=parhigh) #Might want to check this ftol

hfit = multipseudogauss(hlambdas,hparams.params)
dcenter = hparams.params[4]
ecenter = hparams.params[8]
H8center = hparams.params[12]
H9center = hparams.params[16]
H10center = hparams.params[20]

#plt.clf()
#plt.plot(hlambdas,hval,label='data')
#plt.plot(hlambdas,hparams.params[0]*1. + hparams.params[1]*hlambdas + hparams.params[2]*hlambdas**2.)
#plt.plot(hlambdas,hfit,'r',label='fit')
#plt.show()
#sys.exit()

#Fit a line to the fit points  from each end
#Note the line needs to be linear in lambda not in pixel
#bnline is the normalized spectral line. Continuum set to one.

#Start with alpha
#Set the center of the line to the wavelength in vacuum. Models are in vacuum wavelengths.
alambdas = alambdas - (acenter- 6564.60)
#aslope = (alphafit[-1] - alphafit[0] ) / (alambdas[-1] - alambdas[0])
#ali = aslope * (alambdas - alambdas[0]) + alphafit[0]
#anline = dataval[alow:ahi+1] / ali
#asigma = sigmaval[alow:ahi+1] / ali

anormlow = np.min(np.where(alambdas > 6413.))
anormhi = np.min(np.where(alambdas > 6713.))
aslope = (alphafit[anormhi] - alphafit[anormlow] ) / (alambdas[anormhi] - alambdas[anormlow])
alambdasnew = alambdas[anormlow:anormhi+1]
alphavalnew = alphaval[anormlow:anormhi+1]
asigmasnew = asigmas[anormlow:anormhi+1]
ali = aslope * (alambdasnew - alambdas[anormlow]) + alphafit[anormlow]
anline = alphavalnew / ali
asigma = asigmasnew / ali


#plt.clf()
#plt.plot(alambdasnew,anline)
#plt.show()
#sys.exit()

#Now to beta
#Set the center of the line to the wavelength in vacuum. Models are in vacuum wavelengths.
blambdas = blambdas - (bcenter-4862.710)
#bslope = (betafit[-1] - betafit[0] ) / (blambdas[-1] - blambdas[0])
#bli = bslope * (blambdas - blambdas[0]) + betafit[0]
#bnline = dataval[blow:bhi+1] / bli
#bsigma =  sigmaval[blow:bhi+1] / bli

bnormlow = np.min(np.where(blambdas > 4710.))
bnormhi = np.min(np.where(blambdas > 5010.))
bslope = (betafit[bnormhi] - betafit[bnormlow] ) / (blambdas[bnormhi] - blambdas[bnormlow])
blambdasnew = blambdas[bnormlow:bnormhi+1]
betavalnew = betaval[bnormlow:bnormhi+1]
bsigmasnew = bsigmas[bnormlow:bnormhi+1]
bli = bslope * (blambdasnew - blambdas[bnormlow]) + betafit[bnormlow]
bnline = betavalnew / bli
bsigma = bsigmasnew / bli


#plt.clf()
#plt.plot(blambdas,bnline)
#plt.plot(blambdasnew,bnline)
#plt.show()
#sys.exit()

#Now do gamma
#gnline is the normalized spectral line. Continuum set to one.
#Set the center of the line to the wavelength in vacuum
glambdas = glambdas - (gcenter-4341.692)
gnormlow = np.min(np.where(glambdas > 4220.))
gnormhi = np.min(np.where(glambdas > 4490.))
gslope = (gamfit[gnormhi] - gamfit[gnormlow] ) / (glambdas[gnormhi] - glambdas[gnormlow])
glambdasnew = glambdas[gnormlow:gnormhi+1]
gamvalnew = gamval[gnormlow:gnormhi+1]
gsigmasnew = gsigmas[gnormlow:gnormhi+1]
gli = gslope * (glambdasnew - glambdas[gnormlow]) + gamfit[gnormlow]
gnline = gamvalnew / gli
gsigma = gsigmasnew / gli

#plt.clf()
#plt.plot(glambdas,gnline)
#plt.show()
#sys.exit()

#Now normalize the higher order lines (delta, epsilon, H8)
hlambdastemp = hlambdas - (dcenter-4102.892)
#dlambdas = dlambdas- (dcenter-4102.892)
dnormlow = np.min(np.where(hlambdastemp > 4040.))
dnormhi = np.min(np.where(hlambdastemp > 4191.))
dlambdas = hlambdastemp[dnormlow:dnormhi+1]
#dslope = (hfit[np.min(np.where(hlambdas > 4191.))] - hfit[np.min(np.where(hlambdas > 4040.))] ) / (lambdas[dhi] - lambdas[dlow])
dslope = (hfit[dnormhi] - hfit[dnormlow]) / (hlambdastemp[dnormhi] - hlambdastemp[dnormlow])
#dli = dslope * (dlambdas - dlambdas[0]) + hfit[np.min(np.where(hlambdas > 4040.))]
dli = dslope * (dlambdas - dlambdas[0]) + hfit[dnormlow]
#dvaltemp = dataval[dlow:dhi+1]
#dsigtemp = sigmaval[dlow:dhi+1]
dvaltemp = dataval[hlow:hhi+1]
dsigtemp = sigmaval[hlow:hhi+1]
dnline = dvaltemp[dnormlow:dnormhi+1] / dli
dsigma = dsigtemp[dnormlow:dnormhi+1] / dli


#elambdas = elambdas - (ecenter-3971.198)
hlambdastemp = hlambdas - (ecenter-3971.198)
enormlow = np.min(np.where(hlambdastemp > 3930.))
enormhi = np.min(np.where(hlambdastemp > 4030.))
elambdas = hlambdastemp[enormlow:enormhi+1]
#eslope = (hfit[np.min(np.where(hlambdas > 4030.))] - hfit[np.min(np.where(hlambdas > 3930.))] ) / (lambdas[ehi] - lambdas[elow])
#eli = eslope * (elambdas - elambdas[0]) + hfit[np.min(np.where(hlambdas > 3930.))]
eslope = (hfit[enormhi] - hfit[enormlow] ) / (hlambdastemp[enormhi] - hlambdastemp[enormlow])
eli = eslope * (elambdas - elambdas[0]) + hfit[enormlow]
#evaltemp = dataval[elow:ehi+1]
#esigtemp = sigmaval[elow:ehi+1]
evaltemp = dataval[hlow:hhi+1]
esigtemp = sigmaval[hlow:hhi+1]
enline = evaltemp[enormlow:enormhi+1] / eli
esigma = esigtemp[enormlow:enormhi+1] / eli


#H8lambdas = H8lambdas - (H8center-3890.166)
hlambdastemp = hlambdas - (H8center-3890.166)
H8normlow = np.min(np.where(hlambdastemp > 3860.))
H8normhi = np.min(np.where(hlambdastemp > 3930.))
H8lambdas = hlambdastemp[H8normlow:H8normhi+1]
#H8slope = (hfit[np.min(np.where(hlambdas > 3930.))] - hfit[np.min(np.where(hlambdas > 3860.))] ) / (lambdas[H8hi] - lambdas[H8low])
#H8li = H8slope * (H8lambdas - H8lambdas[0]) + hfit[np.min(np.where(hlambdas > 3860.))]
H8slope = (hfit[H8normhi] - hfit[H8normlow] ) / (hlambdastemp[H8normhi] - hlambdastemp[H8normlow])
H8li = H8slope * (H8lambdas - H8lambdas[0]) + hfit[H8normlow]
H8valtemp = dataval[hlow:hhi+1]
H8sigtemp = sigmaval[hlow:hhi+1]
#H8valtemp = dataval[H8low:H8hi+1]
#H8sigtemp = sigmaval[H8low:H8hi+1]
H8nline = H8valtemp[H8normlow:H8normhi+1] / H8li
H8sigma = H8sigtemp[H8normlow:H8normhi+1] / H8li


### To normalize, using points from end of region since it is so small.
#H9lambdas = H9lambdas - (H9center- 3836.485)
hlambdastemp = hlambdas - (H9center- 3836.485)
H9normlow = np.min(np.where(hlambdastemp > 3815.))
H9normhi = np.min(np.where(hlambdastemp > 3855.))
H9lambdas = hlambdastemp[H9normlow:H9normhi+1]
#H9slope = (hfit[np.min(np.where(hlambdas > 3855.))] - hfit[np.min(np.where(hlambdas > 3815.))] ) / (lambdas[H9hi] - lambdas[H9low])
#H9li = H9slope * (H9lambdas - H9lambdas[0]) + hfit[np.min(np.where(hlambdas > 3815.))]
H9slope = (hfit[H9normhi] - hfit[H9normlow] ) / (hlambdastemp[H9normhi] - hlambdastemp[H9normlow])
H9li = H9slope * (H9lambdas - H9lambdas[0]) + hfit[H9normlow]
H9valtemp = dataval[hlow:hhi+1]
H9sigtemp = sigmaval[hlow:hhi+1]

#H9valtemp = dataval[H9low:H9hi+1]
#H9sigtemp = sigmaval[H9low:H9hi+1]
H9nline = H9valtemp[H9normlow:H9normhi+1] / H9li
H9sigma = H9sigtemp[H9normlow:H9normhi+1] / H9li


#H10lambdas = H10lambdas - (H10center-3797.909)
hlambdastemp = hlambdas - (H10center-3797.909)
H10normlow = np.min(np.where(hlambdastemp > 3785.))
H10normhi = np.min(np.where(hlambdastemp > 3815.))
H10lambdas = hlambdastemp[H10normlow:H10normhi+1]
#H10slope = (hfit[np.min(np.where(hlambdas > 3815.))] - hfit[np.min(np.where(hlambdas > 3785.))] ) / (lambdas[H10hi] - lambdas[H10low])
#H10li = H10slope * (H10lambdas - H10lambdas[0]) + hfit[np.min(np.where(hlambdas > 3785.))]
H10slope = (hfit[H10normhi] - hfit[H10normlow] ) / (hlambdastemp[H10normhi] - hlambdastemp[H10normlow])
H10li = H10slope * (H10lambdas - H10lambdas[0]) + hfit[H10normlow]
H10valtemp = dataval[hlow:hhi+1]
H10sigtemp = sigmaval[hlow:hhi+1]

#H10valtemp = dataval[H10low:H10hi+1]
#H10sigtemp = sigmaval[H10low:H10hi+1]
H10nline = H10valtemp[H10normlow:H10normhi+1] / H10li
H10sigma = H10sigtemp[H10normlow:H10normhi+1] / H10li



#plt.clf()
#plt.plot(dlambdas,dnline)
#plt.show()
#plt.clf()
#plt.plot(elambdas,enline)
#plt.show()
#plt.clf()
#plt.plot(dlambdas,dataval[dlow:dhi+1])
#plt.plot(hlambdas,hval)
#plt.plot(hlambdas,hfit,'r',label='fit')
#plt.show()
#plt.clf()
#plt.plot(H8lambdas,H8nline)
#plt.show()
#plt.clf()
#plt.plot(H9lambdas,H9nline)
#plt.plot(H10lambdas,H10nline)
#plt.show()
#sys.exit()

#Combine all the normalized lines together into one array for model fitting
###For Halpha through H10
alllambda = np.concatenate((H10lambdas,H9lambdas,H8lambdas,elambdas,dlambdas,glambdasnew,blambdasnew,alambdasnew))
allnline = np.concatenate((H10nline,H9nline,H8nline,enline,dnline,gnline,bnline,anline))
allsigma = np.concatenate((H10sigma,H9sigma,H8sigma,esigma,dsigma,gsigma,bsigma,asigma))
indices  = [0,len(H10lambdas)-1.,len(H10lambdas),len(H10lambdas)+len(H9lambdas)-1.,len(H10lambdas)+len(H9lambdas),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)+len(blambdasnew)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)+len(blambdasnew),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)+len(blambdasnew)+len(alambdasnew)-1.]
lambdaindex = [afitlow,afithi,alow,ahi,bfitlow,bfithi,blow,bhi,gfitlow,gfithi,glow,ghi,hlow,hhi,dlow,dhi,elow,ehi,H8low,H8hi,H9low,H9hi,H10low,H10hi]
#print len(H10nline),len(H9nline),len(H8nline),len(enline),len(dnline),len(gnline),len(bnline),len(anline)

######For Halpha through H8
#alllambda = np.concatenate((H8lambdas,elambdas,dlambdas,glambdas,blambdas,alambdas))
#allnline = np.concatenate((H8nline,enline,dnline,gnline,bnline,anline))
#allsigma = np.concatenate((H8sigma,esigma,dsigma,gsigma,bsigma,asigma))
#lambdaindex = [0,len(H8lambdas)-1.,len(H8lambdas),len(H8lambdas)+len(elambdas)-1.,len(H8lambdas)+len(elambdas),len(H8lambdas)+len(elambdas)+len(dlambdas)-1.,len(H8lambdas)+len(elambdas)+len(dlambdas),len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)-1.,len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew),len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)+len(blambdasnew)-1.,len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)+len(blambdasnew),len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)+len(blambdasnew)+len(alambdasnew)-1.]

######For Halpha through H9
#alllambda = np.concatenate((H9lambdas,H8lambdas,elambdas,dlambdas,glambdas,blambdas,alambdas))
#allnline = np.concatenate((H9nline,H8nline,enline,dnline,gnline,bnline,anline))
#allsigma = np.concatenate((H9sigma,H8sigma,esigma,dsigma,gsigma,bsigma,asigma))
#lambdaindex = [0,len(H9lambdas)-1.,len(H9lambdas),len(H9lambdas)+len(H8lambdas)-1.,len(H9lambdas)+len(H8lambdas),len(H9lambdas)+len(H8lambdas)+len(elambdas)-1.,len(H9lambdas)+len(H8lambdas)+len(elambdas),len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)-1.,len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas),len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdas)-1.,len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdas),len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdas)+len(blambdas)-1.,len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdas)+len(blambdas),len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdas)+len(blambdas)+len(alambdas)-1.]

######For Hbeta through H8
#alllambda = np.concatenate((H8lambdas,elambdas,dlambdas,glambdas,blambdas))
#allnline = np.concatenate((H8nline,enline,dnline,gnline,bnline))
#allsigma = np.concatenate((H8sigma,esigma,dsigma,gsigma,bsigma))
#lambdaindex = [0,len(H8lambdas)-1.,len(H8lambdas),len(H8lambdas)+len(elambdas)-1.,len(H8lambdas)+len(elambdas),len(H8lambdas)+len(elambdas)+len(dlambdas)-1.,len(H8lambdas)+len(elambdas)+len(dlambdas),len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdas)-1.,len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdas),len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdas)+len(blambdas)-1.]

#########For Hbeta through H9
#alllambda = np.concatenate((H9lambdas,H8lambdas,elambdas,dlambdas,glambdas,blambdas))
#allnline = np.concatenate((H9nline,H8nline,enline,dnline,gnline,bnline))
#allsigma = np.concatenate((H9sigma,H8sigma,esigma,dsigma,gsigma,bsigma))
#lambdaindex = [0,len(H9lambdas)-1.,len(H9lambdas),len(H9lambdas)+len(H8lambdas)-1.,len(H9lambdas)+len(H8lambdas),len(H9lambdas)+len(H8lambdas)+len(elambdas)-1.,len(H9lambdas)+len(H8lambdas)+len(elambdas),len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)-1.,len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas),len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdas)-1.,len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdas),len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdas)+len(blambdas)-1.]

#plt.clf()
#plt.plot(alllambda,allnline)
#plt.show()
#print len(alllambda)
#print lambdaindex
#sys.exit()

measuredcenter = np.array([acenter,bcenter,gcenter,dcenter,ecenter,H8center,H9center])
restwavelength = np.array([6562.79,4862.71,4341.69,4102.89,3971.19,3890.16,3836.48])
c = 2.99792e5
velocity = c * (measuredcenter-restwavelength)/restwavelength

#plt.clf()
#plt.plot(restwavelength,velocity,'b^')
#plt.show()
#alllambdas and allnline are used in intspec.py
#Now call intspec.py to run the model program

print "Starting intspec.py now "
case = 0 #We'll be interpolating Koester's raw models
filenames = 'modelnames.txt'
path = '/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/modelfitting/DA_models'
#np.savetxt('norm_WD0122.dat',np.transpose([blambdas,bnline]))
#ncflux,bestT,bestg = intmodel(alllambda,allnline,allsigma,lambdaindex,case,filenames,lambdas,zzcetiblue,zzcetired,FWHM,indices,path)
#print bestT,bestg
bestT, bestg = 12750, 825
#sys.exit()

#######
# Now we want to compute the finer grid
#######
makefinegrid(bestT,bestg)
sys.exit()
###################
# Compute the Chi-square for
# Each of those
###################

case = 1 #We'll be comparing our new grid to the spectrum.
filenames = 'interpolated_names.txt'
path = '/srv/two/jtfuchs/Interpolated_Models/center' + str(bestT) + '_' + str(bestg)
ncflux,bestT,bestg = intmodel(alllambda,allnline,allsigma,lambdaindex,case,filenames,lambdas,zzcetiblue,zzcetired,FWHM,indices,path)




