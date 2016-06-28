# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 14:16:05 2015

@author: joshfuchs

To do:
- Clean up code and document better

Done:
- Enter spec names and fwhm in command line
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
from matplotlib.backends.backend_pdf import PdfPages
import pyfits as pf # Infierno doesn't support astropy for some reason so using pyfits
#import astropy.io.fits as pf
import mpfit
from intspec import intspecs
from finegrid import makefinegrid
import sys
import os
import datetime
from scipy.optimize import leastsq

# ===========================================================================

#Define pseudogauss to fit one spectral line using parabola for continuum
def pseudogauss(x,p):
    #The model function with parameters p
    return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.)*p[5]))**p[6])


def fitpseudogauss(p,fjac=None,x=None, y=None, err=None):
    #Parameter values are passed in p
    #fjac=None just means partial derivatives will not be computed.
    model = pseudogauss(x,p)
    status = 0
    return([status,(y-model)/err])

# ===========================================================================

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

# ===========================================================================
# ===========================================================================

#Third order continuum plus pseudogaussians for beta through 10. That means 7 pseudogaussians. 
def bigpseudogauss(x,p):
    #The model function with parameters p
    return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*x**3. + p[4]*np.exp(-(np.abs(x-p[5])/(np.sqrt(2.)*p[6]))**p[7]) + p[8]*np.exp(-(np.abs(x-p[9])/(np.sqrt(2.)*p[10]))**p[11]) + p[12]*np.exp(-(np.abs(x-p[13])/(np.sqrt(2.)*p[14]))**p[15]) + p[16]*np.exp(-(np.abs(x-p[17])/(np.sqrt(2.)*p[18]))**p[19]) + p[20]*np.exp(-(np.abs(x-p[21])/(np.sqrt(2.)*p[22]))**p[23]) + p[24]*np.exp(-(np.abs(x-p[25])/(np.sqrt(2.)*p[26]))**p[27]) + p[28]*np.exp(-(np.abs(x-p[29])/(np.sqrt(2.)*p[30]))**p[31])


def fitbigpseudogauss(p,fjac=None,x=None, y=None, err=None):
    #Parameter values are passed in p
    #fjac=None just means partial derivatives will not be computed.
    model = bigpseudogauss(x,p)
    status = 0
    return([status,(y-model)/err])
# ===========================================================================

#Third order continuum plus pseudogaussians for gamma through 10. That means 6 pseudogaussians. 
def bigpseudogaussgamma(x,p):
    #The model function with parameters p
    return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*x**3. + p[4]*np.exp(-(np.abs(x-p[5])/(np.sqrt(2.)*p[6]))**p[7]) + p[8]*np.exp(-(np.abs(x-p[9])/(np.sqrt(2.)*p[10]))**p[11]) + p[12]*np.exp(-(np.abs(x-p[13])/(np.sqrt(2.)*p[14]))**p[15]) + p[16]*np.exp(-(np.abs(x-p[17])/(np.sqrt(2.)*p[18]))**p[19]) + p[20]*np.exp(-(np.abs(x-p[21])/(np.sqrt(2.)*p[22]))**p[23]) + p[24]*np.exp(-(np.abs(x-p[25])/(np.sqrt(2.)*p[26]))**p[27])


def fitbigpseudogaussgamma(p,fjac=None,x=None, y=None, err=None):
    #Parameter values are passed in p
    #fjac=None just means partial derivatives will not be computed.
    model = bigpseudogaussgamma(x,p)
    status = 0
    return([status,(y-model)/err])

# ===========================================================================
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


# ===========================================================================

#Define a function that fits higher order lines simultaneously (delta and higher))
def multipseudogauss(x,p):
    #The model function with parameters p
    #return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.*p[5])))**p[6]) + p[7]*np.exp(-(np.abs(x-p[8])/(np.sqrt(2.*p[9])))**p[10]) + p[11]*np.exp(-(np.abs(x-p[12])/(np.sqrt(2.*p[13])))**p[14]) #This one includes Hdelta, Hepsilon, and H8
    #return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.*p[5])))**p[6]) + p[7]*np.exp(-(np.abs(x-p[8])/(np.sqrt(2.*p[9])))**p[10]) + p[11]*np.exp(-(np.abs(x-p[12])/(np.sqrt(2.*p[13])))**p[14]) + p[15]*np.exp(-(np.abs(x-p[16])/(np.sqrt(2.*p[17])))**p[18]) #This one includes Hdelta, Hepsilon, H8, and H9
    return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.)*p[5]))**p[6]) + p[7]*np.exp(-(np.abs(x-p[8])/(np.sqrt(2.)*p[9]))**p[10]) + p[11]*np.exp(-(np.abs(x-p[12])/(np.sqrt(2.)*p[13]))**p[14]) + p[15]*np.exp(-(np.abs(x-p[16])/(np.sqrt(2.)*p[17]))**p[18]) + p[19]*np.exp(-(np.abs(x-p[20])/(np.sqrt(2.)*p[21]))**p[22]) #This one includes Hdelta, Hepsilon, H8, H9, and H10

def multifitpseudogauss(p,fjac=None,x=None, y=None, err=None):
    #Parameter values are passed in p
    #fjac=None just means partial derivatives will not be computed.
    model = multipseudogauss(x,p)
    status = 0
    return([status,(y-model)/err])

# ===========================================================================

def DispCalc(Pixels, alpha, theta, fr, fd, fl, zPnt):
    # This is the Grating Equation used to calculate the wavelenght of a pixel
    # based on the fitted parameters and angle set up.
    # Inputs: 
    # Pixels= Vector of Pixel Numbers
    # alpha=  of Grating Angle
    # aheta=  Camera Angle
    # fr= fringe density of grating
    # fd= Camera Angle Correction Factor
    # zPnt= Zero point pixel 
    Wavelengths= [] # Vector to store calculated wavelengths 
    for pix in Pixels:    
        beta = np.arctan( (pix-zPnt)*15./fl ) + (fd*theta*np.pi/180.) - (alpha*np.pi/180.) 
        wave = (10**6.)*( np.sin(beta) + np.sin(alpha*np.pi/180.) )/fr
        Wavelengths.append(wave)
    return Wavelengths

# ===========================================================================

#Now we need to read in actual spectrum. This is for Goodman spectra.
#zzcetiblue = 'wtnb.0526.WD1422p095_930_blue.ms.fits'
#zzcetired = 'wtnb.0532.WD1422p095_930_red.ms.fits'
#FWHM = 4.4 #Can read this from header using keyword SPECFWHM
script, zzcetiblue, zzcetired = sys.argv


#Read in the blue spectrum
datalistblue = pf.open(zzcetiblue)
datavalblue = datalistblue[0].data[0,0,:] #Reads in the object spectrum,data[0,0,:] is optimally subtracted, data[1,0,:] is raw extraction,  data[2,0,:] is sky, data[3,0,:] is sigma spectrum
sigmavalblue = datalistblue[0].data[3,0,:] #Sigma spectrum
FWHMpix = datalistblue[0].header['specfwhm']

'''
#Linearized Wavelengths
wav0blue = datalistblue[0].header['crval1']
deltawavblue = datalistblue[0].header['cd1_1']
lambdasblue = np.ones(len(datavalblue))
lambdasblue[0] = wav0blue
ivalblue = np.arange(1,len(datavalblue))

for i in ivalblue:
    lambdasblue[i] = lambdasblue[i-1] + deltawavblue
'''
#Grating equation blue wavelengths
alphablue = float(datalistblue[0].header['GRT_TARG'])
thetablue = float(datalistblue[0].header['CAM_TARG'])
frblue = float(datalistblue[0].header['LINDEN'])
fdblue = float(datalistblue[0].header['CAMFUD'])
flblue = float(datalistblue[0].header['FOCLEN'])
zPntblue = float(datalistblue[0].header['ZPOINT'])

trim_sec_blue= datalistblue[0].header["CCDSEC"]
trim_offset_blue= float( trim_sec_blue[1:len(trim_sec_blue)-1].split(':')[0] )-1
biningblue= float( datalistblue[0].header["PARAM18"] ) 
nxblue= np.size(datavalblue)#spec_data[0]
PixelsBlue= biningblue*(np.arange(0,nxblue,1)+trim_offset_blue)
lambdasblue = DispCalc(PixelsBlue, alphablue, thetablue, frblue, fdblue, flblue, zPntblue)


#Read in the red spectrum
datalistred = pf.open(zzcetired)
datavalred = datalistred[0].data[0,0,:] #data[0,0,:] is optimally extracted, data[2,0,:] is sky
sigmavalred = datalistred[0].data[3,0,:] #Sigma spectrum
'''
#Linearized red wavelengths
wav0red = datalistred[0].header['crval1']
deltawavred = datalistred[0].header['cd1_1']
lambdasred = np.ones(len(datavalred))
lambdasred[0] = wav0red
ivalred = np.arange(1,len(datavalred))

for i in ivalred:
    lambdasred[i] = lambdasred[i-1] + deltawavred

'''
#Grating equation red wavelengths
alphared = float(datalistred[0].header['GRT_TARG'])
thetared = float(datalistred[0].header['CAM_TARG'])
frred = float(datalistred[0].header['LINDEN'])
fdred = float(datalistred[0].header['CAMFUD'])
flred = float(datalistred[0].header['FOCLEN'])
zPntred = float(datalistred[0].header['ZPOINT'])

trim_sec_red= datalistred[0].header["CCDSEC"]
trim_offset_red= float( trim_sec_red[1:len(trim_sec_red)-1].split(':')[0] )-1
biningred= float( datalistred[0].header["PARAM18"] ) 
nxred= np.size(datavalred)#spec_data[0]
PixelsRed= biningred*(np.arange(0,nxred,1)+trim_offset_red)
lambdasred = DispCalc(PixelsRed, alphared, thetared, frred, fdred, flred, zPntred)


#FWHM = FWHMpix * deltawavblue #FWHM in Angstroms linearized
FWHM = FWHMpix * (lambdasblue[-1] - lambdasblue[0])/nxblue #from grating equation

#Concatenate both into two arrays
lambdas = np.concatenate((lambdasblue,lambdasred))
dataval = np.concatenate((datavalblue,datavalred))
sigmaval = 44./87.*np.concatenate((sigmavalblue,sigmavalred))#2.e-17 * np.ones(len(dataval)) small/big

#plot the spectrum
plt.clf()
plt.plot(lambdas,dataval,'k')
#plt.plot(lambdas,sigmaval)
#plt.show()
#sys.exit()

# This sets pixel range for the fitting and normalization
#beta, gamma, delta, epsilon and 8 are the same as the Montreal group.
afitlow = np.min(np.where(lambdas > 6380.))
afithi = np.min(np.where(lambdas > 6760.))
alow = np.min(np.where(lambdas > 6413.))
ahi = np.min(np.where(lambdas > 6713.))

bfitlow = np.min(np.where(lambdas > 4680.))
bfithi = np.min(np.where(lambdas > 5040.))
blow = np.min(np.where(lambdas > 4721.)) #old: 4710
bhi = np.min(np.where(lambdas > 5001.)) #old: 5010

gfitlow = np.min(np.where(lambdas > 4200.))
gfithi = np.min(np.where(lambdas > 4510.))
glow = np.min(np.where(lambdas > 4220.)) #old: 4220
ghi = np.min(np.where(lambdas > 4460.)) #old: 4490


#hlow = np.min(np.where(lambdas > 3860.)) #For H8
hlow = np.min(np.where(lambdas > 3782.)) #Includes H10, normally 3782 
#hlow = np.min(np.where(lambdas > 3755.)) #includes H11
hhi = np.min(np.where(lambdas > 4195.)) #4191 
dlow = np.min(np.where(lambdas > 4031.)) #old: 4040
dhi = np.min(np.where(lambdas > 4191.)) #Montreal: 4171, old: 4191
elow = np.min(np.where(lambdas > 3925.)) #old: 3930
ehi = np.min(np.where(lambdas > 4030.)) #Montreal: 4015, old: 4030
H8low = np.min(np.where(lambdas > 3859.)) #old: 3860
H8hi = np.min(np.where(lambdas > 3925.)) #Montreal: 3919, old: 3930
H9low = np.min(np.where(lambdas > 3815.))
H9hi = np.min(np.where(lambdas > 3855.))
H10low = np.min(np.where(lambdas > 3785.))
H10hi = np.min(np.where(lambdas > 3815.))
#H11low = np.min(np.where(lambdas > 3757.))
#H11hi = np.min(np.where(lambdas > 3785.))

#====================================================
#Set up estimates for pseudo-gaussian fitting
#Make the estimates in a smart and consistent manner.
#====================================================

alambdas = lambdas[afitlow:afithi+1]
asigmas = sigmaval[afitlow:afithi+1]
alphaval = dataval[afitlow:afithi+1]

aest = np.zeros(8)
xes = np.array([lambdas[afitlow],lambdas[alow],lambdas[alow+10],lambdas[ahi-10],lambdas[ahi],lambdas[afithi]])
yes = np.array([dataval[afitlow],dataval[alow],dataval[alow+10],dataval[ahi-10],dataval[ahi],dataval[afithi]])
ap = np.polyfit(xes,yes,3)
app = np.poly1d(ap)
aest[0] = ap[3]
aest[1] = ap[2]
aest[2] = ap[1]
aest[7] = ap[0]
aest[3] = np.min(dataval[alow:ahi+1]) - app(6562.79) #depth of line relative to continuum
aest[4] = 6562.79 #rest wavelength of H alpha
ahalfmax = app(6562.79) + aest[3]/3.
adiff = np.abs(alphaval-ahalfmax)
alowidx = adiff[np.where(alambdas < 6562.79)].argmin()
ahighidx = adiff[np.where(alambdas > 6562.79)].argmin() + len(adiff[np.where(alambdas < 6562.79)])
aest[5] = (alambdas[ahighidx] - alambdas[alowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
aest[6] = 1. #how much of a pseudo-gaussian



blambdas = lambdas[bfitlow:bfithi+1]
bsigmas = sigmaval[bfitlow:bfithi+1]
betaval = dataval[bfitlow:bfithi+1]

best = np.zeros(8)
xes = np.array([lambdas[bfitlow],lambdas[blow],lambdas[blow+10],lambdas[bhi],lambdas[bfithi]])
yes = np.array([dataval[bfitlow],dataval[blow],dataval[blow+10],dataval[bhi],dataval[bfithi]])
bp = np.polyfit(xes,yes,3)
bpp = np.poly1d(bp)
best[0] = bp[3]
best[1] = bp[2]
best[2] = bp[1]
best[7] = bp[0]
best[3] = np.min(dataval[blow:bhi+1]) - bpp(4862.71) #depth of line relative to continuum
best[4] = 4862.71 #rest wavelength of H beta
bhalfmax = bpp(4862.71) + best[3]/2.5
bdiff = np.abs(betaval-bhalfmax)
blowidx = bdiff[np.where(blambdas < 4862.71)].argmin()
bhighidx = bdiff[np.where(blambdas > 4862.71)].argmin() + len(bdiff[np.where(blambdas < 4862.71)])
best[5] = (blambdas[bhighidx] - blambdas[blowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
best[6] = 1. #how much of a pseudo-gaussian



glambdas = lambdas[gfitlow:gfithi+1]
gsigmas = sigmaval[gfitlow:gfithi+1]
gamval = dataval[gfitlow:gfithi+1]

dlambdas = lambdas[dlow:dhi+1]
dval = dataval[dlow:dhi+1]
elambdas = lambdas[elow:ehi+1]
epval = dataval[elow:ehi+1]
H8lambdas = lambdas[H8low:H8hi+1]
H8val = dataval[H8low:H8hi+1]
H9lambdas = lambdas[H9low:H9hi+1]
H9val = dataval[H9low:H9hi+1]
H10lambdas = lambdas[H10low:H10hi+1]
H10val = dataval[H10low:H10hi+1]

'''
gest = np.zeros(8)
#xes = np.array([lambdas[gfitlow],lambdas[gfitlow+10],lambdas[gfithi-10],lambdas[gfithi]])
#yes = np.array([dataval[gfitlow],dataval[gfitlow+10],dataval[gfithi-10],dataval[gfithi]])
#yes += dataval[gfitlow]/50.
#gp = np.polyfit(xes,yes,3)
#gpp = np.poly1d(gp)

xes = np.array([lambdas[H10low],lambdas[H9low],lambdas[H8low],lambdas[elow],lambdas[dlow],lambdas[dhi],lambdas[glow],lambdas[ghi],lambdas[blow],lambdas[bhi]])
yes = np.array([dataval[H10low],dataval[H9low],dataval[H8low],dataval[elow],dataval[dlow],dataval[dhi],dataval[glow],dataval[ghi],dataval[blow],dataval[bhi]])
yes += dataval[H10low]/30. #Trying an offset to make sure the continuum is above the lines
gp = np.polyfit(xes,yes,3)
gpp = np.poly1d(gp)


gest[0] = gp[3]
gest[1] = gp[2]
gest[2] = gp[1]
gest[7] = gp[0]
gest[3] = np.min(dataval[glow:ghi+1]) - gpp(4341.692) #depth of line relative to continuum
gest[4] = 4341.692 #rest wavelength of H gamma
ghalfmax = gpp(4341.69) + gest[3]/3.
gdiff = np.abs(gamval-ghalfmax)
glowidx = gdiff[np.where(glambdas < 4341.69)].argmin()
ghighidx = gdiff[np.where(glambdas > 4341.69)].argmin() + len(gdiff[np.where(glambdas < 4341.69)])
gest[5] = (glambdas[ghighidx] - glambdas[glowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
gest[6] = 1. #how much of a pseudo-gaussian



hest = np.zeros(23)
xes = np.array([lambdas[H10low],lambdas[H9low],lambdas[H8low],lambdas[elow],lambdas[dlow],lambdas[dhi]])
yes = np.array([dataval[H10low],dataval[H9low],dataval[H8low],dataval[elow],dataval[dlow],dataval[dhi]])
yes += dataval[H10low]/25. #Trying an offset to make sure the continuum is above the lines
hp = np.polyfit(xes,yes,2)
hpp = np.poly1d(hp)
hest[0] = hp[2]
hest[1] = hp[1]
hest[2] = hp[0]


#Now delta
hest[3] = np.min(dataval[dlow:dhi+1]) - hpp(4102.892) #depth of line relative to continuum
hest[4] = 4102.892 #rest wavelength of H delta
dhalfmax = hpp(4102.89) + hest[3]/3.
ddiff = np.abs(dval-dhalfmax)
dlowidx = ddiff[np.where(dlambdas < 4102.89)].argmin()
dhighidx = ddiff[np.where(dlambdas > 4102.89)].argmin() + len(ddiff[np.where(dlambdas < 4102.89)])
hest[5] = (dlambdas[dhighidx] - dlambdas[dlowidx]) / (2.*np.sqrt(2.*np.log(2.)))
hest[6] = 1.2 #how much of a pseudo-gaussian

#Now epsilon
hest[7] = np.min(dataval[elow:ehi+1]) - hpp(3971.198) #depth of line relative to continuum
hest[8] = 3971.198  #rest wavelength of H epsilon
ehalfmax = hpp(3971.19) + hest[7]/3.
ediff = np.abs(epval-ehalfmax)
elowidx = ediff[np.where(elambdas < 3971.19)].argmin()
ehighidx = ediff[np.where(elambdas > 3971.19)].argmin() + len(ediff[np.where(elambdas < 3971.19)])
hest[9] = (elambdas[ehighidx] - elambdas[elowidx]) / (2.*np.sqrt(2.*np.log(2.)))
hest[10] = 1.2 #how much of a pseudo-gaussian

#Now H8
hest[11] = np.min(dataval[H8low:H8hi+1]) - hpp(3890.166) #depth of line relative to continuum
hest[12] = 3890.166  #rest wavelength of H8
H8halfmax = hpp(3890.16) + hest[11]/3.
H8diff = np.abs(H8val-H8halfmax)
H8lowidx = H8diff[np.where(H8lambdas < 3890.16)].argmin()
H8highidx = H8diff[np.where(H8lambdas > 3890.16)].argmin() + len(H8diff[np.where(H8lambdas < 3890.16)])
hest[13] = (H8lambdas[H8highidx] - H8lambdas[H8lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
hest[14] = 1.2 #how much of a pseudo-gaussian

#Now H9
hest[15] = np.min(dataval[H9low:H9hi+1]) - hpp(3836.485) #depth of line relative to continuum
hest[16] = 3837.485  #rest wavelength of H9
H9halfmax = hpp(3836.48) + hest[15]/3.
H9diff = np.abs(H9val-H9halfmax)
H9lowidx = H9diff[np.where(H9lambdas < 3836.48)].argmin()
H9highidx = H9diff[np.where(H9lambdas > 3836.48)].argmin() + len(H9diff[np.where(H9lambdas < 3836.48)])
hest[17] = (H9lambdas[H9highidx] - H9lambdas[H9lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
hest[18] = 1.2 #how much of a pseudo-gaussian

#Now H10
hest[19] = np.min(dataval[H10low:H10hi+1]) - hpp(3797.909) #depth of line relative to continuum
hest[20] = 3798.909 #rest wavelength of H10
H10halfmax = hpp(3798.8) + hest[19]/3.
H10diff = np.abs(H10val-H10halfmax)
H10lowidx = H10diff[np.where(H10lambdas < 3798.8)].argmin()
H10highidx = H10diff[np.where(H10lambdas > 3798.8)].argmin() + len(H10diff[np.where(H10lambdas < 3798.8)])
hest[21] = (H10lambdas[H10highidx] - H10lambdas[H10lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
hest[22] = 1.2 #how much of a pseudo-gaussian

#now H11
#hest[23] = np.min(dataval[H11low:H10low+1]) - hpp(3770.63) #depth of line relative to continuum
#hest[24] = 3770.633 #rest wavelength of H10
#hest[25] = 5. #NEED TO CHECK THIS
#hest[26] = 1. #how much of a pseudo-gaussian
'''
'''
#########################
#Fit beta through 10 at once
biglambdas = lambdas[hlow:bfithi+1]
bigval = dataval[hlow:bfithi+1]
bigsig = sigmaval[hlow:bfithi+1]


bigest = np.zeros(32)
xes = np.array([lambdas[H10low],lambdas[H9low],lambdas[H8low],lambdas[elow],lambdas[dlow],lambdas[dhi],lambdas[glow],lambdas[ghi],lambdas[blow],lambdas[bhi]])
yes = np.array([dataval[H10low],dataval[H9low],dataval[H8low],dataval[elow],dataval[dlow],dataval[dhi],dataval[glow],dataval[ghi],dataval[blow],dataval[bhi]])
yes += dataval[H10low]/30. #Trying an offset to make sure the continuum is above the lines
bigp = np.polyfit(xes,yes,3)
bigpp = np.poly1d(bigp)
bigest[0] = bigp[3]
bigest[1] = bigp[2]
bigest[2] = bigp[1]
bigest[3] = bigp[0]

bigest[4] = np.min(dataval[blow:bhi+1]) - bigpp(4862.71) #depth of line relative to continuum
bigest[5] = 4862.71 #rest wavelength of H beta
bhalfmax = bigpp(4862.71) + bigest[4]/2.5
bdiff = np.abs(betaval-bhalfmax)
blowidx = bdiff[np.where(blambdas < 4862.71)].argmin()
bhighidx = bdiff[np.where(blambdas > 4862.71)].argmin() + len(bdiff[np.where(blambdas < 4862.71)])
bigest[6] = (blambdas[bhighidx] - blambdas[blowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
bigest[7] = 1. #how much of a pseudo-gaussian


bigest[8] = np.min(dataval[glow:ghi+1]) - bigpp(4341.692) #depth of line relative to continuum
bigest[9] = 4341.692 #rest wavelength of H gamma
ghalfmax = bigpp(4341.69) + bigest[8]/3.
gdiff = np.abs(gamval-ghalfmax)
glowidx = gdiff[np.where(glambdas < 4341.69)].argmin()
ghighidx = gdiff[np.where(glambdas > 4341.69)].argmin() + len(bdiff[np.where(glambdas < 4341.69)])
bigest[10] = (glambdas[ghighidx] - glambdas[glowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
bigest[11] = 1. #how much of a pseudo-gaussian


#Now delta
bigest[12] = np.min(dataval[dlow:dhi+1]) - bigpp(4102.892) #depth of line relative to continuum
bigest[13] = 4102.892  #rest wavelength of H delta
dhalfmax = bigpp(4102.89) + bigest[12]/3.
ddiff = np.abs(dval-dhalfmax)
dlowidx = ddiff[np.where(dlambdas < 4102.89)].argmin()
dhighidx = ddiff[np.where(dlambdas > 4102.89)].argmin() + len(ddiff[np.where(dlambdas < 4102.89)])
bigest[14] = (dlambdas[dhighidx] - dlambdas[dlowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[15] = 1.2 #how much of a pseudo-gaussian

#Now epsilon
bigest[16] = np.min(dataval[elow:ehi+1]) - bigpp(3971.198) #depth of line relative to continuum
bigest[17] = 3971.198   #rest wavelength of H epsilon
ehalfmax = bigpp(3971.19) + bigest[16]/3.
ediff = np.abs(epval-ehalfmax)
elowidx = ediff[np.where(elambdas < 3971.19)].argmin()
ehighidx = ediff[np.where(elambdas > 3971.19)].argmin() + len(ediff[np.where(elambdas < 3971.19)])
bigest[18] = (elambdas[ehighidx] - elambdas[elowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[19] = 1.2 #how much of a pseudo-gaussian

#Now H8
bigest[20] = np.min(dataval[H8low:H8hi+1]) - bigpp(3890.166) #depth of line relative to continuum
bigest[21] = 3890.166   #rest wavelength of H8
H8halfmax = bigpp(3890.16) + bigest[20]/3.
H8diff = np.abs(H8val-H8halfmax)
H8lowidx = H8diff[np.where(H8lambdas < 3890.16)].argmin()
H8highidx = H8diff[np.where(H8lambdas > 3890.16)].argmin() + len(H8diff[np.where(H8lambdas < 3890.16)])
bigest[22] = (H8lambdas[H8highidx] - H8lambdas[H8lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[23] = 1.2 #how much of a pseudo-gaussian

#Now H9
bigest[24] = np.min(dataval[H9low:H9hi+1]) - bigpp(3836.485) #depth of line relative to continuum
bigest[25] = 3837.485   #rest wavelength of H9
H9halfmax = bigpp(3836.48) + bigest[24]/3.
H9diff = np.abs(H9val-H9halfmax)
H9lowidx = H9diff[np.where(H9lambdas < 3836.48)].argmin()
H9highidx = H9diff[np.where(H9lambdas > 3836.48)].argmin() + len(H9diff[np.where(H9lambdas < 3836.48)])
bigest[26] = (H9lambdas[H9highidx] - H9lambdas[H9lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[27] = 1.2 #how much of a pseudo-gaussian

#Now H10
bigest[28] = np.min(dataval[H10low:H10hi+1]) - bigpp(3797.909) #depth of line relative to continuum
bigest[29] = 3798.909   #rest wavelength of H10
H10halfmax = bigpp(3798.8) + bigest[28]/3.
H10diff = np.abs(H10val-H10halfmax)
H10lowidx = H10diff[np.where(H10lambdas < 3798.8)].argmin()
H10highidx = H10diff[np.where(H10lambdas > 3798.8)].argmin() + len(H10diff[np.where(H10lambdas < 3798.8)])
bigest[30] = (H10lambdas[H10highidx] - H10lambdas[H10lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[31] = 1.2 #how much of a pseudo-gaussian

bigfa = {'x':biglambdas, 'y':bigval, 'err':bigsig}
bigparams = mpfit.mpfit(fitbigpseudogauss,bigest,functkw=bigfa,maxiter=1000,ftol=1e-14,xtol=1e-13)
print bigparams.status, bigparams.niter, bigparams.fnorm
bigfit = bigpseudogauss(biglambdas,bigparams.params)

bigguess = bigpseudogauss(biglambdas,bigest)

#Compute chi-square values for sections
bfitlow2 = np.min(np.where(biglambdas > 4680.))
bfithi2 = np.min(np.where(biglambdas > 5040.))
gfitlow2 = np.min(np.where(biglambdas > 4200.))
gfithi2 = np.min(np.where(biglambdas > 4510.))
hlow2 = np.min(np.where(biglambdas > 3778.)) 
hhi2 = np.min(np.where(biglambdas > 4195.)) 

print np.sum(((bigval-bigfit) / bigsig)**2.,dtype='d')
print np.sum(((bigval[bfitlow2:bfithi2+1]-bigfit[bfitlow2:bfithi2+1]) / bigsig[bfitlow2:bfithi2+1])**2.,dtype='d')
print np.sum(((bigval[gfitlow2:gfithi2+1]-bigfit[gfitlow2:gfithi2+1]) / bigsig[gfitlow2:gfithi2+1])**2.,dtype='d')
print np.sum(((bigval[hlow2:hhi2+1]-bigfit[hlow2:hhi2+1]) / bigsig[hlow2:hhi2+1])**2.,dtype='d')
print bigparams.params
plt.clf()
plt.plot(biglambdas,bigval,'b')
#plt.plot(biglambdas,bigpp(biglambdas),'r')
#plt.plot(biglambdas,bigguess,'r')
plt.plot(biglambdas,bigfit,'r')
plt.show()

sys.exit()
'''
########################
#########################

#Fit gamma through 10 at once
hlambdas = lambdas[hlow:gfithi+1]
hval = dataval[hlow:gfithi+1]
hsig = sigmaval[hlow:gfithi+1]

#temporarydhi = np.min(np.where(lambdas > 4171.)) 


bigest = np.zeros(28)
xes = np.array([lambdas[H10low],lambdas[H9low],lambdas[H8low],lambdas[elow],lambdas[dlow],lambdas[dhi],lambdas[glow],lambdas[ghi]])
yes = np.array([dataval[H10low],dataval[H9low],dataval[H8low],dataval[elow],dataval[dlow],dataval[dhi],dataval[glow],dataval[ghi]])
yes += dataval[H10low]/30. #Trying an offset to make sure the continuum is above the lines
bigp = np.polyfit(xes,yes,3)
bigpp = np.poly1d(bigp)
bigest[0] = bigp[3]
bigest[1] = bigp[2]
bigest[2] = bigp[1]
bigest[3] = bigp[0]


bigest[4] = np.min(dataval[glow:ghi+1]) - bigpp(4341.692) #depth of line relative to continuum
bigest[5] = 4341.692 #rest wavelength of H gamma
ghalfmax = bigpp(4341.69) + bigest[4]/3.
gdiff = np.abs(gamval-ghalfmax)
glowidx = gdiff[np.where(glambdas < 4341.69)].argmin()
ghighidx = gdiff[np.where(glambdas > 4341.69)].argmin() + len(bdiff[np.where(glambdas < 4341.69)])
bigest[6] = (glambdas[ghighidx] - glambdas[glowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
bigest[7] = 1. #how much of a pseudo-gaussian


#Now delta
bigest[8] = np.min(dataval[dlow:dhi+1]) - bigpp(4102.892) #depth of line relative to continuum
bigest[9] = 4102.892  #rest wavelength of H delta
dhalfmax = bigpp(4102.89) + bigest[8]/3.
ddiff = np.abs(dval-dhalfmax)
dlowidx = ddiff[np.where(dlambdas < 4102.89)].argmin()
dhighidx = ddiff[np.where(dlambdas > 4102.89)].argmin() + len(ddiff[np.where(dlambdas < 4102.89)])
bigest[10] = (dlambdas[dhighidx] - dlambdas[dlowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[11] = 1.2 #how much of a pseudo-gaussian

#Now epsilon
bigest[12] = np.min(dataval[elow:ehi+1]) - bigpp(3971.198) #depth of line relative to continuum
bigest[13] = 3971.198   #rest wavelength of H epsilon
ehalfmax = bigpp(3971.19) + bigest[12]/3.
ediff = np.abs(epval-ehalfmax)
elowidx = ediff[np.where(elambdas < 3971.19)].argmin()
ehighidx = ediff[np.where(elambdas > 3971.19)].argmin() + len(ediff[np.where(elambdas < 3971.19)])
bigest[14] = (elambdas[ehighidx] - elambdas[elowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[15] = 1.2 #how much of a pseudo-gaussian

#Now H8
bigest[16] = np.min(dataval[H8low:H8hi+1]) - bigpp(3890.166) #depth of line relative to continuum
bigest[17] = 3890.166   #rest wavelength of H8
H8halfmax = bigpp(3890.16) + bigest[16]/3.
H8diff = np.abs(H8val-H8halfmax)
H8lowidx = H8diff[np.where(H8lambdas < 3890.16)].argmin()
H8highidx = H8diff[np.where(H8lambdas > 3890.16)].argmin() + len(H8diff[np.where(H8lambdas < 3890.16)])
bigest[18] = (H8lambdas[H8highidx] - H8lambdas[H8lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[19] = 1.2 #how much of a pseudo-gaussian

#Now H9
bigest[20] = np.min(dataval[H9low:H9hi+1]) - bigpp(3836.485) #depth of line relative to continuum
bigest[21] = 3837.485   #rest wavelength of H9
H9halfmax = bigpp(3836.48) + bigest[20]/3.
H9diff = np.abs(H9val-H9halfmax)
H9lowidx = H9diff[np.where(H9lambdas < 3836.48)].argmin()
H9highidx = H9diff[np.where(H9lambdas > 3836.48)].argmin() + len(H9diff[np.where(H9lambdas < 3836.48)])
bigest[22] = (H9lambdas[H9highidx] - H9lambdas[H9lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[23] = 1.2 #how much of a pseudo-gaussian

#Now H10
bigest[24] = np.min(dataval[H10low:H10hi+1]) - bigpp(3797.909) #depth of line relative to continuum
bigest[25] = 3798.909   #rest wavelength of H10
H10halfmax = bigpp(3798.8) + bigest[24]/3.
H10diff = np.abs(H10val-H10halfmax)
H10lowidx = H10diff[np.where(H10lambdas < 3798.8)].argmin()
H10highidx = H10diff[np.where(H10lambdas > 3798.8)].argmin() + len(H10diff[np.where(H10lambdas < 3798.8)])
bigest[26] = (H10lambdas[H10highidx] - H10lambdas[H10lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[27] = 1.2 #how much of a pseudo-gaussian

print 'Now fitting H-gamma through H10.'
bigfa = {'x':hlambdas, 'y':hval, 'err':hsig}
hparams = mpfit.mpfit(fitbigpseudogaussgamma,bigest,functkw=bigfa,maxiter=200,ftol=1e-12,xtol=1e-8,quiet=True)#-10,-8
print hparams.status, hparams.niter, hparams.fnorm
hfit = bigpseudogaussgamma(hlambdas,hparams.params)

#Get line centers
gcenter = hparams.params[5]
dcenter = hparams.params[9]
ecenter = hparams.params[13]
H8center = hparams.params[17]
H9center = hparams.params[21]
H10center = hparams.params[25]

#Redefine these variables for quick switching of methods
glambdas = hlambdas
gamval = hval
gsigmas = hsig
gamfit = hfit


bigguess = bigpseudogaussgamma(hlambdas,bigest)

#Compute chi-square values for sections
gfitlow2 = np.min(np.where(hlambdas > 4200.))
gfithi2 = np.min(np.where(hlambdas > 4510.))
hlow2 = np.min(np.where(hlambdas > 3778.)) 
hhi2 = np.min(np.where(hlambdas > 4195.)) 

#print np.sum(((hval-hfit) / hsig)**2.,dtype='d')
#print np.sum(((hval[gfitlow2:gfithi2+1]-hfit[gfitlow2:gfithi2+1]) / hsig[gfitlow2:gfithi2+1])**2.,dtype='d')
#print np.sum(((hval[hlow2:hhi2+1]-hfit[hlow2:hhi2+1]) / hsig[hlow2:hhi2+1])**2.,dtype='d')
print hparams.params
#plt.clf()
#plt.plot(hlambdas,hval,'b')
#plt.plot(biglambdas,bigpp(biglambdas),'r')
#plt.plot(hlambdas,bigguess,'g')
#plt.plot(hlambdas,hfit,'r')
#plt.show()

#sys.exit()
##########################################33


#Fit alpha

#alambdas = lambdas[afitlow:afithi+1]
#asigmas = sigmaval[afitlow:afithi+1]
#alphaval = dataval[afitlow:afithi+1]
print 'Now fitting the H alpha line.'
afa = {'x':alambdas, 'y':alphaval, 'err':asigmas}
aparams = mpfit.mpfit(fitpseudogausscubic,aest,functkw=afa,maxiter=2000,ftol=1e-14,xtol=1e-13,quiet=True)
print 'Number of iterations: ', aparams.niter
acenter = aparams.params[4]
alphafit = pseudogausscubic(alambdas,aparams.params)
alphavariation = np.sum((alphafit - alphaval)**2.)
print aparams.status, aparams.niter, aparams.fnorm


#plt.clf()
#plt.plot(alambdas,alphaval,'b')
#plt.plot(alambdas,alphafit,'g')
#plt.plot(alambdas,pseudogausscubic(alambdas,aest),'k')
#plt.plot(alambdas,aparams.params[0]*1. + aparams.params[1]*alambdas +aparams.params[2]*alambdas**2.)
#plt.show()
#sys.exit()

#Fit beta
#blambdas = lambdas[bfitlow:bfithi+1]
#bsigmas = sigmaval[bfitlow:bfithi+1]
#betaval = dataval[bfitlow:bfithi+1]
print '\nNow fitting the H beta line.'
bfa = {'x':blambdas, 'y':betaval, 'err':bsigmas}
bparams = mpfit.mpfit(fitpseudogausscubic,best,functkw=bfa,maxiter=3000,ftol=1e-16,xtol=1e-10,quiet=True)
print 'Number of iterations: ', bparams.niter
print bparams.status, bparams.niter, bparams.fnorm
bcenter = bparams.params[4]
betafit = pseudogausscubic(blambdas,bparams.params)
betavariation = np.sum((betafit - betaval)**2.)

#plt.clf()
#plt.plot(blambdas,betaval,'b',label='data')
#plt.plot(blambdas,betafit,'r',label='fit')
#plt.plot(blambdas,bparams.params[0]*1. + bparams.params[1]*blambdas+bparams.params[2]*blambdas**2.)
#plt.show()
#sys.exit()

'''
#Fit gamma
#glambdas = lambdas[gfitlow:gfithi+1]
#gsigmas = sigmaval[gfitlow:gfithi+1]
#gamval = dataval[gfitlow:gfithi+1]
#fit to all, fit to only gamma
#gest[0] = -1.126693267e-12#5.49666719e-13
#gest[1] = 7.727728386e-16#-3.48101536e-16
#gest[2] = -1.723903565e-19#7.64647540e-20
#gest[3] = -9.918084972e-15#-8.74012284e-15
#gest[4] = 4341.577618#4.3416886e3
#gest[5] = 30.30251482#2.66664913e1
#gest[6] = 0.8840889689#9.88401271e-1
#gest[7] = 1.265568317e-23#-5.69605495e-24
print '\nNow fitting the H gamma line.'
gfa = {'x':glambdas, 'y':gamval, 'err':gsigmas}
gparams = mpfit.mpfit(fitpseudogausscubic,gest,functkw=gfa,maxiter=3000,ftol=1e-16,xtol=1e-10,quiet=True)
print 'Number of iterations: ', gparams.niter
print gparams.status, gparams.niter, gparams.fnorm
gcenter = gparams.params[4]
gamfit = pseudogausscubic(glambdas,gparams.params)
gammavariation = np.sum((gamfit - gamval)**2.)
print gparams.params

#Try to use scipy.optimize.leastsq
def residuals(p,x,y,e):
    err = (y - pseudogausscubic(x,p))/e
    return err**2.

plsq,cov,infodict,mesg,ier = least_squares(residuals,gest,args=(gamval,glambdas,gsigmas),ftol=1e-16,xtol=1e-16,gtol=1e-10,full_output=True,method='lm')
print plsq
print infodict['nfev']
print mesg
print ier
lsqfit = pseudogausscubic(glambdas,plsq)

#plt.clf()
#plt.plot(glambdas,gamval,'b',label='data')
#plt.plot(glambdas,gamfit,'r',label='fit')
#plt.plot(glambdas,pseudogausscubic(glambdas,gest),'g')
#plt.plot(glambdas,gparams.params[0]*1. + gparams.params[1]*glambdas+gparams.params[2]*glambdas**2.+gparams.params[7]*glambdas**3.,'r')
#plt.plot(glambdas,lsqfit,'c')
#plt.show()
#sys.exit()



#Fit higher order lines
print '\nNow fitting higher order lines.' 

hest[0] = hest[0]
hest[1] = hest[1]
hest[2] = hest[2]
hest[3] = -9.6235e-15
hest[4] = 4093.4601
hest[5] = 29.556
hest[6] = 0.9347
hest[7] = -9.447e-15
hest[8] = 3962.028
hest[9] = 25.635
hest[10] = 1.0651
hest[11] = -8.714e-15
hest[12] = 3880.746
hest[13] = 24.1001
hest[14] = 1.1798
hest[15] = -7.9028e-15
hest[16] = 3826.267
hest[17] = 21.364
hest[18] = 1.411
hest[19] = -8.4607e-15
hest[20] = 3775.064
hest[21] = 24.1268
hest[22] = 2.4862


hlambdas = lambdas[hlow:hhi+1]
#dlambdas = lambdas[dlow:dhi+1]
#elambdas = lambdas[elow:ehi+1]
#H8lambdas = lambdas[H8low:H8hi+1]
#H9lambdas = lambdas[H9low:H9hi+1]
#H10lambdas = lambdas[H10low:H10hi+1]
#H11lambdas = lambdas[H11low:H10low+1]
hsigmas = sigmaval[hlow:hhi+1]
hval = dataval[hlow:hhi+1]
hfa = {'x':hlambdas, 'y':hval, 'err':hsigmas}
hparams = mpfit.mpfit(multifitpseudogauss,hest,functkw=hfa,maxiter=2000,ftol=1e-14,xtol=1e-11,quiet=True) #4e-12
print 'Number of iterations: ', hparams.niter
print hparams.status,hparams.niter, hparams.fnorm

hfit = multipseudogauss(hlambdas,hparams.params)
dcenter = hparams.params[4]
ecenter = hparams.params[8]
H8center = hparams.params[12]
H9center = hparams.params[16]
H10center = hparams.params[20]
#H11center = hparams.params[24]

highervariation = np.sum((hfit - hval)**2.)

#plt.clf()
#plt.plot(hlambdas,hval,'b',label='data')
#plt.plot(hlambdas,hparams.params[0]*1. + hparams.params[1]*hlambdas + hparams.params[2]*hlambdas**2.)
#plt.plot(hlambdas,multipseudogauss(hlambdas,hest),'r')
#plt.plot(hlambdas,hfit,'r',label='fit')
#plt.show()
#sys.exit()
'''
#Fit a line to the fit points  from each end and divide by this line to normalize. 
#Note the line needs to be linear in lambda not in pixel
#bnline is the normalized spectral line. Continuum set to one.



#======================================
#Normalize using the pseudogaussian fits
#======================================

#Start with alpha
#Set the center of the line to the wavelength of the models.
alambdas = alambdas - (acenter- 6564.6047)
#aslope = (alphafit[-1] - alphafit[0] ) / (alambdas[-1] - alambdas[0])
#ali = aslope * (alambdas - alambdas[0]) + alphafit[0]
#anline = dataval[alow:ahi+1] / ali
#asigma = sigmaval[alow:ahi+1] / ali

anormlow = np.min(np.where(alambdas > 6413.)) #Refind the normalization points since we have shifted the line.
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
plt.plot(alambdas[anormlow],alphafit[anormlow],'g^')
plt.plot(alambdas[anormhi],alphafit[anormhi],'g^')
plt.plot(alambdasnew,ali,'g')
#plt.show()
#sys.exit()

#Now to beta
#Set the center of the line to the wavelength of the models.
blambdas = blambdas - (bcenter-4862.6510)
#bslope = (betafit[-1] - betafit[0] ) / (blambdas[-1] - blambdas[0])
#bli = bslope * (blambdas - blambdas[0]) + betafit[0]
#bnline = dataval[blow:bhi+1] / bli
#bsigma =  sigmaval[blow:bhi+1] / bli

bnormlow = np.min(np.where(blambdas > 4721.))
bnormhi = np.min(np.where(blambdas > 5001.))
bslope = (betafit[bnormhi] - betafit[bnormlow] ) / (blambdas[bnormhi] - blambdas[bnormlow])
blambdasnew = blambdas[bnormlow:bnormhi+1]
betavalnew = betaval[bnormlow:bnormhi+1]
bsigmasnew = bsigmas[bnormlow:bnormhi+1]
bli = bslope * (blambdasnew - blambdas[bnormlow]) + betafit[bnormlow]
bnline = betavalnew / bli
bsigma = bsigmasnew / bli


#plt.clf()
#plt.plot(blambdasnew,bnline)
plt.plot(blambdas[bnormlow],betafit[bnormlow],'g^')
plt.plot(blambdas[bnormhi],betafit[bnormhi],'g^')
plt.plot(blambdasnew,bli,'g')
#plt.show()
#sys.exit()

#Now do gamma
#gnline is the normalized spectral line. Continuum set to one.
#Set the center of the line to the wavelength of models
glambdas = glambdas - (gcenter-4341.6550)
gnormlow = np.min(np.where(glambdas > 4220.))
gnormhi = np.min(np.where(glambdas > 4460.))
gslope = (gamfit[gnormhi] - gamfit[gnormlow] ) / (glambdas[gnormhi] - glambdas[gnormlow])
glambdasnew = glambdas[gnormlow:gnormhi+1]
gamvalnew = gamval[gnormlow:gnormhi+1]
gsigmasnew = gsigmas[gnormlow:gnormhi+1]
gli = gslope * (glambdasnew - glambdas[gnormlow]) + gamfit[gnormlow]
gnline = gamvalnew / gli
gsigma = gsigmasnew / gli

gammavariation = np.sum((gamfit[gnormlow:gnormhi+1] - gamval[gnormlow:gnormhi+1])**2.)

#plt.clf()
#plt.plot(glambdasnew,gnline)
plt.plot(glambdas[gnormlow],gamfit[gnormlow],'g^')
plt.plot(glambdas[gnormhi],gamfit[gnormhi],'g^')
plt.plot(glambdasnew,gli,'g')
#plt.show()
#sys.exit()

#Now normalize the higher order lines (delta, epsilon, H8)
hlambdastemp = hlambdas - (dcenter-4102.9071)
dnormlow = np.min(np.where(hlambdastemp > 4031.))
dnormhi = np.min(np.where(hlambdastemp > 4191.))
dlambdas = hlambdastemp[dnormlow:dnormhi+1]
dslope = (hfit[dnormhi] - hfit[dnormlow]) / (hlambdastemp[dnormhi] - hlambdastemp[dnormlow])
dli = dslope * (dlambdas - dlambdas[0]) + hfit[dnormlow]
dvaltemp = dataval[hlow:hhi+1]
dsigtemp = sigmaval[hlow:hhi+1]
dnline = dvaltemp[dnormlow:dnormhi+1] / dli
dsigma = dsigtemp[dnormlow:dnormhi+1] / dli

plt.plot(hlambdastemp[dnormlow],hfit[dnormlow],'g^')
plt.plot(hlambdastemp[dnormhi],hfit[dnormhi],'g^')
plt.plot(dlambdas,dli,'g')

#elambdas = elambdas - (ecenter-3971.198)
hlambdastemp = hlambdas - (ecenter-3971.1751)
enormlow = np.min(np.where(hlambdastemp > 3925.))
enormhi = np.min(np.where(hlambdastemp > 4030.))
elambdas = hlambdastemp[enormlow:enormhi+1]
eslope = (hfit[enormhi] - hfit[enormlow] ) / (hlambdastemp[enormhi] - hlambdastemp[enormlow])
eli = eslope * (elambdas - elambdas[0]) + hfit[enormlow]
evaltemp = dataval[hlow:hhi+1]
esigtemp = sigmaval[hlow:hhi+1]
enline = evaltemp[enormlow:enormhi+1] / eli
esigma = esigtemp[enormlow:enormhi+1] / eli

plt.plot(hlambdastemp[enormlow],hfit[enormlow],'g^')
plt.plot(hlambdastemp[enormhi],hfit[enormhi],'g^')
plt.plot(elambdas,eli,'g')

#H8lambdas = H8lambdas - (H8center-3890.166)
hlambdastemp = hlambdas - (H8center-3890.1461)
H8normlow = np.min(np.where(hlambdastemp > 3859.))
H8normhi = np.min(np.where(hlambdastemp > 3925.))
H8lambdas = hlambdastemp[H8normlow:H8normhi+1]
H8slope = (hfit[H8normhi] - hfit[H8normlow] ) / (hlambdastemp[H8normhi] - hlambdastemp[H8normlow])
H8li = H8slope * (H8lambdas - H8lambdas[0]) + hfit[H8normlow]
H8valtemp = dataval[hlow:hhi+1]
H8sigtemp = sigmaval[hlow:hhi+1]
H8nline = H8valtemp[H8normlow:H8normhi+1] / H8li
H8sigma = H8sigtemp[H8normlow:H8normhi+1] / H8li

plt.plot(hlambdastemp[H8normlow],hfit[H8normlow],'g^')
plt.plot(hlambdastemp[H8normhi],hfit[H8normhi],'g^')
plt.plot(H8lambdas,H8li,'g')

### To normalize, using points from end of region since it is so small.
#H9lambdas = H9lambdas - (H9center- 3836.485)
hlambdastemp = hlambdas - (H9center- 3836.4726)
H9normlow = np.min(np.where(hlambdastemp > 3815.))
H9normhi = np.min(np.where(hlambdastemp > 3855.))
H9lambdas = hlambdastemp[H9normlow:H9normhi+1]
H9slope = (hfit[H9normhi] - hfit[H9normlow] ) / (hlambdastemp[H9normhi] - hlambdastemp[H9normlow])
H9li = H9slope * (H9lambdas - H9lambdas[0]) + hfit[H9normlow]
H9valtemp = dataval[hlow:hhi+1]
H9sigtemp = sigmaval[hlow:hhi+1]
H9nline = H9valtemp[H9normlow:H9normhi+1] / H9li
H9sigma = H9sigtemp[H9normlow:H9normhi+1] / H9li

plt.plot(hlambdastemp[H9normlow],hfit[H9normlow],'g^')
plt.plot(hlambdastemp[H9normhi],hfit[H9normhi],'g^')
plt.plot(H9lambdas,H9li,'g')

#H10lambdas = H10lambdas - (H10center-3797.909)
#print H10center
#H10center = 3797.26
hlambdastemp = hlambdas - (H10center-3798.9799)
H10normlow = np.min(np.where(hlambdastemp > 3785.))
H10normhi = np.min(np.where(hlambdastemp > 3815.))
H10lambdas = hlambdastemp[H10normlow:H10normhi+1]
H10slope = (hfit[H10normhi] - hfit[H10normlow] ) / (hlambdastemp[H10normhi] - hlambdastemp[H10normlow])
H10li = H10slope * (H10lambdas - H10lambdas[0]) + hfit[H10normlow]
H10valtemp = dataval[hlow:hhi+1]
H10sigtemp = sigmaval[hlow:hhi+1]
H10nline = H10valtemp[H10normlow:H10normhi+1] / H10li
H10sigma = H10sigtemp[H10normlow:H10normhi+1] / H10li

plt.plot(hlambdastemp[H10normlow],hfit[H10normlow],'g^')
plt.plot(hlambdastemp[H10normhi],hfit[H10normhi],'g^')
plt.plot(H10lambdas,H10li,'g')

#======================================
#Normalize by averaging continuum points and pseudogaussians for centers
#======================================
alambdas = lambdas[afitlow:afithi+1]
blambdas = lambdas[bfitlow:bfithi+1]
hlambdas = lambdas[hlow:gfithi+1]
glambdas = hlambdas



#Start with alpha
#Set the center of the line to the wavelength of the models.
alambdas = alambdas - (acenter- 6564.6047)
anormlow = np.min(np.where(alambdas > 6413.)) #Refind the normalization points since we have shifted the line.
anormhi = np.min(np.where(alambdas > 6713.))

#Take the average of a few points
anormlowval = np.mean(alphaval[anormlow-5:anormlow+5])
anormhival = np.mean(alphaval[anormhi-5:anormhi+5])

aslope = (anormhival - anormlowval ) / (alambdas[anormhi] - alambdas[anormlow])
alambdasnew = alambdas[anormlow:anormhi+1]
alphavalnew = alphaval[anormlow:anormhi+1]
asigmasnew = asigmas[anormlow:anormhi+1]
ali = aslope * (alambdasnew - alambdas[anormlow]) + anormlowval
anline = alphavalnew / ali
asigma = asigmasnew / ali


#plt.clf()
#plt.plot(alambdas,alphaval)
#plt.plot(alambdasnew,ali)
plt.plot(alambdas[anormlow],anormlowval,'ro')
plt.plot(alambdas[anormhi],anormhival,'ro')
plt.plot(alambdasnew,ali,'r')
#plt.show()
#sys.exit()


#Now to beta
#Set the center of the line to the wavelength of the models.
blambdas = blambdas - (bcenter-4862.6510)
bnormlow = np.min(np.where(blambdas > 4721.))
bnormhi = np.min(np.where(blambdas > 5001.))

bnormlowval = np.mean(betaval[bnormlow-5:bnormlow+5])
bnormhival = np.mean(betaval[bnormhi-5:bnormhi+5])

bslope = (bnormhival - bnormlowval ) / (blambdas[bnormhi] - blambdas[bnormlow])
blambdasnew = blambdas[bnormlow:bnormhi+1]
betavalnew = betaval[bnormlow:bnormhi+1]
bsigmasnew = bsigmas[bnormlow:bnormhi+1]
bli = bslope * (blambdasnew - blambdas[bnormlow]) + bnormlowval
bnline = betavalnew / bli
bsigma = bsigmasnew / bli


#plt.clf()
#plt.plot(blambdasnew,bli)
#plt.plot(blambdas,betaval)
plt.plot(blambdas[bnormlow],bnormlowval,'ro')
plt.plot(blambdas[bnormhi],bnormhival,'ro')
plt.plot(blambdasnew,bli,'r')
#plt.show()
#sys.exit()

#Now do gamma
#gnline is the normalized spectral line. Continuum set to one.
#Set the center of the line to the wavelength of models
glambdas = glambdas - (gcenter-4341.6550)
gnormlow = np.min(np.where(glambdas > 4220.))
gnormhi = np.min(np.where(glambdas > 4460.))

gnormlowval = np.mean(gamval[gnormlow-5:gnormlow+5])
gnormhival = np.mean(gamval[gnormhi-5:gnormhi+5])


gslope = (gnormhival - gnormlowval ) / (glambdas[gnormhi] - glambdas[gnormlow])
glambdasnew = glambdas[gnormlow:gnormhi+1]
gamvalnew = gamval[gnormlow:gnormhi+1]
gsigmasnew = gsigmas[gnormlow:gnormhi+1]
gli = gslope * (glambdasnew - glambdas[gnormlow]) + gnormlowval
gnline = gamvalnew / gli
gsigma = gsigmasnew / gli

#plt.clf()
#plt.plot(glambdasnew,gli,'r')
#plt.plot(glambdas,gamval,'b')
#plt.axvline(x=glambdas[gnormlow-5],color='r')
#plt.axvline(x=glambdas[gnormlow+5],color='r')
#plt.axvline(x=glambdas[gnormhi-5],color='r')
#plt.axvline(x=glambdas[gnormhi+5],color='r')
plt.plot(glambdas[gnormlow],gnormlowval,'ro')
plt.plot(glambdas[gnormhi],gnormhival,'ro')
plt.plot(glambdasnew,gli,'r')
#plt.show()

#Now delta
hlambdastemp = hlambdas - (dcenter-4102.9071)
dnormlow = np.min(np.where(hlambdastemp > 4031.))
dnormhi = np.min(np.where(hlambdastemp > 4191.))

dnormlowval = np.mean(hval[dnormlow-5:dnormlow+5])
dnormhival = np.mean(hval[dnormhi-5:dnormhi+5])

dslope = (dnormhival -dnormlowval ) / (hlambdastemp[dnormhi] - hlambdastemp[dnormlow])
dlambdas = hlambdastemp[dnormlow:dnormhi+1]
dli = dslope * (dlambdas - dlambdas[0]) + dnormlowval
dvaltemp = dataval[hlow:hhi+1]
dsigtemp = sigmaval[hlow:hhi+1]
dnline = dvaltemp[dnormlow:dnormhi+1] / dli
dsigma = dsigtemp[dnormlow:dnormhi+1] / dli

#plt.plot(dlambdas,dli,'g')
#plt.axvline(x=glambdas[dnormlow-5],color='g')
#plt.axvline(x=glambdas[dnormlow+5],color='g')
#plt.axvline(x=glambdas[dnormhi-5],color='g')
#plt.axvline(x=glambdas[dnormhi+5],color='g')
plt.plot(hlambdastemp[dnormlow],dnormlowval,'ro')
plt.plot(hlambdastemp[dnormhi],dnormhival,'ro')
plt.plot(dlambdas,dli,'r')


#Epsilon
hlambdastemp = hlambdas - (ecenter-3971.1751)
enormlow = np.min(np.where(hlambdastemp > 3925.))
enormhi = np.min(np.where(hlambdastemp > 4030.))

enormlowval = np.mean(hval[enormlow-5:enormlow+5])
enormhival = np.mean(hval[enormhi-5:enormhi+5])

eslope = (enormhival - enormlowval ) / (hlambdastemp[enormhi] - hlambdastemp[enormlow])
elambdas = hlambdastemp[enormlow:enormhi+1]
eli = eslope * (elambdas - elambdas[0]) + enormlowval

evaltemp = dataval[hlow:hhi+1]
esigtemp = sigmaval[hlow:hhi+1]
enline = evaltemp[enormlow:enormhi+1] / eli
esigma = esigtemp[enormlow:enormhi+1] / eli

#plt.plot(elambdas,eli,'c')
#plt.axvline(x=glambdas[enormlow-5],color='c')
#plt.axvline(x=glambdas[enormlow+5],color='c')
#plt.axvline(x=glambdas[enormhi-5],color='c')
#plt.axvline(x=glambdas[enormhi+5],color='c')
plt.plot(hlambdastemp[enormlow],enormlowval,'ro')
plt.plot(hlambdastemp[enormhi],enormhival,'ro')
plt.plot(elambdas,eli,'r')

#H8
hlambdastemp = hlambdas - (H8center-3890.1461)
H8normlow = np.min(np.where(hlambdastemp > 3859.))
H8normhi = np.min(np.where(hlambdastemp > 3925.))

H8normlowval = np.mean(hval[H8normlow-5:H8normlow+5])
H8normhival = np.mean(hval[H8normhi-5:H8normhi+5])


H8slope = (H8normhival - H8normlowval ) / (hlambdastemp[H8normhi] - hlambdastemp[H8normlow])
H8lambdas = hlambdastemp[H8normlow:H8normhi+1]
H8li = H8slope * (H8lambdas - H8lambdas[0]) + H8normlowval
H8valtemp = dataval[hlow:hhi+1]
H8sigtemp = sigmaval[hlow:hhi+1]
H8nline = H8valtemp[H8normlow:H8normhi+1] / H8li
H8sigma = H8sigtemp[H8normlow:H8normhi+1] / H8li

#plt.plot(H8lambdas,H8li,'m')
#plt.axvline(x=glambdas[H8normlow-5],color='m')
#plt.axvline(x=glambdas[H8normlow+5],color='m')
#plt.axvline(x=glambdas[H8normhi-5],color='m')
#plt.axvline(x=glambdas[H8normhi+5],color='m')
plt.plot(hlambdastemp[H8normlow],H8normlowval,'ro')
plt.plot(hlambdastemp[H8normhi],H8normhival,'ro')
plt.plot(H8lambdas,H8li,'r')

#H9
hlambdastemp = hlambdas - (H9center- 3836.4726)
H9normlow = np.min(np.where(hlambdastemp > 3815.))
H9normhi = np.min(np.where(hlambdastemp > 3855.))

H9normlowval = np.mean(hval[H9normlow-4:H9normlow+4])
H9normhival = np.mean(hval[H9normhi-5:H9normhi+5])


H9slope = (H9normhival - H9normlowval ) / (hlambdastemp[H9normhi] - hlambdastemp[H9normlow])
H9lambdas = hlambdastemp[H9normlow:H9normhi+1]
H9li = H9slope * (H9lambdas - H9lambdas[0]) + H9normlowval
H9valtemp = dataval[hlow:hhi+1]
H9sigtemp = sigmaval[hlow:hhi+1]

H9nline = H9valtemp[H9normlow:H9normhi+1] / H9li
H9sigma = H9sigtemp[H9normlow:H9normhi+1] / H9li

#plt.plot(H9lambdas,H9li,'k')
#plt.axvline(x=glambdas[H9normlow-5],color='k')
#plt.axvline(x=glambdas[H9normlow+5],color='k')
#plt.axvline(x=glambdas[H9normhi-5],color='k')
#plt.axvline(x=glambdas[H9normhi+5],color='k')
plt.plot(hlambdastemp[H9normlow],H9normlowval,'ro')
plt.plot(hlambdastemp[H9normhi],H9normhival,'ro')
plt.plot(H9lambdas,H9li,'r')


#H10
#H10center = 3797.26
hlambdastemp = hlambdas - (H10center-3798.9799)
H10normlow = np.min(np.where(hlambdastemp > 3785.))
H10normhi = np.min(np.where(hlambdastemp > 3815.))

H10normlowval = np.mean(hval[0:H10normlow+4]) #Since we cut off the spectrum, just start with the leftmost point
H10normhival = np.mean(hval[H10normhi-3:H10normhi+5])

H10slope = (H10normhival - H10normlowval ) / (hlambdastemp[H10normhi] - hlambdastemp[H10normlow])
H10lambdas = hlambdastemp[H10normlow:H10normhi+1]
H10li = H10slope * (H10lambdas - H10lambdas[0]) + H10normlowval
H10valtemp = dataval[hlow:hhi+1]
H10sigtemp = sigmaval[hlow:hhi+1]


H10nline = H10valtemp[H10normlow:H10normhi+1] / H10li
H10sigma = H10sigtemp[H10normlow:H10normhi+1] / H10li

#plt.plot(H10lambdas,H10li,'r')
#plt.axvline(x=glambdas[0],color='r')
#plt.axvline(x=glambdas[H10normlow+5],color='r')
#plt.axvline(x=glambdas[H10normhi-5],color='r')
#plt.axvline(x=glambdas[H10normhi+5],color='r')
plt.plot(hlambdastemp[H10normlow],H10normlowval,'ro')
plt.plot(hlambdastemp[H10normhi],H10normhival,'ro')
plt.plot(H10lambdas,H10li,'r')
#plt.savefig('normalization_points.eps',format='eps',dpi=1200)

plt.show()
sys.exit()
#======================================
#End section to normalize by averaging continuum points and pseudogaussians for centers
#======================================


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
#plt.show()
#plt.clf()
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

#variation = alphavariation + betavariation + gammavariation + highervariation
#print variation
#plt.clf()
#plt.plot(alllambda,allnline)
#plt.show()
#sys.exit()

#measuredcenter = np.array([acenter,bcenter,gcenter,dcenter,ecenter,H8center,H9center])
#restwavelength = np.array([6562.79,4862.71,4341.69,4102.89,3971.19,3890.16,3836.48])
#c = 2.99792e5
#velocity = c * (measuredcenter-restwavelength)/restwavelength

##### Save the normalized spectrum for later use
now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
marker = str(np.round(FWHM,decimals=2)) + '_fit'
endpoint = '.ms.'
savespecname = 'norm_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_' + now[5:10] + '_' + marker + '.txt' 
np.savetxt(savespecname,np.transpose([alllambda,allnline,allsigma]))

#Save the guesses and best-fitting parameters for the pseudogaussians
#aest,best,bigest for guesses and aparams.params, bparams.params, hparams.params
savefitparams = 'params_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_' + now[5:10] + '_' + marker + '.txt'
saveparams = np.zeros([len(bigest),6])
saveparams[0:len(aest),0] = aest
saveparams[0:len(best),1] = best
saveparams[0:len(bigest),2] = bigest
saveparams[0:len(aparams.params),3] = aparams.params
saveparams[0:len(bparams.params),4] = bparams.params
saveparams[0:len(hparams.params),5] = hparams.params

np.savetxt(savefitparams,saveparams)

#Save the pseudogaussian fits to the spectrum as a pdf
savefitspec = 'fit_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_' + now[5:10] + '_' + marker + '.pdf'
fitpdf = PdfPages(savefitspec)
plt.clf()
plt.plot(alambdas,alphaval,'b')
plt.plot(alambdas,alphafit,'r')
fitpdf.savefig()
plt.clf()
plt.plot(blambdas,betaval,'b')
plt.plot(blambdas,betafit,'r')
fitpdf.savefig()
plt.clf()
plt.plot(hlambdas,hval,'b')
plt.plot(hlambdas,hfit,'r')
fitpdf.savefig()
fitpdf.close()
#sys.exit()

print "Starting intspec.py now "
case = 0 #We'll be interpolating Koester's raw models
filenames = 'modelnames.txt'
path = '/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/modelfitting/DA_models'
#ncflux,bestT,bestg = intspecs(alllambda,allnline,allsigma,lambdaindex,case,filenames,lambdas,zzcetiblue,zzcetired,FWHM,indices,path)
#print bestT,bestg
bestT, bestg = 12000, 750
#sys.exit()

#######
# Now we want to compute the finer grid
#######
#makefinegrid(bestT,bestg)
#sys.exit()
###################
# Compute the Chi-square for
# Each of those
###################

case = 1 #We'll be comparing our new grid to the spectrum.
filenames = 'interpolated_names.txt'
#path = '/srv/two/jtfuchs/Interpolated_Models/10teff05logg/center' + str(bestT) + '_' + str(bestg)
path = '/srv/two/jtfuchs/Interpolated_Models/1000K_1g/bottom' + str(bestT) + '_' + str(bestg)
ncflux,bestT,bestg = intspecs(alllambda,allnline,allsigma,lambdaindex,case,filenames,lambdas,zzcetiblue,zzcetired,FWHM,indices,path)

