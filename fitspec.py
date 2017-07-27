"""
Fitting routine to determine Teff and logg for ZZ Cetis

Author: Josh Fuchs, UNC. With substantial initial work done by Bart Dunlap.

:INPUTS:
       zzcetiblue: string, file containing 1D wavelength calibrated 'ZZ Ceti blue' spectrum

:OPTIONAL:
       zzcetired: string, file containing 1D wavelength calibrated 'ZZ Ceti red' spectrum.

       --fitguess: string, sets how to determine initial guesses for pseudogaussian fitting. Options: data, model. Default: data

       --higherlines: string, sets which higher order lines to fit. Either H gamma through H10 or H gamma through H 11. Options: g10, g11. Defauls: g10

       --res: float, resolution in Angstroms. Used to determine how to convolve the models. Only used if inputting a text file. 

:OUTPUTS:
      ALL CAPS BELOW means variable determined by program. WDNAME is the name of the input spectrum. DATE is the date fitspec.py was run. OPTIONS is optional inputs by user.

       fit_WDNAME_DATE_OPTIONS.pdf: pdf containing pseudogaussian fit to spectrum. Each fit is shown on a separate page.

       lambdaoffset_WDNAME_DATE.pdf: pdf showing computed lambda offset. Each individual line fit is shown, then H beta through H epsilon are used to calculate offset.

       offsets_WDNAME_DATE_FWHM.pdf: pdf showing normalization offsets for normalization of Balmer lines.

       params_WDNAME_DATE_OPTIONS.txt: text file containg initial guesses and final fitting parameters from pseudogaussians. Columns are: Halpha guess, Hbeta guess, Hgamma-H10 guess, Halpha best fit, Hbeta best fit, Hgamma-H10 best fit

       norm_WDNAME_DATE_OPTIONS.txt: text file containing normalized spectrum. Columns are wavelength, normalized flux, sigma

       model_WDNAME_DATE_OPTIONS.txt: best-fitting normalized model.

       chi*WDNAME*LINE*txt: text file containing chi-square surface of each individual balmer LINE and the combined fit. Grid information is in first line.

:TO RUN:
       python fitspec.py zzcetiblue zzcetired
       python fitspec.py wtfb.wd1425-811_930_blue_flux_model.ms.fits wtfb.wd1425-811_930_red_flux_model.ms.fits
       python fitspec.py wtfb.wd1425-811_930_blue_flux_model.ms.fits wtfb.wd1425-811_930_red_flux_model.ms.fits --fitguess data --higherlines g10 

"""
#This program uses MPFIT to fit a pseudogaussian to the balmer lines of interest. Therefore you need to have mpfit.py in the same directory. You want to have the version written by Sergei Koposov as that is the most recent and uses numpy.

import numpy as np
import sys
import os
import datetime
from scipy.optimize import leastsq
if os.getcwd()[0:4] == '/pro': #Check if we are on Hatteras
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
else:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
import mpfit
from intspec import intspecs
import astropy.io.fits as fits
#import pyfits as fits # Infierno doesn't support astropy for some reason so using pyfits
from glob import glob
import argparse


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

#Sing pseudogaussian plus cubic for continuum
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

#Third order continuum plus 7 pseudogaussians.
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

#Third order continuum plus 6 pseudogaussians.
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

#Second order continuum plut 5 pseudogaussians
def multipseudogauss(x,p):
    return p[0]*1. + p[1]*x + p[2]*x**2. + p[3]*np.exp(-(np.abs(x-p[4])/(np.sqrt(2.)*p[5]))**p[6]) + p[7]*np.exp(-(np.abs(x-p[8])/(np.sqrt(2.)*p[9]))**p[10]) + p[11]*np.exp(-(np.abs(x-p[12])/(np.sqrt(2.)*p[13]))**p[14]) + p[15]*np.exp(-(np.abs(x-p[16])/(np.sqrt(2.)*p[17]))**p[18]) + p[19]*np.exp(-(np.abs(x-p[20])/(np.sqrt(2.)*p[21]))**p[22])

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


def find_offset(z,p):
    return p[0] + z

def fit_find_offset(p,fjac=None,x=None, y=None, z=None,err=None):
    #Parameter values are passed in p
    #fjac=None just means partial derivatives will not be computed.
    model = find_offset(z,p)
    status = 0
    return([status,(y-model)/err])

# ===========================================================================


def fit_offset(wavelengths,data,fit,wavelength_norm,sigma_data):
    est = np.array([np.mean(data)-np.mean(fit)])
    fa = {'x':wavelengths,'y':data,'z':fit,'err':sigma_data}
    offset_params = mpfit.mpfit(fit_find_offset,est,functkw=fa,maxiter=2000,ftol=1e-14,xtol=1e-13,quiet=True)
    #print offset_params.params

    #Find value at normalization wavelength
    norm_index = np.min(np.where(wavelengths > wavelength_norm))
    refit = find_offset(fit,offset_params.params)
    norm_value = refit[norm_index]
    #print norm_value
    plt.clf()
    plt.plot(wavelengths,data,'k')
    plt.plot(wavelengths,fit,'r',label='Original Fit',linewidth=2.0)
    plt.plot(wavelengths,find_offset(fit,offset_params.params),'b',label='With Offset',linewidth=2.0)
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.legend()
    #plt.show()
    offsetpdf.savefig()
    return norm_value


# ===========================================================================
def shiftinvoverc(lambdas,dataval,sigmaval,plotall=True):
    '''
    Fits H beta through H10 to determine delta lambda offset that is then applied to whole spectrum. Only beta through epsilon are used to determine delta lambda.
    '''
    print 'Determining offset in delta lambda'
    #Ensure arrays
    lambdas = np.asarray(lambdas)
    dataval = np.asarray(dataval)
    sigmaval = np.asarray(sigmaval)


    #First, fit H beta to get rough idea of offset magnitude
    betafitwavelengthlow = 4680. #4680
    betafitwavelengthhigh = 5040. #5040
    
    bfitlow = np.min(np.where(lambdas > betafitwavelengthlow))
    bfithi = np.min(np.where(lambdas > betafitwavelengthhigh))
    blambdas = lambdas[bfitlow:bfithi+1]
    bsigmas = sigmaval[bfitlow:bfithi+1]
    betaval = dataval[bfitlow:bfithi+1]
    
    best = np.zeros(8)
    xes = np.array([lambdas[bfitlow],lambdas[bfitlow+5],lambdas[bfitlow+10],lambdas[bfithi-10],lambdas[bfithi]])
    yes = np.array([dataval[bfitlow],dataval[bfitlow+5],dataval[bfitlow+10],dataval[bfithi-10],dataval[bfithi]])
    bp = np.polyfit(xes,yes,3)
    bpp = np.poly1d(bp)
    best[0] = bp[3]
    best[1] = bp[2]
    best[2] = bp[1]
    best[7] = bp[0]
    best[4] =  blambdas[np.min(np.where(betaval == betaval.min()))]#minimum value in fitting array. Estimate for line center
    best[3] = np.min(dataval[bfitlow:bfithi+1]) - bpp(best[4]) #depth of line relative to continuum
    bhalfmax = bpp(best[4]) + best[3]/2.5
    bdiff = np.abs(betaval-bhalfmax)
    blowidx = bdiff[np.where(blambdas < best[4])].argmin()
    bhighidx = bdiff[np.where(blambdas > best[4])].argmin() + len(bdiff[np.where(blambdas < best[4])])
    best[5] = (blambdas[bhighidx] - blambdas[blowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
    best[6] = 1.0 #how much of a pseudo-gaussian

    bfa = {'x':blambdas, 'y':betaval, 'err':bsigmas}
    bparams = mpfit.mpfit(fitpseudogausscubic,best,functkw=bfa,maxiter=3000,ftol=1e-16,xtol=1e-10,quiet=True)

    betaoffset =  bparams.params[4] - 4862.4555


    ####################
    #Now fit smaller portions of lines
    ####################


    #First, fit H beta
    betafitwavelengthlow = 4815. + betaoffset #4680
    betafitwavelengthhigh = 4930. + betaoffset #5040
    
    bfitlow = np.min(np.where(lambdas > betafitwavelengthlow))
    bfithi = np.min(np.where(lambdas > betafitwavelengthhigh))
    blambdas = lambdas[bfitlow:bfithi+1]
    bsigmas = sigmaval[bfitlow:bfithi+1]
    betaval = dataval[bfitlow:bfithi+1]
    
    best = np.zeros(8)
    xes = np.array([lambdas[bfitlow],lambdas[bfitlow+5],lambdas[bfitlow+10],lambdas[bfithi-10],lambdas[bfithi]])
    yes = np.array([dataval[bfitlow],dataval[bfitlow+5],dataval[bfitlow+10],dataval[bfithi-10],dataval[bfithi]])
    bp = np.polyfit(xes,yes,2)
    bpp = np.poly1d(bp)
    best[0] = bp[2]
    best[1] = bp[1]
    best[2] = bp[0]
    #best[7] = bp[0]
    best[7] = 0.
    best[4] =  blambdas[np.min(np.where(betaval == betaval.min()))]#minimum value in fitting array. Estimate for line center
    best[3] = np.min(dataval[bfitlow:bfithi+1]) - bpp(best[4]) #depth of line relative to continuum
    bhalfmax = bpp(best[4]) + best[3]/2.5
    bdiff = np.abs(betaval-bhalfmax)
    blowidx = bdiff[np.where(blambdas < best[4])].argmin()
    bhighidx = bdiff[np.where(blambdas > best[4])].argmin() + len(bdiff[np.where(blambdas < best[4])])
    best[5] = (blambdas[bhighidx] - blambdas[blowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
    best[6] = 1.0 #how much of a pseudo-gaussian

    vparaminfo = [{'limits':[0,0],'limited':[0,0],'fixed':0} for i in range(8)]
    vparaminfo[7]['fixed'] = 1.
    bfa = {'x':blambdas, 'y':betaval, 'err':bsigmas}
    bparams = mpfit.mpfit(fitpseudogausscubic,best,functkw=bfa,maxiter=3000,ftol=1e-16,xtol=1e-10,quiet=True,parinfo=vparaminfo)

    bline_center = bparams.params[4]
    bline_fit = pseudogausscubic(blambdas,bparams.params)

    betaoffset = bline_center - 4862.4555

    #Save fits to PDF file
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    global idname
    savevoverc = 'lambdaoffset_' + idname[6:] + '_' + now[5:10] + '.pdf'
    savevovercpdf = PdfPages(savevoverc)

    #plt.figure(1)
    plt.clf()
    plt.plot(blambdas,betaval,'k^',label='data')
    plt.plot(blambdas,bline_fit,'r',label='fit')
    plt.title('Beta fit')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    #plt.plot(blambdas,pseudogausscubic(blambdas,best),'g',label='guess')
    savevovercpdf.savefig()
    #plt.plot(blambdas,bparams.params[0]*1. + bparams.params[1]*blambdas+bparams.params[2]*blambdas**2.)
    #plt.show()

    #Now gamma
    gammafitwavelengthlow = 4300. + betaoffset #4200
    gammafitwavelengthhigh = 4400. + betaoffset #4510
    gfitlow = np.min(np.where(lambdas > gammafitwavelengthlow))
    gfithi = np.min(np.where(lambdas > gammafitwavelengthhigh))

    glambdas = lambdas[gfitlow:gfithi+1]
    gsigmas = sigmaval[gfitlow:gfithi+1]
    gval = dataval[gfitlow:gfithi+1]
    
    gest = np.zeros(8)
    xes = np.array([lambdas[gfitlow],lambdas[gfitlow+5],lambdas[gfitlow+10],lambdas[gfithi-10],lambdas[gfithi-5],lambdas[gfithi]])
    yes = np.array([dataval[gfitlow],dataval[gfitlow+5],dataval[gfitlow+10],dataval[gfithi-10],dataval[gfithi-5],dataval[gfithi]])
    gp = np.polyfit(xes,yes,2)
    gpp = np.poly1d(gp)
    gest[0] = gp[2]
    gest[1] = gp[1]
    gest[2] = gp[0]
    #gest[7] = gp[0]
    gest[7] = 0.
    gest[4] =  glambdas[np.min(np.where(gval == gval.min()))]#minimum value in fitting array. Estimate for line center
    gest[3] = np.min(dataval[gfitlow:gfithi+1]) - gpp(gest[4]) #depth of line relative to continuum
    ghalfmax = gpp(gest[4]) + gest[3]/2.5
    gdiff = np.abs(gval-ghalfmax)
    glowidx = gdiff[np.where(glambdas < gest[4])].argmin()
    ghighidx = gdiff[np.where(glambdas > gest[4])].argmin() + len(gdiff[np.where(glambdas < gest[4])])
    gest[5] = (glambdas[ghighidx] - glambdas[glowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
    gest[6] = 1.0 #how much of a pseudo-gaussian
    
    vparaminfo = [{'limits':[0,0],'limited':[0,0],'fixed':0} for i in range(8)]
    vparaminfo[7]['fixed'] = 1.
    gfa = {'x':glambdas, 'y':gval, 'err':gsigmas}
    gparams = mpfit.mpfit(fitpseudogausscubic,gest,functkw=gfa,maxiter=3000,ftol=1e-16,xtol=1e-10,quiet=True,parinfo=vparaminfo)
    
    gline_center = gparams.params[4]
    gline_fit = pseudogausscubic(glambdas,gparams.params)

    #plt.figure(2)
    plt.clf()
    plt.plot(glambdas,gval,'k^',label='data')
    plt.plot(glambdas,gline_fit,'r',label='fit')
    #plt.plot(glambdas,pseudogausscubic(glambdas,gest),'g',label='guess')
    plt.title('Gamma fit')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    savevovercpdf.savefig()
    #plt.plot(blambdas,bparams.params[0]*1. + bparams.params[1]*blambdas+bparams.params[2]*blambdas**2.)
    #plt.show()

    #Now delta
    deltafitwavelengthlow = 4035. + betaoffset#4030
    deltafitwavelengthhigh = 4185. + betaoffset#4210
    dfitlow = np.min(np.where(lambdas > deltafitwavelengthlow))
    dfithi = np.min(np.where(lambdas > deltafitwavelengthhigh))

    dlambdas = lambdas[dfitlow:dfithi+1]
    dsigmas = sigmaval[dfitlow:dfithi+1]
    dval = dataval[dfitlow:dfithi+1]
    
    dest = np.zeros(8)
    xes = np.array([lambdas[dfitlow],lambdas[dfitlow+5],lambdas[dfitlow+10],lambdas[dfithi-10],lambdas[dfithi-5],lambdas[dfithi]])
    yes = np.array([dataval[dfitlow],dataval[dfitlow+5],dataval[dfitlow+10],dataval[dfithi-10],dataval[dfithi-5],dataval[dfithi]])
    dp = np.polyfit(xes,yes,2)
    dpp = np.poly1d(dp)
    dest[0] = dp[2]
    dest[1] = dp[1]
    dest[2] = dp[0]
    #dest[7] = dp[0]
    dest[7] = 0.
    dest[4] =  dlambdas[np.min(np.where(dval == dval.min()))]#minimum value in fitting array. Estimate for line center
    dest[3] = np.min(dataval[dfitlow:dfithi+1]) - dpp(dest[4]) #depth of line relative to continuum
    dhalfmax = dpp(dest[4]) + dest[3]/2.5
    ddiff = np.abs(dval-dhalfmax)
    dlowidx = ddiff[np.where(dlambdas < dest[4])].argmin()
    dhighidx = ddiff[np.where(dlambdas > dest[4])].argmin() + len(ddiff[np.where(dlambdas < dest[4])])
    dest[5] = (dlambdas[dhighidx] - dlambdas[dlowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
    dest[6] = 1.0 #how much of a pseudo-gaussian
    
    vparaminfo = [{'limits':[0,0],'limited':[0,0],'fixed':0} for i in range(8)]
    vparaminfo[7]['fixed'] = 1.
    dfa = {'x':dlambdas, 'y':dval, 'err':dsigmas}
    dparams = mpfit.mpfit(fitpseudogausscubic,dest,functkw=dfa,maxiter=3000,ftol=1e-16,xtol=1e-10,quiet=True,parinfo=vparaminfo)
    
    dline_center = dparams.params[4]
    dline_fit = pseudogausscubic(dlambdas,dparams.params)
    #plt.figure(3)
    plt.clf()
    plt.plot(dlambdas,dval,'k^',label='data')
    plt.plot(dlambdas,dline_fit,'r',label='fit')
    #plt.plot(dlambdas,pseudogausscubic(dlambdas,dest),'g',label='guess')
    plt.title('Delta fit')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    savevovercpdf.savefig()
    #plt.plot(blambdas,bparams.params[0]*1. + bparams.params[1]*blambdas+bparams.params[2]*blambdas**2.)
    #plt.show()

    #Now epsilon
    epfitwavelengthlow = 3930. + betaoffset
    epfitwavelengthhigh = 4015. + betaoffset
    efitlow = np.min(np.where(lambdas > epfitwavelengthlow))
    efithi = np.min(np.where(lambdas > epfitwavelengthhigh))

    elambdas = lambdas[efitlow:efithi+1]
    esigmas = sigmaval[efitlow:efithi+1]
    epval = dataval[efitlow:efithi+1]
    
    eest = np.zeros(8)
    xes = np.array([lambdas[efitlow],lambdas[efitlow+5],lambdas[efitlow+10],lambdas[efithi-10],lambdas[efithi-5],lambdas[efithi]])
    yes = np.array([dataval[efitlow],dataval[efitlow+5],dataval[efitlow+10],dataval[efithi-10],dataval[efithi-5],dataval[efithi]])
    ep = np.polyfit(xes,yes,2)
    epp = np.poly1d(ep)
    eest[0] = ep[2]
    eest[1] = ep[1]
    eest[2] = ep[0]
    #eest[7] = ep[0]
    eest[7] = 0.
    eest[4] =  elambdas[np.min(np.where(epval == epval.min()))]#minimum value in fitting array. Estimate for line center
    eest[3] = np.min(dataval[efitlow:efithi+1]) - epp(eest[4]) #depth of line relative to continuum
    ehalfmax = epp(eest[4]) + eest[3]/2.5
    ediff = np.abs(epval-ehalfmax)
    elowidx = ediff[np.where(elambdas < eest[4])].argmin()
    ehighidx = ediff[np.where(elambdas > eest[4])].argmin() + len(ediff[np.where(elambdas < eest[4])])
    eest[5] = (elambdas[ehighidx] - elambdas[elowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
    eest[6] = 1.0 #how much of a pseudo-gaussian
    
    vparaminfo = [{'limits':[0,0],'limited':[0,0],'fixed':0} for i in range(8)]
    vparaminfo[7]['fixed'] = 1.
    efa = {'x':elambdas, 'y':epval, 'err':esigmas}
    eparams = mpfit.mpfit(fitpseudogausscubic,eest,functkw=efa,maxiter=3000,ftol=1e-16,xtol=1e-10,quiet=True,parinfo=vparaminfo)
    
    eline_center = eparams.params[4]
    eline_fit = pseudogausscubic(elambdas,eparams.params)

    #plt.figure(4)
    plt.clf()
    plt.plot(elambdas,epval,'k^',label='data')
    plt.plot(elambdas,eline_fit,'r',label='fit')
    #plt.plot(elambdas,pseudogausscubic(elambdas,eest),'g',label='guess')
    plt.title('Epsilon fit')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    savevovercpdf.savefig()
    #plt.plot(blambdas,bparams.params[0]*1. + bparams.params[1]*blambdas+bparams.params[2]*blambdas**2.)
    #plt.show()
    
    #Now H8
    H8fitwavelengthlow = 3870. + betaoffset
    H8fitwavelengthhigh = 3910. + betaoffset
    H8fitlow = np.min(np.where(lambdas > H8fitwavelengthlow))
    H8fithi = np.min(np.where(lambdas > H8fitwavelengthhigh))

    H8lambdas = lambdas[H8fitlow:H8fithi+1]
    H8sigmas = sigmaval[H8fitlow:H8fithi+1]
    H8val = dataval[H8fitlow:H8fithi+1]
    
    H8est = np.zeros(8)
    xes = np.array([lambdas[H8fitlow],lambdas[H8fitlow+3],lambdas[H8fithi-3],lambdas[H8fithi]])
    yes = np.array([dataval[H8fitlow],dataval[H8fitlow+3],dataval[H8fithi-3],dataval[H8fithi]])
    H8p = np.polyfit(xes,yes,1)
    H8pp = np.poly1d(H8p)
    H8est[0] = H8p[1]
    H8est[1] = H8p[0]
    #H8est[2] = H8p[0]
    H8est[2] = 0.
    #H8est[7] = H8p[0]
    H8est[7] = 0.
    H8est[4] =  H8lambdas[np.min(np.where(H8val == H8val.min()))]#minimum value in fitting array. Estimate for line center
    H8est[3] = np.min(dataval[H8fitlow:H8fithi+1]) - H8pp(H8est[4]) #depth of line relative to continuum
    H8halfmax = H8pp(H8est[4]) + H8est[3]/2.5
    H8diff = np.abs(H8val-H8halfmax)
    H8lowidx = H8diff[np.where(H8lambdas < H8est[4])].argmin()
    H8highidx = H8diff[np.where(H8lambdas > H8est[4])].argmin() + len(H8diff[np.where(H8lambdas < H8est[4])])
    H8est[5] = (H8lambdas[H8highidx] - H8lambdas[H8lowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
    H8est[6] = 2.0 #how much of a pseudo-gaussian
    
    vparaminfo = [{'limits':[0,0],'limited':[0,0],'fixed':0} for i in range(8)]
    vparaminfo[7]['fixed'] = 1.
    vparaminfo[2]['fixed'] = 1.
    vparaminfo[6]['fixed'] = 1.
    #vparaminfo[0]['fixed'] = 1.
    #vparaminfo[1]['fixed'] = 1.
    H8fa = {'x':H8lambdas, 'y':H8val, 'err':H8sigmas}
    H8params = mpfit.mpfit(fitpseudogausscubic,H8est,functkw=H8fa,maxiter=3000,ftol=1e-16,xtol=1e-10,quiet=True,parinfo=vparaminfo)
    
    H8line_center = H8params.params[4]
    H8line_fit = pseudogausscubic(H8lambdas,H8params.params)

    #plt.figure(5)
    plt.clf()
    plt.plot(H8lambdas,H8val,'k^',label='data')
    plt.plot(H8lambdas,H8line_fit,'r',label='fit')
    #plt.plot(H8lambdas,pseudogausscubic(H8lambdas,H8est),'g',label='guess')
    plt.title('H8 fit')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    savevovercpdf.savefig()
    #plt.plot(blambdas,bparams.params[0]*1. + bparams.params[1]*blambdas+bparams.params[2]*blambdas**2.)
    #plt.show()

    #Now H9
    H9fitwavelengthlow = 3820. + betaoffset
    H9fitwavelengthhigh = 3850. + betaoffset
    H9fitlow = np.min(np.where(lambdas > H9fitwavelengthlow))
    H9fithi = np.min(np.where(lambdas > H9fitwavelengthhigh))

    H9lambdas = lambdas[H9fitlow:H9fithi+1]
    H9sigmas = sigmaval[H9fitlow:H9fithi+1]
    H9val = dataval[H9fitlow:H9fithi+1]
    
    H9est = np.zeros(8)
    xes = np.array([lambdas[H9fitlow],lambdas[H9fitlow+3],lambdas[H9fithi-3],lambdas[H9fithi]])
    yes = np.array([dataval[H9fitlow],dataval[H9fitlow+3],dataval[H9fithi-3],dataval[H9fithi]])
    H9p = np.polyfit(xes,yes,1)
    H9pp = np.poly1d(H9p)
    H9est[0] = H9p[1]
    H9est[1] = H9p[0]
    #H9est[2] = H9p[0]
    H9est[2] = 0.
    #H9est[7] = H9p[0]
    H9est[7] = 0.
    H9est[4] =  H9lambdas[np.min(np.where(H9val == H9val.min()))]#minimum value in fitting array. Estimate for line center
    H9est[3] = np.min(dataval[H9fitlow:H9fithi+1]) - H9pp(H9est[4]) #depth of line relative to continuum
    H9halfmax = H9pp(H9est[4]) + H9est[3]/2.5
    H9diff = np.abs(H9val-H9halfmax)
    H9lowidx = H9diff[np.where(H9lambdas < H9est[4])].argmin()
    H9highidx = H9diff[np.where(H9lambdas > H9est[4])].argmin() + len(H9diff[np.where(H9lambdas < H9est[4])])
    H9est[5] = (H9lambdas[H9highidx] - H9lambdas[H9lowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
    H9est[6] = 2.0 #how much of a pseudo-gaussian
    

    vparaminfo = [{'limits':[0,0],'limited':[0,0],'fixed':0} for i in range(8)]
    vparaminfo[7]['fixed'] = 1.
    vparaminfo[2]['fixed'] = 1.
    vparaminfo[6]['fixed'] = 1.
    H9fa = {'x':H9lambdas, 'y':H9val, 'err':H9sigmas}
    H9params = mpfit.mpfit(fitpseudogausscubic,H9est,functkw=H9fa,maxiter=3000,ftol=1e-16,xtol=1e-10,quiet=True,parinfo=vparaminfo)
    
    H9line_center = H9params.params[4]
    H9line_fit = pseudogausscubic(H9lambdas,H9params.params)

    #plt.figure(6)
    plt.clf()
    plt.plot(H9lambdas,H9val,'k^',label='data')
    plt.plot(H9lambdas,H9line_fit,'r',label='fit')
    #plt.plot(H9lambdas,pseudogausscubic(H9lambdas,H9est),'g',label='guess')
    plt.title('H9 fit')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    savevovercpdf.savefig()
    #plt.plot(blambdas,bparams.params[0]*1. + bparams.params[1]*blambdas+bparams.params[2]*blambdas**2.)
    #plt.show()

    #Now H10
    H10fitwavelengthlow = 3785. + betaoffset
    H10fitwavelengthhigh = 3815. + betaoffset
    H10fitlow = np.min(np.where(lambdas > H10fitwavelengthlow))
    H10fithi = np.min(np.where(lambdas > H10fitwavelengthhigh))

    H10lambdas = lambdas[H10fitlow:H10fithi+1]
    H10sigmas = sigmaval[H10fitlow:H10fithi+1]
    H10val = dataval[H10fitlow:H10fithi+1]
    
    H10est = np.zeros(8)
    xes = np.array([lambdas[H10fitlow],lambdas[H10fitlow+3],lambdas[H10fithi-3],lambdas[H10fithi]])
    yes = np.array([dataval[H10fitlow],dataval[H10fitlow+3],dataval[H10fithi-3],dataval[H10fithi]])
    H10p = np.polyfit(xes,yes,1)
    H10pp = np.poly1d(H10p)
    H10est[0] = H10p[1]
    H10est[1] = H10p[0]
    #H10est[2] = H10p[0]
    H10est[2] = 0.
    #H10est[7] = H10p[0]
    H10est[7] = 0.
    H10est[4] =  H10lambdas[np.min(np.where(H10val == H10val.min()))]#minimum value in fitting array. Estimate for line center
    H10est[3] = np.min(dataval[H10fitlow:H10fithi+1]) - H10pp(H10est[4]) #depth of line relative to continuum
    H10halfmax = H10pp(H10est[4]) + H10est[3]/2.5
    H10diff = np.abs(H10val-H10halfmax)
    H10lowidx = H10diff[np.where(H10lambdas < H10est[4])].argmin()
    H10highidx = H10diff[np.where(H10lambdas > H10est[4])].argmin() + len(H10diff[np.where(H10lambdas < H10est[4])])
    H10est[5] = (H10lambdas[H10highidx] - H10lambdas[H10lowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
    H10est[6] = 2.0 #how much of a pseudo-gaussian
    
    vparaminfo = [{'limits':[0,0],'limited':[0,0],'fixed':0} for i in range(8)]
    vparaminfo[7]['fixed'] = 1.
    vparaminfo[2]['fixed'] = 1.
    vparaminfo[6]['fixed'] = 1.
    H10fa = {'x':H10lambdas, 'y':H10val, 'err':H10sigmas}
    H10params = mpfit.mpfit(fitpseudogausscubic,H10est,functkw=H10fa,maxiter=3000,ftol=1e-16,xtol=1e-10,quiet=True,parinfo=vparaminfo)
    
    H10line_center = H10params.params[4]
    H10line_fit = pseudogausscubic(H10lambdas,H10params.params)

    #plt.figure(7)
    plt.clf()
    plt.plot(H10lambdas,H10val,'k^',label='data')
    plt.plot(H10lambdas,H10line_fit,'r',label='fit')
    #plt.plot(H10lambdas,pseudogausscubic(H10lambdas,H10est),'g',label='guess')
    plt.title('H10 fit')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    savevovercpdf.savefig()
    #plt.plot(blambdas,bparams.params[0]*1. + bparams.params[1]*blambdas+bparams.params[2]*blambdas**2.)
    #plt.show()

    '''
    bvoverc = (bline_center - 4862.4555) / 4862.4555
    gvoverc = (gline_center - 4341.4834) / 4341.4834
    dvoverc = (dline_center - 4103.0343) / 4103.0343
    evoverc = (eline_center - 3971.4475) / 3971.4475
    H8voverc = (H8line_center - 3890.2759) / 3890.2759
    H9voverc = (H9line_center - 3836.1585) / 3836.1585
    H10voverc = (H10line_center - 3799.0785) / 3799.0785
    '''
    deltab = (bline_center - 4862.4555)
    deltag = (gline_center - 4341.4834) 
    deltad = (dline_center - 4103.0343)
    deltae = (eline_center - 3971.4475)
    deltaH8 = (H8line_center - 3890.2759)
    deltaH9 = (H9line_center - 3836.1585)
    deltaH10 = (H10line_center - 3799.0785)

    '''
    print bvoverc,gvoverc,dvoverc, evoverc, H8voverc, H9voverc, H10voverc
    avgvoverc = np.mean([bvoverc,gvoverc,dvoverc,evoverc,H8voverc,H9voverc,H10voverc])
    stdvoverc = np.std([bvoverc,gvoverc,dvoverc,evoverc,H8voverc,H9voverc,H10voverc])
    print avgvoverc
    '''
    
    deltaall = np.array([deltaH10,deltaH9,deltaH8,deltae,deltad,deltag,deltab])
    allfitcenters = np.array([H10line_center,H9line_center,H8line_center,eline_center,dline_center,gline_center,bline_center])
    #allvoverc = np.array([H10voverc,H9voverc,H8voverc,evoverc,dvoverc,gvoverc,bvoverc])
    allcenters = np.array([3799.0785,3836.1585,3890.2759,3971.4475,4103.0343,4341.4834,4862.4555])
    #Fit a line to only beta through epsilon
    line_fit = np.polyfit(allcenters[3:],deltaall[3:],1.)
    line_fitted = np.poly1d(line_fit)

    #The average offset from beta through epsilon will be the delta lambda we apply to the whole wavelength array
    avgdeltalambda = np.mean(deltaall[3:])

    #Determine predictions for line centers of 9 and 10 
    predict9 = allcenters[1] + line_fitted(allcenters[1]) - avgdeltalambda
    predict10 = allcenters[0] + line_fitted(allcenters[0]) - avgdeltalambda
    constraints910 = np.array([predict9,predict10])
    print 'Delta lambda shift:', avgdeltalambda
    print 'Predicted centers of 9 and 10: ', constraints910

    #plt.figure(8)
    plt.clf()
    plt.plot(allcenters,deltaall,'bo')
    plt.plot(allcenters,line_fitted(allcenters),'r--')
    plt.title('Fitted offset')
    plt.xlabel('Wavelength')
    plt.ylabel('Delta lambda')
    savevovercpdf.savefig()
    #plt.show()
    #plt.figure(9)
    #plt.clf()
    #line_fit = np.polyfit(allcenters[3:],allvoverc[3:],1.)
    #line_fitted = np.poly1d(line_fit)
    #plt.plot(allcenters,allvoverc,'bo')
    #plt.plot(allcenters,line_fitted(allcenters),'r--')
    #savevovercpdf.savefig()
    #savevovercpdf.close()
    #plt.show()
    #exit()

    #newlambdas = lambdas - lambdas*avgvoverc
    #newlambdas = lambdas - lambdas*line_fitted(lambdas)
    newlambdas = lambdas - avgdeltalambda

    plt.clf()
    plt.plot(newlambdas,dataval,'b')
    plt.xlim(3765,3945)
    #plt.axvline(3890.2759,ls='--',color='r')
    #plt.axvline(3836.1585,ls='--',color='r')
    #plt.axvline(3799.0785,ls='--',color='r')
    plt.axvline(predict9,ls='--',color='r')
    plt.axvline(predict10,ls='--',color='r')
    plt.title('Predictions')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    savevovercpdf.savefig()
    savevovercpdf.close()
    #exit()

    #plt.clf()
    #plt.plot(lambdas,dataval,'r')
    #plt.plot(newlambdas,dataval,'k')
    #plt.show()
    #exit()

    return newlambdas, constraints910, allfitcenters, avgdeltalambda


# ===========================================================================
# ===========================================================================
# ===========================================================================
# ===========================================================================
# ===========================================================================


def fit_now(zzcetiblue,zzcetired,redfile,fitguess='data',higherlines='g10',res=None):
    if zzcetiblue.lower()[-4:] == 'fits':
        print 'Reading in fits file: ', zzcetiblue
        #Read in the blue spectrum
        datalistblue = fits.open(zzcetiblue)
        datavalblue = datalistblue[0].data[0,0,:] #Reads in the object spectrum,data[0,0,:] is optimally subtracted, data[1,0,:] is raw extraction,  data[2,0,:] is sky, data[3,0,:] is sigma spectrum
        sigmavalblue = datalistblue[0].data[3,0,:] #Sigma spectrum

        #Header values to save
        RA = datalistblue[0].header['RA']
        DEC = datalistblue[0].header['DEC']
        SNR = float(datalistblue[0].header['SNR'])
        airmass = float(datalistblue[0].header['AIRMASS'])
        nexp = float(datalistblue[0].header['NCOMBINE'])
        exptime = float(datalistblue[0].header['EXPTIME'])

        #ID from zzcetiblue for saving and identification of files
        endpoint = '.ms.'
        global idname
        idname = zzcetiblue[zzcetiblue.find('w'):zzcetiblue.find(endpoint)] 
        
        #Read in FWHM of blue spectrum from npy binary file. Use changing values. If no file exists, use linearized wavelengths
        try:
            FWHMbluefilename_struc = zzcetiblue[0:zzcetiblue.find('w')] + '*' + zzcetiblue[zzcetiblue.find('b'):zzcetiblue.find('_flux')] + '_poly.npy'
            FWHMbluefilename = glob(FWHMbluefilename_struc)
            FWHMpixblue = np.load(FWHMbluefilename[0])
            print 'Found file: ', FWHMbluefilename[0]
            print 'Using ', FWHMbluefilename[0], ' for convolution'
        except:
            #Read in FWHM of blue spectrum from header. Use one value only. If you change which one you use, don't forget to change it below too.
            FWHMpixblue_val = datalistblue[0].header['specfwhm'] 
            FWHMpixblue = FWHMpixblue_val * np.ones(len(datavalblue))
            print  'Using ', FWHMpixblue_val, ' for convolution'


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
        try:
            biningblue= float( datalistblue[0].header["PARAM18"] ) 
        except:
            biningblue= float( datalistblue[0].header["PG3_2"] ) 
        nxblue= np.size(datavalblue)#spec_data[0]
        PixelsBlue= biningblue*(np.arange(0,nxblue,1)+trim_offset_blue)
        lambdasblue = DispCalc(PixelsBlue, alphablue, thetablue, frblue, fdblue, flblue, zPntblue)

        #Offset in v/c
        lambdasblue, constraints910, firstfitcenters, avgdelta = shiftinvoverc(lambdasblue,datavalblue,sigmavalblue,plotall=True)
        print 'Done offsetting in delta lambda.'
        
        
        #Mask out the Littrow Ghost if 'LTTROW' is in the image header
        try:
            print 'Masking littrow ghost.'
            littrow_ghost = datalistblue[0].header['LITTROW']
            littrow_mask_low = int(float(littrow_ghost[1:len(littrow_ghost)-1].split(',')[0]))# - float(trim_offset_blue))
            littrow_mask_high = int(float(littrow_ghost[1:len(littrow_ghost)-1].split(',')[1]))# - float(trim_offset_blue))
            lambdasblue = np.concatenate((lambdasblue[:littrow_mask_low+1],lambdasblue[littrow_mask_high:]))
            datavalblue = np.concatenate((datavalblue[:littrow_mask_low+1],datavalblue[littrow_mask_high:]))
            sigmavalblue = np.concatenate((sigmavalblue[:littrow_mask_low+1],sigmavalblue[littrow_mask_high:]))
            FWHMpixblue = np.concatenate((FWHMpixblue[:littrow_mask_low+1],FWHMpixblue[littrow_mask_high:]))
        except:
            print 'No mask for Littrow ghost'
            pass

        #Read in the red spectrum
        if redfile:
            datalistred = fits.open(zzcetired)
            datavalred = datalistred[0].data[0,0,:] #data[0,0,:] is optimally extracted, data[2,0,:] is sky
            sigmavalred = datalistred[0].data[3,0,:] #Sigma spectrum
            #Read in FWHM of red spectrum from npy binary file. Use changing values. If no file exists, use linearized wavelengths
            try:
                FWHMredfilename_struc = zzcetired[0:zzcetired.find('w')] + '*' + zzcetired[zzcetired.find('b'):zzcetired.find('_flux')] + '_poly.npy'
                FWHMredfilename = glob(FWHMredfilename_struc)
                FWHMpixred = np.load(FWHMredfilename[0])
                print 'Found file: ', FWHMredfilename[0]
                print 'Using ', FWHMredfilename[0], ' for convolution'
            except:
                #Read in FWHM of red spectrum from header. Use one value only. If you change which one you use, don't forget to change it below too.
                FWHMpixred_val = datalistred[0].header['specfwhm'] 
                FWHMpixred = FWHMpixred_val * np.ones(len(datavalred))
                print  'Using ', FWHMpixred_val, ' for convolution'
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
            try:
                biningred= float( datalistred[0].header["PARAM18"] ) 
            except:
                biningred= float( datalistred[0].header["PG3_2"] ) 
            nxred= np.size(datavalred)#spec_data[0]
            PixelsRed= biningred*(np.arange(0,nxred,1)+trim_offset_red)
            lambdasred = DispCalc(PixelsRed, alphared, thetared, frred, fdred, flred, zPntred)

        #Concatenate both into two arrays
        if redfile:
            lambdas = np.concatenate((lambdasblue,lambdasred))
            dataval = np.concatenate((datavalblue,datavalred))
            sigmaval = np.concatenate((sigmavalblue,sigmavalred))#2.e-17 * np.ones(len(dataval))
            FWHM = (lambdasblue[-1] - lambdasblue[0])/nxblue * np.concatenate((FWHMpixblue,FWHMpixred)) #from grating equation
            #FWHM = deltawavblue * np.concatenate((FWHMpixblue,FWHMpixred)) #FWHM in Angstroms linearized
        else:
            lambdas = np.array(lambdasblue)
            dataval = np.array(datavalblue)
            sigmaval = np.array(sigmavalblue)
            FWHM = FWHMpixblue * (lambdasblue[-1] - lambdasblue[0])/nxblue #from grating equation
            #FWHM = FWHMpixblue * deltawavblue #FWHM in Angstroms linearized
    
    else:
        print 'Reading in text file: ', zzcetiblue
        try:
            lambdas, dataval, sigmaval = np.genfromtxt(zzcetiblue,unpack=True)
        except:
            lambdas, dataval = np.genfromtxt(zzcetiblue,unpack=True)
            print 'Setting sigma = 1'
            sigmaval = np.ones(len(dataval))
        FWHM = res * np.ones(len(lambdas))

        #Header values to save
        RA = 'UNKNOWN'
        DEC = 'UNKNOWN'
        SNR = 0.
        airmass = 0.
        nexp = 0.
        exptime = 0.

        #ID for naming of files
        global idname
        idname = zzcetiblue[:-4]

        #Offset in v/c
        
        print 'Offsetting in delta lambda.'
        lambdas, stdvoverc = shiftinvoverc(lambdas,dataval,sigmaval,plotall=True)
        print 'Done offsetting in delta lambda.'

    #plot the spectrum
    #plt.clf()
    #plt.plot(lambdas,dataval,'k^')
    #plt.plot(lambdas,sigmaval,'b^')
    #plt.plot(FWHM)
    #plt.show()
    #sys.exit()

    #Save spectrum
    #np.savetxt('VPHAS1813-2138_spectrum.txt',np.transpose([lambdas,dataval,sigmaval]))

    #First define the fitting and normalization wavelengths,
    # Then sets pixel range using those points

    #Normal wavelengths
    alphafitwavelengthlow = 6380.#6380
    alphafitwavelengthhigh = 6760.#6760
    alphanormwavelengthlow = 6413. #6413
    alphanormwavelengthhigh = 6713. #6713
    
    betafitwavelengthlow = 4680. #4680
    betafitwavelengthhigh = 5040. #5040
    betanormwavelengthlow = 4721. #4721
    betanormwavelengthhigh = 5001. #5001
    
    gammafitwavelengthlow = 4200. #4200
    gammafitwavelengthhigh = 4510. #4510
    gammanormwavelengthlow = 4220. #4220
    gammanormwavelengthhigh = 4460. #4460
    
    highwavelengthlow = 3782. #3782 for H10 and 3755 for H11
    highwavelenghthigh = 4191. #4191 
    deltawavelengthlow = 4031. #4031
    deltawavelengthhigh = 4191. #4191
    epsilonwavelengthlow = 3925. #3925
    epsilonwavelengthhigh = 4030. # 4030
    heightwavelengthlow = 3859. #3859
    heightwavelengthhigh = 3925. # 3925
    hninewavelengthlow = 3815. #3815
    hninewavelengthhigh = 3855. #3855
    htenwavelengthlow = 3785. #3785
    htenwavelengthhigh = 3815. #3815
    helevenwavelengthlow = 3757.
    helevenwavelengthhigh = 3785.
    '''
    #Wavelengths for possible ELM fitting
    #plt.clf()
    #plt.plot(lambdas,dataval)
    #plt.show()
    #Mask out Calcium line and others
    features_to_mask = [[3920,3940],[4016,4036],[4911,4938]]
    mask = lambdas == lambdas
    for waverange in features_to_mask:
        inds = np.where((lambdas > waverange[0]) & (lambdas < waverange[1]))
        mask[inds] = False
                    
    lambdas = lambdas[mask]
    dataval = dataval[mask]
    sigmaval = sigmaval[mask]
    FWHM = FWHM[mask]
    
    #plt.plot(lambdas,dataval-1e-14)
    #plt.show()
    #exit()

    betafitwavelengthlow = 4680. #4680
    betafitwavelengthhigh = 4961. #5040
    betanormwavelengthlow = 4812. #4721
    betanormwavelengthhigh = 4911. #4982
    
    gammafitwavelengthlow = 4200. #4200
    gammafitwavelengthhigh = 4380. #4510
    gammanormwavelengthlow = 4281. #4291
    gammanormwavelengthhigh = 4374. #4411

    highwavelengthlow = 3755.
    highwavelenghthigh = 4222. #4191 
    deltawavelengthlow = 4052. #4031
    deltawavelengthhigh = 4152. #4191
    epsilonwavelengthlow = 3940. #3925
    epsilonwavelengthhigh = 4007. # 4021
    heightwavelengthlow = 3865. #3859
    heightwavelengthhigh = 3920. # 3925
    hninewavelengthlow = 3815. #3815
    hninewavelengthhigh = 3855. #3855
    htenwavelengthlow = 3785. #3785
    htenwavelengthhigh = 3815. #3815
    helevenwavelengthlow = 3757.
    helevenwavelengthhigh = 3785.
    '''
    #Find indices for normalization
    if redfile:
        afitlow = np.min(np.where(lambdas > alphafitwavelengthlow)) 
        afithi = np.min(np.where(lambdas > alphafitwavelengthhigh)) 
        alow = np.min(np.where(lambdas > alphanormwavelengthlow))
        ahi = np.min(np.where(lambdas > alphanormwavelengthhigh))

    bfitlow = np.min(np.where(lambdas > betafitwavelengthlow))
    bfithi = np.min(np.where(lambdas > betafitwavelengthhigh))
    blow = np.min(np.where(lambdas > betanormwavelengthlow))
    bhi = np.min(np.where(lambdas > betanormwavelengthhigh))

    gfitlow = np.min(np.where(lambdas > gammafitwavelengthlow))
    gfithi = np.min(np.where(lambdas > gammafitwavelengthhigh))
    glow = np.min(np.where(lambdas > gammanormwavelengthlow))
    ghi = np.min(np.where(lambdas > gammanormwavelengthhigh))


    hlow = np.min(np.where(lambdas > highwavelengthlow)) 
    hhi = np.min(np.where(lambdas > highwavelenghthigh))
    dlow = np.min(np.where(lambdas > deltawavelengthlow))
    dhi = np.min(np.where(lambdas > deltawavelengthhigh))
    elow = np.min(np.where(lambdas > epsilonwavelengthlow))
    ehi = np.min(np.where(lambdas > epsilonwavelengthhigh))
    H8low = np.min(np.where(lambdas > heightwavelengthlow))
    H8hi = np.min(np.where(lambdas > heightwavelengthhigh))
    H9low = np.min(np.where(lambdas > hninewavelengthlow))
    H9hi = np.min(np.where(lambdas > hninewavelengthhigh))
    H10low = np.min(np.where(lambdas > htenwavelengthlow))
    H10hi = np.min(np.where(lambdas > htenwavelengthhigh))
    H11low = np.min(np.where(lambdas > helevenwavelengthlow))
    H11hi = np.min(np.where(lambdas > helevenwavelengthhigh))


    #====================================================
    #Set up estimates for pseudo-gaussian fitting
    #Make the estimates in a smart and consistent manner.
    #====================================================
    if redfile:
        alambdas = lambdas[afitlow:afithi+1]
        asigmas = sigmaval[afitlow:afithi+1]
        alphaval = dataval[afitlow:afithi+1]
        
        aest = np.zeros(8)
        ######
        if fitguess == 'data':
            print 'Using data for H alpha pseudogaussian guess.'
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
            aest[6] = 1.0 #how much of a pseudo-gaussian
        elif fitguess == 'model':
            print 'Using model for H alpha pseudogaussian guess.'
            ######
            #From fit to GD 165 on 2015-04-26
            aest[0] = 6.55710438e+04
            aest[1] =-3.01859244e+01
            aest[2] = 4.65757107e-03
            aest[3] = -1.03978059e+02
            aest[4] = 6.56457033e+03
            aest[5] =3.71607024e+01
            aest[6] = 7.29274380e-01
            aest[7] =-2.40131150e-07
            

    blambdas = lambdas[bfitlow:bfithi+1]
    bsigmas = sigmaval[bfitlow:bfithi+1]
    betaval = dataval[bfitlow:bfithi+1]
    
    best = np.zeros(8)
    #######
    if fitguess == 'data':
        print 'Using data for H beta pseudogaussian guess.'
        xes = np.array([lambdas[bfitlow],lambdas[blow],lambdas[blow+10],lambdas[bhi],lambdas[bfithi]])
        yes = np.array([dataval[bfitlow],dataval[blow],dataval[blow+10],dataval[bhi],dataval[bfithi]])
        bp = np.polyfit(xes,yes,3)
        bpp = np.poly1d(bp)
        best[0] = bp[3]
        best[1] = bp[2]
        best[2] = bp[1]
        best[7] = bp[0]
        best[3] = np.min(dataval[blow:bhi+1]) - bpp(4862.4555) #depth of line relative to continuum
        best[4] = 4862.4555 #rest wavelength of H beta
        bhalfmax = bpp(4862.4555) + best[3]/2.5
        bdiff = np.abs(betaval-bhalfmax)
        blowidx = bdiff[np.where(blambdas < 4862.4555)].argmin()
        bhighidx = bdiff[np.where(blambdas > 4862.4555)].argmin() + len(bdiff[np.where(blambdas < 4862.4555)])
        best[5] = (blambdas[bhighidx] - blambdas[blowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
        best[6] = 1.0 #how much of a pseudo-gaussian
    elif fitguess == 'model':
            print 'Using model for H beta pseudogaussian guess.'
            ##########
            #From fit to GD 165 on 2015-04-26
            best[0] = 2.94474405e+05
            best[1] = -1.81068634e+02
            best[2] = 3.72112471e-02
            best[3] = -2.94650570e+02
            best[4] =4.86191879e+03
            best[5] = 3.22064185e+01
            best[6] =8.86479782e-01
            best[7] = -2.55179334e-06

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
    H11lambdas = lambdas[H11low:H11hi+1]
    H11val = dataval[H11low:H11hi+1]


    ########################################
    #Begin fitting pseudogaussians to observed spectrum

    if higherlines == 'g11':
        print 'Fitting H gamma through H 11 for the higher order lines.'
        #########################
        #Fit gamma through 11
        print 'Now fitting gamma through 11'
        highwavelengthlow = 3755. #3782 for H10 and 3755 for H11
        hlow = np.min(np.where(lambdas > highwavelengthlow)) 

        hlambdas = lambdas[hlow:gfithi+1]
        hval = dataval[hlow:gfithi+1]
        hsig = sigmaval[hlow:gfithi+1]


        bigest = np.zeros(32) #Array for guess parameters
        if fitguess == 'model':
            print 'Using model for higher order lines pseudogaussian guess.'
            #Best fit parameters from GD 165 on 2015-04-26
            bigest[0] = -1.41357212e+05
            bigest[1] = 1.02070253e+02
            bigest[2] = -2.43244563e-02
            bigest[3] = 1.92248498e-06
            bigpp = np.poly1d([bigest[3],bigest[2],bigest[1],bigest[0]])
            bigest[4] = -3.96618275e+02
            bigest[5] = 4.34090612e+03
            bigest[6] = 2.64573080e+01
            bigest[7] = 1.00480981e+00
            bigest[8] = -4.88251832e+02
            bigest[9] = 4.10262307e+03
            bigest[10] = 2.77876123e+01
            bigest[11] = 9.66143639e-01
            bigest[12] = -4.48296148e+02
            bigest[13] = 3.97136338e+03
            bigest[14] = 2.30877387e+01
            bigest[15] = 1.09539929e+00
            bigest[16] = -3.60256002e+02
            bigest[17] = 3.88986126e+03
            bigest[18] = 1.98910307e+01
            bigest[19] = 1.22023094e+00
            bigest[20] = -2.18227322e+02
            bigest[21] = 3.83618537e+03
            bigest[22] = 1.63076240e+01
            bigest[23] = 1.37277188e+00
            bigest[24] = -5.94775596e+01
            bigest[25] = 3.80089099e+03
            bigest[26] = 8.80022809e+00
            bigest[27] = 2.18181124e+00
            bigest[28] = -2.33480022e+02
            bigest[29] = 3.77059291e+03
            bigest[30] = 3.98766671e+01
            bigest[31] = 1.63020905e+00
    
        elif fitguess == 'data':
            print 'Using data for higher order lines pseudogaussian guess.'

            #Guess for the continuum
            xes = np.array([lambdas[H10low],lambdas[H9low],lambdas[H8low],lambdas[elow],lambdas[dlow],lambdas[dhi],lambdas[glow],lambdas[ghi]])
            yes = np.array([dataval[H10low],dataval[H9low],dataval[H8low],dataval[elow],dataval[dlow],dataval[dhi],dataval[glow],dataval[ghi]])
            yes += dataval[H10low]/30. #Trying an offset to make sure the continuum is above the lines
            bigp = np.polyfit(xes,yes,3)
            bigpp = np.poly1d(bigp)
            bigest[0] = bigp[3]
            bigest[1] = bigp[2]
            bigest[2] = bigp[1]
            bigest[3] = bigp[0]
            #bigest[0] = -3.85045934e+04
            #bigest[1] = 2.73530286e+01
            #bigest[2] = -6.42031052e-03
            #bigest[3] = 5.00111153e-07
            #bigpp = np.poly1d([bigest[3],bigest[2],bigest[1],bigest[0]])

            #Gamma
            bigest[4] = np.min(dataval[glow:ghi+1]) - bigpp(4341.4834) #depth of line relative to continuum
            bigest[5] = 4341.4834 #rest wavelength of H gamma
            ghalfmax = bigpp(4341.4834) + bigest[4]/2.0
            gdiff = np.abs(gamval-ghalfmax)
            glowidx = gdiff[np.where(glambdas < 4341.4834)].argmin()
            ghighidx = gdiff[np.where(glambdas > 4341.4834)].argmin() + len(bdiff[np.where(glambdas < 4341.4834)])
            bigest[6] = (glambdas[ghighidx] - glambdas[glowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
            bigest[7] = 1.0 #how much of a pseudo-gaussian


            #Now delta
            bigest[8] = np.min(dataval[dlow:dhi+1]) - bigpp(4103.0343) #depth of line relative to continuum
            bigest[9] = 4103.0343  #rest wavelength of H delta
            dhalfmax = bigpp(4103.0343) + bigest[8]/2.0
            ddiff = np.abs(dval-dhalfmax)
            dlowidx = ddiff[np.where(dlambdas < 4103.0343)].argmin()
            dhighidx = ddiff[np.where(dlambdas > 4103.0343)].argmin() + len(ddiff[np.where(dlambdas < 4103.0343)])
            bigest[10] = (dlambdas[dhighidx] - dlambdas[dlowidx]) / (2.*np.sqrt(2.*np.log(2.)))
            bigest[11] = 1.2 #how much of a pseudo-gaussian

            #Now epsilon
            bigest[12] = np.min(dataval[elow:ehi+1]) - bigpp(3971.4475) #depth of line relative to continuum
            bigest[13] = 3971.4475   #rest wavelength of H epsilon
            ehalfmax = bigpp(3971.4475) + bigest[12]/2.0
            ediff = np.abs(epval-ehalfmax)
            elowidx = ediff[np.where(elambdas < 3971.4475)].argmin()
            ehighidx = ediff[np.where(elambdas > 3971.4475)].argmin() + len(ediff[np.where(elambdas < 3971.4475)])
            bigest[14] = (elambdas[ehighidx] - elambdas[elowidx]) / (2.*np.sqrt(2.*np.log(2.)))
            bigest[15] = 1.2 #how much of a pseudo-gaussian

            #Now H8
            bigest[16] = np.min(dataval[H8low:H8hi+1]) - bigpp(3890.2759) #depth of line relative to continuum
            bigest[17] = 3890.2759   #rest wavelength of H8
            H8halfmax = bigpp(3890.2759) + bigest[16]/2.0
            H8diff = np.abs(H8val-H8halfmax)
            H8lowidx = H8diff[np.where(H8lambdas < 3890.2759)].argmin()
            H8highidx = H8diff[np.where(H8lambdas > 3890.2759)].argmin() + len(H8diff[np.where(H8lambdas < 3890.2759)])
            bigest[18] = (H8lambdas[H8highidx] - H8lambdas[H8lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
            bigest[19] = 1.2 #how much of a pseudo-gaussian

            #Now H9
            bigest[20] = np.min(dataval[H9low:H9hi+1]) - bigpp(3836.1585) #depth of line relative to continuum
            bigest[21] = 3836.1585   #rest wavelength of H9
            H9halfmax = bigpp(3836.1585) + bigest[20]/2.0
            H9diff = np.abs(H9val-H9halfmax)
            H9lowidx = H9diff[np.where(H9lambdas < 3836.1585)].argmin()
            H9highidx = H9diff[np.where(H9lambdas > 3836.1585)].argmin() + len(H9diff[np.where(H9lambdas < 3836.1585)])
            bigest[22] = (H9lambdas[H9highidx] - H9lambdas[H9lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
            bigest[23] = 1.2 #how much of a pseudo-gaussian

            #Now H10
            bigest[24] = np.min(dataval[H10low:H10hi+1]) - bigpp(3797.909) #depth of line relative to continuum
            bigest[25] = 3799.0785   #rest wavelength of H10
            H10halfmax = bigpp(3799.0785) + bigest[24]/2.0
            H10diff = np.abs(H10val-H10halfmax)
            H10lowidx = H10diff[np.where(H10lambdas < 3799.0785)].argmin()
            H10highidx = H10diff[np.where(H10lambdas > 3799.0785)].argmin() + len(H10diff[np.where(H10lambdas < 3799.0785)])
            bigest[26] = (H10lambdas[H10highidx] - H10lambdas[H10lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
            bigest[27] = 1.2 #how much of a pseudo-gaussian

            #Now H11
            bigest[28] = np.min(dataval[H11low:H11hi+1]) - bigpp(3770.636) #depth of line relative to continuum
            bigest[29] = 3770.636   #rest wavelength of H11
            H11halfmax = bigpp(3770.6) + bigest[28]/2.0
            H11diff = np.abs(H11val-H11halfmax)
            H11lowidx = H11diff[np.where(H11lambdas < 3770.6)].argmin()
            H11highidx = H11diff[np.where(H11lambdas > 3770.6)].argmin() + len(H11diff[np.where(H11lambdas < 3770.6)])
            bigest[30] = (H11lambdas[H11highidx] - H11lambdas[H11lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
            bigest[31] = 1.2 #how much of a pseudo-gaussian
        
    
        #Constrain width of H11 pseudogaussian to be smaller than H10 pseudogaussian
        paraminfo = [{'limits':[0,0],'limited':[0,0]} for i in range(32)]
        #paraminfo[27]['limited'] = [0,1]
        #paraminfo[27]['limits'] = [0,3.]

        bigfa = {'x':hlambdas, 'y':hval, 'err':hsig}
        hparams = mpfit.mpfit(fitbigpseudogauss,bigest,functkw=bigfa,maxiter=200,ftol=1e-12,xtol=1e-8,parinfo=paraminfo,quiet=True)
        print hparams.status, hparams.niter, hparams.fnorm, hparams.dof, hparams.fnorm/hparams.dof
        hfit = bigpseudogauss(hlambdas,hparams.params)

        bigguess = bigpseudogauss(hlambdas,bigest)

        #compute reduced chi-square for gamma through 10
        low2 = np.min(np.where(hlambdas > 3782.)) 
        high2 =  np.min(np.where(hlambdas > 4378.))

        chisquare2 = np.sum(((hval[low2:high2+1]-hfit[low2:high2+1]) / hsig[low2:high2+1])**2.,dtype='d')
        dof2 = float(len(hval[low2:high2+1])) - 28.

        chisquare3 = np.sum(((hval-hfit) / hsig)**2.,dtype='d')
        dof3 = float(len(hval)) - 32.
    
        #print chisquare3, dof3, chisquare3/dof3
        #print chisquare2, dof2, chisquare2/dof2

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
    
    
        #print bigest
        print hparams.params
        plt.clf()
        plt.plot(hlambdas,hval,'b')
        plt.plot(hlambdas,bigpp(hlambdas),'g')
        plt.plot(hlambdas,bigguess,'g')
        plt.plot(hlambdas,hfit,'r')
        plt.plot(hlambdas,hparams.params[0] + hparams.params[1]*hlambdas + hparams.params[2]*hlambdas**2. + hparams.params[3]*hlambdas**3.,'r')
        #plt.plot(hlambdas,hval-hfit+(hfit.min()-15.),'k')
        #bigparams.params[0] = 0.
        #bigparams.params[1] = 0.
        #bigparams.params[2] = 0.
        #bigparams.params[3] = 0.
        #plt.plot(biglambdas,bigpseudogauss(biglambdas,bigparams.params),'k')
        #plt.plot(biglambdas,bigparams.params[4]*np.exp(-(np.abs(biglambdas-bigparams.params[5])/(np.sqrt(2.)*bigparams.params[6]))**bigparams.params[7]),'c')
        #plt.plot(biglambdas,bigparams.params[20]*np.exp(-(np.abs(biglambdas-bigparams.params[21])/(np.sqrt(2.)*bigparams.params[22]))**bigparams.params[23]),'c')
        #plt.plot(biglambdas,25.+bigest[4]*np.exp(-(np.abs(biglambdas-bigest[5])/(np.sqrt(2.)*bigest[6]))**bigest[7]),'c')
        #plt.plot(biglambdas,25.+bigest[20]*np.exp(-(np.abs(biglambdas-bigest[21])/(np.sqrt(2.)*bigest[22]))**bigest[23]),'c')

        #plt.show()
    
        '''
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        endpoint = '.ms.'
        savefitspec = 'fit_' + idname + '_' + now[5:10] + '_H11.pdf'
        fitpdf = PdfPages(savefitspec)
        plt.clf()
        plt.plot(hlambdas,hval,'b')
        plt.plot(hlambdas,bigguess,'g')
        plt.plot(hlambdas,bigpp(hlambdas),'g')
        plt.title('Guess')
        fitpdf.savefig()
        plt.clf()
        plt.plot(hlambdas,hval,'b')
        plt.plot(hlambdas,hfit,'r')
        plt.plot(hlambdas,hparams.params[0] + hparams.params[1]*hlambdas + hparams.params[2]*hlambdas**2. + hparams.params[3]*hlambdas**3.,'r')
        #plt.title(np.round(hparams.fnorm/hparams.dof,decimals=4))
        plt.title(np.round(chisquare2/dof2,decimals=4))
        fitpdf.savefig()
        fitpdf.close()
        '''

        #sys.exit()

        ########################
        #########################
    elif higherlines == 'g10':
        print 'Fitting H gamma through H 10 for the higher order lines.'
        #Fit gamma through 10 at once
        #highwavelengthlow = 3782. #3782 for H10 and 3755 for H11
        #hlow = np.min(np.where(lambdas > highwavelengthlow)) 


        hlambdas = lambdas[hlow:gfithi+1]
        hval = dataval[hlow:gfithi+1]
        hsig = sigmaval[hlow:gfithi+1]

        bigest = np.zeros(28)

        if fitguess == 'model':
            print 'Using model for higher order lines pseudogaussian guess.'
            #Guesses from GD 165: 2015-04-26
            bigest[0] = -1.40049986e+05
            bigest[1] = 1.00197634e+02
            bigest[2] = -2.36678694e-02
            bigest[3] = 1.85471480e-06
            bigpp = np.poly1d([bigest[3],bigest[2],bigest[1],bigest[0]])
            bigest[4] = -4.06230663e+02
            bigest[5] = 4.34109176e+03
            bigest[6] = 2.76921973e+01
            bigest[7] = 9.78626650e-01
            bigest[8] =-4.85132845e+02
            bigest[9] = 4.10278586e+03
            bigest[10] =2.76129635e+01
            bigest[11] =9.52884929e-01
            bigest[12] =-4.22115667e+02
            bigest[13] = 3.97153282e+03
            bigest[14] =2.11737451e+01
            bigest[15] = 1.12639940e+00
            bigest[16] =-3.16990734e+02
            bigest[17] =3.89012946e+03
            bigest[18] =1.80330308e+01
            bigest[19] = 1.18346199e+00
            bigest[20] =-1.47793093e+02
            bigest[21] =3.83743302e+03
            bigest[22] =1.13154407e+01
            bigest[23] =1.35033896e+00
            bigest[24] =-2.17402577e+02
            bigest[25] =3.79732448e+03
            bigest[26] =3.98047434e+01
            bigest[27] =1.13178448e+00

        elif fitguess == 'data':
            print 'Using data for higher order lines pseudogaussian guess.'

            #Guess for continuum
            xes = np.array([lambdas[H10low],lambdas[H9low],lambdas[H8low],lambdas[elow],lambdas[dlow],lambdas[dhi],lambdas[glow],lambdas[ghi]])
            yes = np.array([dataval[H10low],dataval[H9low],dataval[H8low],dataval[elow],dataval[dlow],dataval[dhi],dataval[glow],dataval[ghi]])
            yes += dataval[H10low]/30. #offset to make sure the continuum is above the lines
            bigp = np.polyfit(xes,yes,3)
            bigpp = np.poly1d(bigp)
            bigest[0] = bigp[3]
            bigest[1] = bigp[2]
            bigest[2] = bigp[1]
            bigest[3] = bigp[0]
            #bigest[0] = -1.41210158e+05
            #bigest[1] = 1.02004150e+02
            #bigest[2] = -2.43173657e-02
            #bigest[3] = 1.92255983e-06
            #bigpp = np.poly1d([bigest[3],bigest[2],bigest[1],bigest[0]])


            bigest[4] = np.min(dataval[glow:ghi+1]) - bigpp(4341.4834) #depth of line relative to continuum
            bigest[5] = 4341.4834 #rest wavelength of H gamma
            ghalfmax = bigpp(4341.4834) + bigest[4]/2.0
            gdiff = np.abs(gamval-ghalfmax)
            glowidx = gdiff[np.where(glambdas < 4341.4834)].argmin()
            ghighidx = gdiff[np.where(glambdas > 4341.4834)].argmin() + len(bdiff[np.where(glambdas < 4341.4834)])
            bigest[6] = (glambdas[ghighidx] - glambdas[glowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
            bigest[7] = 1.0 #how much of a pseudo-gaussian

            plt.clf()
            plt.plot(glambdas[glowidx],gamval[glowidx],'k^')
            plt.plot(glambdas[ghighidx],gamval[ghighidx],'k^')

            #Now delta
            bigest[8] = np.min(dataval[dlow:dhi+1]) - bigpp(4103.0343) #depth of line relative to continuum
            bigest[9] = 4103.0343  #rest wavelength of H delta
            dhalfmax = bigpp(4103.0343) + bigest[8]/2.0
            ddiff = np.abs(dval-dhalfmax)
            dlowidx = ddiff[np.where(dlambdas < 4103.0343)].argmin()
            dhighidx = ddiff[np.where(dlambdas > 4103.0343)].argmin() + len(ddiff[np.where(dlambdas < 4103.0343)])
            bigest[10] = (dlambdas[dhighidx] - dlambdas[dlowidx]) / (2.*np.sqrt(2.*np.log(2.)))
            bigest[11] = 1.2 #how much of a pseudo-gaussian

            #Now epsilon
            bigest[12] = np.min(dataval[elow:ehi+1]) - bigpp(3971.4475) #depth of line relative to continuum
            bigest[13] = 3971.4475   #rest wavelength of H epsilon
            ehalfmax = bigpp(3971.4475) + bigest[12]/2.0
            ediff = np.abs(epval-ehalfmax)
            elowidx = ediff[np.where(elambdas < 3971.4475)].argmin()
            ehighidx = ediff[np.where(elambdas > 3971.4475)].argmin() + len(ediff[np.where(elambdas < 3971.4475)])
            bigest[14] = (elambdas[ehighidx] - elambdas[elowidx]) / (2.*np.sqrt(2.*np.log(2.)))
            bigest[15] = 1.2 #how much of a pseudo-gaussian

            #Now H8
            bigest[16] = np.min(dataval[H8low:H8hi+1]) - bigpp(3890.2759) #depth of line relative to continuum
            bigest[17] = 3890.2759   #rest wavelength of H8
            H8halfmax = bigpp(3890.2759) + bigest[16]/2.0
            H8diff = np.abs(H8val-H8halfmax)
            H8lowidx = H8diff[np.where(H8lambdas < 3890.2759)].argmin()
            H8highidx = H8diff[np.where(H8lambdas > 3890.2759)].argmin() + len(H8diff[np.where(H8lambdas < 3890.2759)])
            bigest[18] = (H8lambdas[H8highidx] - H8lambdas[H8lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
            bigest[19] = 1.2 #how much of a pseudo-gaussian

            #Now H9
            bigest[20] = np.min(dataval[H9low:H9hi+1]) - bigpp(3836.1585) #depth of line relative to continuum
            try:
                bigest[21] = constraints910[0]
            except:
                bigest[21] = 3836.1585   #rest wavelength of H9
            H9halfmax = bigpp(3836.1585) + bigest[20]/2.0
            H9diff = np.abs(H9val-H9halfmax)
            H9lowidx = H9diff[np.where(H9lambdas < 3836.1585)].argmin()
            H9highidx = H9diff[np.where(H9lambdas > 3836.1585)].argmin() + len(H9diff[np.where(H9lambdas < 3836.1585)])
            bigest[22] = (H9lambdas[H9highidx] - H9lambdas[H9lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
            bigest[23] = 1.2 #how much of a pseudo-gaussian
            plt.plot(H9lambdas[H9lowidx],H9val[H9lowidx],'k^')
            plt.plot(H9lambdas[H9highidx],H9val[H9highidx],'k^')

            #Now H10
            bigest[24] = np.min(dataval[H10low:H10hi+1]) - bigpp(3799.0785) #depth of line relative to continuum
            try:
                bigest[25] = constraints910[1]
            except:
                bigest[25] = 3799.0785   #rest wavelength of H10 in models
            H10halfmax = bigpp(3799.0785) + bigest[24]/2.0
            H10diff = np.abs(H10val-H10halfmax)
            H10lowidx = H10diff[np.where(H10lambdas < 3799.0785)].argmin()
            H10highidx = H10diff[np.where(H10lambdas > 3799.0785)].argmin() + len(H10diff[np.where(H10lambdas < 3799.0785)])
            bigest[26] = (H10lambdas[H10highidx] - H10lambdas[H10lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
            bigest[27] = 1.2 #how much of a pseudo-gaussian

        #Set up wavelength limits based on v/c errors
        #Central wavelengths are bigest 5, 9, 13, 17, 21, 25
        #stdvoverc
        #siglimit = 3.
        #print stdvoverc
        paraminfo = [{'limits':[0,0],'limited':[0,0],'fixed':0} for i in range(28)]
        #paraminfo[5]['limited'] = [1,1]
        #paraminfo[5]['limits'] = [4341.4834-siglimit*stdvoverc*4341.4834,4341.4834+siglimit*stdvoverc*4341.3834]
        #paraminfo[9]['limited'] = [1,1]
        #paraminfo[9]['limits'] = [4103.0343-siglimit*stdvoverc*4103.0343,4103.0343+siglimit*stdvoverc*4103.0343]
        #paraminfo[13]['limited'] = [1,1]
        #paraminfo[13]['limits'] = [3971.4475-siglimit*stdvoverc*3971.4475,3971.4475+siglimit*stdvoverc*3971.4475]
        #paraminfo[17]['limited'] = [1,1]
        #paraminfo[17]['limits'] = [3890.2759-siglimit*stdvoverc*3890.2759,3890.2759+siglimit*stdvoverc*3890.2759]
        #paraminfo[21]['limited'] = [1,1]
        #paraminfo[21]['limits'] = [3836.1585-siglimit*stdvoverc*3836.1585,3836.1585+siglimit*stdvoverc*3836.1585]
        #paraminfo[25]['limited'] = [1,1]
        #paraminfo[25]['limits'] = [3799.0785-siglimit*stdvoverc*3799.0785,3799.0785+siglimit*stdvoverc*3799.0785]
        paraminfo[21]['fixed'] = 1. #H9
        paraminfo[25]['fixed'] = 1. #H10

        print 'Holding H9 and H10 centers fixed.'
        print bigest[21], bigest[25]
        print paraminfo[21], paraminfo[25]
        '''
        plt.clf()
        plt.plot(hlambdas,hval,'k',linewidth=2.0)
        plt.plot(hlambdas,bigpseudogaussgamma(hlambdas,bigest),'g')
        plt.axvline(bigest[5])
        plt.axvline(bigest[9])
        plt.axvline(bigest[13])
        plt.axvline(bigest[17])
        plt.axvline(bigest[21])
        plt.axvline(bigest[25])
        plt.show()
        #exit()
        '''
        

        print 'Now fitting H-gamma through H10.'
        bigfa = {'x':hlambdas, 'y':hval, 'err':hsig}
        ##hparams = mpfit.mpfit(fitbigpseudogaussgamma,bigest,functkw=bigfa,maxiter=300,ftol=1e-12,xtol=1e-8,quiet=True)#-10,-8
        hparams = mpfit.mpfit(fitbigpseudogaussgamma,bigest,functkw=bigfa,maxiter=300,ftol=1e-12,xtol=1e-8,quiet=True,parinfo=paraminfo)#-10,-8
        #print bigest
        print hparams.status, hparams.niter, hparams.fnorm, hparams.dof, hparams.fnorm/hparams.dof
        #print bigest
        print hparams.params
        print ''
        hfit = bigpseudogaussgamma(hlambdas,hparams.params)

        #print 'Differences in fitted line location from limits:'
        #print 'Gamma: ', (hparams.params[5]-paraminfo[5]['limits'][0]), (hparams.params[5]-paraminfo[5]['limits'][1])
        #print 'Delta: ', (hparams.params[9]-paraminfo[9]['limits'][0]), (hparams.params[9]-paraminfo[9]['limits'][1])
        #print 'Epsilon: ', (hparams.params[13]-paraminfo[13]['limits'][0]), (hparams.params[13]-paraminfo[13]['limits'][1])
        #print 'H8: ', (hparams.params[17]-paraminfo[17]['limits'][0]), (hparams.params[17]-paraminfo[17]['limits'][1])
        #print 'H9: ', (hparams.params[21]-paraminfo[21]['limits'][0]), (hparams.params[21]-paraminfo[21]['limits'][1])
        #print 'H10: ', (hparams.params[25]-paraminfo[25]['limits'][0]), (hparams.params[25]-paraminfo[25]['limits'][1])
        #print 'H9: ', (hparams.params[21] - bigest[21])
        #print 'H10: ', (hparams.params[25] - bigest[25])


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
        gfithi2 = np.min(np.where(hlambdas > 4378.))
        hlow2 = np.min(np.where(hlambdas > 3778.)) 
        hhi2 = np.min(np.where(hlambdas > 4195.)) 


        #plt.clf()
        #plt.plot(hlambdas,hval,'k',linewidth=2.0)
        #plt.plot(biglambdas,bigpp(biglambdas),'r')
        #plt.plot(hlambdas,bigguess,'g')
        #plt.axvline(bigest[21])
        #plt.axvline(bigest[25])
        #plt.plot(hlambdas,bigpp(hlambdas),'g')
        #plt.plot(hlambdas,hfit,'r',linewidth=2.0)
        #plt.plot(hlambdas,hparams.params[0] + hparams.params[1]*hlambdas + hparams.params[2]*hlambdas**2. + hparams.params[3]*hlambdas**3.,'r')
        #plt.title(np.round(hparams.fnorm/hparams.dof,decimals=4))
        #plt.xlabel('Wavelength')
        #plt.ylabel('Flux')
        #plt.show()
        #exit()

        '''
        #Save the pseudogaussian fits to the spectrum as a pdf
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        endpoint = '.ms.'
        savefitspec = 'fit_' + idname + '_' + now[5:10] + '_H10.pdf'
        fitpdf = PdfPages(savefitspec)
        plt.clf()
        plt.plot(hlambdas,hval,'b')
        plt.plot(hlambdas,bigguess,'g')
        plt.plot(hlambdas,bigpp(hlambdas),'g')
        plt.title('Guess')
        fitpdf.savefig()
        plt.clf()
        plt.plot(hlambdas,hval,'b')
        plt.plot(hlambdas,hfit,'r')
        plt.plot(hlambdas,hparams.params[0] + hparams.params[1]*hlambdas + hparams.params[2]*hlambdas**2. + hparams.params[3]*hlambdas**3.,'r')
        plt.title(np.round(hparams.fnorm/hparams.dof,decimals=4))
        fitpdf.savefig()
        fitpdf.close()
        '''

        #sys.exit()

        ##########################################33

    #Fit alpha
    if redfile:
        #alambdas = lambdas[afitlow:afithi+1]
        #asigmas = sigmaval[afitlow:afithi+1]
        #alphaval = dataval[afitlow:afithi+1]
        print 'Now fitting the H alpha line.'
        afa = {'x':alambdas, 'y':alphaval, 'err':asigmas}
        aparams = mpfit.mpfit(fitpseudogausscubic,aest,functkw=afa,maxiter=3000,ftol=1e-14,xtol=1e-13,quiet=True)
        print 'Number of iterations: ', aparams.niter
        acenter = aparams.params[4]
        alphafit = pseudogausscubic(alambdas,aparams.params)
        alphavariation = np.sum((alphafit - alphaval)**2.)
        print aparams.status, aparams.niter, aparams.fnorm, aparams.dof
        print aparams.params

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
    bparams = mpfit.mpfit(fitpseudogausscubic,best,functkw=bfa,maxiter=4000,ftol=1e-16,xtol=1e-10,quiet=True)
    print 'Number of iterations: ', bparams.niter
    print bparams.status, bparams.niter, bparams.fnorm, bparams.dof
    print bparams.params

    bcenter = bparams.params[4]
    betafit = pseudogausscubic(blambdas,bparams.params)
    betavariation = np.sum((betafit - betaval)**2.)

    #plt.clf()
    #plt.plot(blambdas,betaval,'b^',label='data')
    #plt.plot(blambdas,betafit,'r',label='fit')
    #plt.plot(blambdas,bparams.params[0]*1. + bparams.params[1]*blambdas+bparams.params[2]*blambdas**2.)
    #plt.show()
    #sys.exit()

    #Plot delta lambda over lambda
    
    #lams = np.array([4862.4555,4341.4834,4103.0343,3971.4475,3890.2759,3836.1585,3799.0785])
    #deltalams = np.array([(bcenter-lams[0])/lams[0],(gcenter-lams[1])/lams[1],(dcenter-lams[2])/lams[2],(ecenter-lams[3])/lams[3],(H8center-lams[4])/lams[4],(H9center-lams[5])/lams[5],(H10center-lams[6])/lams[6]])
    #deltalams = np.array([(bcenter-lams[0]),(gcenter-lams[1]),(dcenter-lams[2]),(ecenter-lams[3]),(H8center-lams[4]),(H9center-lams[5]),(H10center-lams[6])])
    #line_fit = np.polyfit(lams[:-1],deltalams[:-1],1.)
    #line_fitted = np.poly1d(line_fit)
    #print lams[:-1]
    #plt.clf()
    #plt.plot(lams,deltalams,'bo')
    #plt.plot(lams,line_fitted(lams),'g--')
    #plt.show()
    #exit()
    

    #Fit a line to the fit points  from each end and divide by this line to normalize. 
    #Note the line needs to be linear in lambda not in pixel
    #bnline is the normalized spectral line. Continuum set to one.

    #======================================
    #Redefine normalization wavelengths so you can narrow or widen the lines but keep the psedugaussian fits the same.
    #======================================
    '''
    #Narrow wavelengths
    alphafitwavelengthlow = 6380.#6380
    alphafitwavelengthhigh = 6760.#6760
    alphanormwavelengthlow = 6435.5 #6413
    alphanormwavelengthhigh = 6690.5 #6713

    betafitwavelengthlow = 4680. #4680
    betafitwavelengthhigh = 5040. #5040
    betanormwavelengthlow = 4742. #4721
    betanormwavelengthhigh = 4980. #5001

    gammafitwavelengthlow = 4200. #4200
    gammafitwavelengthhigh = 4510. #4510
    gammanormwavelengthlow = 4238. #4220
    gammanormwavelengthhigh = 4442. #4460

    highwavelengthlow = 3782.
    highwavelenghthigh = 4191. #4191 
    deltawavelengthlow = 4043. #4031
    deltawavelengthhigh = 4179. #4191
    epsilonwavelengthlow = 3932.9 #3925
    epsilonwavelengthhigh = 4022.1 # 4030
    heightwavelengthlow = 3864. #3859
    heightwavelengthhigh = 3920. # 3925
    hninewavelengthlow = 3818. #3815
    hninewavelengthhigh = 3852. #3855
    htenwavelengthlow = 3787.3 #3785
    htenwavelengthhigh = 3812.7 #3815
    '''
    '''
    #Wide wavelengths
    alphafitwavelengthlow = 6380.#6380
    alphafitwavelengthhigh = 6760.#6760
    alphanormwavelengthlow = 6390.5 #6413
    alphanormwavelengthhigh = 6735.5 #6713

    betafitwavelengthlow = 4680. #4680
    betafitwavelengthhigh = 5040. #5040
    betanormwavelengthlow = 4700. #4721
    betanormwavelengthhigh = 5022. #5001

    gammafitwavelengthlow = 4200. #4200
    gammafitwavelengthhigh = 4510. #4510
    gammanormwavelengthlow = 4202. #4220
    gammanormwavelengthhigh = 4478. #4460

    highwavelengthlow = 3782.
    highwavelenghthigh = 4205. #4191 
    deltawavelengthlow = 4019. #4031
    deltawavelengthhigh = 4203. #4191
    epsilonwavelengthlow = 3917.1 #3925
    epsilonwavelengthhigh = 4037.8 # 4030
    heightwavelengthlow = 3854.1 #3859
    heightwavelengthhigh = 3930.1 # 3925
    hninewavelengthlow = 3812. #3815
    hninewavelengthhigh = 3858. #3855
    htenwavelengthlow = 3782.7 #3785
    htenwavelengthhigh = 3817.2 #3815
    '''



    #======================================
    #Normalize using the pseudogaussian fits
    #======================================
    print 'Now normalizing the models using the pseudogaussian fits.'
    #Save offests of pseudogaussians
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    marker = str(np.round(FWHM[0],decimals=2))
    endpoint = '.ms.'

    #If spectra are in a different directory, change to that directory
    home_directory = os.getcwd()
    if zzcetiblue[0] == '.':
        os.chdir(zzcetiblue[0:zzcetiblue.find('w')])
    saveoffsets = 'offsets_' + idname + '_' + now[5:10] + '_' + marker + '.pdf'
    global offsetpdf
    offsetpdf = PdfPages(saveoffsets)
    if redfile:
        #Start with alpha
        #Set the center of the line to the wavelength of the models.
        alambdas = alambdas - (acenter- 6564.6047)
        #aslope = (alphafit[-1] - alphafit[0] ) / (alambdas[-1] - alambdas[0])
        #ali = aslope * (alambdas - alambdas[0]) + alphafit[0]
        #anline = dataval[alow:ahi+1] / ali
        #asigma = sigmaval[alow:ahi+1] / ali

        anormlow = np.min(np.where(alambdas > alphanormwavelengthlow)) #Refind the normalization points since we have shifted the line.
        anormhi = np.min(np.where(alambdas > alphanormwavelengthhigh))
        #Find the offset between the pseudogaussian fit and the actual data
        arefit_width = 10
        afit_high = fit_offset(alambdas[anormhi-arefit_width:anormhi+arefit_width],alphaval[anormhi-arefit_width:anormhi+arefit_width],alphafit[anormhi-arefit_width:anormhi+arefit_width],alphanormwavelengthhigh,asigmas[anormhi-arefit_width:anormhi+arefit_width])
        afit_low = fit_offset(alambdas[anormlow-arefit_width:anormlow+arefit_width],alphaval[anormlow-arefit_width:anormlow+arefit_width],alphafit[anormlow-arefit_width:anormlow+arefit_width],alphanormwavelengthlow,asigmas[anormlow-arefit_width:anormlow+arefit_width])
        #print afit_high, afit_low
        #afit_high = alphafit[anormhi]
        #afit_low = alphafit[anormlow]
        aslope = (afit_high - afit_low ) / (alambdas[anormhi] - alambdas[anormlow])
        alambdasnew = alambdas[anormlow:anormhi+1]
        alphavalnew = alphaval[anormlow:anormhi+1]
        asigmasnew = asigmas[anormlow:anormhi+1]
        ali = aslope * (alambdasnew - alambdas[anormlow]) + afit_low
        anline = alphavalnew / ali
        asigma = asigmasnew / ali


        #plt.clf()
        #plt.plot(alambdasnew,anline)
        #plt.plot(alambdasnew,alphavalnew)
        #plt.plot(alambdas[anormlow],alphafit[anormlow],'g^')
        #plt.plot(alambdas[anormhi],alphafit[anormhi],'g^')
        #plt.plot(alambdasnew,ali,'g')
        #plt.show()
        #sys.exit()

    #Now to beta
    #Set the center of the line to the wavelength of the models.
    blambdas = blambdas - (bcenter-4862.4555)
    #bslope = (betafit[-1] - betafit[0] ) / (blambdas[-1] - blambdas[0])
    #bli = bslope * (blambdas - blambdas[0]) + betafit[0]
    #bnline = dataval[blow:bhi+1] / bli
    #bsigma =  sigmaval[blow:bhi+1] / bli

    bnormlow = np.min(np.where(blambdas > betanormwavelengthlow))
    bnormhi = np.min(np.where(blambdas > betanormwavelengthhigh))
    #Find the offset between the pseudogaussian fit and the actual data
    brefit_width = 10
    bfit_high = fit_offset(blambdas[bnormhi-brefit_width:bnormhi+brefit_width],betaval[bnormhi-brefit_width:bnormhi+brefit_width],betafit[bnormhi-brefit_width:bnormhi+brefit_width],betanormwavelengthhigh,bsigmas[bnormhi-brefit_width:bnormhi+brefit_width])
    bfit_low = fit_offset(blambdas[bnormlow-brefit_width:bnormlow+brefit_width],betaval[bnormlow-brefit_width:bnormlow+brefit_width],betafit[bnormlow-brefit_width:bnormlow+brefit_width],betanormwavelengthlow,bsigmas[bnormlow-brefit_width:bnormlow+brefit_width])
    #print bfit_high, bfit_low
    #bfit_high = betafit[bnormhi]
    #bfit_low = betafit[bnormlow]
    bslope = (bfit_high - bfit_low ) / (blambdas[bnormhi] - blambdas[bnormlow])
    blambdasnew = blambdas[bnormlow:bnormhi+1]
    betavalnew = betaval[bnormlow:bnormhi+1]
    bsigmasnew = bsigmas[bnormlow:bnormhi+1]
    bli = bslope * (blambdasnew - blambdas[bnormlow]) + bfit_low
    bnline = betavalnew / bli
    bsigma = bsigmasnew / bli


    #plt.clf()
    #plt.plot(blambdasnew,bnline)
    #plt.plot(blambdasnew,betavalnew)
    #plt.plot(blambdas[bnormlow],betafit[bnormlow],'g^')
    #plt.plot(blambdas[bnormhi],betafit[bnormhi],'g^')
    #plt.plot(blambdasnew,bli,'g')
    #plt.show()


    #Now do gamma
    #gnline is the normalized spectral line. Continuum set to one.
    #Set the center of the line to the wavelength of models
    glambdas = glambdas - (gcenter-4341.4834)
    gnormlow = np.min(np.where(glambdas > gammanormwavelengthlow))
    gnormhi = np.min(np.where(glambdas > gammanormwavelengthhigh))
    #Find the offset between the pseudogaussian fit and the actual data
    grefit_width = 10
    gfit_high = fit_offset(glambdas[gnormhi-grefit_width:gnormhi+grefit_width],gamval[gnormhi-grefit_width:gnormhi+grefit_width],gamfit[gnormhi-grefit_width:gnormhi+grefit_width],gammanormwavelengthhigh,gsigmas[gnormhi-grefit_width:gnormhi+grefit_width])
    gfit_low = fit_offset(glambdas[gnormlow-grefit_width:gnormlow+grefit_width],gamval[gnormlow-grefit_width:gnormlow+grefit_width],gamfit[gnormlow-grefit_width:gnormlow+grefit_width],gammanormwavelengthlow,gsigmas[gnormlow-grefit_width:gnormlow+grefit_width])
    #print gfit_high, gfit_low
    #gfit_high = gamfit[gnormhi]
    #gfit_low = gamfit[gnormlow]
    gslope = (gfit_high - gfit_low ) / (glambdas[gnormhi] - glambdas[gnormlow])
    glambdasnew = glambdas[gnormlow:gnormhi+1]
    gamvalnew = gamval[gnormlow:gnormhi+1]
    gsigmasnew = gsigmas[gnormlow:gnormhi+1]
    gli = gslope * (glambdasnew - glambdas[gnormlow]) + gfit_low
    gnline = gamvalnew / gli
    gsigma = gsigmasnew / gli

    gammavariation = np.sum((gamfit[gnormlow:gnormhi+1] - gamval[gnormlow:gnormhi+1])**2.)

    #plt.clf()
    #plt.plot(glambdasnew,gnline)
    #plt.plot(glambdas[gnormlow],gamfit[gnormlow],'g^')
    #plt.plot(glambdas[gnormhi],gamfit[gnormhi],'g^')
    #plt.plot(glambdasnew,gli,'g')
    #plt.show()
    #sys.exit()

    #Now normalize the higher order lines (delta, epsilon, H8)
    hlambdastemp = hlambdas - (dcenter-4103.0343)
    dnormlow = np.min(np.where(hlambdastemp > deltawavelengthlow))
    dnormhi = np.min(np.where(hlambdastemp > deltawavelengthhigh))
    dlambdas = hlambdastemp[dnormlow:dnormhi+1]
    dvaltemp = dataval[hlow:gfithi+1]
    dsigtemp = sigmaval[hlow:gfithi+1]
    #Find the offset between the pseudogaussian fit and the actual data
    drefit_width = 10
    dfit_high = fit_offset(hlambdastemp[dnormhi-drefit_width:dnormhi+drefit_width],dvaltemp[dnormhi-drefit_width:dnormhi+drefit_width],hfit[dnormhi-drefit_width:dnormhi+drefit_width],deltawavelengthhigh,dsigtemp[dnormhi-drefit_width:dnormhi+drefit_width])
    dfit_low = fit_offset(hlambdastemp[dnormlow-drefit_width:dnormlow+drefit_width],dvaltemp[dnormlow-drefit_width:dnormlow+drefit_width],hfit[dnormlow-drefit_width:dnormlow+drefit_width],deltawavelengthlow,dsigtemp[dnormlow-drefit_width:dnormlow+drefit_width])
    #print dfit_high, dfit_low
    #dfit_high = hfit[dnormhi]
    #dfit_low = hfit[dnormlow]
    dslope = (dfit_high - dfit_low) / (hlambdastemp[dnormhi] - hlambdastemp[dnormlow])
    dli = dslope * (dlambdas - dlambdas[0]) + dfit_low
    dnline = dvaltemp[dnormlow:dnormhi+1] / dli
    dsigma = dsigtemp[dnormlow:dnormhi+1] / dli

    #plt.plot(hlambdastemp,dvaltemp,'k')
    #plt.plot(hlambdastemp[dnormlow],hfit[dnormlow],'g^')
    #plt.plot(hlambdastemp[dnormhi],hfit[dnormhi],'g^')
    #plt.plot(dlambdas,dli,'g')
    #plt.show()
    #exit()

    #elambdas = elambdas - (ecenter-3971.4475)
    hlambdastemp = hlambdas - (ecenter-3971.4475)
    enormlow = np.min(np.where(hlambdastemp > epsilonwavelengthlow))
    enormhi = np.min(np.where(hlambdastemp > epsilonwavelengthhigh))
    elambdas = hlambdastemp[enormlow:enormhi+1]
    evaltemp = dataval[hlow:hhi+1]
    esigtemp = sigmaval[hlow:hhi+1]
    #Find the offset between the pseudogaussian fit and the actual data
    erefit_width = 10 #pixels
    efit_high = fit_offset(hlambdastemp[enormhi-erefit_width:enormhi+erefit_width],evaltemp[enormhi-erefit_width:enormhi+erefit_width],hfit[enormhi-erefit_width:enormhi+erefit_width],epsilonwavelengthhigh,esigtemp[enormhi-erefit_width:enormhi+erefit_width])
    efit_low = fit_offset(hlambdastemp[enormlow-erefit_width:enormlow+erefit_width],evaltemp[enormlow-erefit_width:enormlow+erefit_width],hfit[enormlow-erefit_width:enormlow+erefit_width],epsilonwavelengthlow,esigtemp[enormlow-erefit_width:enormlow+erefit_width])
    #print efit_high, efit_low
    #efit_high = hfit[enormhi]
    #efit_low = hfit[enormlow]
    eslope = (efit_high - efit_low ) / (hlambdastemp[enormhi] - hlambdastemp[enormlow])
    eli = eslope * (elambdas - elambdas[0]) + efit_low
    enline = evaltemp[enormlow:enormhi+1] / eli
    esigma = esigtemp[enormlow:enormhi+1] / eli

    #plt.plot(hlambdastemp[enormlow],hfit[enormlow],'g^')
    #plt.plot(hlambdastemp[enormhi],hfit[enormhi],'g^')
    #plt.plot(elambdas,eli,'g')

    #H8lambdas = H8lambdas - (H8center-3890.2759)
    hlambdastemp = hlambdas - (H8center-3890.2759)
    H8normlow = np.min(np.where(hlambdastemp > heightwavelengthlow))
    H8normhi = np.min(np.where(hlambdastemp > heightwavelengthhigh))
    H8lambdas = hlambdastemp[H8normlow:H8normhi+1]
    H8valtemp = dataval[hlow:hhi+1]
    H8sigtemp = sigmaval[hlow:hhi+1]
    #Find the offset between the pseudogaussian fit and the actual data
    H8refit_width = 10 #pixels
    H8fit_high = fit_offset(hlambdastemp[H8normhi-H8refit_width:H8normhi+H8refit_width],H8valtemp[H8normhi-H8refit_width:H8normhi+H8refit_width],hfit[H8normhi-H8refit_width:H8normhi+H8refit_width],heightwavelengthhigh,H8sigtemp[H8normhi-H8refit_width:H8normhi+H8refit_width])
    H8fit_low = fit_offset(hlambdastemp[H8normlow-H8refit_width:H8normlow+H8refit_width],H8valtemp[H8normlow-H8refit_width:H8normlow+H8refit_width],hfit[H8normlow-H8refit_width:H8normlow+H8refit_width],heightwavelengthlow,H8sigtemp[H8normlow-H8refit_width:H8normlow+H8refit_width])
    #print H8fit_high, H8fit_low
    #H8fit_high = hfit[H8normhi]
    #H8fit_low = hfit[H8normlow]
    H8slope = (H8fit_high - H8fit_low ) / (hlambdastemp[H8normhi] - hlambdastemp[H8normlow])
    H8li = H8slope * (H8lambdas - H8lambdas[0]) + H8fit_low
    H8nline = H8valtemp[H8normlow:H8normhi+1] / H8li
    H8sigma = H8sigtemp[H8normlow:H8normhi+1] / H8li

    #plt.plot(hlambdastemp[H8normlow],hfit[H8normlow],'g^')
    #plt.plot(hlambdastemp[H8normhi],hfit[H8normhi],'g^')
    #plt.plot(H8lambdas,H8li,'g')

    ### To normalize, using points from end of region since it is so small.
    #H9lambdas = H9lambdas - (H9center- 3836.1585)
    hlambdastemp = hlambdas - (H9center- 3836.1585)
    H9normlow = np.min(np.where(hlambdastemp > hninewavelengthlow))
    H9normhi = np.min(np.where(hlambdastemp > hninewavelengthhigh))
    H9lambdas = hlambdastemp[H9normlow:H9normhi+1]
    H9valtemp = dataval[hlow:hhi+1]
    H9sigtemp = sigmaval[hlow:hhi+1]
    #Find the offset between the pseudogaussian fit and the actual data
    H9refit_width = 10 #pixels
    H9fit_high = fit_offset(hlambdastemp[H9normhi-H9refit_width:H9normhi+H9refit_width],H9valtemp[H9normhi-H9refit_width:H9normhi+H9refit_width],hfit[H9normhi-H9refit_width:H9normhi+H9refit_width],hninewavelengthhigh,H9sigtemp[H9normhi-H9refit_width:H9normhi+H9refit_width])
    H9fit_low = fit_offset(hlambdastemp[H9normlow-H9refit_width:H9normlow+H9refit_width],H9valtemp[H9normlow-H9refit_width:H9normlow+H9refit_width],hfit[H9normlow-H9refit_width:H9normlow+H9refit_width],hninewavelengthlow,H9sigtemp[H9normlow-H9refit_width:H9normlow+H9refit_width])
    #print H9fit_high, H9fit_low
    #H9fit_high = hfit[H9normhi]
    #H9fit_low = hfit[H9normlow]
    H9slope = (H9fit_high - H9fit_low ) / (hlambdastemp[H9normhi] - hlambdastemp[H9normlow])
    H9li = H9slope * (H9lambdas - H9lambdas[0]) + H9fit_low
    H9nline = H9valtemp[H9normlow:H9normhi+1] / H9li
    H9sigma = H9sigtemp[H9normlow:H9normhi+1] / H9li

    #plt.plot(hlambdastemp[H9normlow],hfit[H9normlow],'g^')
    #plt.plot(hlambdastemp[H9normhi],hfit[H9normhi],'g^')
    #plt.plot(H9lambdas,H9li,'g')

    #Check if H10 fit is too low, if so, offset from H9
    if H10center < 3780.:
        H10center = H9center - 37.493

    #H10lambdas = H10lambdas - (H10center-3799.0785)
    hlambdastemp = hlambdas - (H10center-3799.0785)
    H10normlow = np.min(np.where(hlambdastemp > htenwavelengthlow))
    H10normhi = np.min(np.where(hlambdastemp > htenwavelengthhigh))
    H10lambdas = hlambdastemp[H10normlow:H10normhi+1]
    H10valtemp = dataval[hlow:hhi+1]
    H10sigtemp = sigmaval[hlow:hhi+1]
    #Find the offset between the pseudogaussian fit and the actual data
    H10refit_width = 10 #pixels
    H10fit_high = fit_offset(hlambdastemp[H10normhi-H10refit_width:H10normhi+H10refit_width],H10valtemp[H10normhi-H10refit_width:H10normhi+H10refit_width],hfit[H10normhi-H10refit_width:H10normhi+H10refit_width],htenwavelengthhigh,H10sigtemp[H10normhi-H10refit_width:H10normhi+H10refit_width])


    #Since H10 is close to the bottom of our array, we must be careful with setting the index limits
    if H10normlow-H10refit_width < 0:
        H10index_low = 0
        H10index_high = 11
        #print 'using: ', H10index_low, H10index_high
    else:
        H10index_low = H10normlow-H10refit_width
        H10index_high = H10normlow+H10refit_width
        #print 'Now using: ', H10index_low, H10index_high
    H10fit_low = fit_offset(hlambdastemp[H10index_low:H10index_high],H10valtemp[H10index_low:H10index_high],hfit[H10index_low:H10index_high],htenwavelengthlow,H10sigtemp[H10index_low:H10index_high])
    #print H10fit_high, H10fit_low
    #H10fit_high = hfit[H10normhi]
    #H10fit_low = hfit[H10normlow]
    H10slope = (H10fit_high - H10fit_low ) / (hlambdastemp[H10normhi] - hlambdastemp[H10normlow])
    H10li = H10slope * (H10lambdas - H10lambdas[0]) + H10fit_low
    H10nline = H10valtemp[H10normlow:H10normhi+1] / H10li
    H10sigma = H10sigtemp[H10normlow:H10normhi+1] / H10li
    offsetpdf.close()
    if zzcetiblue[0] == '.':
        os.chdir(home_directory)
    #plt.plot(hlambdastemp[H10normlow],hfit[H10normlow],'g^')
    #plt.plot(hlambdastemp[H10normhi],hfit[H10normhi],'g^')
    #plt.plot(H10lambdas,H10li,'g')

    #======================================
    #Normalize by averaging continuum points and pseudogaussians for centers
    #======================================
    '''
    print 'Now normalizing the models by averaging.'
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
    blambdas = blambdas - (bcenter-4862.4555)
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
    glambdas = glambdas - (gcenter-4341.4834)
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
    hlambdastemp = hlambdas - (dcenter-4103.0343)
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
    hlambdastemp = hlambdas - (ecenter-3971.4475)
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
    hlambdastemp = hlambdas - (H8center-3890.2759)
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
    hlambdastemp = hlambdas - (H9center- 3836.1585)
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
    hlambdastemp = hlambdas - (H10center-3799.0785)
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
    '''
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


    #Save and compare fitted line centers
    pseudocenters = np.array([H10center,H9center,H8center,ecenter,dcenter,gcenter,bcenter])
    centerdiffs = pseudocenters - firstfitcenters
    allcenters = np.array([3799.0785,3836.1585,3890.2759,3971.4475,4103.0343,4341.4834,4862.4555])


    #Combine all the normalized lines together into one array for model fitting
    ###For Halpha through H10
    if redfile:
        alllambda = np.concatenate((H10lambdas,H9lambdas,H8lambdas,elambdas,dlambdas,glambdasnew,blambdasnew,alambdasnew))
        allnline = np.concatenate((H10nline,H9nline,H8nline,enline,dnline,gnline,bnline,anline))
        allsigma = np.concatenate((H10sigma,H9sigma,H8sigma,esigma,dsigma,gsigma,bsigma,asigma))
        indices  = [0,len(H10lambdas)-1.,len(H10lambdas),len(H10lambdas)+len(H9lambdas)-1.,len(H10lambdas)+len(H9lambdas),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)+len(blambdasnew)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)+len(blambdasnew),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)+len(blambdasnew)+len(alambdasnew)-1.]
        lambdaindex = [H10low,H10hi,H9low,H9hi,H8low,H8hi,elow,ehi,dlow,dhi,hlow,hhi,gfitlow,gfithi,glow,ghi,bfitlow,bfithi,blow,bhi,afitlow,afithi,alow,ahi]
        #lambdaindex = [afitlow,afithi,alow,ahi,bfitlow,bfithi,blow,bhi,gfitlow,gfithi,glow,ghi,hlow,hhi,dlow,dhi,elow,ehi,H8low,H8hi,H9low,H9hi,H10low,H10hi]
        #print len(H10nline),len(H9nline),len(H8nline),len(enline),len(dnline),len(gnline),len(bnline),len(anline)
    else:
        alllambda = np.concatenate((H10lambdas,H9lambdas,H8lambdas,elambdas,dlambdas,glambdasnew,blambdasnew))
        allnline = np.concatenate((H10nline,H9nline,H8nline,enline,dnline,gnline,bnline))
        allsigma = np.concatenate((H10sigma,H9sigma,H8sigma,esigma,dsigma,gsigma,bsigma))
        indices  = [0,len(H10lambdas)-1.,len(H10lambdas),len(H10lambdas)+len(H9lambdas)-1.,len(H10lambdas)+len(H9lambdas),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)-1.,len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew),len(H10lambdas)+len(H9lambdas)+len(H8lambdas)+len(elambdas)+len(dlambdas)+len(glambdasnew)+len(blambdasnew)-1.]
        lambdaindex = [H10low,H10hi,H9low,H9hi,H8low,H8hi,elow,ehi,dlow,dhi,hlow,hhi,gfitlow,gfithi,glow,ghi,bfitlow,bfithi,blow,bhi]
        #print len(H10nline),len(H9nline),len(H8nline),len(enline),len(dnline),len(gnline),len(bnline),len(anline)


    #variation = alphavariation + betavariation + gammavariation + highervariation
    #print variation
    #plt.clf()
    #plt.plot(alllambda,allnline,'b^')
    #plt.show()
    #sys.exit()

    #If spectra are in a different directory, change to that directory
    home_directory = os.getcwd()
    if zzcetiblue[0] == '.':
        os.chdir(zzcetiblue[0:zzcetiblue.find('w')])

    ##### Save the normalized spectrum for later use
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    marker = str(np.round(FWHM[0],decimals=2))
    endpoint = '.ms.'
    savespecname = 'norm_' + idname + '_' + now[5:10] + '_' + marker + '.txt'
    header = 'Normalized spectrum. Columns: wavelength, normalized flux, sigma' 
    np.savetxt(savespecname,np.transpose([alllambda,allnline,allsigma]),header=header)

    #Save the guesses and best-fitting parameters for the pseudogaussians
    #aest,best,bigest for guesses and aparams.params, bparams.params, hparams.params
    savefitparams = 'params_' + idname + '_' + now[5:10] + '_' + marker + '.txt'
    saveparams = np.zeros([len(bigest),6])
    saveparams[0:len(best),1] = best
    saveparams[0:len(bigest),2] = bigest
    saveparams[0:len(bparams.params),4] = bparams.params
    saveparams[0:len(hparams.params),5] = hparams.params
    if redfile:
        saveparams[0:len(aest),0] = aest
        saveparams[0:len(aparams.params),3] = aparams.params

    header = 'Guesses and best-fitting parameters from pseudogaussian fits to spectrum. Columns: Halpha guess, Hbeta guess, Hgamma-H10 guess, Halpha best fit, Hbeta best fit, Hgamma-H10 best fit '
    np.savetxt(savefitparams,saveparams,header=header)

    #Save the pseudogaussian fits to the spectrum as a pdf
    savefitspec = 'fit_' + idname  + '_' + now[5:10] + '_' + marker + '.pdf'
    fitpdf = PdfPages(savefitspec)
    if redfile:
        plt.clf()
        plt.plot(alambdas,alphaval,'b')
        plt.plot(alambdas,alphafit,'r')
        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        plt.plot(alambdas,alphaval-alphafit + (alphafit.min()+ymin)/2.5,'k')
        plt.title(zzcetired[zzcetired.find('w'):zzcetired.find(endpoint)] + ', R. chi^2: ' + str(np.round(aparams.fnorm/aparams.dof,decimals=4)))
        fitpdf.savefig()
    plt.clf()
    plt.plot(blambdas,betaval,'b')
    plt.plot(blambdas,betafit,'r')
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    plt.plot(blambdas,betaval-betafit + (betafit.min()+ymin)/2.5,'k')
    plt.title(idname + ', R. chi^2: ' + str(np.round(bparams.fnorm/bparams.dof,decimals=4)))
    fitpdf.savefig()
    plt.clf()
    plt.plot(hlambdas,hval,'b')
    plt.plot(hlambdas,hfit,'r')
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    plt.plot(hlambdas,hval-hfit + (hfit.min()+ymin)/2.5,'k')
    plt.title(idname + ', R. chi^2: ' + str(np.round(hparams.fnorm/hparams.dof,decimals=4)))
    fitpdf.savefig()
    plt.clf()
    plt.plot(hlambdas,hval,'b')
    plt.plot(hlambdas,hfit,'r')
    plt.xlim(3775,3935)
    plt.title(idname + ', Higher order lines only ' )
    fitpdf.savefig()
    plt.clf()
    plt.plot(allcenters,centerdiffs,'bo')
    plt.axhline(-1.*avgdelta,ls='--',color='r')
    plt.xlabel('Wavelength')
    plt.ylabel('Difference in Angstroms. Red in average lambda offset.')
    plt.title('Difference between individual pseudo fits and all pseudo fits')
    fitpdf.savefig()
    try:
        stitchlocation = datalistblue[0].header['STITCHLO']
        stitchpoint = stitchlocation-hlow #hlambdas starts at pixel hlow, but STITCHLO is from pixel 0. So we need that difference to use the correct location.
        plt.clf()
        plt.plot(hlambdas[stitchpoint-25:stitchpoint+26],hval[stitchpoint-25:stitchpoint+26],'b')
        plt.plot(hlambdas[stitchpoint-25:stitchpoint+26],hfit[stitchpoint-25:stitchpoint+26],'r')
        plt.axvline(hlambdas[stitchpoint],ymin=0,ymax=0.2,linewidth=4,color='k')
        plt.title('Data and fit surrounding stitch location in flat field')
        fitpdf.savefig()
    except:
        pass
    fitpdf.close()
    if zzcetiblue[0] == '.':
        os.chdir(home_directory)

    #=================
    #Run the spectrum through the coarse grid
    if not redfile:
        zzcetired = 'not_fitting_Halpha'

    print "Starting intspec.py now "
    '''
    case = 0 #We'll be interpolating Koester's raw models
    filenames = 'modelnames.txt'
    if os.getcwd()[0:4] == '/pro': #Check if we are on Hatteras
        path = '/projects/stars/uncphysics/josh/DA_models'
    elif os.getcwd()[0:4] == '/afs': #Check if we are on Infierno
        path = '/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/modelfitting/Koester_06'
    ncflux,bestT,bestg = intspecs(alllambda,allnline,allsigma,lambdaindex,case,filenames,lambdas,zzcetiblue,zzcetired,FWHM,indices,path,marker,redfile,RA,DEC,SNR,airmass,nexp,exptime)
    sys.exit()
    '''

    #================
    #Run the spectrum through the fine grid
    case = 1 #We'll be comparing our new grid to the spectrum.
    filenames = 'interpolated_names.txt'
    #filenames = 'short_list.txt'
    if os.getcwd()[0:4] == '/pro': #Check if we are on Hatteras
        path = '/projects/stars/uncphysics/josh/Koester_ML2alpha08'
        modelwavelengths = 'vacuum'
        #path = '/projects/stars/uncphysics/josh/bergeron_new'
        #modelwavelengths = 'air'
    elif os.getcwd()[0:4] == '/afs': #Check if we are on Infierno
        #path = '/srv/two/jtfuchs/Interpolated_Models/Koester_ML2alpha06/bottom11500_750'
        path = '/srv/two/jtfuchs/Interpolated_Models/Koester_ML2alpha08/bottom10000_700'
        modelwavelengths = 'vacuum'
        #path = '/srv/two/jtfuchs/Interpolated_Models/Bergeron_new/bottom10000_700'
        #modelwavelengths = 'air'

    ncflux,bestT,bestg = intspecs(alllambda,allnline,allsigma,lambdaindex,case,filenames,lambdas,zzcetiblue,zzcetired,FWHM,indices,path,marker,redfile,RA,DEC,SNR,airmass,nexp,exptime,modelwavelengths,idname)



if __name__ == '__main__':
    #Read in spectral file names from command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('specfiles',type=str,nargs='+',help='Blue/Red fits files')
    parser.add_argument('--fitguess',type=str,default='data',help='Either data or model',choices=['data','model'])
    parser.add_argument('--higherlines',type=str,default='g10',help='Either g10 or g11',choices=['g10','g11'])
    parser.add_argument('--res',type=float,help='Resolution in Angstroms')
    args = parser.parse_args()
    if len(args.specfiles) == 1:
        zzcetiblue = args.specfiles[0]
        zzcetired = None
        redfile = False
    elif len(args.specfiles) ==2:
        zzcetiblue = args.specfiles[0]
        zzcetired = args.specfiles[1]
        redfile = True
    else:
        print '\n Incorrect number of spectra. \n'

    fit_now(zzcetiblue,zzcetired,redfile,args.fitguess,args.higherlines,args.res)
