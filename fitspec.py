"""
Fitting routine to determine Teff and logg for ZZ Cetis

Author: Josh Fuchs, UNC. With substantial initial work done by Bart Dunlap.

:INPUTS:
       zzcetiblue: string, file containing 1D wavelength calibrated 'ZZ Ceti blue' spectrum

:OPTIONAL:
       zzcetired: string, file containing 1D wavelength calibrated 'ZZ Ceti red' spectrum.

:OUTPUTS:
      ALL CAPS BELOW means variable determined by program. WDNAME is the name of the input spectrum. DATE is the date fitspec.py was run. OPTIONS is optional inputs by user.

       fit_WDNAME_DATE_OPTIONS.pdf: pdf containing pseudogaussian fit to spectrum. Each fit is shown on a separate page.

       params_WDNAME_DATE_OPTIONS.txt: text file containg initial guesses and final fitting parameters from pseudogaussians. Columns are: Halpha guess, Hbeta guess, Hgamma-H10 guess, Halpha best fit, Hbeta best fit, Hgamma-H10 best fit

       norm_WDNAME_DATE_OPTIONS.txt: text file containing normalized spectrum. Columns are wavelength, normalized flux, sigma

:TO RUN:
       python fitspec.py zzcetiblue zzcetired
       python fitspec.py wtfb.wd1425-811_930_blue_flux_model.ms.fits wtfb.wd1425-811_930_red_flux_model.ms.fits

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
import pyfits as fits # Infierno doesn't support astropy for some reason so using pyfits
from glob import glob
#import astropy.io.fits as pf


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
    plt.plot(wavelengths,fit,'r',label='old')
    plt.plot(wavelengths,find_offset(fit,offset_params.params),'b',label='with offset')
    plt.legend()
    #plt.show()
    offsetpdf.savefig()
    return norm_value



# ===========================================================================
# ===========================================================================
# ===========================================================================
# ===========================================================================
# ===========================================================================
# Begin the actual code here
#Now we need to read in actual spectrum.
if len(sys.argv) == 3: 
    script, zzcetiblue, zzcetired = sys.argv
    redfile = True
    print 'Now fitting %s and %s \n' % (zzcetiblue, zzcetired)
elif len(sys.argv) == 2:
    script, zzcetiblue = sys.argv
    redfile = False
    print 'Now fitting %s \n' % zzcetiblue
else:
    print '\n Incorrect number of arguments. \n'

#Read in the blue spectrum
datalistblue = fits.open(zzcetiblue)
datavalblue = datalistblue[0].data[0,0,:] #Reads in the object spectrum,data[0,0,:] is optimally subtracted, data[1,0,:] is raw extraction,  data[2,0,:] is sky, data[3,0,:] is sigma spectrum
sigmavalblue = datalistblue[0].data[3,0,:] #Sigma spectrum

#Header values to save
RA = datalistblue[0].header['RA']
DEC = datalistblue[0].header['DEC']
SNR = 1.#float(datalistblue[0].header['SNR'])
airmass = float(datalistblue[0].header['AIRMASS'])
nexp = float(datalistblue[0].header['NCOMBINE'])
exptime = float(datalistblue[0].header['EXPTIME'])


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
biningblue= float( datalistblue[0].header["PARAM18"] ) 
nxblue= np.size(datavalblue)#spec_data[0]
PixelsBlue= biningblue*(np.arange(0,nxblue,1)+trim_offset_blue)
lambdasblue = DispCalc(PixelsBlue, alphablue, thetablue, frblue, fdblue, flblue, zPntblue)

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
    biningred= float( datalistred[0].header["PARAM18"] ) 
    nxred= np.size(datavalred)#spec_data[0]
    PixelsRed= biningred*(np.arange(0,nxred,1)+trim_offset_red)
    lambdasred = DispCalc(PixelsRed, alphared, thetared, frred, fdred, flred, zPntred)


#Concatenate both into two arrays
if redfile:
    lambdas = np.concatenate((lambdasblue,lambdasred))
    dataval = np.concatenate((datavalblue,datavalred))
    sigmaval = np.concatenate((sigmavalblue,sigmavalred))#2.e-17 * np.ones(len(dataval)) small/big44./87.*
    FWHM = (lambdasblue[-1] - lambdasblue[0])/nxblue * np.concatenate((FWHMpixblue,FWHMpixred)) #from grating equation
    #FWHM = deltawavblue * np.concatenate((FWHMpixblue,FWHMpixred)) #FWHM in Angstroms linearized
else:
    lambdas = np.array(lambdasblue)
    dataval = np.array(datavalblue)
    sigmaval = np.array(sigmavalblue)
    FWHM = FWHMpixblue * (lambdasblue[-1] - lambdasblue[0])/nxblue #from grating equation
    #FWHM = FWHMpixblue * deltawavblue #FWHM in Angstroms linearized


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
    
    ######
    #From fit to GD 165 on 2015-04-26
    '''
    aest[0] = -3.329793545118666952e+04
    aest[1] = 1.452254867177559028e+01
    aest[2] = -2.080336009400552289e-03
    aest[3] = -1.068782717847220596e+02
    aest[4] = 6.564426653206536685e+03
    aest[5] = 3.961723338240169312e+01
    aest[6] = 7.322514919203364503e-01
    aest[7] = 9.830814524561716704e-08
    '''
    


blambdas = lambdas[bfitlow:bfithi+1]
bsigmas = sigmaval[bfitlow:bfithi+1]
betaval = dataval[bfitlow:bfithi+1]

best = np.zeros(8)
#######

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
best[6] = 1.0 #how much of a pseudo-gaussian

##########
#From fit to GD 165 on 2015-04-26
'''
best[0] = 3.464194462449746788e+05
best[1] = -2.130872658512382429e+02
best[2] = 4.378414747018435915e-02
best[3] = -2.988719360409691035e+02
best[4] = 4.861631463570591222e+03
best[5] = 3.362166147167607733e+01
best[6] = 8.720131814693605765e-01
best[7] = -3.001151458131578997e-06
'''

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

#########################
#Fit gamma through 11
'''
print 'Now fitting gamma through 11'
highwavelengthlow = 3755. #3782 for H10 and 3755 for H11
hlow = np.min(np.where(lambdas > highwavelengthlow)) 

hlambdas = lambdas[hlow:gfithi+1]
hval = dataval[hlow:gfithi+1]
hsig = sigmaval[hlow:gfithi+1]


bigest = np.zeros(32) #Array for guess parameters
'''
'''
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
'''
'''
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
bigest[4] = np.min(dataval[glow:ghi+1]) - bigpp(4341.692) #depth of line relative to continuum
bigest[5] = 4341.692 #rest wavelength of H gamma
ghalfmax = bigpp(4341.69) + bigest[4]/2.0
gdiff = np.abs(gamval-ghalfmax)
glowidx = gdiff[np.where(glambdas < 4341.69)].argmin()
ghighidx = gdiff[np.where(glambdas > 4341.69)].argmin() + len(bdiff[np.where(glambdas < 4341.69)])
bigest[6] = (glambdas[ghighidx] - glambdas[glowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
bigest[7] = 1.0 #how much of a pseudo-gaussian


#Now delta
bigest[8] = np.min(dataval[dlow:dhi+1]) - bigpp(4102.892) #depth of line relative to continuum
bigest[9] = 4102.892  #rest wavelength of H delta
dhalfmax = bigpp(4102.89) + bigest[8]/2.0
ddiff = np.abs(dval-dhalfmax)
dlowidx = ddiff[np.where(dlambdas < 4102.89)].argmin()
dhighidx = ddiff[np.where(dlambdas > 4102.89)].argmin() + len(ddiff[np.where(dlambdas < 4102.89)])
bigest[10] = (dlambdas[dhighidx] - dlambdas[dlowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[11] = 1.2 #how much of a pseudo-gaussian

#Now epsilon
bigest[12] = np.min(dataval[elow:ehi+1]) - bigpp(3971.198) #depth of line relative to continuum
bigest[13] = 3971.198   #rest wavelength of H epsilon
ehalfmax = bigpp(3971.19) + bigest[12]/2.0
ediff = np.abs(epval-ehalfmax)
elowidx = ediff[np.where(elambdas < 3971.19)].argmin()
ehighidx = ediff[np.where(elambdas > 3971.19)].argmin() + len(ediff[np.where(elambdas < 3971.19)])
bigest[14] = (elambdas[ehighidx] - elambdas[elowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[15] = 1.2 #how much of a pseudo-gaussian

#Now H8
bigest[16] = np.min(dataval[H8low:H8hi+1]) - bigpp(3890.166) #depth of line relative to continuum
bigest[17] = 3890.166   #rest wavelength of H8
H8halfmax = bigpp(3890.16) + bigest[16]/2.0
H8diff = np.abs(H8val-H8halfmax)
H8lowidx = H8diff[np.where(H8lambdas < 3890.16)].argmin()
H8highidx = H8diff[np.where(H8lambdas > 3890.16)].argmin() + len(H8diff[np.where(H8lambdas < 3890.16)])
bigest[18] = (H8lambdas[H8highidx] - H8lambdas[H8lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[19] = 1.2 #how much of a pseudo-gaussian

#Now H9
bigest[20] = np.min(dataval[H9low:H9hi+1]) - bigpp(3836.485) #depth of line relative to continuum
bigest[21] = 3837.485   #rest wavelength of H9
H9halfmax = bigpp(3836.48) + bigest[20]/2.0
H9diff = np.abs(H9val-H9halfmax)
H9lowidx = H9diff[np.where(H9lambdas < 3836.48)].argmin()
H9highidx = H9diff[np.where(H9lambdas > 3836.48)].argmin() + len(H9diff[np.where(H9lambdas < 3836.48)])
bigest[22] = (H9lambdas[H9highidx] - H9lambdas[H9lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[23] = 1.2 #how much of a pseudo-gaussian

#Now H10
bigest[24] = np.min(dataval[H10low:H10hi+1]) - bigpp(3797.909) #depth of line relative to continuum
bigest[25] = 3798.909   #rest wavelength of H10
H10halfmax = bigpp(3798.8) + bigest[24]/2.0
H10diff = np.abs(H10val-H10halfmax)
H10lowidx = H10diff[np.where(H10lambdas < 3798.8)].argmin()
H10highidx = H10diff[np.where(H10lambdas > 3798.8)].argmin() + len(H10diff[np.where(H10lambdas < 3798.8)])
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
'''
'''
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
'''
'''
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
'''
now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
endpoint = '.ms.'
savefitspec = 'fit_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_' + now[5:10] + '_H11.pdf'
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

#Fit gamma through 10 at once
#highwavelengthlow = 3782. #3782 for H10 and 3755 for H11
#hlow = np.min(np.where(lambdas > highwavelengthlow)) 

hlambdas = lambdas[hlow:gfithi+1]
hval = dataval[hlow:gfithi+1]
hsig = sigmaval[hlow:gfithi+1]

bigest = np.zeros(28)

#Guesses from GD 165: 2015-04-26
'''
bigest[0] = -1.406063761484372953e+05#-1.41159057e+05
bigest[1] = 1.003170676291885854e+02#1.00443047e+02
bigest[2] = -2.363770735364889922e-02#-2.36076932e-02
bigest[3] = 1.848214768237999889e-06#1.84139110e-06
bigpp = np.poly1d([bigest[3],bigest[2],bigest[1],bigest[0]])
bigest[4] = -4.040590087943886033e+02#-4.06035255e+02
bigest[5] = 4.340925249063438059e+03#4.34092583e+03
bigest[6] = 2.708817417035567132e+01#2.72522312e+01
bigest[7] = 9.878402835308270902e-01#9.83112306e-01
bigest[8] = -4.794659311360872493e+02#-4.78134615e+02
bigest[9] = 4.102701783225264080e+03#4.10272287e+03
bigest[10] = 2.709696664077381456e+01#2.69840743e+01
bigest[11] = 9.631399125660804472e-01#9.61424635e-01
bigest[12] = -4.166094422675674878e+02#-4.11042483e+02
bigest[13] = 3.971486630382756175e+03#3.97149906e+03
bigest[14] = 2.116735070719211720e+01#2.08403895e+01
bigest[15] = 1.130689652723139815e+00#1.14000247e+00
bigest[16] = -3.106269362989938259e+02#-3.07128449e+02
bigest[17] = 3.889903866559738617e+03#3.88991856e+03
bigest[18] = 1.782651953604086259e+01#1.76330401e+01
bigest[19] = 1.223808191795874301e+00#1.22352091e+00
bigest[20] = -1.487979807794951057e+02#-1.48658941e+02
bigest[21] = 3.837355731272535195e+03#3.83730915e+03
bigest[22] = 1.148214333621755578e+01#1.14755484e+01
bigest[23] = 1.425091059538869054e+00#1.42376957e+00
bigest[24] = -1.930436817526830851e+02#-1.78719907e+02
bigest[25] = 3.797740412884535090e+03#3.79796636e+03
bigest[26] = 3.532527100706699485e+01#3.30098176e+01
bigest[27] = 1.092804870594776379e+00#1.07062679e+00
'''

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


bigest[4] = np.min(dataval[glow:ghi+1]) - bigpp(4341.692) #depth of line relative to continuum
bigest[5] = 4341.692 #rest wavelength of H gamma
ghalfmax = bigpp(4341.69) + bigest[4]/2.0
gdiff = np.abs(gamval-ghalfmax)
glowidx = gdiff[np.where(glambdas < 4341.69)].argmin()
ghighidx = gdiff[np.where(glambdas > 4341.69)].argmin() + len(bdiff[np.where(glambdas < 4341.69)])
bigest[6] = (glambdas[ghighidx] - glambdas[glowidx]) / (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma
bigest[7] = 1.0 #how much of a pseudo-gaussian

plt.clf()
plt.plot(glambdas[glowidx],gamval[glowidx],'k^')
plt.plot(glambdas[ghighidx],gamval[ghighidx],'k^')

#Now delta
bigest[8] = np.min(dataval[dlow:dhi+1]) - bigpp(4102.892) #depth of line relative to continuum
bigest[9] = 4102.892  #rest wavelength of H delta
dhalfmax = bigpp(4102.89) + bigest[8]/2.0
ddiff = np.abs(dval-dhalfmax)
dlowidx = ddiff[np.where(dlambdas < 4102.89)].argmin()
dhighidx = ddiff[np.where(dlambdas > 4102.89)].argmin() + len(ddiff[np.where(dlambdas < 4102.89)])
bigest[10] = (dlambdas[dhighidx] - dlambdas[dlowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[11] = 1.2 #how much of a pseudo-gaussian

#Now epsilon
bigest[12] = np.min(dataval[elow:ehi+1]) - bigpp(3971.198) #depth of line relative to continuum
bigest[13] = 3971.198   #rest wavelength of H epsilon
ehalfmax = bigpp(3971.19) + bigest[12]/2.0
ediff = np.abs(epval-ehalfmax)
elowidx = ediff[np.where(elambdas < 3971.19)].argmin()
ehighidx = ediff[np.where(elambdas > 3971.19)].argmin() + len(ediff[np.where(elambdas < 3971.19)])
bigest[14] = (elambdas[ehighidx] - elambdas[elowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[15] = 1.2 #how much of a pseudo-gaussian

#Now H8
bigest[16] = np.min(dataval[H8low:H8hi+1]) - bigpp(3890.166) #depth of line relative to continuum
bigest[17] = 3890.166   #rest wavelength of H8
H8halfmax = bigpp(3890.16) + bigest[16]/2.0
H8diff = np.abs(H8val-H8halfmax)
H8lowidx = H8diff[np.where(H8lambdas < 3890.16)].argmin()
H8highidx = H8diff[np.where(H8lambdas > 3890.16)].argmin() + len(H8diff[np.where(H8lambdas < 3890.16)])
bigest[18] = (H8lambdas[H8highidx] - H8lambdas[H8lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[19] = 1.2 #how much of a pseudo-gaussian

#Now H9
bigest[20] = np.min(dataval[H9low:H9hi+1]) - bigpp(3836.485) #depth of line relative to continuum
bigest[21] = 3837.485   #rest wavelength of H9
H9halfmax = bigpp(3836.48) + bigest[20]/2.0
H9diff = np.abs(H9val-H9halfmax)
H9lowidx = H9diff[np.where(H9lambdas < 3836.48)].argmin()
H9highidx = H9diff[np.where(H9lambdas > 3836.48)].argmin() + len(H9diff[np.where(H9lambdas < 3836.48)])
bigest[22] = (H9lambdas[H9highidx] - H9lambdas[H9lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[23] = 1.2 #how much of a pseudo-gaussian
plt.plot(H9lambdas[H9lowidx],H9val[H9lowidx],'k^')
plt.plot(H9lambdas[H9highidx],H9val[H9highidx],'k^')

#Now H10
bigest[24] = np.min(dataval[H10low:H10hi+1]) - bigpp(3797.909) #depth of line relative to continuum
bigest[25] = 3798.909   #rest wavelength of H10
H10halfmax = bigpp(3798.8) + bigest[24]/2.0
H10diff = np.abs(H10val-H10halfmax)
H10lowidx = H10diff[np.where(H10lambdas < 3798.8)].argmin()
H10highidx = H10diff[np.where(H10lambdas > 3798.8)].argmin() + len(H10diff[np.where(H10lambdas < 3798.8)])
bigest[26] = (H10lambdas[H10highidx] - H10lambdas[H10lowidx]) / (2.*np.sqrt(2.*np.log(2.)))
bigest[27] = 1.2 #how much of a pseudo-gaussian


print 'Now fitting H-gamma through H10.'
bigfa = {'x':hlambdas, 'y':hval, 'err':hsig}
hparams = mpfit.mpfit(fitbigpseudogaussgamma,bigest,functkw=bigfa,maxiter=300,ftol=1e-12,xtol=1e-8,quiet=True)#-10,-8
#print bigest
print hparams.status, hparams.niter, hparams.fnorm, hparams.dof, hparams.fnorm/hparams.dof
#print bigest
print hparams.params
print ''
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
gfithi2 = np.min(np.where(hlambdas > 4378.))
hlow2 = np.min(np.where(hlambdas > 3778.)) 
hhi2 = np.min(np.where(hlambdas > 4195.)) 


#plt.clf()
#plt.plot(hlambdas,hval,'b')
#plt.plot(biglambdas,bigpp(biglambdas),'r')
#plt.plot(hlambdas,bigguess,'g')
#plt.plot(hlambdas,bigpp(hlambdas),'g')
#plt.plot(hlambdas,hfit,'r')
#plt.plot(hlambdas,hparams.params[0] + hparams.params[1]*hlambdas + hparams.params[2]*hlambdas**2. + hparams.params[3]*hlambdas**3.,'r')
#plt.title(np.round(hparams.fnorm/hparams.dof,decimals=4))
#plt.show()


'''
#Save the pseudogaussian fits to the spectrum as a pdf
now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
endpoint = '.ms.'
savefitspec = 'fit_' + zzcetiblue[5:zzcetiblue.find(endpoint)] + '_' + now[5:10] + '_H10.pdf'
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
bcenter = bparams.params[4]
betafit = pseudogausscubic(blambdas,bparams.params)
betavariation = np.sum((betafit - betaval)**2.)

#plt.clf()
#plt.plot(blambdas,betaval,'b^',label='data')
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
saveoffsets = 'offsets_' + zzcetiblue[zzcetiblue.find('w'):zzcetiblue.find(endpoint)] + '_' + now[5:10] + '_' + marker + '.pdf'
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
blambdas = blambdas - (bcenter-4862.6510)
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
#sys.exit()

#Now do gamma
#gnline is the normalized spectral line. Continuum set to one.
#Set the center of the line to the wavelength of models
glambdas = glambdas - (gcenter-4341.6550)
gnormlow = np.min(np.where(glambdas > gammanormwavelengthlow))
gnormhi = np.min(np.where(glambdas > gammanormwavelengthhigh))
#Find the offset between the pseudogaussian fit and the actual data
grefit_width = 10
gfit_high = fit_offset(glambdas[gnormhi-grefit_width:gnormhi+grefit_width],gamval[gnormhi-grefit_width:gnormhi+grefit_width],gamfit[gnormhi-grefit_width:gnormhi+grefit_width],gammanormwavelengthhigh,gsigmas[gnormhi-grefit_width:gnormhi+grefit_width])
gfit_low = fit_offset(glambdas[gnormlow-grefit_width:gnormlow+grefit_width],gamval[gnormlow-grefit_width:gnormlow+grefit_width],gamfit[gnormlow-grefit_width:gnormlow+grefit_width],gammanormwavelengthlow,gsigmas[gnormlow-grefit_width:gnormlow+grefit_width])
#print gfit_high, gfit_low
#gfit_high = gamfit[bnormhi]
#gfit_low = gamfit[bnormlow]
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
hlambdastemp = hlambdas - (dcenter-4102.9071)
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

#elambdas = elambdas - (ecenter-3971.198)
hlambdastemp = hlambdas - (ecenter-3971.1751)
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

#H8lambdas = H8lambdas - (H8center-3890.166)
hlambdastemp = hlambdas - (H8center-3890.1461)
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
#H9lambdas = H9lambdas - (H9center- 3836.485)
hlambdastemp = hlambdas - (H9center- 3836.4726)
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

#H10lambdas = H10lambdas - (H10center-3797.909)
hlambdastemp = hlambdas - (H10center-3798.9799)
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
#plt.plot(alllambda,allnline)
#plt.show()
#sys.exit()

#measuredcenter = np.array([acenter,bcenter,gcenter,dcenter,ecenter,H8center,H9center])
#restwavelength = np.array([6562.79,4862.71,4341.69,4102.89,3971.19,3890.16,3836.48])
#c = 2.99792e5
#velocity = c * (measuredcenter-restwavelength)/restwavelength

#If spectra are in a different directory, change to that directory
home_directory = os.getcwd()
if zzcetiblue[0] == '.':
    os.chdir(zzcetiblue[0:zzcetiblue.find('w')])

##### Save the normalized spectrum for later use
now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
marker = str(np.round(FWHM[0],decimals=2))
endpoint = '.ms.'
savespecname = 'norm_' + zzcetiblue[zzcetiblue.find('w'):zzcetiblue.find(endpoint)] + '_' + now[5:10] + '_' + marker + '.txt'
header = 'Normalized spectrum. Columns: wavelength, normalized flux, sigma' 
np.savetxt(savespecname,np.transpose([alllambda,allnline,allsigma]),header=header)

#Save the guesses and best-fitting parameters for the pseudogaussians
#aest,best,bigest for guesses and aparams.params, bparams.params, hparams.params
savefitparams = 'params_' + zzcetiblue[zzcetiblue.find('w'):zzcetiblue.find(endpoint)] + '_' + now[5:10] + '_' + marker + '.txt'
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
savefitspec = 'fit_' + zzcetiblue[zzcetiblue.find('w'):zzcetiblue.find(endpoint)] + '_' + now[5:10] + '_' + marker + '.pdf'
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
plt.title(zzcetiblue[zzcetiblue.find('w'):zzcetiblue.find(endpoint)] + ', R. chi^2: ' + str(np.round(bparams.fnorm/bparams.dof,decimals=4)))
fitpdf.savefig()
plt.clf()
plt.plot(hlambdas,hval,'b')
plt.plot(hlambdas,hfit,'r')
axes = plt.gca()
ymin, ymax = axes.get_ylim()
plt.plot(hlambdas,hval-hfit + (hfit.min()+ymin)/2.5,'k')
plt.title(zzcetiblue[zzcetiblue.find('w'):zzcetiblue.find(endpoint)] + ', R. chi^2: ' + str(np.round(hparams.fnorm/hparams.dof,decimals=4)))
fitpdf.savefig()
try:
    stitchpoint = datalistblue[0].header['STITCHLO']
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
#filenames = 'interpolated_names.txt'
filenames = 'short_list.txt'
if os.getcwd()[0:4] == '/pro': #Check if we are on Hatteras
    path = '/projects/stars/uncphysics/josh/Koester_ML2alpha08'
elif os.getcwd()[0:4] == '/afs': #Check if we are on Infierno
    #path = '/srv/two/jtfuchs/Interpolated_Models/Koester_ML2alpha06/bottom11500_750'
    path = '/srv/two/jtfuchs/Interpolated_Models/Koester_ML2alpha08/bottom10000_700'

ncflux,bestT,bestg = intspecs(alllambda,allnline,allsigma,lambdaindex,case,filenames,lambdas,zzcetiblue,zzcetired,FWHM,indices,path,marker,redfile,RA,DEC,SNR,airmass,nexp,exptime)

