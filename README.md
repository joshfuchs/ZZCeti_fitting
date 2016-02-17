Written in python by JT Fuchs
Parts originally done in IDL by BH Dunlap

As of February 2016, these modules are still a work in progress.

These modules take spectra of white dwarfs and determine a temperature and surface gravity by comparing the spectra to DA models from D. Koester. The modules will interpolate Koester's models.

fitspec.py: Reads in spectrum and fits the Balmer lines with pseudo-gausians. Uses the fit to normalize each line.

intspec.py: Compares the models to the observed spectrum. The models are interpolated and convolved to the seeing of the observed spectrum. The model Balmer lines are fitted with pseudo-gaussians like the observed spectrum and then normalized in the same way.

finegrid.py: Takes the best log(g) and Teff result from intspec.py and sets up the model grid for interpolation to a smaller grid. 

intmodels.py: Interpolates the coarse model grid to a smaller model grid. 

Dependencies:
- numpy
- scipy
- pyfits 
- mpfit (can be found at http://code.google.com/p/astrolibpy/source/browse/trunk/)