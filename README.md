# Pre-processing pipeline for ARIES/MMT
Pre-process raw detector images from ARIES/MMT combination and reduce them to aligned and wavelength calibrated spectral time series.

## Credit
Developed by Lennart van Sluijs. For more information on the pre-processing pipeline please have a look at the Methods section in [van Sluijs et al. 2022](https://arxiv.org/abs/2203.13234). Please cite this paper if you make use of this pipeline. Please get in touch in case of further questions lennart.vansluijs@physics.ox.ac.uk.

## Dependencies
I use Anaconda 3 to manage the dependencies. The main repository contains two ```.yml``` files. These contain all dependencies. After cloning and entering the repository, run the following lines from a terminal
```bash
conda env create -f ariespipe_py27.yml
conda env create -f ariespipe_py3.yml
```
to install all dependencies. Furthermore, I included the packages for [CERES](https://github.com/rabrahm/ceres) and [TelFit](https://github.com/kgullikson88/Telluric-Fitter) packages in the ```\lib``` folder for convenience, but all credit goes to the corresponding authors of these software packages. Please have a look at their own documentation if you run into any issues with these packages. It also uses the [Corquad](http://66.194.178.32/~rfinn/pisces.html) routine to correct for ghost-like detector images (so-called crosstalk). A Python based version of this program can be found [here](https://github.com/jordan-stone/ARIES).

## Run the pre-processing pipeline
After installing all dependencies one should be able to run the pre-processing pipeline. The pipeline requires input on the target and observing night. Using the following command
```python
./ARIESpipe "wasp33" "15102016"
```
should start all subsequent data reduction steps on one night (15/10/2016) of WASP-33 observations. The input should be in the ```/data/wasp33/10152016/raw``` directory and science/flat/dark frames should have respectively "science"/"flat"/"dark" in their filenames. There are two more nights in the ```\data\wasp33``` folder.

## Visualising the output
I have included a short Jupyter notebook which can be used to load and plot the final aligned and wavelength calibrated spectral time series.
```bash
jupyter-notebook plot_wasp33_15102016.ipynb
```

## Notes
The flats in the raw data directories were all taken at different elevation angles. These angles can be accessed through the ```.fits``` file header. The alignment of the spectra is done by alignment of all spectra to the first spectrum of the night. This implicitly assumes a stable wavelength solution for the course of one observing night.
