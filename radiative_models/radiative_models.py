import os, random

import pandas as pd
import numpy as np

def generate_opticallythin_spectra(nr, nr_bins=150):

    def rebin(s, bins, wavelength_col = "w", value_col = "a", handle_nan = False):
        s['w_bins'] = pd.cut(s[wavelength_col], bins) # Cut the wavelength into bins
        s_bin = s.groupby(['w_bins'])[value_col].mean().reset_index() # Group by bin and average in the bin
        s = np.array(s_bin[value_col])
        if handle_nan:
            found = np.isnan(s)
            if found.any():
                s = [0 if found[i] else s[i] for i in range(len(s))]
        return s

    def B(lambd, T, micron=True):
        c = 299792458 # metres per second.
        h = 6.62607015e-34 # joule second
        k = 1.380649e-23 # joule per kelvin (K)

        if micron:
            lambd = lambd*1e-6
        return (2*h*c**2/lambd**5) * 1 / (np.exp(h*c/(k*T*lambd))-1)

    def syn_spec(w, a, opac, T):
        spec = np.zeros(opac[0].shape)
        for i in range(len(a)):
            spec = spec + a[i]*opac[i]
        f = B(w, T) * spec
        f = f/f.max()
        return f
    
    # Read in the opacities
    directory = "opacities"
    file_forsterite = 'Forsterite0.1.Kabs'
    opac_fo = np.loadtxt(os.path.join(directory, file_forsterite))
    file_amorphSilicate = 'AmorphousOlivineX0.5_0.1.Kabs'
    opac_am = np.loadtxt(os.path.join(directory, file_amorphSilicate))
    file_enstatite = 'Enstatite0.1.Kabs'
    opac_en = np.loadtxt(os.path.join(directory, file_enstatite))
    
    # Rebin the opacities
    bins = np.linspace(5.,35., num=nr_bins)#121)#[0, 5,10,15,20,25,30,35,40]
    new_wavelength = []
    for i in range(len(bins)):
        if i < len(bins)-1: 
            new_wavelength.append(bins[i]+(bins[i+1]-bins[i])/2)
    new_wavelength = np.array(new_wavelength)

    res = new_wavelength[1:-1] - new_wavelength[0:-2]
    print("New resolution: ",round(res.mean(),4),"+/-", round(res.std(),4))
    opac_fo_df = pd.DataFrame(opac_fo, columns=['w', 'a'])
    opac_am_df = pd.DataFrame(opac_am, columns=['w', 'a'])
    opac_en_df = pd.DataFrame(opac_en, columns=['w', 'a'])

    opac_fo_bin = rebin(opac_fo_df, bins)
    opac_en_bin = rebin(opac_en_df, bins)
    opac_am_bin = rebin(opac_am_df, bins)
    
    print("Nan values: ", np.isnan(opac_fo_bin).any())
    #print(list(zip(new_wavelength, opac_fo_bin)))

#    print("Length of the new wavelength grid: ", opac_fo_bin.shape)
    
    # Generate parameters of the radiative model
    cr_a_min, cr_a_max = 0,8
    cr_threshold = 3
    T_min, T_max = 100, 300
    a_fo = np.array([random.uniform(cr_a_min,cr_a_max) for i in range(nr)])
    a_en = np.array([random.uniform(cr_a_min,cr_a_max) for i in range(nr)])
    a_am = [100-(a_fo[i]+a_en[i]) for i in range(nr)]
    Ts = [random.uniform(T_min,T_max) for i in range(nr)]
    
    # Determine the labels of the spectra
    labels = np.array([1 if a_fo[i] > cr_threshold else 0 for i in range(nr)])

    
    grid = np.array([a_fo, a_en, a_am, Ts]).T
    specs = np.array([syn_spec(new_wavelength, [a_fo[i], a_en[i], a_am[i]], 
                               [opac_fo_bin, opac_en_bin, opac_am_bin], Ts[i]) for i in range(len(a_fo))])
    

    
    return specs, labels, grid