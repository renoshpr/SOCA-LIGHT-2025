#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


def find_Ze(chl_sat):
    
    ## Calcul de Ze
    ## Calcul du Kd
    ##--------------
    # K: coefficient d'attenuation global
    # Kw: coefficient d'attenuation de l'eau pure (m^-1)(490 nm)
    # Kbio: coefficient d'attenuation biologique (490 nm)
    # K490: coefficient d'attenuation à 490 nm.
    
    # pour la longueur d'onde lambda = 490 nm  #
    Kw = 0.01660
    e = 0.68955
    X = 0.07242
    
    # d'aprés A. Morel 2001  #
    K490 = Kw + X*chl_sat**e
    
    #d'apres Rochford et al. 2001
    Kd = 0.0085 + 1.6243*K490
    
    Ze = -np.log(0.01)/Kd
    return Ze


# In[ ]:


def find_Ze(chl_sat):
    """
    Calculate Ze based on chlorophyll-a concentration.
    
    Parameters:
        chl_sat (np.ndarray): Array of chlorophyll-a concentrations.
        
    Returns:
        np.ndarray: Array of Ze values.
    """
    # Constants
    Kw = 0.01660
    e = 0.68955
    X = 0.07242

    # Calculate K490
    K490 = Kw + X * chl_sat**e

    # Calculate Kd
    Kd = 0.0085 + 1.6243 * K490

    # Calculate Ze
    Ze = -np.log(0.01) / Kd
    return Ze

