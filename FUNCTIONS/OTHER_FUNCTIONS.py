#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# <center><h1> Functions other than light model (DOY, LC, ZPD, and MLD) </h1></center>
# 

# <div style="text-align: right"><i> Renosh - Nov 2023 </i></div>
# <div style="text-align: right"><i> Email: renosh.pr@imev-mer.fr </i></div>
# <div style="text-align: right"><i> Email: pr.renosh@gmail.com </i></div>

# ## 1) Load all libraries

# In[3]:


import os
import numpy as np
import pandas as pd


# ## 2) Functions other than light model (DOY, LC, ZPD, and MLD)

# In[4]:


#######################################################################################
#######################################################################################
# Function to convert DOY into Radians
# No need to change anything in this section by the user
def rad_date(date):
    date_radians = (date*np.pi)/182.5
    return date_radians
#######################################################################################
#######################################################################################
# Function to convert Local Hour into Radians
# No need to change anything in this section by the user
def rad_LC_hour(LC_hour):
    LC_hour_radians = (LC_hour*np.pi)/12 # 
    return LC_hour_radians
#######################################################################################
# Function to derive ZPD from KD490
# No need to change anything in this section by the user
def derive_ZPD(KD490):
    ZPD = 1/KD490 
    return ZPD
#######################################################################################
############################################################
##  function TO FIND THE MIXED LAYER DEPTH (DEBOYER)       ##
#############################################################
# No need to change anything in this section by the user
# Ref: DeBoyer Montegut et al. 2004
# Usual global threshold 0.03
def find_MLD(Z, dens, threshold):
    Zmax = np.nanmax(Z)
    Zint = np.arange(0, Zmax+1,1)
    
    dens1 = dens[~np.isnan(dens)]
    pres1= Z[~np.isnan(dens)]
    if (Zmax <20): 
        MLD = np.nan
    else:
    
        D = np.interp(Zint, pres1, dens1)
        delta = D - D[21]
        MLD1 = Zint[(abs(delta) > threshold) & (Zint >= 10)]

        if (np.size(MLD1)<1):
            MLD = np.nan
        else:
            MLD = np.nanmin(MLD1)
    return MLD
#######################################################################################
#######################################################################################


# In[ ]:




