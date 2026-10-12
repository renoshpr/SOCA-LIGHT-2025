#!/usr/bin/env python
# coding: utf-8

# ## Function to apply the bathymetric mask

# ### Install libraries

# In[1]:


import numpy as np


# ### Define function to get bathy condition

# In[2]:


def mask_bathy(bathy_mat, lon_bathy, lat_bathy, lon, lat, threshold):
    i_bath = np.argmin(np.abs(lon_bathy-lon))
    j_bath = np.argmin(np.abs(lat_bathy-lat))
    height = bathy_mat[j_bath,i_bath]
    if height>threshold:
        bathy = False
    else:
        bathy = True
    return bathy

