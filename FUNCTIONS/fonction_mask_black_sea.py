#!/usr/bin/env python
# coding: utf-8

# ## Function to apply a mask and remove the data in the Black Sea

# In[4]:


def mask_black_sea(lon, lat):
    black_sea = False
    if lon>27 and lon<42 and lat>40 and lat<48:
        black_sea = True
    return black_sea

