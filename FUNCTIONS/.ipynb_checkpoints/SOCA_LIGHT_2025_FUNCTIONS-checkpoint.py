#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# <center><h1> Functions of all light variable (PAR, ED380, ED412 and ED490) profiles from Temp, Sal profiles and corresponding Satellite Ocean Color Matchups </h1></center>
# 
# 

# <div style="text-align: right"><i> Renosh - June 2025 </i></div>
# <div style="text-align: right"><i> Email: renosh.pr@imev-mer.fr </i></div>
# <div style="text-align: right"><i> Email: pr.renosh@gmail.com </i></div>

# In[10]:


import warnings
warnings.filterwarnings('ignore')


# In[11]:


import pathlib
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import time as time
import torch
import torch.nn as nn
import torch.optim as optim
import pickle


# In[12]:


# MODEL PATHS
# MODEL PATH---->

# get the current working directory
curr_work_dir = Path.cwd()
# home_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
# print output to the console
# print(curr_work_dir)



#######################################################################################
MODEL_PAR_PATH = (''.join([str(curr_work_dir),'/MODELS/PAR/']))
MODEL_ED380_PATH = (''.join([str(curr_work_dir),'/MODELS/ED380/']))
MODEL_ED412_PATH = (''.join([str(curr_work_dir),'/MODELS/ED412/']))
MODEL_ED490_PATH = (''.join([str(curr_work_dir),'/MODELS/ED490/']))
#######################################################################################


# 1) SOCA PAR 2025 FUNCTION

# In[13]:


#######################################################################################
# Load the PCAs models (Temp and Sal)
par_pkl_filename_temp_pca = ''.join([str(MODEL_PAR_PATH),str("/temp_pca_51.pkl")])
par_pca_temp = pickle.load(open(par_pkl_filename_temp_pca, "rb"))
par_pkl_filename_sal_pca = ''.join([str(MODEL_PAR_PATH),str("/sal_pca_51.pkl")])
par_pca_sal = pickle.load(open(par_pkl_filename_sal_pca, "rb"))

# Load the x-scaler from the file 
par_pkl_filename_x_scaler = ''.join([str(MODEL_PAR_PATH),str("/x_scaler.pkl")]) 
par_x_scaler = pickle.load(open(par_pkl_filename_x_scaler, 'rb')) 
# Load the x-scaler from the file 
par_pkl_filename_y_scaler = ''.join([str(MODEL_PAR_PATH),str("/y_scaler.pkl")])
par_y_scaler = pickle.load(open(par_pkl_filename_y_scaler, 'rb'))
#######################################################################################


#######################################################################################
# CREATE SCALED INPUTS
def INPUTS_PAR_ADJUSTED_2025(PAR_SAT, MLD, ZPD_MOREL, RRS412, RRS443, RRS490, RRS555, RRS670,
                               sin_doy, cos_doy, solar_elevation_pvlib, temp_cmems, sal_cmems, depths_cmems, depth_output):
                          
    
    INPUTS1 = [PAR_SAT, MLD, ZPD_MOREL, RRS412, RRS443, RRS490, RRS555, RRS670,
              sin_doy, cos_doy, solar_elevation_pvlib]
    
    temp_51 = np.interp(depth_output, depths_cmems, temp_cmems) # made into 51 depths
    temp_51 = temp_51.reshape(-1, 1)
    sal_51 = np.interp(depth_output, depths_cmems, sal_cmems) # made into 51 depths
    sal_51 = sal_51.reshape(-1, 1) 
    
       

    # Apply PCA on temperature, salinity and spiciness profiles to get the principal components
    PrincipalComponentsTemp_Test = par_pca_temp.transform(np.transpose(temp_51))[:,:5]
    principalComponentsSal_Test = par_pca_sal.transform(np.transpose(sal_51))[:,:4]   

         
    #INPUTS = INPUTS1 + np.concatenate(PrincipalComponentsTemp_Test).ravel().tolist() + np.concatenate(principalComponentsSal_Test).ravel().tolist()

    combined_inputs_list = INPUTS1 + \
                           np.concatenate(PrincipalComponentsTemp_Test).ravel().tolist() + \
                           np.concatenate(principalComponentsSal_Test).ravel().tolist()

    INPUTS = pd.DataFrame([combined_inputs_list]) # <--- Fix applied here: wrap combined_inputs_list in a list

    

    input_features = ['PAR_SAT','MLD', 'ZPD_MOREL', 'RRS412','RRS443','RRS490','RRS555', 'RRS670',
                      'sin_doy', 'cos_doy','solar_elevation_pvlib',
#                       'x_cart', 'y_cart', 'z_cart',
                      'TEMP_PC_N_0', 'TEMP_PC_N_1', 'TEMP_PC_N_2', 'TEMP_PC_N_3','TEMP_PC_N_4',
                      'SAL_PC_N_0', 'SAL_PC_N_1', 'SAL_PC_N_2', 'SAL_PC_N_3']

    INPUTS.columns = input_features
    # INPUTS = pd.DataFrame(INPUTS)
    # INPUTS = INPUTS.transpose()    
    # INPUTS.columns = input_features
    INPUTS_scaled = pd.DataFrame(par_x_scaler.transform(INPUTS), columns=INPUTS.columns)
    
    return INPUTS_scaled
#######################################################################################


#######################################################################################
# Define the PyTorch MLP model
class PAR_MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(PAR_MLPModel, self).__init__()
        layers = []
        in_features = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.Tanh())
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
#######################################################################################


#######################################################################################
# Initialize an empty list to store the models (STORE all models in a list)
par_models = []
nc1 = [40,53,25,25,36,41,31,29,27,41]
nc2 = [28,29,12,19,34,27,25,14,17,15]


# Assuming nc1 and nc2 are defined and contain the necessary indices
for x in np.arange(len(nc1)):
    par_model_path = f'{MODEL_PAR_PATH}/PAR_2025_model_n1_{nc1[x]}_n2_{nc2[x]}.pth'
 
    # Initialize the model and load the saved state_dict
    model = PAR_MLPModel(input_size=20, hidden_layers=[nc1[x],nc2[x]], output_size=51)
    model.load_state_dict(torch.load(par_model_path))
    model.eval()  # Set the model to evaluation mode
    
    # Append the loaded model to the list
    par_models.append(model)
#######################################################################################


#######################################################################################
# FUNCTION to derive ED380_ADJUSTED profile
def SOCA_PAR_ADJUSTED_2025(INPUTS_PAR_ADJUSTED_2025,depth_output):
    
    n_sample = INPUTS_PAR_ADJUSTED_2025.shape[0]
    YPRED_TOTAL = pd.DataFrame()    
    nc1 = [40,53,25,25,36,41,31,29,27,41]
    nc2 = [28,29,12,19,34,27,25,14,17,15]

    for model in par_models:
        
        INPUTS_PAR_ADJUSTED_2025_tensor = torch.tensor(INPUTS_PAR_ADJUSTED_2025.values, dtype=torch.float32)
        # Evaluate the model
        with torch.no_grad():
            ypred_scaled = model(INPUTS_PAR_ADJUSTED_2025_tensor).numpy()

        ypred_scaled_eval = ypred_scaled

        # Detransform the outputs
        ypred = pd.DataFrame(
            par_y_scaler.inverse_transform(ypred_scaled_eval),
            columns=np.arange(51),
            index=np.arange(n_sample)
        )
        # Apply transformation

        # make vectors
        PAR_PRED = ypred 
        PAR_PRED_vec = PAR_PRED.values.ravel()  
        PAR_PRED_vec = pow(10,PAR_PRED_vec) 
        YPRED_TOTAL = pd.concat([YPRED_TOTAL, pd.DataFrame(PAR_PRED_vec)], axis=1)

    soca_par=np.mean(YPRED_TOTAL, axis=1)
    soca_par_err=np.std(YPRED_TOTAL, axis=1)    

    return soca_par, soca_par_err
#######################################################################################  


# 2) SOCA ED380 2025 FUNCTION

# In[14]:


#######################################################################################
# Load the PCAs models (Temp aand Sal)
ed380_pkl_filename_temp_pca = ''.join([str(MODEL_ED380_PATH),str("/temp_pca_51.pkl")])
ed380_pca_temp = pickle.load(open(ed380_pkl_filename_temp_pca, "rb"))
ed380_pkl_filename_sal_pca = ''.join([str(MODEL_ED380_PATH),str("/sal_pca_51.pkl")])
ed380_pca_sal = pickle.load(open(ed380_pkl_filename_sal_pca, "rb"))

# Load the x-scaler from the file 
ed380_pkl_filename_x_scaler = ''.join([str(MODEL_ED380_PATH),str("/x_scaler.pkl")]) 
ed380_x_scaler = pickle.load(open(ed380_pkl_filename_x_scaler, 'rb')) 
# Load the x-scaler from the file 
ed380_pkl_filename_y_scaler = ''.join([str(MODEL_ED380_PATH),str("/y_scaler.pkl")])
ed380_y_scaler = pickle.load(open(ed380_pkl_filename_y_scaler, 'rb'))  
#######################################################################################


#######################################################################################
# CREATE SCALED INPUTS
def INPUTS_ED380_ADJUSTED_2025(PAR_SAT, MLD, ZPD_MOREL, RRS412, RRS443, RRS490, RRS555, RRS670,
                               sin_doy, cos_doy, solar_elevation_pvlib, temp_cmems, sal_cmems, depths_cmems, depth_output):
                          
    
    INPUTS1 = [PAR_SAT, MLD, ZPD_MOREL, RRS412, RRS443, RRS490, RRS555, RRS670,
              sin_doy, cos_doy, solar_elevation_pvlib]
    
    temp_51 = np.interp(depth_output, depths_cmems, temp_cmems) # made into 51 depths
    temp_51 = temp_51.reshape(-1, 1)
    sal_51 = np.interp(depth_output, depths_cmems, sal_cmems) # made into 51 depths
    sal_51 = sal_51.reshape(-1, 1) 
    
       

    # Apply PCA on temperature, salinity and spiciness profiles to get the principal components
    PrincipalComponentsTemp_Test = ed380_pca_temp.transform(np.transpose(temp_51))[:,:5]
    principalComponentsSal_Test = ed380_pca_sal.transform(np.transpose(sal_51))[:,:4]   

         
    #INPUTS = INPUTS1 + np.concatenate(PrincipalComponentsTemp_Test).ravel().tolist() + np.concatenate(principalComponentsSal_Test).ravel().tolist()

    combined_inputs_list = INPUTS1 + \
                           np.concatenate(PrincipalComponentsTemp_Test).ravel().tolist() + \
                           np.concatenate(principalComponentsSal_Test).ravel().tolist()

    INPUTS = pd.DataFrame([combined_inputs_list]) # <--- Fix applied here: wrap combined_inputs_list in a list
    

    input_features = ['PAR_SAT','MLD', 'ZPD_MOREL', 'RRS412','RRS443','RRS490','RRS555', 'RRS670',
                      'sin_doy', 'cos_doy','solar_elevation_pvlib',
#                       'x_cart', 'y_cart', 'z_cart',
                      'TEMP_PC_N_0', 'TEMP_PC_N_1', 'TEMP_PC_N_2', 'TEMP_PC_N_3','TEMP_PC_N_4',
                      'SAL_PC_N_0', 'SAL_PC_N_1', 'SAL_PC_N_2', 'SAL_PC_N_3']

    # INPUTS.columns = input_features
    # INPUTS = pd.DataFrame(INPUTS)
    # INPUTS = INPUTS.transpose()    
    INPUTS.columns = input_features
    INPUTS_scaled = pd.DataFrame(ed380_x_scaler.transform(INPUTS), columns=INPUTS.columns)
    
    return INPUTS_scaled
#######################################################################################


#######################################################################################
# Define the PyTorch MLP model
class ED380_MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(ED380_MLPModel, self).__init__()
        layers = []
        in_features = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.Tanh())
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
#######################################################################################


#######################################################################################
# Initialize an empty list to store the models (STORE all models in a list)
ed380_models = []
nc1 = [64,109,106,67,40,74,93,96,67,93]
nc2 = [35,64,52,39,34,42,58,52,56,63]
# Assuming nc1 and nc2 are defined and contain the necessary indices
for x in np.arange(len(nc1)):
    ed380_model_path = f'{MODEL_ED380_PATH}/ED380_2025_model_n1_{nc1[x]}_n2_{nc2[x]}.pth'
 
    # Initialize the model and load the saved state_dict
    model = ED380_MLPModel(input_size=20, hidden_layers=[nc1[x],nc2[x]], output_size=51)
    model.load_state_dict(torch.load(ed380_model_path))
    model.eval()  # Set the model to evaluation mode
    
    # Append the loaded model to the list
    ed380_models.append(model)
#######################################################################################


#######################################################################################
# FUNCTION to derive ED380_ADJUSTED profile
def SOCA_ED380_ADJUSTED_2025(INPUTS_ED380_ADJUSTED_2025,depth_output):
    
    n_sample = INPUTS_ED380_ADJUSTED_2025.shape[0]
    YPRED_TOTAL = pd.DataFrame()    
    nc1 = [64,109,106,67,40,74,93,96,67,93]
    nc2 = [35,64,52,39,34,42,58,52,56,63]

    for model in ed380_models:
        
        INPUTS_ED380_ADJUSTED_2025_tensor = torch.tensor(INPUTS_ED380_ADJUSTED_2025.values, dtype=torch.float32)
        # Evaluate the model
        with torch.no_grad():
            ypred_scaled = model(INPUTS_ED380_ADJUSTED_2025_tensor).numpy()

        ypred_scaled_eval = ypred_scaled

        # Detransform the outputs
        ypred = pd.DataFrame(
            ed380_y_scaler.inverse_transform(ypred_scaled_eval),
            columns=np.arange(51),
            index=np.arange(n_sample)
        )
        # Apply transformation

        # make vectors
        ED380_PRED = ypred 
        ED380_PRED_vec = ED380_PRED.values.ravel()  
        ED380_PRED_vec = pow(10,ED380_PRED_vec) 
        YPRED_TOTAL = pd.concat([YPRED_TOTAL, pd.DataFrame(ED380_PRED_vec)], axis=1)

    soca_ed380=np.mean(YPRED_TOTAL, axis=1)
    soca_ed380_err=np.std(YPRED_TOTAL, axis=1)    

    return soca_ed380, soca_ed380_err
#######################################################################################


# 3) SOCA ED412 2025 FUNCTION

# In[15]:


#######################################################################################
# Load the PCAs models (Temp aand Sal)
ed412_pkl_filename_temp_pca = ''.join([str(MODEL_ED412_PATH),str("/temp_pca_51.pkl")])
ed412_pca_temp = pickle.load(open(ed412_pkl_filename_temp_pca, "rb"))
ed412_pkl_filename_sal_pca = ''.join([str(MODEL_ED412_PATH),str("/sal_pca_51.pkl")])
ed412_pca_sal = pickle.load(open(ed412_pkl_filename_sal_pca, "rb"))

# Load the x-scaler from the file 
ed412_pkl_filename_x_scaler = ''.join([str(MODEL_ED412_PATH),str("/x_scaler.pkl")]) 
ed412_x_scaler = pickle.load(open(ed412_pkl_filename_x_scaler, 'rb')) 
# Load the x-scaler from the file 
ed412_pkl_filename_y_scaler = ''.join([str(MODEL_ED412_PATH),str("/y_scaler.pkl")])
ed412_y_scaler = pickle.load(open(ed412_pkl_filename_y_scaler, 'rb'))
#######################################################################################


#######################################################################################
# CREATE SCALED INPUTS
def INPUTS_ED412_ADJUSTED_2025(PAR_SAT, MLD, ZPD_MOREL, RRS412, RRS443, RRS490, RRS555, RRS670,
                               sin_doy, cos_doy, solar_elevation_pvlib, temp_cmems, sal_cmems, depths_cmems, depth_output):
                          
    
    INPUTS1 = [PAR_SAT, MLD, ZPD_MOREL, RRS412, RRS443, RRS490, RRS555, RRS670,
              sin_doy, cos_doy, solar_elevation_pvlib]
    
    temp_51 = np.interp(depth_output, depths_cmems, temp_cmems) # made into 51 depths
    temp_51 = temp_51.reshape(-1, 1)
    sal_51 = np.interp(depth_output, depths_cmems, sal_cmems) # made into 51 depths
    sal_51 = sal_51.reshape(-1, 1) 
    
       

    # Apply PCA on temperature, salinity and spiciness profiles to get the principal components
    PrincipalComponentsTemp_Test = ed412_pca_temp.transform(np.transpose(temp_51))[:,:5]
    principalComponentsSal_Test = ed412_pca_sal.transform(np.transpose(sal_51))[:,:4]   

         
    #INPUTS = INPUTS1 + np.concatenate(PrincipalComponentsTemp_Test).ravel().tolist() + np.concatenate(principalComponentsSal_Test).ravel().tolist()

    combined_inputs_list = INPUTS1 + \
                           np.concatenate(PrincipalComponentsTemp_Test).ravel().tolist() + \
                           np.concatenate(principalComponentsSal_Test).ravel().tolist()

    INPUTS = pd.DataFrame([combined_inputs_list]) # <--- Fix applied here: wrap combined_inputs_list in a list
    

    input_features = ['PAR_SAT','MLD', 'ZPD_MOREL', 'RRS412','RRS443','RRS490','RRS555', 'RRS670',
                      'sin_doy', 'cos_doy','solar_elevation_pvlib',
#                       'x_cart', 'y_cart', 'z_cart',
                      'TEMP_PC_N_0', 'TEMP_PC_N_1', 'TEMP_PC_N_2', 'TEMP_PC_N_3','TEMP_PC_N_4',
                      'SAL_PC_N_0', 'SAL_PC_N_1', 'SAL_PC_N_2', 'SAL_PC_N_3']

    # INPUTS = pd.DataFrame(INPUTS)
    # INPUTS = INPUTS.transpose()    
    INPUTS.columns = input_features
    INPUTS_scaled = pd.DataFrame(ed412_x_scaler.transform(INPUTS), columns=INPUTS.columns)
    
    return INPUTS_scaled
#######################################################################################


#######################################################################################
# Define the PyTorch MLP model
class ED412_MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(ED412_MLPModel, self).__init__()
        layers = []
        in_features = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.Tanh())
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
#######################################################################################


#######################################################################################
# Initialize an empty list to store the models (STORE all models in a list)
ed412_models = []
nc1 = [62,66,137,71,47,110,87,79,104,81]
nc2 = [34,16,20,37,40,50,21,45,39,41]
# Assuming nc1 and nc2 are defined and contain the necessary indices
for x in np.arange(len(nc1)):
    ed412_model_path = f'{MODEL_ED412_PATH}/ED412_2025_model_n1_{nc1[x]}_n2_{nc2[x]}.pth'
 
    # Initialize the model and load the saved state_dict
    model = ED412_MLPModel(input_size=20, hidden_layers=[nc1[x],nc2[x]], output_size=51)
    model.load_state_dict(torch.load(ed412_model_path))
    model.eval()  # Set the model to evaluation mode
    
    # Append the loaded model to the list
    ed412_models.append(model)
#######################################################################################


#######################################################################################
# FUNCTION to derive ED412_ADJUSTED profile
def SOCA_ED412_ADJUSTED_2025(INPUTS_ED412_ADJUSTED_2025,depth_output):
    
    n_sample = INPUTS_ED412_ADJUSTED_2025.shape[0]
    YPRED_TOTAL = pd.DataFrame()    
    nc1 = [62,66,137,71,47,110,87,79,104,81]
    nc2 = [34,16,20,37,40,50,21,45,39,41]

    for model in ed412_models:
        
        INPUTS_ED412_ADJUSTED_2025_tensor = torch.tensor(INPUTS_ED412_ADJUSTED_2025.values, dtype=torch.float32)
        # Evaluate the model
        with torch.no_grad():
            ypred_scaled = model(INPUTS_ED412_ADJUSTED_2025_tensor).numpy()

        ypred_scaled_eval = ypred_scaled

        # Detransform the outputs
        ypred = pd.DataFrame(
            ed412_y_scaler.inverse_transform(ypred_scaled_eval),
            columns=np.arange(51),
            index=np.arange(n_sample)
        )
        # Apply transformation

        # make vectors
        ED412_PRED = ypred 
        ED412_PRED_vec = ED412_PRED.values.ravel()  
        ED412_PRED_vec = pow(10,ED412_PRED_vec) 
        YPRED_TOTAL = pd.concat([YPRED_TOTAL, pd.DataFrame(ED412_PRED_vec)], axis=1)

    soca_ed412=np.mean(YPRED_TOTAL, axis=1)
    soca_ed412_err=np.std(YPRED_TOTAL, axis=1)    

    return soca_ed412, soca_ed412_err
#######################################################################################


# 4) SOCA ED490 2025 FUNCTION

# In[16]:


#######################################################################################
# Load the PCAs models (Temp aand Sal)
ed490_pkl_filename_temp_pca = ''.join([str(MODEL_ED490_PATH),str("/temp_pca_51.pkl")])
ed490_pca_temp = pickle.load(open(ed490_pkl_filename_temp_pca, "rb"))
ed490_pkl_filename_sal_pca = ''.join([str(MODEL_ED490_PATH),str("/sal_pca_51.pkl")])
ed490_pca_sal = pickle.load(open(ed490_pkl_filename_sal_pca, "rb"))

# Load the x-scaler from the file 
ed490_pkl_filename_x_scaler = ''.join([str(MODEL_ED490_PATH),str("/x_scaler.pkl")]) 
ed490_x_scaler = pickle.load(open(ed490_pkl_filename_x_scaler, 'rb')) 
# Load the x-scaler from the file 
ed490_pkl_filename_y_scaler = ''.join([str(MODEL_ED490_PATH),str("/y_scaler.pkl")])
ed490_y_scaler = pickle.load(open(ed490_pkl_filename_y_scaler, 'rb')) 
#######################################################################################


#######################################################################################
# CREATE SCALED INPUTS
def INPUTS_ED490_ADJUSTED_2025(PAR_SAT, MLD, ZPD_MOREL, RRS412, RRS443, RRS490, RRS555, RRS670,
                               sin_doy, cos_doy, solar_elevation_pvlib, temp_cmems, sal_cmems, depths_cmems, depth_output):
                          
    
    INPUTS1 = [PAR_SAT, MLD, ZPD_MOREL, RRS412, RRS443, RRS490, RRS555, RRS670,
              sin_doy, cos_doy, solar_elevation_pvlib]
    
    temp_51 = np.interp(depth_output, depths_cmems, temp_cmems) # made into 51 depths
    temp_51 = temp_51.reshape(-1, 1)
    sal_51 = np.interp(depth_output, depths_cmems, sal_cmems) # made into 51 depths
    sal_51 = sal_51.reshape(-1, 1) 
    
       

    # Apply PCA on temperature, salinity and spiciness profiles to get the principal components
    PrincipalComponentsTemp_Test = ed490_pca_temp.transform(np.transpose(temp_51))[:,:5]
    principalComponentsSal_Test = ed490_pca_sal.transform(np.transpose(sal_51))[:,:4]   

         
    #INPUTS = INPUTS1 + np.concatenate(PrincipalComponentsTemp_Test).ravel().tolist() + np.concatenate(principalComponentsSal_Test).ravel().tolist()
    combined_inputs_list = INPUTS1 + \
                           np.concatenate(PrincipalComponentsTemp_Test).ravel().tolist() + \
                           np.concatenate(principalComponentsSal_Test).ravel().tolist()

    INPUTS = pd.DataFrame([combined_inputs_list]) # <--- Fix applied here: wrap combined_inputs_list in a list
    

    input_features = ['PAR_SAT','MLD', 'ZPD_MOREL', 'RRS412','RRS443','RRS490','RRS555', 'RRS670',
                      'sin_doy', 'cos_doy','solar_elevation_pvlib',
#                       'x_cart', 'y_cart', 'z_cart',
                      'TEMP_PC_N_0', 'TEMP_PC_N_1', 'TEMP_PC_N_2', 'TEMP_PC_N_3','TEMP_PC_N_4',
                      'SAL_PC_N_0', 'SAL_PC_N_1', 'SAL_PC_N_2', 'SAL_PC_N_3']

    # INPUTS = pd.DataFrame(INPUTS)
    # INPUTS = INPUTS.transpose()    
    INPUTS.columns = input_features
    INPUTS_scaled = pd.DataFrame(ed490_x_scaler.transform(INPUTS), columns=INPUTS.columns)
    
    return INPUTS_scaled
#######################################################################################


#######################################################################################
# Define the PyTorch MLP model
class ED490_MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(ED490_MLPModel, self).__init__()
        layers = []
        in_features = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.Tanh())
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
#######################################################################################


#######################################################################################
# Initialize an empty list to store the models (STORE all models in a list)
ed490_models = []
nc1 = [20,22,25,36,23,18,23,46,19,55]
nc2 = [18,9,23,21,15,14,17,27,11,48]

# Assuming nc1 and nc2 are defined and contain the necessary indices
for x in np.arange(len(nc1)):
    ed490_model_path = f'{MODEL_ED490_PATH}/ED490_2025_model_n1_{nc1[x]}_n2_{nc2[x]}.pth'
 
    # Initialize the model and load the saved state_dict
    model = ED490_MLPModel(input_size=20, hidden_layers=[nc1[x],nc2[x]], output_size=51)
    model.load_state_dict(torch.load(ed490_model_path))
    model.eval()  # Set the model to evaluation mode
    
    # Append the loaded model to the list
    ed490_models.append(model)
#######################################################################################


#######################################################################################
# FUNCTION to derive ED490_ADJUSTED profile
def SOCA_ED490_ADJUSTED_2025(INPUTS_ED490_ADJUSTED_2025,depth_output):
    
    n_sample = INPUTS_ED490_ADJUSTED_2025.shape[0]
    YPRED_TOTAL = pd.DataFrame()    
    nc1 = [20,22,25,36,23,18,23,46,19,55]
    nc2 = [18,9,23,21,15,14,17,27,11,48]

    for model in ed490_models:
        
        INPUTS_ED490_ADJUSTED_2025_tensor = torch.tensor(INPUTS_ED490_ADJUSTED_2025.values, dtype=torch.float32)
        # Evaluate the model
        with torch.no_grad():
            ypred_scaled = model(INPUTS_ED490_ADJUSTED_2025_tensor).numpy()

        ypred_scaled_eval = ypred_scaled

        # Detransform the outputs
        ypred = pd.DataFrame(
            ed490_y_scaler.inverse_transform(ypred_scaled_eval),
            columns=np.arange(51),
            index=np.arange(n_sample)
        )
        # Apply transformation

        # make vectors
        ED490_PRED = ypred 
        ED490_PRED_vec = ED490_PRED.values.ravel()  
        ED490_PRED_vec = pow(10,ED490_PRED_vec) 
        YPRED_TOTAL = pd.concat([YPRED_TOTAL, pd.DataFrame(ED490_PRED_vec)], axis=1)

    soca_ed490=np.mean(YPRED_TOTAL, axis=1)
    soca_ed490_err=np.std(YPRED_TOTAL, axis=1)    

    return soca_ed490, soca_ed490_err
####################################################################################### 







