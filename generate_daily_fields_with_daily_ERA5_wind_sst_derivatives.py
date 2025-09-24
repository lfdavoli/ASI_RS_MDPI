# %% [markdown]
# # Gathering data for SST-wind coupling analysis (DMM and PA) depending on environmental conditions from ERA5 daily data.
# ## Description
# 
# ## To do's
# * 
# ## Details
# 
# @author: Lorenzo Francesco Davoli
# 
# @creation: 11/07/2024
# 
# @project: Phase-SST-wind-coupling
# 
# @github: https://github.com/lfdavoli/ASI_RS_MDPI
# 
# @contact: l.davoli@campus.unimib.it
# 
# @notes: adaptation of the code from _glauco_ project for the paper Meroni et al. (2023)

# %% [markdown]
# # Packages, functions and controls

# %% [markdown]
# ## Packages

# %%
import sys
import os
from pathlib import Path
# My utility funtions from "miscellanea" directory
from my_utility_functions import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
# from matplotlib.pyplot import cm
import numpy as np
import xarray as xr
import pandas as pd
import glob
import geometry as gm # grad_sphere, div_sphere, nan_gaussian_filter, L2wind_2_regular_grid_mask
import geopy.distance
import seaborn as sns
#import copy
import time as tm
# Trace memory usage
import tracemalloc

import warnings
warnings.filterwarnings("ignore")


# %% [markdown]
# ## Controls

# %%
# This is to compute systematically the statistics for the DMM over different
# areas of the globe and over single seasons. It can also be used for the PA.
# The standard deviation of the gaussian filter used to determine the
# background wind is sigma. The background wind is saved in the output files.

# Set the geographical parameters for the analysis and the figures.
# Look at the areas of the WBCs defined by O'Neill et al, J. Cli (2012).
# The western Pacific area is defined following Li and Carbone, JAS (2012).
# The eastern Pacific cold tongue are follows Chelton et al., J. Cli (2001).
area_configs = [
    #{'area_str': 'extended_eureca', 'area_name': 'extended EUREC4A', 'minlon': -62., 'maxlon': -40., 'minlat': 0, 'maxlat': 20.},
    #{'area_str': 'gulf_stream', 'area_name': 'Gulf Stream', 'minlon': -83., 'maxlon': -30., 'minlat': 30., 'maxlat': 55.},
    #{'area_str': 'malvinas', 'area_name': 'Malvinas current', 'minlon': -70., 'maxlon': 0., 'minlat': -60., 'maxlat': -30.},
    #{'area_str': 'agulhas', 'area_name': 'Agulhas current', 'minlon': 0., 'maxlon': 100., 'minlat': -60., 'maxlat': -30.},
    #{'area_str': 'kuroshio', 'area_name': 'Kuroshio current', 'minlon': 140., 'maxlon': 180., 'minlat': 30., 'maxlat': 50.},
    #{'area_str': 'southern_indian', 'area_name': 'Southern Indian', 'minlon': 55., 'maxlon': 100., 'minlat': -60., 'maxlat': -30.},
    #{'area_str': 'south_east_atlantic', 'area_name': 'South-east Atlantic', 'minlon': -10., 'maxlon': 15., 'minlat': -25., 'maxlat': -5.},
    #{'area_str': 'artic', 'area_name': 'Artic', 'minlon': -12., 'maxlon': 10., 'minlat': 67., 'maxlat': 75.50},
    {'area_str': 'global', 'area_name': 'Global', 'minlon': -180., 'maxlon': 180., 'minlat': -60., 'maxlat': 60.},
]

# Geographical areas to remove, in the format
# [[minlon,maxlon],[minlat,maxlat]]
geo_masks = {
    'gulf_stream' : [
        [[-83,-50],[47,55]],
        ], # North land and lakes
    'malvinas' : [
        ], 
    'extended_eureca' : [
        ], 
    'agulhas' : [
        ], 
    'kuroshio' : [
        ],
    'southern_indian' : [
        ],
    'south_east_atlantic' : [
        ],
    'artic' : [
        ],
    'global' : [
        ],
}


# Select here the fields to be analysed.
str_mech = 'DMM'
#str_mech = 'PA'
if str_mech == 'DMM':
    sst_deriv = 'dsst_dr' # Choose between: 'dsst_dr', 'lapl_sst', 'd2sst_ds2', 'sst_prime'
    wind_deriv = 'dr_dot_prime_dr' # Choose between: 'wind_div', 'dr_dot_prime_dr', 'ds_dot_prime_ds', 'ws_prime'
elif str_mech == 'PA':
    sst_deriv = 'd2sst_ds2'
    wind_deriv = 'ds_dot_prime_ds'
else: 
    raise NameError('Mechanism not recognised')

# Select the standard deviation of the Gaussian filter used to determine the background wind field.
# The unit of measure is the sst-product gridstep (for MetOp A AVHRR it's 5km, ERA5 25km).
# In Desbiolles et al. 2023 Lanczos filter with 10° ~ gaussian with 4.37° ~ 450km (see email ago 18/12/2024)
sigma = 18 # 18 gridsteps = 18*25km = 450 km
#sigma = 9 # 9 gridsteps = 9*25km = 225 km

# Define the extremes of the period to analyze
period_start = '2000-01-01'
#period_end = '2015-12-31'
period_end = '2021-12-31'
#period_end = '2004-09-30'

# We split the whole period into sub periods to reduce the memory usage. 
# timestep represents the length of the period to be considered at the time.
timestep = pd.DateOffset(months=1) 

# ERA5 variables to save
# For ERA5 data with surface and press level info
'''
vars_to_save = [
    'u10','v10','d2m','t2m','sst','sp','slhf','sshf','zust',
    'smoothed_air_sea_temp_diff','hcc','tp','cbh',
    'cp','lsp','tcc','lcc','skt','msl',
    'd_850','r_850','q_850','t_850',
    'u_850','v_850','w_850','z_850',
    'd_900','r_900','q_900','t_900',
    'u_900','v_900','w_900','z_900',
    'd_1000','r_1000','q_1000','t_1000',
    'u_1000','v_1000','w_1000','z_1000',
    ]
'''
# For basic ERA5 with only surface
vars_to_save = [
    'u10','v10','d2m','t2m','sst','sp','slhf','sshf','zust',
    ]

# Set some relevant paths.
# ERA5 data with surface and press level info
#era5_dataset_name = 'surf_u10_v10_d2m_t2m_sst_sp_slhf_sshf_zust_hcc_tp_cbh_cp_lsp_tcc_lcc_skt_msl_1000hPA_900hPA_850hPA_d_r_q_t_u_v_w_z'
# Basic ERA5 with only surface
era5_dataset_name = 'u10_v10_d2m_t2m_sst_sp_slhf_sshf_zust'
path2era5 = ''

for area_config in area_configs:
    area_str = area_config['area_str']
    area_name = area_config['area_name']
    minlon = area_config['minlon']
    maxlon = area_config['maxlon']
    minlat = area_config['minlat']
    maxlat = area_config['maxlat']

    print(f"### {area_name} ###")

    # Output folder
    path2output = ''
    Path(path2output).mkdir(parents=True, exist_ok=True)

    # Set some parameters for the maps.
    region_extent = [minlon, maxlon, minlat, maxlat]
    crs = ccrs.PlateCarree()

    # Enable saving outputs
    enable_save_files = True 
    merge_files = False

    # %% [markdown]
    # ## Functions

    # %%
    def pre_process_variable(database,var):
        '''
            Apply an hardcoded preprocessing to the selected database and variable, then return the updated database.
        '''
        if var == 'datetime_imagette':
            database[var] = pd.to_datetime(database[var],utc=True)
            database = database.assign(time=database[var])
            database.drop([var],axis=1,inplace=True)
        elif var == 'lon_sar':
            # Rename lon_sar and lat_sar
            database = database.assign(lon=database.lon_sar.values)
            database.drop(['lon_sar'],axis=1,inplace=True)
        elif var == 'lat_sar':
            # Rename lon_sar and lat_sar
            database = database.assign(lat=database.lat_sar.values)
            database.drop(['lat_sar'],axis=1,inplace=True)
        elif var == 'y_pred_aoi':
            raise NameError('Take care of y_preo_aoi earlier.')
        elif  var == 'class_1':  
            # Rename class_1 as Wang_class
            database = database.assign(Wang_class=database.class_1.values)
            database.drop(['class_1'],axis=1,inplace=True)
        elif var == 'tsea_era5':
            # Rename tsea_era5 as sst_era5
            database = database.assign(sst_era5=database.tsea_era5.values)
            database.drop(['tsea_era5'],axis=1,inplace=True)
        elif var == 'tsea_era5':
            # Rename wspd_era5
            database = database.assign(ws_era5=database.wspd_era5.values)
            database.drop(['wspd_era5'],axis=1,inplace=True)
        else: 
            print(f'WARNING: no pre-processing found for {var}')

        return database



    def split_time_interval(period_start : str, period_end : str, timestep : pd.DateOffset):
        """ 
            Split the time interval into subintervals of size timestep.
            timestep must be larger than 1h.
        """
        if timestep == pd.DateOffset(hours=1): 
            raise NameError('Timestep is too short')
        # Convert the input strings to pandas Timestamps
        start_date = pd.to_datetime(period_start)
        end_date = pd.to_datetime(period_end)
        
        # Initialize lists to store the interval starting and ending points
        period_interval_starting_points = []
        period_interval_ending_points = []
        
        # Create intervals of `timestep` length from `start_date` to `end_date`
        current_start = start_date
        while current_start < end_date:
            current_end = (current_start + timestep)
            
            # Ensure that the interval end does not go beyond `end_date`
            if current_end > end_date:
                current_end = end_date+pd.DateOffset(1,'days')
                
            # Append the current interval to the lists
            period_interval_starting_points.append(current_start)
            period_interval_ending_points.append(current_end)
            
            # Move to the next interval
            current_start = current_start + timestep
        
        return period_interval_starting_points, period_interval_ending_points



    def create_nan_dummy_row(df_row):
        """
        Create a new row with the same columns as the input row, but with all values as NaN
        """
        nan_dummy = pd.DataFrame(np.nan, index=[0], columns=df_row.index)
        return nan_dummy



    def stack_txt_files(filenames, output_file):
        """
        Takes a list of txt files and merges them by stacking them vertically.
        Only the header of the first file is included in the output file.
        """
        
        header_written = False
        
        with open(output_file, 'w') as outfile:
            for txt_file in filenames:
                with open(txt_file, 'r') as infile:
                    lines = infile.readlines()
                    
                    # Write the header only if it hasn't been written yet
                    if not header_written:
                        outfile.write(lines[0])  # Write the header line
                        header_written = True
                    
                    # Write the rest of the lines (skip header)
                    outfile.writelines(lines[1:])
                    outfile.write("\n")  # Optional: Add newline between files


    def compute_two_fields(str_a,str_b,sigma,llon,llat,l3_sst,u_interp,v_interp):
        """
        Compute the fields defined by str_a and str_b using the l3_sst, u_interp and v_interp variables,
        which are all defined on the same grid (llon,llat).
        Note that the fields are treated as lists, because the append function
        is much faster with respect to the numpy.append function. The lists are converted to numpy arrays at the end.
        """
        # Get the background wind field.
        smooth_u = gm.nan_gaussian_filter(u_interp,sigma)
        smooth_v = gm.nan_gaussian_filter(v_interp,sigma)
        smooth_ws = np.sqrt(smooth_u**2+smooth_v**2)

        cosphi = smooth_u/smooth_ws
        sinphi = smooth_v/smooth_ws

        # Get the anomalies with respect to the background wind field.
        u_prime = u_interp-smooth_u
        v_prime = v_interp-smooth_v

        dsst_dx, dsst_dy = gm.grad_sphere(l3_sst,llon,llat)
        if str_a=='gamma':
            a_prime = u_interp*dsst_dx + v_interp*dsst_dy
        elif str_a=='dsst_dr':
            a_prime = dsst_dx*cosphi + dsst_dy*sinphi
        elif str_a=='lapl_sst':
            a_prime = gm.div_sphere(dsst_dx,dsst_dy,llon,llat)
        elif str_a=='d2sst_ds2':
            dsst_ds = -dsst_dx*sinphi + dsst_dy*cosphi
            ddsst_ds_dx, ddsst_ds_dy = gm.grad_sphere(dsst_ds,llon,llat)
            a_prime = -ddsst_ds_dx*sinphi + ddsst_ds_dy*cosphi
        elif str_a=='sst_prime':
            smooth_sst = gm.nan_gaussian_filter(l3_sst,sigma)
            a_prime = l3_sst-smooth_sst
            
        if str_b=='wind_div':
            b_prime = gm.div_sphere(u_interp,v_interp,llon,llat)
        elif str_b=='dr_dot_prime_dr':
            r_dot_prime = u_prime*cosphi + v_prime*sinphi
            dr_dot_prime_dx, dr_dot_prime_dy = gm.grad_sphere(r_dot_prime,llon,llat)
            b_prime = dr_dot_prime_dx*cosphi + dr_dot_prime_dy*sinphi 
        elif str_b=='ds_dot_prime_ds':
            s_dot_prime = -u_prime*sinphi + v_prime*cosphi
            ds_dot_prime_dx, ds_dot_prime_dy = gm.grad_sphere(s_dot_prime,llon,llat)
            b_prime = -ds_dot_prime_dx*sinphi + ds_dot_prime_dy*cosphi
        elif str_b=='ws_prime':
            b_prime = np.sqrt(u_interp**2+v_interp**2)-smooth_ws


        # Remove the NaNs, from the variables to be concatenated (with no subsampling).
        #a_to_be_concat = a_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
        #b_to_be_concat = b_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]    
        #U_to_be_concat = smooth_ws[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]    

        return a_prime, b_prime, smooth_ws #a_to_be_concat, b_to_be_concat, U_to_be_concat



    def lonlat_to_indices(lat, lon, llat, llon):
        '''
        Transform lon e lat in the equivalent indices in llon and llat meshgrid
        '''
        latitudes = llat[:, 0]
        longitudes = llon[0, :]
        
        i_lat = np.abs(latitudes - lat).argmin()
        j_lon = np.abs(longitudes - lon).argmin()
        
        return i_lat, j_lon



    def transform_array(entries, llat, llon):
        '''
        Transform the entries = (lon,lat,log_L,time) in (i,j,log_L,time) with i_lat,j_lon indices of llat/llon meshgrid
        '''
        transformed_entries = []
        # Structure of the entries: lat,lon,log_L,time
        for entry in entries:
            i_lat, j_lon = lonlat_to_indices(entry[0], entry[1], llat, llon)
            # Compose new entry
            transformed_entry = [i_lat,j_lon]
            # Add the remaining elements (log_L and time)
            for entry_elem in entry[2:]:
                transformed_entry.append(entry_elem)
            transformed_entries.append(list(transformed_entry))
        return transformed_entries

    # %% [markdown]
    # # ERA5 wind and sst analysis period by period

    # %%
    # Set the periods to analyse
    period_interval_starting_points, period_interval_ending_points = split_time_interval(period_start, period_end, timestep)

    # Save the list of files that are produced to merge them in the end
    generated_filenames = [] 

    # Loop over different periods, needed to reduce memory usage
    for tt,_ in enumerate(period_interval_starting_points):
        str_start = period_interval_starting_points[tt]
        str_end = period_interval_ending_points[tt]
        instant_start = pd.to_datetime(str_start)
        instant_end = pd.to_datetime(str_end)

        a = []
        b = []
        U = []
        llon_container = []
        llat_container = []
        era5_dict = dict()
        for var in vars_to_save:
            era5_dict[var] = []

        # Loop over all the days of the time interval
        instant = instant_start
        while instant<instant_end:
            print(instant)
            if era5_dataset_name == 'u10_v10_d2m_t2m_sst_sp_slhf_sshf_zust':
                era5_filename = os.path.join(
                        path2era5,
                        instant.strftime("%Y"),
                        instant.strftime("%m"),
                        f'era5_{era5_dataset_name}_{instant.strftime("%Y%m%d")}.nc'
                        )
            else:
                era5_filename = os.path.join(
                        path2era5,
                        instant.strftime("%Y"),
                        instant.strftime("%m"),
                        f'{era5_dataset_name}_{instant.strftime("%Y%m%d")}.nc'
                        )

            era5_day_ds = xr.open_dataset(era5_filename).isel(valid_time = 0)

            # Shift longitude from [0, 360] to [-180, 180]
            era5_day_ds = era5_day_ds.assign_coords(
                longitude=((era5_day_ds.longitude + 180) % 360) - 180
            )
            # Sort the dataset by the new longitude values
            era5_day_ds = era5_day_ds.sortby("longitude")
            # Spatial filter
            era5_day_ds = era5_day_ds.sel(
                latitude=slice(maxlat, minlat),  # Note: latitude usually decreases
                longitude=slice(minlon, maxlon),
                )

            # Filter out some geographical areas.
            if len(geo_masks[area_str]) != 0:
                for lon_bnds,lat_bnds in geo_masks[area_str]:
                    era5_day_ds = era5_day_ds.where((era5_day_ds['longitude']<lon_bnds[0]) | (era5_day_ds['longitude']>lon_bnds[1]) | (era5_day_ds['latitude']<lat_bnds[0]) | (era5_day_ds['latitude']>lat_bnds[1]))
            
            # Filter out land
            era5_day_ds = era5_day_ds.where(era5_day_ds['sst'].notnull())

            # Filter out sea/lake ice (sst < 275 K)
            era5_day_ds = era5_day_ds.where(era5_day_ds['sst']>275)
                    
            # Get the background wind field.
            # Here sigma is given in gridsteps of the sst product
            # For AVHRR MetOp A 1 step == 5km
            smooth_u = gm.nan_gaussian_filter(era5_day_ds['u10'],sigma)
            smooth_v = gm.nan_gaussian_filter(era5_day_ds['v10'],sigma)
            smooth_sst = gm.nan_gaussian_filter(era5_day_ds['sst'],sigma)
            smooth_t2m = gm.nan_gaussian_filter(era5_day_ds['t2m'],sigma)
            smooth_air_sea_temp_diff = smooth_t2m - smooth_sst
            smooth_ws = np.sqrt(smooth_u**2+smooth_v**2)

            era5_day_ds['smoothed_u10'] = (['latitude', 'longitude'], smooth_u)
            era5_day_ds['smoothed_u10'].attrs['units'] = 'm s**-1'
            era5_day_ds['smoothed_u10'].attrs['long_name'] = f'smoothed 10 metre U wind component'

            era5_day_ds['smoothed_v10'] = (['latitude', 'longitude'], smooth_v)
            era5_day_ds['smoothed_v10'].attrs['units'] = 'm s**-1'
            era5_day_ds['smoothed_v10'].attrs['long_name'] = f'smoothed 10 metre V wind component'

            era5_day_ds['smoothed_ws10'] = (['latitude', 'longitude'], smooth_ws)
            era5_day_ds['smoothed_ws10'].attrs['units'] = 'm s**-1'
            era5_day_ds['smoothed_ws10'].attrs['long_name'] = f'smoothed 10 metre wind speed'
            
            era5_day_ds['smoothed_air_sea_temp_diff'] = (['latitude', 'longitude'], smooth_air_sea_temp_diff)
            era5_day_ds['smoothed_air_sea_temp_diff'].attrs['units'] = 'K'
            era5_day_ds['smoothed_air_sea_temp_diff'].attrs['long_name'] = f'smoothed air-sea temperature difference'

            # Get the anomalies
            era5_day_ds['u10_prime'] = era5_day_ds['u10'] - era5_day_ds['smoothed_u10']
            era5_day_ds['u10_prime'].attrs['units'] = 'm s**-1'
            era5_day_ds['u10_prime'].attrs['long_name'] = f'anomaly 10 metre U wind component'

            era5_day_ds['v10_prime'] = era5_day_ds['v10'] - era5_day_ds['smoothed_v10']
            era5_day_ds['v10_prime'].attrs['units'] = 'm s**-1'
            era5_day_ds['v10_prime'].attrs['long_name'] = f'anomaly 10 metre V wind component'

            # Create a lon-lat meshgrid
            llon, llat = np.meshgrid(era5_day_ds['longitude'],era5_day_ds['latitude'])

            # SST gradient field
            dsst_dx, dsst_dy = gm.grad_sphere(era5_day_ds['sst'],llon,llat)

            era5_day_ds['dsst_dx'] = (['latitude', 'longitude'], dsst_dx)
            era5_day_ds['dsst_dx'].attrs['units'] = 'K m**-1'
            era5_day_ds['dsst_dx'].attrs['long_name'] = r'$\frac{dSST}{dx}$'

            era5_day_ds['dsst_dy'] = (['latitude', 'longitude'], dsst_dx)
            era5_day_ds['dsst_dy'].attrs['units'] = 'K m**-1'
            era5_day_ds['dsst_dy'].attrs['long_name'] = r'$\frac{dSST}{dy}$'

            # Background wind direction cos and sin
            era5_day_ds['cosphi'] = era5_day_ds['smoothed_u10']/era5_day_ds['smoothed_ws10']
            era5_day_ds['sinphi'] = era5_day_ds['smoothed_v10']/era5_day_ds['smoothed_ws10']

            # Apply metrics to compute control and response fields
            if sst_deriv=='gamma':
                raise NameError('Not implemented')
                a_prime = u_interp*dsst_dx + v_interp*dsst_dy
                x_string = 'u.grad(SST) [K/s]'; vmin_a=-2.2e-4; vmax_a=2.2e-4
            elif sst_deriv=='dsst_dr':
                era5_day_ds['a_prime'] = era5_day_ds['dsst_dx']*era5_day_ds['cosphi'] + era5_day_ds['dsst_dy']*era5_day_ds['sinphi']
                era5_day_ds['a_prime'].attrs['units'] = 'K m**-1'
                era5_day_ds['a_prime'].attrs['long_name'] = r'$\frac{dSST}{dr}$'
                vmin_a=-2.2e-5; vmax_a=2.2e-5
            elif sst_deriv=='lapl_sst':
                raise NameError('Not implemented')
                a_prime = gm.div_sphere(dsst_dx,dsst_dy,llon,llat)
                x_string = 'lapl SST [K/m^2]'; vmin_a=-1e-9; vmax_a=1e-9
            elif sst_deriv=='d2sst_ds2':
                dsst_ds = -era5_day_ds['dsst_dx']*era5_day_ds['sinphi'] + era5_day_ds['dsst_dy']*era5_day_ds['cosphi']
                ddsst_ds_dx, ddsst_ds_dy = gm.grad_sphere(dsst_ds,llon,llat)
                era5_day_ds['a_prime'] = (['latitude', 'longitude'], -ddsst_ds_dx*era5_day_ds['sinphi'] + ddsst_ds_dy*era5_day_ds['cosphi'])
                era5_day_ds['a_prime'].attrs['units'] = 'K m**-2'
                era5_day_ds['a_prime'].attrs['long_name'] = r'$\frac{d2SST}{ds2}$'
                vmin_a=-1e-9; vmax_a=1e-9
            elif sst_deriv=='sst_prime':
                raise NameError('Not implemented')
                smooth_sst = gm.nan_gaussian_filter(l3c_sst,sigma)
                a_prime = l3c_sst-smooth_sst
                x_string = 'SST_prime [K]'; vmin_a=-5; vmax_a=5

            if wind_deriv=='wind_div':
                raise NameError('Not implemented')
                b_prime = gm.div_sphere(u_interp,v_interp,llon,llat)
                y_string = 'Wind divergence [1/s]'; vmin_b=-2.2e-4; vmax_b=2.2e-4
            elif wind_deriv=='dr_dot_prime_dr':
                r_dot_prime = era5_day_ds['u10_prime']*era5_day_ds['cosphi'] + era5_day_ds['v10_prime']*era5_day_ds['sinphi']
                dr_dot_prime_dx, dr_dot_prime_dy = gm.grad_sphere(r_dot_prime,llon,llat)
                era5_day_ds['b_prime'] = (['latitude', 'longitude'], dr_dot_prime_dx*era5_day_ds['cosphi'].data + dr_dot_prime_dy*era5_day_ds['sinphi'].data) 
                era5_day_ds['b_prime'].attrs['units'] = 's**-1'
                era5_day_ds['b_prime'].attrs['long_name'] = r"$\frac{d\dot{r}'}{dr}$"
                vmin_b=-2.2e-4; vmax_b=2.2e-4
            elif wind_deriv=='ds_dot_prime_ds':
                s_dot_prime = -era5_day_ds['u10_prime'].data*era5_day_ds['sinphi'].data + era5_day_ds['v10_prime'].data*era5_day_ds['cosphi'].data
                ds_dot_prime_dx, ds_dot_prime_dy = gm.grad_sphere(s_dot_prime,llon,llat)
                era5_day_ds['b_prime'] = (['latitude', 'longitude'], -ds_dot_prime_dx*era5_day_ds['sinphi'].data + ds_dot_prime_dy*era5_day_ds['cosphi'].data) 
                era5_day_ds['b_prime'].attrs['units'] = 's**-1'
                era5_day_ds['b_prime'].attrs['long_name'] = r"$\frac{d\dot{s}'}{ds}$"
                vmin_b=-2.2e-4; vmax_b=2.2e-4
            elif wind_deriv=='ws_prime':
                raise NameError('Not implemented')
                b_prime = np.sqrt(u_interp**2+v_interp**2)-smooth_ws
                y_string = 'ws_prime [m/s]'; vmin_b=-5; vmax_b=5;
            
            
            controlname = sst_deriv
            varname = wind_deriv
        
            # Drop nans
            a_prime = era5_day_ds['a_prime'].data
            b_prime = era5_day_ds['b_prime'].data
            smooth_ws = era5_day_ds['smoothed_ws10'].data
                    
            
            
            # Concatenate the variables (with no subsampling) removing the NaNs.
            a_to_be_concat = a_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
            b_to_be_concat = b_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]    
            U_to_be_concat = smooth_ws[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
            llon_to_be_concat = llon[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
            llat_to_be_concat = llat[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]

            a.extend(a_to_be_concat)
            b.extend(b_to_be_concat)
            U.extend(U_to_be_concat)
            llon_container.extend(llon_to_be_concat)
            llat_container.extend(llat_to_be_concat)
            
            for var in era5_dict.keys():
                era5_dict[var].extend(era5_day_ds[var].data[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))])
            
            # Go to the next hour
            instant += pd.Timedelta(1,'days')
        
        ### Save variables as text files.
        a = np.array(a)
        b = np.array(b)
        U = np.array(U)
        llon_container = np.array(llon_container)
        llat_container = np.array(llat_container)
        for var in era5_dict.keys():
            era5_dict[var] = np.array(era5_dict[var])

        # Take care of inf
        a[np.isinf(a)] = np.nan
        b[np.isinf(b)] = np.nan
        U[np.isinf(U)] = np.nan
        
        # Remove the NaN to make the files smaller, otherwise the final files are unreadable.
        llon_container = llon_container[(~np.isnan(a))&(~np.isnan(b))&(~np.isnan(U))] 
        llat_container = llat_container[(~np.isnan(a))&(~np.isnan(b))&(~np.isnan(U))] 
        for var in era5_dict.keys():
            era5_dict[var] = era5_dict[var][(~np.isnan(a))&(~np.isnan(b))&(~np.isnan(U))]
        
        # Finally you clean up all the fields involved in the condition AT ONCE.
        # If you don't they mess up the indices.
        a, b, U = a[(~np.isnan(a))&(~np.isnan(b))&(~np.isnan(U))], b[(~np.isnan(a))&(~np.isnan(b))&(~np.isnan(U))], U[(~np.isnan(a))&(~np.isnan(b))&(~np.isnan(U))] 

        # Output dictionary
        d = {
            'control':np.transpose(a), 
            'response':np.transpose(b), 
            'background_ws':np.transpose(U),
            'lon':np.transpose(llon_container),
            'lat':np.transpose(llat_container),
            }
        
        for var in era5_dict.keys():
            d[var+'_era5'] = np.transpose(era5_dict[var])

        df = pd.DataFrame(data=d)
            
        #plt.scatter(df.control,df.response,s=0.1)
        #plt.show()

        if enable_save_files:
            filename = f'{area_str}_daily_era5_env_cond_era5_{wind_deriv}_vs_era5_{sst_deriv}_from_{instant_start.strftime('%Y-%m-%d')}_to_{(instant_end-pd.Timedelta(1,'d')).strftime('%Y-%m-%d')}_sigma{sigma}.txt'
            df.to_csv(path2output+'/'+filename, index=False)
            generated_filenames.append(path2output+'/'+filename)


