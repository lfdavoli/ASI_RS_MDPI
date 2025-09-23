# %% [markdown]
# # Gathering data for SST-wind coupling analysis (DMM and PA) depending on environmental conditions from MetOp A data and ERA5.
# ## Description
# 
# ## To do's
# 
# ## Details
# 
# @author: Lorenzo Francesco Davoli (original from Agostino Meroni)
# 
# @creation: 02/07/2024
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
import glob
from multiprocessing import Pool
from datetime import datetime

# My utility funtions from "miscellanea" directory
# append a new directory to sys.path
# pyright: ignore[reportMissingImports]
sys.path.append('/home/lorenzo/ASI_RS_MDPI')
from my_utility_functions import test_if_dataset_empty
import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs
from global_land_mask import globe

# from matplotlib.pyplot import cm
import numpy as np
import xarray as xr
import pandas as pd
import glob
import geometry as gm # grad_sphere, div_sphere, nan_gaussian_filter, L2wind_2_regular_grid_mask

import warnings
warnings.filterwarnings("ignore")




# %% [markdown]
# ## Controls

# %%
# This is to compute systematically the statistics for the DMM over different
# areas of the globe and over single seasons. It can also be used for the PA.
# The standard deviation of the gaussian filter used to determine the
# background wind is sigma.

# Set the geographical parameters for the analysis and the figures.
# Look at the areas of the WBCs defined by O'Neill et al, J. Cli (2012).
# The western Pacific area is defined following Li and Carbone, JAS (2012).
# The eastern Pacific cold tongue are follows Chelton et al., J. Cli (2001).
# lat in [-90,90], lon in [-180,180]
area_configs = [
    #{'area_str': 'extended_eureca', 'area_name': 'extended EUREC4A', 'minlon': -62., 'maxlon': -40., 'minlat': 0, 'maxlat': 20.},
    {'area_str': 'gulf_stream', 'area_name': 'Gulf Stream', 'minlon': -83., 'maxlon': -30., 'minlat': 30., 'maxlat': 55.},
    #{'area_str': 'malvinas', 'area_name': 'Malvinas current', 'minlon': -70., 'maxlon': 0., 'minlat': -60., 'maxlat': -30.},
    #{'area_str': 'agulhas', 'area_name': 'Agulhas current', 'minlon': 0., 'maxlon': 100., 'minlat': -60., 'maxlat': -30.},
    #{'area_str': 'kuroshio', 'area_name': 'Kuroshio current', 'minlon': 140., 'maxlon': 180., 'minlat': 30., 'maxlat': 50.},
    #{'area_str': 'southern_indian', 'area_name': 'Southern Indian', 'minlon': 55., 'maxlon': 100., 'minlat': -60., 'maxlat': -30.},
    #{'area_str': 'south_east_atlantic', 'area_name': 'South-east Atlantic', 'minlon': -10., 'maxlon': 15., 'minlat': -25., 'maxlat': -5.},
    #{'area_str': 'artic', 'area_name': 'Artic', 'minlon': -12., 'maxlon': 10., 'minlat': 67., 'maxlat': 75.50},
    #{'area_str': 'global', 'area_name': 'Global', 'minlon': -180., 'maxlon': 180., 'minlat': -60., 'maxlat': 60.},
]

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

wind_data_resolution = '12_5km_coastal'
sst_data_resolution = '5km'

# Select the standard deviation of the low-pass Gaussian filter used to determine the background wind field.
# The unit of measure is the sst-product gridstep (for MetOp A AVHRR it's 5km).
# If we take 5 the correlation seems to be less significant: we problably go to too fine scales... 
# Standard value: sigma = 10 == 50km.
# In the tropics the SST structures are large: check how the results change for different sigmas.
#Take a relatively local sigma, to highlight the small scale features. 
sigma = 10 # *5km. 

# Select the smoothings to be applied to the sst field through a gaussian filter.
# The unit is the sst-product gridstep.
psi = 2 # *5km

# Select the smoothing to be applied to the wind field through a gaussian filter.
enable_wind_smoothing = True
if enable_wind_smoothing:
    sigma = 90 # *5km. 
    psi = 3 # *5km
    psi_wind = 3 # *5km
    if psi_wind != psi:
        raise NameError('Same SST-wind smoothing needed')
    if 5*psi_wind > sigma:
        raise NameError('No signal left for the anomaly field')

# Define the extremes of the period to analyze
str_start = '2020-01-01'
#str_start = '2020-12-25'
#period_start = '2021-05-01'
#period_end = '2020-04-12'
#period_end = '2020-03-01'
str_end = '2021-09-30'
#period_end = '2020-09-01'
#period_end = '2021-06-01'
# We split the whole period into intervals to reduce the memory cost. 
time_interval_length = pd.DateOffset(days=0) 
# Number of processes to start, each one receives a time
# interval to process. 
# Note: too many processes imply large memory usage. Tune 
# it together with time_inte
num_processes = 8

# Description of the env conditions in the file
description = 'env_cond_era5_derivatives'

# Set some relevant paths.
path2C3S_ICDR = '/pacific/c3s'
path2SST = path2C3S_ICDR + '/sst/ICDR_v2/AVHRR/L3C/v2.1/AVHRRMTA_G'
path2ascat = '/atlantic/podaac/OceanWinds'
if wind_data_resolution == '12_5km_coastal':
    path2ascat = path2ascat + '/ascat/L2/metop_a/12.5km_coastal'
elif wind_data_resolution == '25km':
    path2ascat = path2ascat + '/ascat/L2/metop_a/25km'
path2era5 = '/pacific/ERA5/hourly/u10_v10_d2m_t2m_sst_sp_slhf_sshf_zust_blh_ssrd_strd_crr_lsrr'

# SST variables to drop
sst_vars_to_drop = [
    'lat_bnds',
    'lon_bnds',
    'time_bnds',
    'sea_surface_temperature_depth',
    'sea_surface_temperature_depth_anomaly',
    'sst_depth_dtime',
    'sses_bias',
    'sses_standard_deviation',
    'sea_surface_temperature_total_uncertainty',
    'sea_surface_temperature_depth_total_uncertainty',
    'depth_adjustment',
    'adjustment_alt',
    'l2p_flags',
    'wind_speed',
    'sea_surface_temperature_retrieval_type',
    'alt_sst_retrieval_type',
    'uncertainty_random',
    'uncertainty_correlated',
    'uncertainty_systematic',
    'uncertainty_correlated_time_and_depth_adjustment',
    'uncertainty_random_alt',
    'uncertainty_correlated_alt',
    'uncertainty_systematic_alt',
    'sst_sensitivity',
    'empirical_adjustment',
    ]
# Wind variables to drop
wind_vars_to_drop = [
    'wvc_index',
    'model_speed',
    'model_dir',
    'ice_prob',
    'ice_age',
    'bs_distance',
    ]

# Enable saving outputs
enable_save_files = True 

# Enable all the tests, print stuff, plot
enable_debug_mode = False
enable_debug_plots = enable_debug_mode and True

# %% [markdown]
# ## Functions

# %%
def split_time_interval(period_start, period_end, time_interval_length : pd.DateOffset):
    """ 
    Split the period into intervals of fixed length.
    """
    # Convert the input period strings to pandas Timestamps
    period_start_date = pd.to_datetime(period_start)
    period_end_date = pd.to_datetime(period_end)
    
    # Initialize lists to store the intervals starting and ending points
    period_interval_starting_points = []
    period_interval_ending_points = []
    
    # Create intervals of "time_interval_length" from "start_date" to "end_date"
    current_interval_start = period_start_date
    while current_interval_start < period_end_date:
        # Increment by the interval length
        current_interval_end = (current_interval_start + time_interval_length)

        # Ensure that the interval end does not go beyond "end_date"
        if current_interval_end >= period_end_date:
            current_interval_end = period_end_date
            
        # Append the current interval to the lists
        period_interval_starting_points.append(current_interval_start.strftime('%Y-%m-%d'))
        period_interval_ending_points.append(current_interval_end.strftime('%Y-%m-%d'))
        
        # Move to the next interval
        current_interval_start = current_interval_end + pd.DateOffset(days=1)
    
    return period_interval_starting_points, period_interval_ending_points


def get_time_interval(array):
    '''
    Take an array of Timestamps and None elements, return the min and max
    time in the array.
    '''
    # Flatten the array and filter out None values
    timestamps = [elem for elem in array.ravel() if elem is not None]
    
    if not timestamps:
        return None, None  # Return None if no timestamps are present
    
    # Find the minimum and maximum timestampswind_ds.where(wind_ds['des']==False)
    min_time = min(timestamps)
    max_time = max(timestamps)
    
    return min_time, max_time

def create_nan_dummy_row(df_row):
    """
    Create a new row with the same columns as the input row, but with all values as NaN.
    Be careful: if tossed in an iterrows() loop all the nans value will be promoted to
    NaT.
    """
    nan_dummy = pd.DataFrame(np.nan, index=[0], columns=df_row.index)
    #nan_dummy['time'] = pd.NaT 
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
                    #outfile.write('control,response,background_ws,lon,lat,u10_era5,v10_era5,d2m_era5,t2m_era5,sst_era5,sp_era5,slhf_era5,sshf_era5,zust_era5,L_coare_50,Wang_class50,time_coare_50,lat_coare_50,lon_coare_50,log_L_ML_50\n')
                    header_written = True
                
                # Write the rest of the lines (skip header)
                outfile.writelines(lines[1:])
                outfile.write("\n")  # Optional: Add newline between files
 

def get_wind_files_in_period(path2ascat, int_str_start, int_str_end):
    """
    Find all files in a directory structure 'YYYY/MM/ascat_YYYYMMDD*.nc' falling within the given period.

    Args:
        path2ascat (str): The base directory containing the files.
        int_str_start (str): The start of the period in 'YYYY-MM-DD' format.
        int_str_end (str): The end of the period in 'YYYY-MM-DD' format.

    Returns:
        list: A list of file paths matching the criteria.
    """
    # Convert period_start and period_end to datetime objects
    start_date = datetime.strptime(int_str_start, '%Y-%m-%d')
    end_date = datetime.strptime(int_str_end, '%Y-%m-%d')

    # Initialize an empty list to store matching file paths
    matching_files = []

    # Iterate through the years in the range
    for year in range(start_date.year, end_date.year + 1):
        year_path = os.path.join(path2ascat, str(year))
        if not os.path.exists(year_path):
            continue

        # Iterate through the months in the range
        for month in range(1, 13):
            month_path = os.path.join(year_path, str(month).zfill(2))
            if not os.path.exists(month_path):
                continue

            # Generate the glob pattern for files in the month
            file_pattern = os.path.join(month_path, f"ascat_{year}{str(month).zfill(2)}*.nc")
            files_in_month = sorted(glob.glob(file_pattern))

            # Filter files by date range
            for file_path in files_in_month:
                file_date_str = os.path.basename(file_path).split('_')[1]
                file_date = datetime.strptime(file_date_str, '%Y%m%d')
                if start_date <= file_date <= end_date:
                    matching_files.append(file_path)

    return matching_files

def get_sst_files_in_wind_file_period(path2SST, int_str_start, int_str_end):
    """
    Find all files in a directory structure 'YYYY/MM/YYYYMMDD*.nc' falling within the given period.

    Args:
        path2SST (str): The base directory containing the files.
        str_start (str): The start of the period in 'YYYY-MM-DD' format.
        str_end (str): The end of the period in 'YYYY-MM-DD' format.

    Returns:
        list: A list of file paths matching the criteria.
    """
    # Convert period_start and period_end to datetime objects.
    # Neglect the hour.
    start_date = datetime.strptime(int_str_start[:10], '%Y-%m-%d')
    end_date = datetime.strptime(int_str_end[:10], '%Y-%m-%d')

    # Initialize an empty list to store matching file paths
    matching_files = []

    # Iterate through the years in the range
    for year in range(start_date.year, end_date.year + 1):
        year_path = os.path.join(path2SST, str(year))
        if not os.path.exists(year_path):
            continue

        # Iterate through the months in the range
        for month in range(1, 13):
            month_path = os.path.join(year_path, str(month).zfill(2))
            if not os.path.exists(month_path):
                continue

            # Generate the glob pattern for files in the month
            file_pattern = os.path.join(month_path, f"{year}{str(month).zfill(2)}*.nc")
            files_in_month = sorted(glob.glob(file_pattern))

            # Filter files by date range
            for file_path in files_in_month:
                file_date_str = os.path.basename(file_path).split('-')[0][:8]
                file_date = datetime.strptime(file_date_str, '%Y%m%d')
                if start_date <= file_date <= end_date:
                    matching_files.append(file_path)

    return matching_files

def get_era5_file_in_wind_file_period(path2era5, wind_time_mode):
    # Get rounded mode hour
    wind_time_mode = pd.to_datetime(wind_time_mode,utc=True).round('H')
    
    era5_file = os.path.join(
        path2era5,
        str(wind_time_mode.year),
        str(wind_time_mode.month).zfill(2),
        str(wind_time_mode.day).zfill(2),
        f"era5_u10_v10_d2m_t2m_sst_sp_slhf_sshf_zust_blh_ssrd_strd_crr_lsrr_{wind_time_mode.year}{str(wind_time_mode.month).zfill(2)}{str(wind_time_mode.day).zfill(2)}_{str(wind_time_mode.hour).zfill(2)}0000.nc"
        )
    return era5_file

def compute_fields_per_period(int_str_start_end):
    print(f'{datetime.now()} Interval {int_str_start_end}')
    int_str_start = int_str_start_end[0]
    int_str_end = int_str_start_end[1]
    a = [] # Forcing field.
    b = [] # Atmospheric response field.
    U = [] # Background wind speed.
    asc = [] # ascending (1) or descending (0) orbit
    llon_container = [] # longitude
    llat_container = [] # latitude

    # Initialize container for era5 variables
    era5_dict = dict()
    with xr.open_dataset(sorted(glob.glob(path2era5 +f'/{int_str_start[:4]}/{int_str_start[5:7]}/{int_str_start[-2:]}/*.nc'))[0]) as ds_era5_example:
        for var, _ in ds_era5_example.data_vars.items():
            era5_dict[var] = []
        #era5_dict['time'] = []

    # Find wind files corresponding to the period
    wind_files = get_wind_files_in_period(path2ascat,int_str_start,int_str_end)

    # Loop over wind files
    for wind_file in wind_files:
        if enable_debug_mode: print(f"{datetime.now()} {os.path.basename(wind_file)}")
        wind_ds = xr.open_dataset(wind_file,chunks='auto',drop_variables=wind_vars_to_drop)
        # Fix lon notation to be compatible with sst files. 
        # In the end: wind lon in [-180,180], lat in [-90,90]
        wind_ds = wind_ds.assign_coords(lon=(wind_ds.lon + 180) % 360 - 180)   
        
        if enable_debug_mode and enable_debug_plots:
            plt.figure(figsize=(12, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            pcm = ax.pcolormesh(wind_ds['lon'],wind_ds['lat'],wind_ds['wind_speed'], cmap=cmocean.cm.speed, shading='auto', transform=ccrs.PlateCarree())
            plt.colorbar(pcm, orientation='horizontal', label='ws')
            plt.show()
            
        # Filter to region extent. If no data there, then skip file
        if enable_debug_mode: print(f"\t{datetime.now()} - Region check")

        wind_speed = wind_ds['wind_speed'].where(
            (wind_ds['lon'] > region_extent[0]) & (wind_ds['lon'] < region_extent[1]) & (wind_ds['lat'] > region_extent[2]) & (wind_ds['lat'] < region_extent[3])
            )
        
        if np.isnan(wind_speed).all() or np.isnan(wind_speed.values).sum() == 1:
            if enable_debug_mode: print(f"\t{datetime.now()} {area_name} {os.path.basename(wind_file)} SKIP - Out of region")
            continue
        else: 
            if enable_debug_mode and enable_debug_plots:
                plt.figure(figsize=(12, 6))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.coastlines()
                pcm = ax.pcolormesh(wind_ds['lon'],wind_ds['lat'],wind_speed, cmap=cmocean.cm.speed, shading='auto', transform=ccrs.PlateCarree())
                plt.title(f'Wind speed in region ({np.isfinite(wind_speed.values).sum()}/{np.size(wind_speed)})')
                plt.colorbar(pcm, orientation='horizontal', label='ws')
                plt.show()
        if enable_debug_mode: print(f"\t{datetime.now()} - Region filter - DONE")

        # Info on ascending or descending orbit
        # If True then descending orbit
        nrows = len(wind_ds['NUMROWS'])
        asc_rows_start = slice(0,round(nrows/4))
        des_rows = slice(round(nrows/4),3*round(nrows/4))
        asc_rows_end = slice(3*round(nrows/4),-1)

        
        if enable_debug_mode and enable_debug_plots:
            plt.figure(figsize=(12, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            pcm = ax.pcolormesh(wind_ds.sel(NUMROWS=asc_rows_start)['lon'],wind_ds.sel(NUMROWS=asc_rows_start)['lat'],wind_ds.sel(NUMROWS=asc_rows_start)['wind_speed'], cmap=cmocean.cm.speed, shading='auto', transform=ccrs.PlateCarree())
            ax.set_title('ascending start')
            plt.colorbar(pcm, orientation='horizontal', label='ws')
            plt.show()
            
            plt.figure(figsize=(12, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            pcm = ax.pcolormesh(wind_ds.sel(NUMROWS=des_rows)['lon'],wind_ds.sel(NUMROWS=des_rows)['lat'],wind_ds.sel(NUMROWS=des_rows)['wind_speed'], cmap=cmocean.cm.speed, shading='auto', transform=ccrs.PlateCarree())
            ax.set_title('descending')
            plt.colorbar(pcm, orientation='horizontal', label='ws')
            plt.show()

            plt.figure(figsize=(12, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            pcm = ax.pcolormesh(wind_ds.sel(NUMROWS=asc_rows_end)['lon'],wind_ds.sel(NUMROWS=asc_rows_end)['lat'],wind_ds.sel(NUMROWS=asc_rows_end)['wind_speed'], cmap=cmocean.cm.speed, shading='auto', transform=ccrs.PlateCarree())
            ax.set_title('ascending end')
            plt.colorbar(pcm, orientation='horizontal', label='ws')
            plt.show()

        if enable_debug_mode: print(f"\t{datetime.now()} Extract wind fields")
        # Read the variables of interest.
        lon_wind = wind_ds['lon'].values
        lat_wind = wind_ds['lat'].values
        wind_speed = wind_ds['wind_speed'].values
        wind_dir = wind_ds['wind_dir'].values
        u = wind_speed * np.cos(np.pi*0.5-wind_dir*np.pi/180)
        v = wind_speed * np.sin(np.pi*0.5-wind_dir*np.pi/180)
        del wind_speed,wind_dir

        # Remove the wind data with any high quality flag.
        wvc_quality_flag = wind_ds['wvc_quality_flag'].values
        u[np.log2(wvc_quality_flag)>5]=np.nan
        v[np.log2(wvc_quality_flag)>5]=np.nan
        del wvc_quality_flag
        if enable_debug_mode: print(f"\t{datetime.now()} Extract wind fields - DONE")
        

        if enable_debug_mode and enable_debug_plots:

            llat_wind,llon_wind = wind_ds['lat'],wind_ds['lon']
            
            plt.figure(figsize=(12, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            pcm = ax.pcolormesh(llon_wind[asc_rows_start,:], llat_wind[asc_rows_start,:], u[asc_rows_start,:], cmap=cmocean.cm.speed, shading='auto', transform=ccrs.PlateCarree())
            plt.colorbar(pcm, orientation='horizontal', label='u10 - ascending start')
            plt.show()
            
            plt.figure(figsize=(12, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            pcm = ax.pcolormesh(llon_wind[des_rows,:], llat_wind[des_rows,:], u[des_rows,:], cmap=cmocean.cm.speed, shading='auto', transform=ccrs.PlateCarree())
            plt.colorbar(pcm, orientation='horizontal', label='u10 - descending')
            plt.show()
            
            plt.figure(figsize=(12, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            pcm = ax.pcolormesh(llon_wind[asc_rows_end,:], llat_wind[asc_rows_end,:], u[asc_rows_end,:], cmap=cmocean.cm.speed, shading='auto', transform=ccrs.PlateCarree())
            plt.colorbar(pcm, orientation='horizontal', label='u10 - ascending end')
            plt.show()            

        if enable_debug_mode: print(f"\t{datetime.now()} Extract ERA5")
        # Wind mode for ERA5 coloc
        wind_time_mode = wind_ds['time'].values[round(len(wind_ds['time'])/2),0]    
        # Find closest hourly ERA5 file
        era5_file = get_era5_file_in_wind_file_period(path2era5,wind_time_mode)
        era5_ds = xr.open_dataset(era5_file)
        if enable_debug_mode: print(f"\t{datetime.now()} Extract ERA5 - DONE")
        
        if enable_debug_mode: print(f"\t{datetime.now()} Fix ERA5")
        # Inverse the order of latitude coordinate to fit the structure of sst_lat
        era5_ds = era5_ds.reindex(latitude = era5_ds.latitude[::-1])
        # Shift longitude from [0,360] to [-180,180]
        era5_ds.coords['longitude'] = (era5_ds.coords['longitude'] + 180) % 360 - 180
        era5_ds = era5_ds.sortby(era5_ds.longitude) 
        # In the end: lon in [-180,180], lat in [-90,90]

        # Crop to region of interest
        era5_ds = era5_ds.sel(longitude=slice(region_extent[0],region_extent[1]),latitude=slice(region_extent[2],region_extent[3]))
        if enable_debug_mode: print(f"\t{datetime.now()} Fix ERA5 - DONE")
        
        # Find wind data time span
        wind_time_bnds = np.nanmin(wind_ds['time']),np.nanmax(wind_ds['time'])

        # Clean-up
        wind_ds.close()

        if enable_debug_mode: print(f"\t{datetime.now()} SST files")
        
        # Select the corresponding sst files
        sst_files = get_sst_files_in_wind_file_period(path2SST, str(wind_time_bnds[0]),str(wind_time_bnds[1]))
        
        # Loop over the SST files
        for sst_file in sst_files:
            if enable_debug_mode: 
                print(f"\t\t{datetime.now()} {sst_file}")
            sst_ds = xr.open_dataset(sst_file,chunks='auto',drop_variables=sst_vars_to_drop, decode_cf=False)
            
            if enable_debug_mode: print(f"\t\t{datetime.now()} Decoding SST file data")
            # Decode time properly
            decoded_time = xr.decode_cf(xr.Dataset({'time': sst_ds['time']})).time.values 
            sst_ds['time'] = (('time',), decoded_time)
            # Decode sst properly    
            decoded_sst = xr.decode_cf(sst_ds)['sea_surface_temperature'].values 
            sst_ds['sea_surface_temperature'].data = decoded_sst
            if enable_debug_mode: print(f"\t\t{datetime.now()} Decoding SST file data - DONE")

            # Select only the region of interest. sst lon in [-180,180], lat in [-90,90]
            if enable_debug_mode: print(f"\t\t{datetime.now()} Regional filter")
            sst_ds = sst_ds.sel(lon=slice(region_extent[0],region_extent[1]),lat=slice(region_extent[2],region_extent[3]))
            if enable_debug_mode: print(f"\t\t{datetime.now()} Regional filter - DONE")

            # Filter according to quality level, following Merchant et al., 2019
            if enable_debug_mode: print(f"\t\t{datetime.now()} Quality filter SST")
            sst_ds['sea_surface_temperature'].data = sst_ds['sea_surface_temperature'].where(sst_ds['quality_level'] >= 3).values
            if enable_debug_mode: print(f"\t\t{datetime.now()} Quality filter SST: keep { np.array(sst_ds['quality_level'] >= 3).sum()}/{sst_ds['sea_surface_temperature'].data.size}")

            # First remove all areas near land (50km) by taking a landmask and 
            # extending it 10 gridsteps in each direction.
            lon_sst = sst_ds.lon
            lat_sst = sst_ds.lat
            # Transform into grids 
            llon_sst, llat_sst = np.meshgrid(lon_sst,lat_sst)

            if enable_debug_mode:
                print(f"\t\t{datetime.now()} llon: {llon_sst.shape}, llat: {llat_sst.shape}")
                print(f"\t\t{datetime.now()} lon: {lon_sst.shape}, lat: {lat_sst.shape}")

            if enable_debug_mode: 
                print(f"\t\t{datetime.now()} Coast mask")
            
            ocean_mask = globe.is_ocean(llat_sst, llon_sst)
            if enable_debug_mode and enable_debug_plots:
                plt.figure(figsize=(12, 6))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.coastlines()
                pcm = ax.pcolormesh(llon_sst,llat_sst,ocean_mask, cmap=cmocean.cm.gray, shading='auto', transform=ccrs.PlateCarree())
                plt.colorbar(pcm, orientation='horizontal', label='ocean mask')
                plt.show()

            np.putmask(ocean_mask,ocean_mask,np.roll(ocean_mask,10,axis=0)) # change only water values into land ones with shifts
            np.putmask(ocean_mask,ocean_mask,np.roll(ocean_mask,-10,axis=0))
            np.putmask(ocean_mask,ocean_mask,np.roll(ocean_mask,10,axis=1))
            np.putmask(ocean_mask,ocean_mask,np.roll(ocean_mask,-10,axis=1))
            
            if enable_debug_mode and enable_debug_plots:
                plt.figure(figsize=(12, 6))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.coastlines()
                pcm = ax.pcolormesh(llon_sst,llat_sst, globe.is_ocean(llat_sst, llon_sst)^ocean_mask, cmap=cmocean.cm.gray, shading='auto', transform=ccrs.PlateCarree())
                plt.colorbar(pcm, orientation='horizontal', label='coast mask')
                plt.show()

            sst_ds['sea_surface_temperature'] = sst_ds.where(ocean_mask)['sea_surface_temperature']  # Set land points to NaN

            if enable_debug_mode and enable_debug_plots:
                sst_ds['sea_surface_temperature'].isel(time=0).plot(cmap=cmocean.cm.thermal)
                plt.xlim((-180,180))
                plt.ylim((-90,90))
                plt.title('regional SST with no coastal areas')
                plt.show()
        
            if enable_debug_mode: 
                print(f"\t\t{datetime.now()} Coast mask - DONE")

            # We need to extract only values corresponding to the wind file orbit.
            # We find the interval in the sst_dtime dimension.
            min_dtime_wind = np.timedelta64(int((wind_time_bnds[0]-sst_ds['time'].data[0])/np.timedelta64(1,'s')),'s')
            max_dtime_wind = np.timedelta64(int((wind_time_bnds[1]-sst_ds['time'].data[0])/np.timedelta64(1,'s')),'s')
            # Select only the wind-colocated values in the sst fields
            sst_ds = sst_ds.where((sst_ds['sst_dtime'] < max_dtime_wind) & (sst_ds['sst_dtime'] > min_dtime_wind))
            
            # Finally, drop dims entries for which we have all nans values
            sst_ds = sst_ds.dropna(dim='lat',how='all')
            sst_ds = sst_ds.dropna(dim='lon',how='all')

            # Skip if the result of the filtering is not enough data (considering 
            # interpolation with 25km ERA5 and interp artifacts for gapped data, 
            # it's better to take a large threshold), or 1D, or all nans.
            if sst_ds['sea_surface_temperature'].isel(time=0).size < 100 or 1 in sst_ds['sea_surface_temperature'].isel(time=0).shape or np.isnan(sst_ds['sea_surface_temperature'].isel(time=0).data).all():
                if enable_debug_mode: print(f'\t\t{datetime.now()} {area_name} {os.path.basename(sst_file)} SKIP - not enough SST points after filtering, dims: {sst_ds['sea_surface_temperature'].dims}, shape: {sst_ds['sea_surface_temperature'].shape}')
                continue
                
            # Update sst grids
            lon_sst, lat_sst = sst_ds['lon'],sst_ds['lat']
            llon_sst, llat_sst = np.meshgrid(lon_sst,lat_sst)

            if enable_debug_mode:
                print(f"\t\t{datetime.now()} Updated llon: {llon_sst.shape}, llat: {llat_sst.shape}")
                print(f"\t\t{datetime.now()} Updated lon: {lon_sst.shape}, lat: {lat_sst.shape}")

            if enable_debug_mode and enable_debug_plots:
                plt.scatter(lon_wind,lat_wind,c=u,cmap=cmocean.cm.speed,s=0.01)
                plt.title('u')
                plt.xlim((-180,180))
                plt.ylim((-90,90))
                plt.show()
                plt.scatter(lon_wind,lat_wind,c=v,cmap=cmocean.cm.speed,s=0.01)
                plt.title('v')
                plt.xlim((-180,180))
                plt.ylim((-90,90))
                plt.show()
                if 1 not in sst_ds['sea_surface_temperature'].isel(time=0).shape:
                    sc=sst_ds['sea_surface_temperature'].isel(time=0).plot(cmap=cmocean.cm.thermal)
                    plt.xlim((-180,180))
                    plt.ylim((-90,90))
                    plt.title('colocated SST')
                    plt.show()
                else: 
                    print(f"\t\t{datetime.now()} SST shape : {sst_ds['sea_surface_temperature'].isel(time=0).shape}")

            # Interpolate ERA5
            if enable_debug_mode: print(f'\t\t{datetime.now()} Interp ERA5')
            era5_interp_ds = era5_ds.interp(
                latitude=lat_sst,
                longitude=lon_sst,
                method="linear"  # Options include 'linear', 'nearest', 'cubic'
                )
            
            # Stop if the interpolation fails
            try: 
                test_if_dataset_empty(era5_interp_ds) 
            except:
                print(f"\t{datetime.now()} {area_name} {os.path.basename(wind_file)} SKIP - Empty ERA5 interpolation")
                continue

            if enable_debug_mode: print(f'\t\t{datetime.now()} Interp ERA5 - DONE')
            
            # SST field smoothing
            if enable_debug_mode: print(f'\t\t{datetime.now()} Smooth sst')
            sst = gm.nan_gaussian_filter(sst_ds['sea_surface_temperature'].isel(time=0).data,psi)
            if enable_debug_mode: print(f'\t\t{datetime.now()} Smooth sst - DONE')
            
            # Clean-up
            sst_ds.close()

            # Get the background wind field.
            # Here sigma is given in gridsteps of the sst product
            # For AVHRR MetOp A 1 step == 5km
            # Interpolate wind to this grid
            if enable_debug_mode: print(f'\t\t{datetime.now()} Interp wind')

            # We want to regrid the wind into the sst grid.
            # The interpolation introduces artifacts if you don't separate the different branches.
            u_branches = dict()
            v_branches = dict()
            u_branches['interp_asc_start'], v_branches['interp_asc_start'] = gm.L2wind_2_regular_grid_mask(lon_wind[asc_rows_start,:],lat_wind[asc_rows_start,:],u[asc_rows_start,:],v[asc_rows_start,:],lon_sst,lat_sst,region_extent)
            u_branches['interp_asc_end'], v_branches['interp_asc_end'] = gm.L2wind_2_regular_grid_mask(lon_wind[asc_rows_end,:],lat_wind[asc_rows_end,:],u[asc_rows_end,:],v[asc_rows_end,:],lon_sst,lat_sst,region_extent)
            u_branches['interp_des'], v_branches['interp_des'] = gm.L2wind_2_regular_grid_mask(lon_wind[des_rows,:],lat_wind[des_rows,:],u[des_rows,:],v[des_rows,:],lon_sst,lat_sst,region_extent)

            if enable_debug_mode and enable_debug_plots:
                for key in u_branches.keys():    
                    #print(key,u_branches[key])
                    if np.size(u_branches[key]) > 1:
                        print(f"\t\t{datetime.now()} llon: {llon_sst.shape}, llat: {llat_sst.shape}, u_branch[{key}]: {u_branches[key].shape}")
                        sc=plt.scatter(llon_sst,llat_sst,c=u_branches[key],cmap=cmocean.cm.speed,s=0.01)
                        plt.colorbar(sc, orientation='horizontal', label='u '+key)
                        plt.xlim((-180,180))
                        plt.ylim((-90,90))
                        plt.show()
                        del sc
                        
            del lon_sst,lat_sst

            # Merge all branches into the complete u_interp field
            u_interp = np.full_like(llon_sst,fill_value=np.nan,dtype=float)
            v_interp = np.full_like(llon_sst,fill_value=np.nan,dtype=float)
            for key in u_branches.keys():    
                if np.size(u_branches[key]) > 1:
                    np.putmask(u_interp,~np.isnan(u_branches[key]),u_branches[key])
                    if enable_debug_mode: print(f'\t\t{datetime.now()} In branch management: {key}, add to u_interp')
                else:
                    if enable_debug_mode: print(f"\t\t{datetime.now()} branch {key}, len {np.size(u_branches[key])}: empty")

                if np.size(v_branches[key]) > 1:
                    np.putmask(v_interp,~np.isnan(v_branches[key]),v_branches[key])
                    if enable_debug_mode: print(f'\t\t{datetime.now()} In branch management: {key}, add to v_interp')
                else:
                    if enable_debug_mode: print(f"\t\t{datetime.now()} branch {key}, len {np.size(u_branches[key])}: empty")

            # Try if u_interp has been created. If not, skip.
            try:
                if np.isnan(u_interp).all():
                    if enable_debug_mode:
                        print(f"\t{datetime.now()} {area_name} {os.path.basename(wind_file)} SKIP - All-nans wind interpolation")
                    continue
            except:
                if enable_debug_mode: 
                    print(f"\t{datetime.now()} {area_name} {os.path.basename(wind_file)} SKIP - No wind interpolation")
                continue

            # Define an ascending-descending-no_data map
            ascending = np.full_like(u_interp,fill_value=-1,dtype=int)
            for key in u_branches.keys():    
                if np.size(u_branches[key]) > 1:
                    np.putmask(ascending,~np.isnan(u_branches[key]),  1 if key[7:10] == 'asc' else 0 )
                    
            del u_branches,v_branches
                
            if enable_debug_mode and enable_debug_plots:
                print(f"lon: {llon_sst.shape}, lat: {llat_sst.shape}, ascending: {ascending.shape}")
                sc=plt.scatter(llon_sst,llat_sst,c=ascending,cmap=cmocean.cm.thermal,s=0.01)
                plt.colorbar(sc, orientation='horizontal', label='asc')
                plt.xlim((-180,180))
                plt.ylim((-90,90))
                plt.show()
            

            if enable_debug_mode: print(f'\t\t{datetime.now()} Interp wind - DONE')

            if enable_debug_mode and enable_debug_plots:
                sc=plt.scatter(llon_sst,llat_sst,c=u_interp,cmap=cmocean.cm.speed,s=0.1)
                plt.colorbar(sc, orientation='horizontal', label='u_interp')
                #plt.xlim((-180,180))
                #plt.ylim((-90,90))
                plt.show()

            # Smooth wind field
            if enable_wind_smoothing:
                if enable_debug_mode: print(f'\t\t{datetime.now()} Smooth wind')
                u_interp = gm.nan_gaussian_filter(u_interp,psi_wind)
                v_interp = gm.nan_gaussian_filter(v_interp,psi_wind)
                if enable_debug_mode: print(f'\t\t{datetime.now()} Smooth wind - DONE')

            if enable_debug_mode and enable_debug_plots and enable_wind_smoothing:
                sc=plt.scatter(llon_sst,llat_sst,c=u_interp,cmap=cmocean.cm.speed,s=0.1)
                plt.colorbar(sc, orientation='horizontal', label='smoothed u_interp')
                #plt.xlim((-180,180))
                #plt.ylim((-90,90))
                plt.show()

            if enable_debug_mode: print(f'\t\t{datetime.now()} Extract background wind')
            smooth_u = gm.nan_gaussian_filter(u_interp,sigma)
            smooth_v = gm.nan_gaussian_filter(v_interp,sigma)
            smooth_ws = np.sqrt(smooth_u**2+smooth_v**2)
            if enable_debug_mode: print(f'\t\t{datetime.now()} Extract background wind - DONE')

            if enable_debug_mode and enable_debug_plots:
                sc=plt.scatter(llon_sst,llat_sst,c=smooth_u,cmap=cmocean.cm.speed,s=0.1)
                plt.colorbar(sc, orientation='horizontal', label='U')
                #plt.xlim((-180,180))
                #plt.ylim((-90,90))
                plt.show()

            cosphi = smooth_u/smooth_ws
            sinphi = smooth_v/smooth_ws

            # Get the anomalies with respect to the background wind field.
            u_prime = u_interp-smooth_u
            v_prime = v_interp-smooth_v

            del smooth_u,smooth_v

            if enable_debug_mode and enable_debug_plots:
                sc=plt.scatter(llon_sst,llat_sst,c=u_prime,cmap=cmocean.cm.speed,s=0.1)
                plt.colorbar(sc, orientation='horizontal', label='u_prime')
                #plt.xlim((-180,180))
                #plt.ylim((-90,90))
                plt.show()
                sc=plt.scatter(llon_sst,llat_sst,c=v_prime,cmap=cmocean.cm.speed,s=0.1)
                plt.colorbar(sc, orientation='horizontal', label='v_interp')
                #plt.xlim((-180,180))
                #plt.ylim((-90,90))
                plt.show()
                sc=plt.scatter(llon_sst,llat_sst,c=sst,cmap=cmocean.cm.thermal,s=0.1)
                plt.colorbar(sc, orientation='horizontal', label='sst')
                #plt.xlim((-180,180))
                #plt.ylim((-90,90))
                plt.show()
                

            if enable_debug_mode: print(f'\t\t{datetime.now()} Derivatives')

            dsst_dx, dsst_dy = gm.grad_sphere(sst,llon_sst,llat_sst)
            
            a_prime = []
            b_prime = []
            if sst_deriv=='gamma':
                a_prime = u_interp*dsst_dx + v_interp*dsst_dy
                x_string = 'u.grad(SST) [K/s]'; vmin_a=-2.2e-4; vmax_a=2.2e-4
            elif sst_deriv=='dsst_dr':
                a_prime = dsst_dx*cosphi + dsst_dy*sinphi
                x_string = 'dSST/dr [K/m]'; vmin_a=-2.2e-5; vmax_a=2.2e-5
            elif sst_deriv=='lapl_sst':
                a_prime = gm.div_sphere(dsst_dx,dsst_dy,llon_sst,llat_sst)
                x_string = 'lapl SST [K/m^2]'; vmin_a=-1e-9; vmax_a=1e-9
            elif sst_deriv=='d2sst_ds2':
                dsst_ds = -dsst_dx*sinphi + dsst_dy*cosphi
                ddsst_ds_dx, ddsst_ds_dy = gm.grad_sphere(dsst_ds,llon_sst,llat_sst)
                a_prime = -ddsst_ds_dx*sinphi + ddsst_ds_dy*cosphi
                del ddsst_ds_dx, ddsst_ds_dy
                x_string = 'd2SST/ds2 [K/m^2]'; vmin_a=-1e-9; vmax_a=1e-9
            elif sst_deriv=='sst_prime':
                smooth_sst = gm.nan_gaussian_filter(sst,sigma)
                a_prime = sst-smooth_sst
                x_string = 'SST_prime [K]'; vmin_a=-5; vmax_a=5
                del smooth_sst
            else:
                raise NameError('Unknown derivative')

            if wind_deriv=='wind_div':
                b_prime = gm.div_sphere(u_interp,v_interp,llon_sst,llat_sst)
                y_string = 'Wind divergence [1/s]'; vmin_b=-2.2e-4; vmax_b=2.2e-4
            elif wind_deriv=='dr_dot_prime_dr':
                r_dot_prime = u_prime*cosphi + v_prime*sinphi
                dr_dot_prime_dx, dr_dot_prime_dy = gm.grad_sphere(r_dot_prime,llon_sst,llat_sst)
                b_prime = dr_dot_prime_dx*cosphi + dr_dot_prime_dy*sinphi 
                y_string = 'dr dot prime/dr [1/s]'; vmin_b=-2.2e-4; vmax_b=2.2e-4
                del dr_dot_prime_dx, dr_dot_prime_dy, r_dot_prime
            elif wind_deriv=='ds_dot_prime_ds':
                s_dot_prime = -u_prime*sinphi + v_prime*cosphi
                ds_dot_prime_dx, ds_dot_prime_dy = gm.grad_sphere(s_dot_prime,llon_sst,llat_sst)
                b_prime = -ds_dot_prime_dx*sinphi + ds_dot_prime_dy*cosphi
                y_string = 'ds dot prime/ds [1/s]'; vmin_b=-2.2e-4; vmax_b=2.2e-4
                del s_dot_prime,ds_dot_prime_dx, ds_dot_prime_dy
            elif wind_deriv=='ws_prime':
                b_prime = np.sqrt(u_interp**2+v_interp**2)-smooth_ws
                y_string = 'ws_prime [m/s]'; vmin_b=-5; vmax_b=5;
            else:
                raise NameError('Unknown derivative')
            del sst,cosphi,sinphi,u_prime,v_prime,dsst_dx,dsst_dy

            if len(a_prime)==0 or len(b_prime) == 0 or np.isnan(a_prime).all() or np.isnan(b_prime).all():
                print(f"\t{datetime.now()} {area_name} {os.path.basename(wind_file)} SKIP -  no a_prime or b_prime")
                continue

            if enable_debug_mode: print(f'\t\t{datetime.now()} Derivatives - DONE')

            if enable_debug_mode and enable_debug_plots:
                print(f"a_prime shape : {a_prime.shape}")
                sc=plt.scatter(llon_sst,llat_sst,c=a_prime,cmap=cmocean.cm.balance,s=0.1)
                plt.colorbar(sc, orientation='horizontal', label=sst_deriv)
                #plt.xlim((-180,180))
                #plt.ylim((-90,90))
                plt.show()
                sc=plt.scatter(llon_sst,llat_sst,c=b_prime,cmap=cmocean.cm.balance,s=0.1)
                plt.colorbar(sc, orientation='horizontal', label=wind_deriv)
                #plt.xlim((-180,180))
                #plt.ylim((-90,90))
                plt.show()

            # Concatenate the variables (with no subsampling) removing the NaNs.
            if enable_debug_mode: print(f'\t\t{datetime.now()} Formatting')
            a_to_be_concat = a_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
            b_to_be_concat = b_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]    
            U_to_be_concat = smooth_ws[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
            asc_to_be_concat = ascending[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
            llon_to_be_concat = llon_sst[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
            llat_to_be_concat = llat_sst[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
            del llon_sst,llat_sst,ascending
            if enable_debug_mode: print(f'\t\t{datetime.now()} Formatting - DONE')
            

            if enable_debug_mode and enable_debug_plots:
                sc=plt.scatter(llon_to_be_concat,llat_to_be_concat,c=a_to_be_concat,cmap=cmocean.cm.balance,vmin=-np.nanmax(np.abs(a_to_be_concat.flatten())),vmax=np.nanmax(np.abs(a_to_be_concat.flatten())),s=0.1)
                plt.colorbar(sc, orientation='horizontal', label=sst_deriv)
                plt.show()
                plt.hist(a_to_be_concat.flatten())
                plt.title(sst_deriv)
                plt.show()
                sc=plt.scatter(llon_to_be_concat,llat_to_be_concat,c=b_to_be_concat,cmap=cmocean.cm.balance,vmin=-np.nanmax(np.abs(b_to_be_concat.flatten())),vmax=np.nanmax(np.abs(b_to_be_concat.flatten())),s=0.1)
                plt.colorbar(sc, orientation='horizontal', label=wind_deriv)
                plt.show()
                plt.hist(b_to_be_concat.flatten())
                plt.title(wind_deriv)
                plt.show()
                sc=plt.scatter(llon_to_be_concat,llat_to_be_concat,c=asc_to_be_concat,cmap=cmocean.cm.thermal,s=0.1)
                plt.colorbar(sc, orientation='horizontal', label='asc')
                plt.show()
                

            if enable_debug_mode and enable_debug_plots and False:
                for var, era5_var_ds in era5_interp_ds.data_vars.items():
                    sc=plt.scatter(llon_to_be_concat,llat_to_be_concat,c=era5_var_ds.data[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))],s=0.01)
                    plt.colorbar(sc, orientation='horizontal', label=var)
                    plt.title(var)
                    plt.show()

            if enable_debug_mode: print(f'\t\t{datetime.now()} Update containers')
                    
            a.extend(a_to_be_concat)
            b.extend(b_to_be_concat)
            U.extend(U_to_be_concat)
            asc.extend(asc_to_be_concat)
            llon_container.extend(llon_to_be_concat)
            llat_container.extend(llat_to_be_concat)

            del a_to_be_concat,b_to_be_concat,llon_to_be_concat,llat_to_be_concat
            
            for var, era5_var_ds in era5_interp_ds.data_vars.items():
                era5_dict[var].extend(era5_var_ds.data[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))])
                if len(a) != len(era5_dict[var]): 
                    print(era5_var_ds)
                    print(era5_dict[var]) 
                    raise NameError(f'Size mismatch {var}')
                
            del era5_interp_ds,smooth_ws,a_prime,b_prime
            if enable_debug_mode: print(f'\t\t{datetime.now()} Update containers - DONE')

        # Clean-up
        era5_ds.close()

        if enable_debug_mode: print(f"\t{datetime.now()} SST files - DONE")


    ### Save variables as text files.
    a = np.array(a)
    b = np.array(b)
    U = np.array(U)
    asc = np.array(asc)
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
        'asc' : np.transpose(asc),
        'lon':np.transpose(llon_container),
        'lat':np.transpose(llat_container),
        }

    for var in era5_dict.keys():
        d[var+'_era5'] = np.transpose(era5_dict[var])

    df = pd.DataFrame(data=d)
    del d,a,b,U,llon_container,llat_container,era5_dict    
    df = df.reindex(sorted(df.columns), axis=1)

    if enable_wind_smoothing:
        filename = f'{area_str}_{description}_L2_{wind_data_resolution}_{wind_deriv}_vs_L3C_{sst_data_resolution}_{sst_deriv}_from_{int_str_start}_to_{int_str_end}_sigma{sigma}_psi{psi}_psi_wind{psi_wind}.txt'
    else:
        filename = f'{area_str}_{description}_L2_{wind_data_resolution}_{wind_deriv}_vs_L3C_{sst_data_resolution}_{sst_deriv}_from_{int_str_start}_to_{int_str_end}_sigma{sigma}_psi{psi}.txt'
    if enable_save_files:
        Path(path2output).mkdir(parents=True, exist_ok=True)
        df.to_csv(path2output+'/'+filename, index=False)


    # Check output 
    if enable_debug_plots and enable_save_files:
        fig,axes = plt.subplots(1,2)
        axes[0].scatter(df['lon'],df['lat'],c=df['control'],s=0.01)
        axes[0].set_title('Runtime')
        saved_df = pd.read_csv(path2output+'/'+filename)
        axes[1].scatter(saved_df['lon'],saved_df['lat'],c=saved_df['control'],s=0.01)
        axes[1].set_title('Saved')
        plt.show()
        del saved_df        
    
    del df 

    #raise NameError()

    return path2output+'/'+filename

# %% [markdown]
# # L3C analysis interval by region and interval

# %%
if __name__ == "__main__":
    # Loop over each area configuration
    for area in area_configs:

        area_str = area['area_str']
        area_name = area['area_name']
        minlon = area['minlon']
        maxlon = area['maxlon']
        minlat = area['minlat']
        maxlat = area['maxlat']

        print(f"### {area_name} ###")
        
        # Set region_extent for maps
        region_extent = [minlon, maxlon, minlat, maxlat]

        # Set output path for this area
        if enable_wind_smoothing:
            path2output = os.path.join(
                '/pacific/data_lorenzo/sst_wind_derivatives_fields_output',
                f'{area_str}_{description}_L2_{wind_data_resolution}_{wind_deriv}_vs_L3C_{sst_data_resolution}_{sst_deriv}_from_{str_start}_to_{str_end}_sigma{sigma}_psi{psi}_psi_wind{psi_wind}')
        else:
            path2output = os.path.join(
                '/pacific/data_lorenzo/sst_wind_derivatives_fields_output',
                f'{area_str}_{description}_L2_{wind_data_resolution}_{wind_deriv}_vs_L3C_{sst_data_resolution}_{sst_deriv}_from_{str_start}_to_{str_end}_sigma{sigma}_psi{psi}')

        # Set the periods to analyse
        period_interval_starting_points, period_interval_ending_points = split_time_interval(str_start, str_end, time_interval_length)

        generated_filenames = []
        interval_str_start_end_ar = []

        # Transform into an array of (start,end) as strings,
        for tt,_ in enumerate(period_interval_starting_points):
            interval_str_start = period_interval_starting_points[tt]
            interval_str_end = period_interval_ending_points[tt]
            interval_str_start_end_ar.append((interval_str_start,interval_str_end))
        
        # Limit the number of periods to be simultaneously computed
        for i in range(0,len(interval_str_start_end_ar),num_processes):
            interval_str_start_end_ar_i = interval_str_start_end_ar[i:i+num_processes]
            with Pool(processes=num_processes) as pool:
                generated_filenames.append(pool.map(compute_fields_per_period, interval_str_start_end_ar_i))



