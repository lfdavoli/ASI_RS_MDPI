### Packages ###
import numpy as np
import pandas as pd
import xarray as xr
import typing 
import sys
import os
from pathlib import Path 


### Functions ###
def Get_list_duplicates(mylist):
    '''
    Print all the duplicates in a list.
    '''
    # the given list contains duplicates
    newlist = [] # empty list to hold unique elements from the list
    duplist = [] # empty list to hold the duplicate elements from the list
    for i in mylist:
        if i not in newlist:
            newlist.append(i)
        else:
            duplist.append(i) # this method catches the first duplicate entries, and appends them to the list
    # The next step is to print the duplicate entries, and the unique entries
    print("List of duplicates", duplist)
    return duplist


def classif_y_in_cat(y,cats: dict) -> str:
    '''
    Classify value "y" within cathegories defined in dict "cats".
    '''    
    for key in cats:
        if (y >= cats[key][0]) & (y<=cats[key][1]):
            return key
    return 'unclassified'


# Calculate the lon,lat definition of a square of side side_length
# centered in lon,lat.
import math 
def calculate_square_bnds_deltas(latitude, side_length):
    # Earth's radia in kilometers (https://imagine.gsfc.nasa.gov/features/cosmic/earth_info.html#:~:text=Note%3A%20The%20Earth%20is%20almost,the%20polar%20and%20equatorial%20values.)
    # Equatorial radius
    earth_radius_eq = 6378 # (km)
    # Polar radius
    earth_radius_pol = 6357 # (km)
    # Use avg
    earth_radius = (earth_radius_eq+earth_radius_pol)/2

    # Convert latitude from degrees to radians
    lat_radians = math.radians(latitude)
    # Radius of the circumference at lat == latitude 
    azimuthal_radius = earth_radius * math.cos(lat_radians) #(km)

    # Deltas in rad units
    delta_lat_rad = side_length / azimuthal_radius #(rad)
    delta_lon_rad = side_length / earth_radius #(rad)
    
    # Deltas in deg units
    delta_lat_deg = math.degrees(delta_lat_rad) #(deg)
    delta_lon_deg = math.degrees(delta_lon_rad) #(deg)

    return delta_lon_deg,delta_lat_deg



import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot_lat_lon_on_global_map(latitudes, longitudes, region=None):
    """
    Plots given latitude and longitude points on a map. Optionally restricts to a specified region.

    Args:
        latitudes (list or array): Array of latitude values.
        longitudes (list or array): Array of longitude values.
        region_ar (list): Optional array [[minlat, maxlat], [minlon, maxlon]] defining the map region.
    """
    if len(latitudes) != len(longitudes):
        raise ValueError("Latitude and longitude arrays must have the same length.")

    # Set default region to global if none is specified
    if region is None:
        region = [[-90, 90], [-180, 180]]

    minlat, maxlat = region[0]
    minlon, maxlon = region[1]

    # Create a figure and a Basemap instance
    fig, ax = plt.subplots(figsize=(12, 6))
    m = Basemap(projection='mill', llcrnrlat=minlat, urcrnrlat=maxlat,
                llcrnrlon=minlon, urcrnrlon=maxlon, resolution='c', ax=ax)

    # Draw coastlines, countries, and map boundary
    m.drawcoastlines(color="gray")
    m.drawcountries(color="darkgray")
    m.drawmapboundary(fill_color='#A6CAE0')
    m.fillcontinents(color='#D0F1BF', lake_color='#A6CAE0', zorder=1)

    # Draw lat/lon grid lines
    m.drawparallels(range(int(minlat), int(maxlat) + 1, 10), labels=[1, 0, 0, 0], color='lightgray', fontsize=10)
    m.drawmeridians(range(int(minlon), int(maxlon) + 1, 20), labels=[0, 0, 0, 1], color='lightgray', fontsize=10)

    # Convert latitude and longitude to map projection coordinates
    x, y = m(longitudes, latitudes)

    # Plot the points on the map
    m.scatter(x, y, marker='x', color='orange', edgecolor=None, s=10, zorder=5)

    # Show the plot
    plt.show()


from scipy.special import gammaincc

def weighted_linear_regression_with_pvalues(x, y, sigma_y):
    '''         
        Compute a weighted linear regressio of (x,y+/-sigma_y),
        where the uncertainty on y observations is taken into 
        account.
        Returns a dictionary in the form:
                "intercept" : (intercept, sigma_b),
                "slope" : (slope, sigma_m),
                "chi2" : chi square value,
                "q" : goodness of fit

        From Numerical Recipe 15.2
        If q is larger than, say, 0.1,then the goodness-of-fit is believable. 
        If it is larger than, say, 0.001, then the fit may be acceptable if the 
        errors are nonnormal or have been moderately underestimated. If q is 
        less than 0.001, then the model and/or estimation procedure can rightly
        be called into question. In this latter case, turn to 15.7 to proceed 
        further.
    '''
    
    '''
    ### Numerical recipe 15.2 (https://numerical.recipes/book.html), handmade
    # y = a + bx
    S = np.sum(np.power(sigma_y,-2))
    S_X = np.sum(np.power(sigma_y,-2)*x)
    S_y = np.sum(np.power(sigma_y,-2)*y)
    S_XX = np.sum(np.power(sigma_y,-2)*np.power(x,2))
    S_Xy = np.sum(np.power(sigma_y,-2)*np.multiply(x,y))
    
    Delta = S*S_XX - S_X**2
    a = (S_XX*S_y - S_X*S_Xy)/Delta
    b = (S*S_Xy - S_X*S_y)/Delta
    var_a = S_XX/Delta
    var_b = S/Delta
    sigma_a = np.sqrt(var_a)
    sigma_b = np.sqrt(var_b)
    '''

    # Numerical recipe code 15.2 (https://numerical.recipes/book.html)
    # Translated to python by copilot and checked.
    ndata = len(x)
    x = np.array(x)
    y = np.array(y)
    sigma_y = np.array(sigma_y)
    
    ss = 0.0
    sx = 0.0
    sy = 0.0
    st2 = 0.0
    b = 0.0
    chi2 = 0.0
    q = 1.0 # Estimator for goodness of fit.
    '''
        From Numerical Recipe 15.2
        If Q is larger than, say, 0.1,then the goodness-of-fit is believable. 
        If it is larger than, say, 0.001, then the fit may be acceptable if the 
        errors are nonnormal or have been moderately underestimated. If Q is 
        less than 0.001, then the model and/or estimation procedure can rightly
        be called into question. In this latter case, turn to 15.7 to proceed 
        further.
    '''

    for i in range(ndata):
        wt = 1.0 / (sigma_y[i] ** 2)
        ss += wt
        sx += x[i] * wt
        sy += y[i] * wt

    sxoss = sx / ss

    for i in range(ndata):
        t = (x[i] - sxoss) / sigma_y[i]
        st2 += t * t
        b += t * y[i] / sigma_y[i]

    b /= st2
    a = (sy - sx * b) / ss
    sigma_a = np.sqrt((1.0 + sx * sx / (ss * st2)) / ss)
    sigma_b = np.sqrt(1.0 / st2)

    for i in range(ndata):
        chi2 += ((y[i] - a - b * x[i]) / sigma_y[i]) ** 2

    if ndata > 2:
        q = gammaincc(0.5 * (ndata - 2), 0.5 * chi2)

    return {
        "intercept": (a, sigma_a),
        "slope": (b, sigma_b),
        "chi2" : chi2,
        "q" : q,
    }


def extract_elements_from_bool_list(var_ar,bool_ar):
    '''
        Extract elements from var_ar array according to bool criterion in bool_ar
    '''
    return [ value for i,value in enumerate(var_ar) if bool_ar[i] ]

# def compute_specific_humidy(RH,T,p):
#     '''
#         Compute the specific humidity q starting from the relative humidity RH, 
#         the air temperature T, the air pressure p)
#     '''
#     # Use Tetens formula for computation of saturation vapor pressure e_s
#     e_s = 6.122*np.exp((17.67*T)/(T+243.5))
#     # Compute vapor pressure e
#     e = RH*e_s/100
#     # Compute the specific humidity
#     q = 0.633*e/(p-0.378*e)

#     return q

def test_if_dataset_empty(ds: xr.Dataset):
    for var_name, da in ds.data_vars.items():
        if da.size == 0:
            raise ValueError(f"Variable '{var_name}' is empty.")
        if np.isnan(da.values).all():
            raise ValueError(f"Variable '{var_name}' contains only NaNs. Dims: {da.dims}")


def get_pandas_intervals_from_params(start : np.float64, end : np.float64, n_intervals : np.int16 = 20, extend_to_pos_inf : bool = True, extend_to_neg_inf : bool = True):
    """
    Generate a `pandas.IntervalIndex` object with evenly spaced intervals between a specified range.
    """
    intervals = pd.interval_range(start,end,periods=n_intervals)
    if extend_to_pos_inf:
        intervals = intervals.append(pd.IntervalIndex.from_tuples([(end,np.inf)])).sort_values()
    if extend_to_neg_inf:
        intervals = intervals.append(pd.IntervalIndex.from_tuples([(-np.inf,start)])).sort_values()
    return intervals

def get_pandas_intervals_from_list(intervals_list : tuple):
    """
    Create a `pandas.IntervalIndex` object from a list of interval tuples.
    """
    return pd.IntervalIndex.from_tuples(intervals_list)


import pandas as pd

def get_season(period):
    """
    Assigns a season ('DJF', 'MAM', 'JJA', 'SON') to a given period based on its start and end dates.
    """
    start, end = period
    if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp):
        raise ValueError("Both start and end must be pd.Timestamp objects.")

    # Check the month of the start date to determine the season
    month = start.month
    if month in [12, 1, 2]:
        return 'DJF'  # December, January, February
    elif month in [3, 4, 5]:
        return 'MAM'  # March, April, May
    elif month in [6, 7, 8]:
        return 'JJA'  # June, July, August
    elif month in [9, 10, 11]:
        return 'SON'  # September, October, November
    else:
        raise ValueError("Invalid month in the start date.")
