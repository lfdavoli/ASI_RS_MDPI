### Packages ###

import pandas as pd
import sys
import os
from pathlib import Path 
import numpy as np

# Official COARE3.6
from coare36vn_zrf_et import coare36vn_zrf_et
# Unofficial COARE4
from coare4_mod import coare4_mod

### Constants ###
# Specific heat dry air at const p
c_p = 1005 #[J kg-1 K-1]
# Specific gas constant for dry air
R_d = 287.05 #[J/(kg K)]
# Specific gas constant for water vapor
R_v = 461.52 #[J/(kg K)]
# Water vaporization specific latent heat (Stull)
L_v = 2500000 #2268000 #[J/kg] 
# L_v/R_v, Specific heat moist air at const p (source: Stull, pag. 89, https://www.eoas.ubc.ca/books/Practical_Meteorology/prmet/PracticalMet_WholeBook-v1_00b.pdf)
c_water = 5423 #[K] 
# Gravit accel
g = 9.81 #[m s-2]
# Von Karman constant
kappa = 0.4 #[1]
# epsilon = R_d/R_v
eps = 0.622 #[g/g]
# beta ~ (R_v/R_d)-1
beta = 0.608 
# saturation vapor pressure 
e0 = 611.3 #[Pa]
# Water freezing temperature
T0 = 273.15 # [K]
# Constants of Tetens' formula for saturated air water content (eq 4.2, Stull, pag. 89, https://www.eoas.ubc.ca/books/Practical_Meteorology/prmet/PracticalMet_WholeBook-v1_00b.pdf)
b = 17.2694
T1 = T0 #[K]
T2 = 35.86 #[K]



### Functions ###

def compute_specific_humidity(d,sp): # 
    """
    Calculates specific humidity q from the dew-point temperature d, the temperature T and the surface pressure sp. 
    Source: equation 4.24, Pg 96 Practical Meteorolgy (Roland Stull)
    from https://www.eoas.ubc.ca/books/Practical_Meteorology/prmet/PracticalMet_WholeBook-v1_00b.pdf
    """
    q = (eps * e0 * np.exp(c_water * (1/T0 - 1/d)))/sp # [g/kg] 
    return q #[g/g]

def compute_relative_humidity(T,d):
    """
    Calculates relative humidity RH (%) from  the air temperature T and the dew-point temperature d. 
    Source: equation 4.2+4.14a, Pg 89 Practical Meteorolgy (Roland Stull)
    from https://www.eoas.ubc.ca/books/Practical_Meteorology/prmet/PracticalMet_WholeBook-v1_00b.pdf
    """
    e = e0 * np.exp(c_water * (1/T0 - 1/d)) # [Pa] (4.1b) 
    e_s = e0*np.exp(b*(T-T1)/(T-T2)) # [Pa]
    RH = e/e_s*100 # [%]
    return RH

def compute_virtual_temperature(T,q):
    '''
        Compute virtual temperature T_v from surface air temperature T, specific humidity q
    '''
    T_v = T * (1+beta*q)

    return T_v

def compute_air_density(sp,T_v):
    '''
        Compute air density rho given surface pressure sp and virtual temperature T_v (ideal gas law)
    '''
    rho = sp / (R_d * T_v)

    return rho

def compute_buoyancy_flux(rho,T_v,T,H_s,H_l,q):
    '''
        Compute the buoyancy flux B at the surface starting from air-density rho, 
        virtual air temperature T_v, sensible and latent heat fluxes, 
        specific humidity q. 
        Positive B => positive heat flux into the ocean => stable conditions.
        The reference (De Szoeke) computes B as a kinematic flux, so I have to 
        multiply by rho*c_p to obtain a dynamic flux (as the one in B_ML_50)
        Source: De Szoeke et al. (2020) for the comput of the kinetic flux wT, 
        and then multiplied by rho*c_p according to glossary of meteorology to 
        obtain a density flux (virt temperature per unit of area per second),
        as the one compute by COARE.
    '''
    
    B = c_p * ( H_s*(1+beta*q)/c_p + H_l*beta*T/L_v )

    return B


def compute_buoyancy_flux_Q_contribution(rho,T_v,H_s,H_l,q):
    '''
        Returns the sensible heat term contributing to buoyancy flux B at the surface starting from air-density rho, 
        virtual air temperature T_v, sensible heat flux, 
        specific humidity q
    '''
    
    Q = g/(rho*T_v) * H_s*(1+beta*q)/c_p 

    return Q


def compute_buoyancy_flux_L_contribution(rho,T_v,T,H_s,H_l,q):
    '''
        Returns the latent heat term contributing to buoyancy flux B at the surface starting from air-density rho, 
        virtual air temperature T_v, sensible heat flux, 
        specific humidity q
    '''
    
    L = g/(rho*T_v) * H_l*beta*T/L_v 

    return L


def compute_Obukhov_length(T_v,u_star,B,rho):
    '''
        Returns Obukhov length from friction velocity u_star and B (i.e. virtual temperature dynamic flux B) 
        as L = - T_v * u_star^3 / (kappa * B / (rho*c_p) ).
        Sources: Stull (1988) for L formula + De Szoeke (2020) for wT_v formula + gloss of meteo for dyn flux.
        Note: B is positive (negative) for unstable (stable) conditions. Depending on how B is defined
        the minus signs is removed.
    '''
    L_Obukhov = T_v * np.power(u_star,3) * rho * c_p / (kappa * g * B)

    return L_Obukhov

def compute_rescaled_wind(U,lat):
    '''
        Rescale wind speed with sin(phi) to take into account the meridional 
        variability in characteristic length scale induced by Rossby number.
    '''
    return np.abs(U*np.sin(np.deg2rad(lat)))

def get_coare_era5(inputs_df: pd.DataFrame,coare_version : str = 'coare4'):
    if coare_version == 'coare4':
        inputs_complet = dict()
        
        inputs_complet['zu'] = 10*np.ones_like(inputs_df.u10_era5)  # zu  (float): height of wind speed (m)
        inputs_complet['zt'] = 2*np.ones_like(inputs_df.t2m_era5)  # zt  (float): height of air temperature (m)
        inputs_complet['zq'] = 2*np.ones_like(inputs_df.d2m_era5)  # zq  (float): height of relative humidity (m)
        inputs_complet['zi'] = inputs_df.blh_era5  # zi  (float): PBL height (m) (default = 600m)
        inputs_complet['lat'] = inputs_df.latitude # lat (float): latitude (default = +45 N)

        inputs_complet['u'] = np.sqrt(inputs_df.u10_era5**2 + inputs_df.v10_era5**2)    # u   (float): relative wind speed (m/s) at height zu
        inputs_complet['t'] = inputs_df.t2m_era5 - 273.16   # t   (float): bulk air temperature (degC) at height zt
        inputs_complet['rh'] = compute_relative_humidity(inputs_df.t2m_era5,inputs_df.d2m_era5)/100  # rh  (float): relative humidity (#) at height zq [0-100] (given in %)
        inputs_complet['P'] = inputs_df.sp_era5 * 10**(-2)   # P   (float): surface air pressure (mb) (default = 1015)
        inputs_complet['ts'] = inputs_df.sst_era5 - 273.16 # ts  (float): water temperature (degC) see jcool below
        inputs_complet['Rs'] = inputs_df.ssrd_era5/3600 # Rs  (float): downward shortwave radiation (W/m^2) (default = 150)
        inputs_complet['Rl'] = inputs_df.strd_era5/3600 # Rl  (float): downward longwave radiation (W/m^2) (default = 370)
        print('\t Run coare')
        return coare4_mod(inputs_complet)
    
    elif coare_version == 'coare36vn_zrf_et':
        coare_output_ar = coare36vn_zrf_et(
            np.sqrt(inputs_df.u10_era5.to_numpy()**2 + inputs_df.v10_era5.to_numpy()**2), 
            10*np.ones_like(inputs_df.u10_era5), 
            inputs_df.t2m_era5.to_numpy() - 273.16, 
            2*np.ones_like(inputs_df.t2m_era5), 
            compute_relative_humidity(inputs_df.t2m_era5,inputs_df.d2m_era5).to_numpy()/100, 
            2*np.ones_like(inputs_df.d2m_era5), 
            inputs_df.sp_era5.to_numpy() * 10**(-2), 
            inputs_df.sst_era5.to_numpy() - 273.16, 
            inputs_df.ssrd_era5.to_numpy()/3600, 
            inputs_df.strd_era5.to_numpy()/3600, 
            inputs_df.latitude.to_numpy(), 
            inputs_df.longitude.to_numpy(), 
            inputs_df['time_era5'].astype('datetime64[ns]').dt.dayofyear.to_numpy(), 
            inputs_df.blh_era5.to_numpy(), 
            (inputs_df.crr_era5.to_numpy() + inputs_df.lsrr_era5.to_numpy())*3600, 
            35*np.ones_like(inputs_df.u10_era5.to_numpy()) # Whatever works
            )         

        coare_output_dict = dict()
        output_columns = ['usr','tau','hsb','hlb','hbb','hsbb','hlwebb','tsr','qsr','zo','zot','zoq','Cd','Ch','Ce','L','zeta','dT_skinx','dq_skinx','dz_skin','Urf','Trf','Qrf','RHrf','UrfN','TrfN','QrfN','lw_net','sw_net','Le','rhoa','UN','U10','U10N','Cdn_10','Chn_10','Cen_10','hrain','Qs','Evap','T10','T10N','Q10','Q10N','RH10','P10','rhoa10','gust','wc_frac','Edis']
        for i, column_name in enumerate(output_columns):
            coare_output_dict[column_name] = coare_output_ar[:, i]

        return coare_output_dict
