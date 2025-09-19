#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:35:30 2023

@author: amore
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import datetime
from dateutil.relativedelta import relativedelta
from scipy import optimize
import sys
sys.path.append(r"/home/amore/Dropbox/Documents/library_code/GEOD2GEOC")
from numpy import meshgrid
from scipy.interpolate import griddata
from chaosmagpy import load_CHAOS_matfile
from geod2geoc import geod2geocconv
import os
from spacepy import pycdf
import brewer2mpl
import seaborn
import julian
import pandas as pd
import pyIGRF
from datetime import timedelta, date
import chaosmagpy as cp
cp.basicConfig['file.RC_index'] = '/home/amore/Documents/00Data/RC_1997-2025_augmented.dat'


FILEPATH_CHAOS = '/home/amore/Dropbox/Documents/library_methods_models/CHAOS7_Model/data/CHAOS-8.2.mat' 

def year_fraction(dater):
    start = datetime.date(dater.year, 1, 1).toordinal()
    year_length = datetime.date(dater.year+1, 1, 1).toordinal() - start
    return dater.year + float(dater.toordinal() - start) / year_length

def datetime_to_decimal_year(dt):
    year_start = datetime.datetime(dt.year, 1, 1)
    year_end = datetime.datetime(dt.year + 1, 1, 1)
    year_length = (year_end - year_start).total_seconds()
    seconds_into_year = (dt - year_start).total_seconds()
    return dt.year + seconds_into_year / year_length


def chaos(date, la, lo, alt):
    
    R_REF = 6371.2   
    # Ensure 'date' is a datetime object
    # Work out year 2000 in Julian date
    year2000 = datetime.datetime(year = 2000, month = 1, day = 1, hour = 0, minute = 0, second = 0)
    jd2000 = julian.to_jd(year2000, fmt='jd') 
  # work out epoch in julian date. This looks messy because datetime is an annoying library: epoch is a float that needs to be converted to datetime 
  # and then is converted back to float for subtraction purposes.                         
    year = int(date)
    rem = date - year
    base = datetime.datetime(year, 1, 1)
    result = base + datetime.timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds()* rem)  
    jd = julian.to_jd(result, fmt='jd')
    # subtract the two julian dates to obtain MJD2000
    finalres = jd - jd2000
    alpha = la
    
    alpha_rad = np.radians(alpha) # degrees to radians for geod2geoc, alpha is latitude
    [radius, theta_rad] = geod2geocconv(alpha_rad, alt/1000) # geodetic to geocentric in radians
    # convert back to degrees honestly why is this in radians in the first place
    theta_deg = np.degrees(theta_rad)
#    print(90-theta[0])
   
    phi = lo # longitude in deg   
    time = finalres  # time in mjd2000
    
    # load the CHAOS model
    model = load_CHAOS_matfile(FILEPATH_CHAOS)
    
    B_core = model.synth_values_tdep(time, radius, theta_deg, phi)
    B_crust = model.synth_values_static(radius, theta_deg, phi, nmax=110)

    # complete internal contribution
    B_theta_int =  B_core[1]
    B_phi_int =  B_core[2]
    B_radius_int =  B_core[0]

    # Convert spherical coordi to geocentric B_XYZ
    B_geocX = B_theta_int    
    B_geocY = B_phi_int
    B_geocZ = B_radius_int
    
    # Rotate geocentric B_XYZ  to geodetic B_XYZ
    psi = np.sin(np.deg2rad(alpha))*np.sin(theta_rad) - np.cos(np.deg2rad(alpha))*np.cos(theta_rad)
    
    B_geodX = -np.cos(psi)*B_geocX - np.sin(psi)*B_geocZ # X    
    B_geodY = B_geocY    
    B_geodZ = +np.sin(psi)*B_geocX - np.cos(psi)*B_geocZ # Z   
      
    B_D = -np.degrees(np.arctan(B_geocY/B_geocX))     
    B_geodH = np.sqrt(B_geodX**2 + B_geodY**2)
    B_F = np.sqrt(B_geocX**2 + B_geocY**2 + B_geocZ**2)
    
    return B_geodX, B_geodY, B_geodZ



def chaos_ext(date, la, lo, alt): 
    

    R_REF = 6371.2
  
    # Work out year 2000 in Julian date
    year2000 = datetime.datetime(year = 2000, month = 1, day = 1, hour = 0, minute = 0, second = 0)
    jd2000 = julian.to_jd(year2000, fmt='jd') 
  # work out epoch in julian date. This looks messy because datetime is an annoying library: epoch is a float that needs to be converted to datetime 
  # and then is converted back to float for subtraction purposes.                         
    year = int(date)
    rem = date - year
    base = datetime.datetime(year, 1, 1)
    result = base + datetime.timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds()* rem)  
    jd = julian.to_jd(result, fmt='jd')
    # subtract the two julian dates to obtain MJD2000
    finalres = jd - jd2000
    alpha = la
    
    alpha_rad = np.radians(alpha) # degrees to radians for geod2geoc, alpha is latitude
    [radius, theta_rad] = geod2geocconv(alpha_rad, alt/1000) # geodetic to geocentric in radians
    # convert back to degrees honestly why is this in radians in the first place
    theta_deg = np.degrees(theta_rad)
#    print(90-theta[0])
   
    phi = lo  # longitude in deg   
    time = finalres  # time in mjd2000
    
    # load the CHAOS model
    model = load_CHAOS_matfile(FILEPATH_CHAOS)    
  
    # print('Computing field due to external sources, incl. induced field: GSM.')
    B_gsm = model.synth_values_gsm(time, radius, theta_deg, phi, source='all')
    
    # # print('Computing field due to external sources, incl. induced field: SM.')
    B_sm = model.synth_values_sm(time, radius, theta_deg, phi, source='all')

    # # complete external field contribution
    B_radius_ext = B_gsm[0] + B_sm[0]
    B_theta_ext = B_gsm[1] + B_sm[1]
    B_phi_ext = B_gsm[2] + B_sm[2]
    
    # Convert spherical coordi to geocentric B_XYZ
    B_geocX = B_theta_ext    
    B_geocY = B_phi_ext
    B_geocZ = B_radius_ext
    
    # Rotate geocentric B_XYZ  to geodetic B_XYZ
    psi = np.sin(np.deg2rad(alpha))*np.sin(theta_rad) - np.cos(np.deg2rad(alpha))*np.cos(theta_rad)
    
    B_geodX_ext = -np.cos(psi)*B_geocX - np.sin(psi)*B_geocZ # X    
    B_geodY_ext = B_geocY    
    B_geodZ_ext = +np.sin(psi)*B_geocX - np.cos(psi)*B_geocZ # Z     
      
    
    # B_D = -np.degrees(np.arctan(B_geocY_ext/B_geocX_ext))     
    # B_geodH = np.sqrt(B_geodX_ext**2 + B_geodY_ext**2)
    # B_F = np.sqrt(B_geocX_ext**2 + B_geocY_ext**2 + B_geocZ_ext**2)

    return B_geodX_ext, B_geodY_ext, B_geodZ_ext
    


