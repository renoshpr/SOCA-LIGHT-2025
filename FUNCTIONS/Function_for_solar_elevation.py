#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ## Function for solar elevation

# In[2]:


#######################################################################################
# FUNCTION TO FIND SOLAR ELEVATION AT LOCAL HOUR OF A LATITUDE AND LONGITUDE
import pvlib
import pandas as pd
import numpy as np
from datetime import timedelta
import pytz # Still needed for timezone operations by pandas
from timezonefinder import TimezoneFinder # New import

# Instantiate TimezoneFinder once for efficiency if you plan to call the function many times
# If you only call it rarely, instantiating inside the function is fine too.
# For this example, let's instantiate it globally.
tf = TimezoneFinder()

# def calculate_solar_elevation(date_str, local_hour, latitude, longitude):
#     """
#     Calculates the apparent solar elevation for a given date string, local hour,
#     latitude, and longitude, by inferring the timezone.

#     Args:
#         date_str (str): The date in 'yyyymmdd' format (e.g., '19980115').
#         local_hour (float or int): The local hour (e.g., 10.5 for 10:30 AM, 14 for 2 PM).
#         latitude (float): The latitude in degrees.
#         longitude (float): The longitude in degrees (East is positive, West is negative).

#     Returns:
#         float: The apparent solar elevation in degrees.
#     """
#     # Parse the date string
#     year = int(date_str[:4])
#     month = int(date_str[4:6])
#     day = int(date_str[6:8])

#     # Create a *naive* datetime object for the specified local time
#     naive_time = pd.Timestamp(year=year, month=month, day=day,
#                               hour=int(local_hour), minute=int((local_hour % 1) * 60),
#                               second=int((local_hour * 3600) % 60))

#     # Infer the timezone from latitude and longitude
#     timezone_str = tf.timezone_at(lng=longitude, lat=latitude)

#     if timezone_str is None:
#         #print(f"Warning: Could not determine timezone for Latitude {latitude}, Longitude {longitude}.")
#         #print("Falling back to longitude-based UTC conversion (less accurate, no DST).")
#         # Fallback if no timezone is found (e.g., in open ocean or invalid coordinates)
#         utc_offset_hours = longitude / 15.0
#         time_utc = naive_time - timedelta(hours=utc_offset_hours)
#         time_utc = time_utc.tz_localize('UTC') # Make it UTC-aware
#     else:
#         try:
#             # Localize the naive timestamp to the inferred timezone
#             time_local = naive_time.tz_localize(timezone_str, ambiguous='infer')

#             # Convert the localized time to UTC
#             time_utc = time_local.tz_convert('UTC')

#         except Exception as e:
#             #print(f"An unexpected error occurred during timezone conversion using '{timezone_str}': {e}")
#             #print("Falling back to longitude-based UTC conversion (less accurate, no DST).")
#             # Fallback for any other conversion errors (e.g., pytz issues)
#             utc_offset_hours = longitude / 15.0
#             time_utc = naive_time - timedelta(hours=utc_offset_hours)
#             time_utc = time_utc.tz_localize('UTC') # Make it UTC-aware

#     # Calculate solar position using the UTC time and the provided latitude and longitude
#     solar_position = pvlib.solarposition.get_solarposition(time_utc, latitude, longitude)

#     # Extract the apparent elevation
#     solar_elevation = solar_position['apparent_elevation'].values[0]
#     return solar_elevation
#######################################################################################


#######################################################################################
def calculate_solar_elevation(date_str, local_hour, latitude, longitude):
    """
    Calculates the apparent solar elevation for a given date string, local hour,
    latitude, and longitude.

    This function determines UTC time solely based on the longitude, assuming
    15 degrees of longitude equals 1 hour. It does NOT account for official
    time zones, political boundaries, or Daylight Saving Time (DST).
    Therefore, the 'local_hour' input should represent the standard time
    at that longitude, not necessarily the current official local time.

    Args:
        date_str (str): The date in 'yyyymmdd' format (e.g., '19980115').
        local_hour (float or int): The local hour (e.g., 10.5 for 10:30 AM, 14 for 2 PM).
                                    This is assumed to be the local mean solar time
                                    or standard time based directly on longitude.
        latitude (float): The latitude in degrees.
        longitude (float): The longitude in degrees (East is positive, West is negative).

    Returns:
        float: The apparent solar elevation in degrees.
    """
    # Parse the date string
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    # Create a *naive* datetime object for the specified local time.
    # This naive time is treated as the 'local mean solar time' for the given longitude.
    naive_time_local = pd.Timestamp(year=year, month=month, day=day,
                                    hour=int(local_hour), minute=int((local_hour % 1) * 60),
                                    second=int((local_hour * 3600) % 60),
                                    microsecond=int((local_hour * 3600 * 1000000) % 1000000))

    # Calculate the UTC offset based on longitude only (15 degrees = 1 hour)
    # Longitude is positive for East, negative for West.
    # A positive longitude (East) means the local time is *ahead* of UTC, so we subtract
    # to get UTC. A negative longitude (West) means local time is *behind* UTC, so we add.
    utc_offset_hours = longitude / 15.0

    # Convert the local time to UTC time by applying the longitude-based offset.
    # We subtract the offset because local time (at positive longitude) is ahead of UTC,
    # and we add (effectively subtract negative offset) for negative longitude.
    time_utc_naive = naive_time_local - timedelta(hours=utc_offset_hours)

    # Localize the resulting naive UTC time to be UTC-aware.
    time_utc = time_utc_naive.tz_localize('UTC')

    # Calculate solar position using the UTC time and the provided latitude and longitude
    solar_position = pvlib.solarposition.get_solarposition(time_utc, latitude, longitude)

    # Extract the apparent elevation
    # pvlib returns a pandas Series/DataFrame, so we access the value.
    solar_elevation = solar_position['apparent_elevation'].values[0]
    return solar_elevation
#######################################################################################


# In[ ]:




