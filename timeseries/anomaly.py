# Function to calculate temperature and salinity anomaly using world ocean atlas data
# Created by Jace Marquardt
# Last updated 01-08-2026

import numpy as np

def temperature_anomaly(transect_dict, woa_dict):
    """
    Returns a dict of temperature anomalies keyed by transect name.
    
    Inputs:

        transect_dict : dict
            A dictionary where each key is a transect name and each value is another
            dictionary containing 'temp' (temperature data) and 'mean_time'.

        woa_dict : dict
            A dictionary where each key is a month number (as string) and each value
            is the corresponding World Ocean Atlas temperature data for that month.

    Outputs:

        temp_anomaly : dict
            A dictionary where each key is a transect name and each value is another
            dictionary containing 'temp_anomaly' and 'mean_time'.
    """

    temp_anomaly = {}
    woa_map = {str(k): v for k, v in woa_dict.items()}

    for key, value in transect_dict.items():
        month_number = key.split('_')[0]

        if month_number in woa_map:
            temp_anomaly[key] = {
                "temp_anomaly": value["temp"][:, np.newaxis] - woa_map[month_number],
                "mean_time": value["mean_time"],
            }

    return temp_anomaly

def salinity_anomaly(transect_dict, woa_dict):
    """
    Returns a dict of salinity anomalies keyed by transect name.
    
    Inputs:

        transect_dict : dict
            A dictionary where each key is a transect name and each value is another
            dictionary containing 'salt' (salinity data) and 'mean_time'.

        woa_dict : dict
            A dictionary where each key is a month number (as string) and each value
            is the corresponding World Ocean Atlas salinity data for that month.

    Outputs:

        salt_anomaly : dict
            A dictionary where each key is a transect name and each value is another
            dictionary containing 'salt_anomaly' and 'mean_time'.
    """

    salt_anomaly = {}
    woa_map = {str(k): v for k, v in woa_dict.items()}

    for key, value in transect_dict.items():
        month_number = key.split('_')[0]

        if month_number in woa_map:
            salt_anomaly[key] = {
                "salt_anomaly": value["salt"][:, np.newaxis] - woa_map[month_number],
                "mean_time": value["mean_time"],
            }

    return salt_anomaly