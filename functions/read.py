import xarray as xr
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime
import sys


def is_variable_too_big(data_lists):
    total_sum = 0
    for key in data_lists:
        total_sum += sys.getsizeof(data_lists[key])
    size_mb = total_sum / (1024 * 1024)
    # print(size_mb)
    if size_mb >= 500:
        return True
    else:
        return False


def get_date(ds):
    complete_date = str(ds['date'].values)
    gmt_time = str(ds['GMT_time'].values)
    year = int(complete_date[:4])
    month = int(complete_date[4:6])
    day = int(complete_date[6:])
    hour = int(float(gmt_time) // 1)
    minute = int((float(gmt_time) % 1) * 60)

    return year, month, day, hour, minute


def initialize_variables():
    string_attrs = {'orig_profile_ID': 'wod_unique_cast',
                    'orig_cruise_id': 'originators_cruise_identifier',
                    'access_no': 'Access_no',
                    'platform': 'Platform',
                    'station_no': 'Orig_Stat_Num',
                    'instrument_type': 'dataset',
                    'lat': 'lat',
                    'lon': 'lon',
                    'bottom_depth': 'Bottom_Depth',
                    'datestr': '',
                    'timestamp': '',
                    'orig_filename': '',
                    'shallowest_depth': '',
                    'deepest_depth': '',
                    'parent_index': '',
                    }
    obs_attrs = {'depth': 'z',
                 'press': 'Pressure',
                 'temp': 'Temperature',
                 'psal': 'Salinity',
                 'depth_flag': 'z_WODflag',
                 'temp_flag': 'Temperature_WODflag',
                 'psal_flag': 'Salinity_WODflag',
                 }
    data_lists = {attr: [] for attr in list(string_attrs.keys()) + list(obs_attrs.keys())}
    i = 0
    return string_attrs, obs_attrs, data_lists, i


def read_raw_data(datasets, data_path, save_path, file_counter):
    string_attrs, obs_attrs, data_lists, i = initialize_variables()
    for ds, filename in tqdm(datasets, colour='GREEN'):
        if 'z' in ds:
            if len(ds['z'].values) > 1:
                data_lists['shallowest_depth'].append(min(ds['z'][ds['z'] != 0]))
            else:
                data_lists['shallowest_depth'].append(min(ds['z']))
            data_lists['deepest_depth'].append(max(ds['z']))
            data_lists['parent_index'].extend([i] * len(ds['z']))
        elif 'Pressure' in ds:
            data_lists['depth'].extend([np.nan] * len(ds['Pressure']))
            data_lists['parent_index'].extend([i] * len(ds['Pressure']))
        else:
            continue

        if 'date' in ds and 'GMT_time' in ds:
            year, month, day, hour, minute = get_date(ds)
            datestr = datetime(year, month, day, hour, minute)
            data_lists['timestamp'].append(datestr.timestamp())
            data_lists['datestr'].append(datetime.strftime(datestr, "%Y/%m/%d %H:%M:%S"))
        else:
            continue

        for key, value in string_attrs.items():
            if value != '':
                try:
                    data_lists[key].append(ds[value].values)
                except KeyError:
                    data_lists[key].append(np.nan)
        for key, value in obs_attrs.items():
            if value != '':
                try:
                    data_lists[key].extend(ds[value].values)
                except KeyError:
                    data_lists[key].extend([np.nan] * len(ds['z']))

        data_lists['orig_filename'].append(filename)
        i += 1
        if is_variable_too_big(data_lists):
            create_dataset(data_path, save_path, data_lists, string_attrs, file_counter)
            string_attrs, obs_attrs, data_lists, i = initialize_variables()
            file_counter += 1
    create_dataset(data_path, save_path, data_lists, string_attrs, file_counter)


def create_dataset(data_path, save_path, data_list, string_attrs, file_counter):
    os.chdir(data_path[:data_path.rfind("/")])
    os.chdir("../")
    dataset = os.getcwd()
    dataset = dataset[dataset.rfind("/") + 1:]

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    os.chdir(save_path)
    ds = xr.Dataset(
        coords=dict(
            timestamp=(['profile'], data_list['timestamp']),
            lat=(['profile', ], data_list['lat']),
            lon=(['profile', ], data_list['lon']),
        ),
        data_vars=dict(
            **{attr: xr.DataArray(data_list[attr], dims=['profile']) for attr in string_attrs.keys() if
               attr not in ['lat', 'lon', 'timestamp', 'parent_index']},
            # measurements
            parent_index=xr.DataArray(data_list['parent_index'], dims=['obs']),
            depth=xr.DataArray(data_list['depth'], dims=['obs']),
            depth_flag=xr.DataArray(data_list['depth_flag'], dims=['obs']),
            press=xr.DataArray(data_list['press'], dims=['obs']),
            temp=xr.DataArray(data_list['temp'], dims=['obs']),
            temp_flag=xr.DataArray(data_list['temp_flag'], dims=['obs']),
            psal=xr.DataArray(data_list['psal'], dims=['obs']),
            psal_flag=xr.DataArray(data_list['psal_flag'], dims=['obs']),
        ),
        attrs=dict(
            dataset_name=dataset,
            creation_date=str(datetime.now().strftime("%Y-%m-%d %H:%M")),
        ),
    )
    ds.to_netcdf(f"WOD_2022_{file_counter}_raw.nc")


def read_WOD(data_path, save_path):
    os.chdir(data_path)
    datasets = [[xr.open_dataset(file), file] for file in os.listdir() if file.endswith(".nc")]
    file_counter = 0
    read_raw_data(datasets, data_path, save_path, file_counter)
