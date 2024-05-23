"""
This code reads data from World Ocean Database (WOD) NetCDF files, processes it, and saves the processed data 
as new NetCDF files. The script handles various data attributes including depth, pressure, temperature, and salinity,
organizing them into structured lists and then into an Xarray Dataset before saving.

Steps followed in the script:
1. Initialize the WODReader class.
2. Read raw data from WOD NetCDF files.
3. Process the data, extracting observational and metadata.
4. Organize the data into an Xarray Dataset.
5. Save the Dataset as a new NetCDF file.

Functions and methods in the script:
1. WODReader.is_variable_too_big: Checks if the size of the data variables exceeds a certain threshold.
2. WODReader.get_date: Extracts and parses date and time information from the dataset.
3. WODReader.initialize_variables: Initializes dictionaries for data storage.
4. WODReader.read_raw_data: Reads and processes raw data from WOD NetCDF files.
5. WODReader.create_dataset: Creates an Xarray Dataset from processed data and saves it as a new NetCDF file.
6. WODReader.run: Executes the data processing workflow.
7. main: Main function to run the WODReader for multiple data files.
"""

import xarray as xr
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime
import threading
import sys


class WODReader:
    def __init__(self):
        """Initialize the WODReader class"""
        pass

    def is_variable_too_big(self, data_lists):
        """
        Check if the size of the data variables exceeds a certain threshold.

        Args:
            data_lists (dict): Dictionary containing data variables.

        Returns:
            bool: True if the total size exceeds 150 MB, False otherwise.
        """
        total_sum = 0
        for key in data_lists:
            total_sum += sys.getsizeof(data_lists[key])
        size_mb = total_sum / (1024 * 1024)
        # print(f"size mb: {size_mb}")
        if size_mb >= 150:
            return True
        else:
            return False

    def get_date(self, ds):
        """
        Extract and parse the date and time information from the dataset.

        Args:
            ds (xarray.Dataset): Dataset containing date and time information.

        Returns:
            tuple: A tuple containing year, month, day, hour, and minute.
        """
        complete_date = str(ds["date"].values)
        gmt_time = str(ds["GMT_time"].values)
        year = int(complete_date[:4])
        month = int(complete_date[4:6])
        day = int(complete_date[6:])
        hour = int(float(gmt_time) // 1)
        minute = int((float(gmt_time) % 1) * 60)

        return year, month, day, hour, minute

    def initialize_variables(self):
        """
        Initialize dictionaries to store string attributes, observational attributes,
        and data lists.

        Returns:
            tuple: A tuple containing string_attrs, obs_attrs, data_lists, and i.
        """
        string_attrs = {
            "orig_profile_ID": "wod_unique_cast",
            "orig_cruise_id": "originators_cruise_identifier",
            "access_no": "Access_no",
            "platform": "Platform",
            "station_no": "Orig_Stat_Num",
            "instrument_type": "dataset",
            "lat": "lat",
            "lon": "lon",
            "bottom_depth": "Bottom_Depth",
            "datestr": "",
            "timestamp": "",
            "orig_filename": "",
            "shallowest_depth": "",
            "deepest_depth": "",
            "parent_index": "",
        }
        obs_attrs = {
            "depth": "z",
            "press": "Pressure",
            "temp": "Temperature",
            "psal": "Salinity",
            "depth_flag": "z_WODflag",
            "temp_flag": "Temperature_WODflag",
            "psal_flag": "Salinity_WODflag",
        }
        data_lists = {attr: [] for attr in list(string_attrs.keys()) + list(obs_attrs.keys())}
        i = 0
        return string_attrs, obs_attrs, data_lists, i

    def read_raw_data(self, datasets, data_path, save_path, file_counter):
        """
        Read raw data from NetCDF files, process it, and save as new NetCDF files.

        Args:
            datasets (list): List of NetCDF files.
            data_path (str): Path to the original data.
            save_path (str): Path to save the processed data.
            file_counter (int): Counter to track processed files.
        """
        string_attrs, obs_attrs, data_lists, i = self.initialize_variables()
        for file in tqdm(datasets, colour="GREEN"):
            ds = xr.open_dataset(file)

            # get observational data
            depth_list = None
            press_list = None
            temp_list = None
            psal_list = None
            if "z" and "Pressure" in ds:
                depth_list = ds["z"].values
                press_list = ds["Pressure"].values
                data_lists["depth"].extend(depth_list)
                data_lists["press"].extend(press_list)
            elif "z" in ds and "Pressure" not in ds:
                depth_list = ds["z"].values
                press_list = [np.nan] * len(depth_list)
                data_lists["depth"].extend(depth_list)
                data_lists["press"].extend(press_list)
            elif "Pressure" in ds and "z" not in ds:
                press_list = ds["Pressure"].values
                depth_list = [np.nan] * press_list
                data_lists["press"].extend(press_list)
                data_lists["depth"].extend(depth_list)
            else:
                continue

            if "z_WODflag" in ds:
                data_lists["depth_flag"].extend(ds["z_WODflag"].values)
            else:
                data_lists["depth_flag"].extend([np.nan] * depth_list)

            if len(depth_list) > 1:
                data_lists["shallowest_depth"].append(min(depth_list[depth_list != 0]))
            else:
                data_lists["shallowest_depth"].append(min(depth_list))
            data_lists["deepest_depth"].append(max(depth_list))
            data_lists["parent_index"].extend([i] * len(depth_list))

            if "Salinity" in ds:
                psal_list = ds["Salinity"].values
                data_lists["psal"].extend(ds["Salinity"].values)
                if "Salinity_WODflag" in ds:
                    data_lists["psal_flag"].extend(ds["Salinity_WODflag"].values)
                else:
                    data_lists["psal_flag"].extend([np.nan] * len(psal_list))
            else:
                psal_list = [np.nan] * len(depth_list)
                data_lists["psal"].extend(psal_list)
                data_lists["psal_flag"].extend(psal_list)

            if "Temperature" in ds:
                temp_list = ds["Temperature"].values
                data_lists["temp"].extend(temp_list)
                if "Temperature_WODflag" in ds:
                    data_lists["temp_flag"].extend(ds["Temperature_WODflag"].values)
                else:
                    data_lists["temp_flag"].extend([np.nan] * temp_list)
            else:
                temp_list = [np.nan] * len(depth_list)
                data_lists["temp"].extend(temp_list)
                data_lists["temp_flag"].extend(temp_list)

            # get metadata
            if "orig_filename" in ds and len(ds["orig_filename"]) > 0:
                data_lists["orig_filename"].append(file)
            else:
                data_lists["orig_filename"].append(np.nan)

            if "date" in ds and "GMT_time" in ds:
                try:
                    year, month, day, hour, minute = self.get_date(ds)
                    datestr = datetime(year, month, day, hour, minute)
                    data_lists["timestamp"].append(datestr.timestamp())
                    data_lists["datestr"].append(datetime.strftime(datestr, "%Y/%m/%d %H:%M:%S"))
                except Exception:
                    data_lists["datestr"].append(np.nan)
                    data_lists["timestamp"].append(np.nan)
            else:
                data_lists["datestr"].append(np.nan)
                data_lists["timestamp"].append(np.nan)

            for key, value in string_attrs.items():
                if value != "":
                    if value in ds:
                        data_lists[key].append(ds[value].values)
                    else:
                        data_lists[key].append(np.nan)

            # check if the file is too big. If so, save the file and start again
            i += 1
            if self.is_variable_too_big(data_lists):
                for attr in data_lists:
                    print(f"attr: {attr} - len: {len(data_lists[attr])}")
                print("-" * 100)
                self.create_dataset(data_path, save_path, data_lists, string_attrs, file_counter)
                string_attrs, obs_attrs, data_lists, i = self.initialize_variables()
                file_counter += 1
        self.create_dataset(data_path, save_path, data_lists, string_attrs, file_counter)

    def create_dataset(self, data_path, save_path, data_list, string_attrs, file_counter):
        """
        Create a new xarray Dataset from processed data and save it as a NetCDF file.

        Args:
            data_path (str): Path to the original data.
            save_path (str): Path to save the processed data.
            data_list (dict): Dictionary containing processed data.
            string_attrs (dict): Dictionary containing string attributes.
            file_counter (int): Counter to track processed files.
        """
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        os.chdir(save_path)
        ds = xr.Dataset(
            coords=dict(
                timestamp=(["profile"], data_list["timestamp"]),
                lat=(
                    [
                        "profile",
                    ],
                    data_list["lat"],
                ),
                lon=(
                    [
                        "profile",
                    ],
                    data_list["lon"],
                ),
            ),
            data_vars=dict(
                **{
                    attr: xr.DataArray(data_list[attr], dims=["profile"])
                    for attr in string_attrs.keys()
                    if attr not in ["lat", "lon", "timestamp", "parent_index"]
                },
                # measurements
                parent_index=xr.DataArray(data_list["parent_index"], dims=["obs"]),
                depth=xr.DataArray(data_list["depth"], dims=["obs"]),
                depth_flag=xr.DataArray(data_list["depth_flag"], dims=["obs"]),
                press=xr.DataArray(data_list["press"], dims=["obs"]),
                temp=xr.DataArray(data_list["temp"], dims=["obs"]),
                temp_flag=xr.DataArray(data_list["temp_flag"], dims=["obs"]),
                psal=xr.DataArray(data_list["psal"], dims=["obs"]),
                psal_flag=xr.DataArray(data_list["psal_flag"], dims=["obs"]),
            ),
            attrs=dict(
                dataset_name="WOD_2022",
                creation_date=str(datetime.now().strftime("%Y-%m-%d %H:%M")),
            ),
        )
        ds.to_netcdf(f"WOD_2022_{file_counter}_raw.nc")
        os.chdir(data_path)

    def run(self, data_path, save_path, datasets):
        """
        Main function to run the data processing pipeline.

        Args:
            data_path (str): Path to the original data.
            save_path (str): Path to save the processed data.
            datasets (list): List of NetCDF files.
        """
        os.chdir(data_path)
        file_counter = 0
        self.read_raw_data(datasets, data_path, save_path, file_counter)


def main():
    """Main function to initiate the data processing pipeline."""
    wod_reader = WODReader()
    data_path = "/mnt/storage6/caio/AW_CAA/CTD_DATA/WOD_2022/original_data/netcdf/ocldb1663004073.18805.CTD"
    save_path = "/mnt/storage6/caio/AW_CAA/CTD_DATA/WOD_2022/ncfiles_raw"
    os.chdir(data_path)
    datasets = [file for file in os.listdir() if file.endswith(".nc")]

    wod_reader.run(data_path=data_path, save_path=save_path, datasets=datasets)


if __name__ == "__main__":
    main()
