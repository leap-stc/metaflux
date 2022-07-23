import os
import pandas as pd
from glob import glob
import xarray
import numpy as np
import argparse

def read_lai(data_dir, meta_dir, stations_dir):
    """
    Read LAI data from a NetCDF4 file for each of the valid station
    """

    # read metadata of stations
    meta = pd.read_csv(meta_dir)
    stations = glob(f"{stations_dir}/*csv")
    stations = [station.split("/")[-1].split(".")[0] for station in stations]
    station_ids = meta[meta["Site_name"].isin(stations)]["No"].to_list()

    # read LAI data
    data = xarray.open_dataset(data_dir)
    lai = data["LAI"].values
    station = data["SiteNumber"].values

    # append LAI data to each station
    for i, st_id in enumerate(station_ids):
        try:
            st_name = meta[meta["No"] == st_id]["Site_name"].values[0]
            station_df = pd.read_csv(f"{stations_dir}/{st_name}.csv", index_col=0)
            print(f"Preparing LAI for {st_id}-{st_name}...")
            
            # retrieve corresponding station and its LAI values
            valid_st = np.where(np.isin(station, st_id) == True)[0]
            valid_lai = lai[valid_st]

            # write and save to current df
            station_df["lai"] = valid_lai

            assert len(station_df) == len(valid_lai), f"{st_id} - {st_name} fails to retrieve LAI values"
            station_df.to_csv(f"{stations_dir}/{st_name}.csv", index=False)
        except:
            print(f"{st_id} - {st_name} fails to retrieve LAI values")
            os.remove(os.path.join(stations_dir, f'{st_name}.csv')) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assign LAI data to valid FLUXNET stations')
    parser.add_argument('--lai_file', type=str, required=True, help='LAI data with global coverage')
    parser.add_argument('--meta_file', type=str, required=True, help='Stations details CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory where the processed CSV files are located')

    args = parser.parse_args()

    read_lai(
        data_dir= args.lai_file, #"../../Data/FLUXNET/Inputva_G_11.nc", 
        meta_dir= args.meta_file, #"../../Data/FLUXNET/Fluxnet_sites_Hysteresis.csv", 
        stations_dir= args.output_dir #"../Data/fluxnet"
    )
