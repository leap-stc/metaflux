"""
Script to preprocess the original FLUX FULLSET data

Input: List of zipped FLUXNET data
Output: 
- Collection of site-specific Dataframes 
- Updated canopy CSV file with FLAGS
"""
import os
import shutil
import zipfile
import glob
import math
from datetime import datetime
import pandas as pd
import numpy as np
import argparse

def read_data(data_dir, output_dir):
    """
    Driver function to start the data preprocessing steps

    Input:
    - data_dir <string>: specifying the directory containing zipped FLUX FULLSET data
    - height_dir <string>: specifying the file location of canopy_height CSV data

    Output: None
    """
    #FLAGS
    FLAG_USE_HALFHOURLY = 1

    # Assertion for the validity of data and directory
    assert os.path.isdir(data_dir)
    #assert os.path.isfile(height_dir)
    
    # Reading files
    list_files = [file for file in os.listdir(data_dir) if file.endswith("zip")]
    n_files = len(list_files)
    #height_file = pd.read_csv(height_dir)
    #assert len(list_files) > 0 and len(height_file) > 0
    #assert len(list_files) == len(height_file)

    for i in range(n_files):
        # Create temporary directory
        tempdir = "../data/tempdir"
        if os.path.exists(tempdir): shutil.rmtree(tempdir)
        os.mkdir(tempdir)

        # Unzip
        with zipfile.ZipFile(os.path.join(data_dir, list_files[i]), "r") as zip_ref:
            zip_ref.extractall(tempdir)

        # Read CSV and XLS file
        filename_csv = [csv for csv in os.listdir(tempdir) if csv.endswith("csv")]
        PREFIX = filename_csv[0][0:3]
        assert PREFIX == "FLX"
        sitecode = filename_csv[0][4:10]
        print(f"Processing {i + 1} out of {n_files} sites: {sitecode}...")

        # Extract canopy information
        #h_measure, h_canopy, cover_type, latitude, longitude, altitude = extract_canopy(height_file, sitecode)

        # Import hourly or half-hourly fluxes data
        try:
            if FLAG_USE_HALFHOURLY:
                flux_df = pd.read_csv(glob.glob(tempdir + "/*FULLSET_HH*.csv")[0]) # Use half hourly data
                FLAG_FILE = 0

            if len(flux_df) == 0 or not FLAG_USE_HALFHOURLY:
                flux_df = pd.read_csv(glob.glob(tempdir + "/*FULLSET_HR*.csv")[0]) # Use hourly data
                FLAG_FILE = 1
        except:
            continue # skip current iteration if no corresponding file is found

        # Process FLUXNET variable data
        """NOTE: *_QC columns indicate the quality of the gapfill, where:
        0 = measured; 1 = good quality gapfill; 2 = medium; 3 = poor
        refer to https://fluxnet.org/data/fluxnet2015-dataset/fullset-data-product/ for more information
        """
        columns = ["TIMESTAMP_START"]
        flux_df["TIMESTAMP_START"]= pd.to_datetime(flux_df["TIMESTAMP_START"], format="%Y%m%d%H%S")

        """Water and carbon fluxes
        1) H: sensible heat flux, W/m2
        2) LE: latent heat flux, W/m2
        """
        H = "H_F_MDS" if len(flux_df["H_CORR"].unique()) == 1 and flux_df["H_CORR"][0] == -9999 else "H_CORR"
        LE = "LE_F_MDS" if len(flux_df["LE_CORR"].unique()) == 1 and flux_df["LE_CORR"][0] == -9999 else "LE_CORR"
        columns.extend([H, LE, "H_F_MDS_QC", "LE_F_MDS_QC", "G_F_MDS", "G_F_MDS_QC"])

        
        """
        GPP: Gross primary production
        1) GPP_NT_VUT_REF: NT-partitioned GPP, warning some sites have 2.1 scaling becasue of weight, umol CO2/m2/s
        2) GPP_DT_VUT_REF: DT-partitioned GPP

        NEE: Net ecosystem exchange
        1) NEE_VUT_REF
        """
        columns.extend(["GPP_NT_VUT_REF", "GPP_DT_VUT_REF", "NEE_VUT_REF", "NEE_VUT_REF_QC"])

        
        """Meteorological variables
        1) TA_F: air temperature, deg C
        2) SW_IN_F: shortwave radiation incoming, W/m2
        3) VPD_F: vapor pressure decifit, hPa
        4) PA_F: atmospheric pressure, kPa
        5) P_F: precipitation, mm
        6) WS_F: wind speed, m/s
        7) NETRAD: net radiation, Rn, W/m2
        8) USTAR: friction velocity, m/s
        """
        columns.extend(["TA_F", "TA_F_QC", "SW_IN_F", "SW_IN_F_QC", "VPD_F", "VPD_F_QC", "PA_F", "PA_F_QC", "P_F", "P_F_QC", "WS_F", "WS_F_QC", "NETRAD", "USTAR"])

        
        """RH: relative humidity
        """
        FLAG_RH = 1 if "RH" in flux_df.columns else 0
        columns.extend(["RH"])
        if not FLAG_RH:
            for index, row in flux_df.iterrows():
                flux_df.loc[index, "RH"] = 100*(1.-row["VPD_F"]/(0.6108*math.exp((17.27*row["TA_F"]/row["TA_F"] + 237.3))))

        
        """PAR
        1) PPFD_IN: photosynthetic photon flux density incoming, m-2 s-1 (PAR)
        2) PPFD_OUT: photosynthetic photon flux density outgoing, m-2 s-1 (PAR)
        """
        FLAG_PAR = 1 if set(["PPFD_IN", "PPFD_OUT"]).issubset(flux_df.columns) else 0
        if FLAG_PAR:
            columns.extend(["PPFD_IN", "PPFD_OUT"])

        
        """SWC: soil water content
        1) SWC_F_MDS_1: soil water content (volumetric), (SWC layer 1 - upper)
        2) SWC_F_MDS_2: soil water content (volumetric), (SWC layer 2 - lower)
        """
        if "SWC_F_MDS_1" in flux_df.columns:
            columns.extend(["SWC_F_MDS_1","SWC_F_MDS_1_QC"])
        
        if "SWC_F_MDS_2" in flux_df.columns:
            columns.extend(["SWC_F_MDS_2","SWC_F_MDS_2_QC"])

        ## SWC layer1 takes precedence over layer2 
        columns.extend(["SWC"])
        if "SWC_F_MDS_1" in flux_df.columns:
            SWC, FLAG_SWC = "SWC_F_MDS_1", 1
            if len(flux_df[SWC].unique()) == 1 and flux_df[SWC][0] == -9999:
                try:
                    SWC, FLAG_SWC = "SWC_F_MDS_2", 2
                    if len(flux_df[SWC].unique()) == 1 and flux_df[SWC][0] == -9999:
                        SWC, FLAG_SWC = None, 0
                except:
                    FLAG_SWC = 0

        elif "SWC_F_MDS_2" in flux_df.columns:
            SWC, FLAG_SWC = "SWC_F_MDS_2", 2
        else:
            FLAG_SWC = 0
        
        flux_df["SWC"] = flux_df[SWC]

        
        """CO2_F_MDS: CO2 concentration
        """
        if "CO2_F_MDS" in flux_df.columns:
            columns.extend(["CO2_F_MDS", "CO2_F_MDS_QC"])
            FLAG_CA = 1
        else:
            FLAG_CA = 0

        
        """SW_IN_POT: shortwave radiation incoming, potential, W/m2
        """
        if "SW_IN_POT" in flux_df.columns:
            columns.extend(["SW_IN_POT"])
            FLAG_SW_IN_POT = 1
        else:
            FLAG_SW_IN_POT = 0

        # COMBINE ALL RELATED COLUMNS
        flux_df = flux_df[columns]


        # Step 1: Exclude rainy hours and 6 hours data after rain
        rain_hour = flux_df.index[flux_df["P_F"] > 0].tolist()

        if FLAG_FILE == 0: # half-hourly data
            rain_hour_end = [rain_start + 12 for rain_start in rain_hour]
        else: # hourly data
            rain_hour_end = [rain_start + 6 for rain_start in rain_hour]

        ## Generate rain flag or mask
        FLAG_RAIN = []
        for k in range(len(rain_hour_end)):
            if rain_hour_end[k] + 1 < len(flux_df):
                FLAG_RAIN.extend([idx for idx in range(rain_hour[k], rain_hour_end[k] + 1)])

        FLAG_RAIN = list(set(FLAG_RAIN))

        # Step 2: remove data of poor quality & within the rain period
        flux_df[H] = np.where((abs(flux_df[H]) > 9000), np.nan, flux_df[H])
        flux_df[H] = np.where(flux_df["H_F_MDS_QC"] >= 2, np.nan, flux_df[H])
        flux_df.loc[FLAG_RAIN, H] = np.nan
        flux_df[LE] = np.where((flux_df[LE] <= 0) | (abs(flux_df[LE] > 9000)), np.nan, flux_df[LE])
        flux_df[LE] = np.where(flux_df["LE_F_MDS_QC"] >= 2, np.nan, flux_df[LE])
        flux_df.loc[FLAG_RAIN, LE] = np.nan
        flux_df["G_F_MDS"] = np.where(abs(flux_df["G_F_MDS"]) > 9000, np.nan, flux_df["G_F_MDS"])
        flux_df["G_F_MDS"] = np.where(flux_df["G_F_MDS_QC"] >= 2, np.nan, flux_df["G_F_MDS"])
        flux_df.loc[FLAG_RAIN, "G_F_MDS"] = np.nan
        flux_df["GPP_NT_VUT_REF"] = np.where((flux_df["GPP_NT_VUT_REF"] <= 0) | (abs(flux_df["GPP_NT_VUT_REF"]) > 9000), np.nan, flux_df["GPP_NT_VUT_REF"])
        flux_df["GPP_NT_VUT_REF"] = np.where(flux_df["NEE_VUT_REF_QC"] >= 2, np.nan, flux_df["GPP_NT_VUT_REF"])
        flux_df.loc[FLAG_RAIN, "GPP_NT_VUT_REF"] = np.nan
        flux_df["GPP_DT_VUT_REF"] = np.where((flux_df["GPP_DT_VUT_REF"] <= 0) | (abs(flux_df["GPP_DT_VUT_REF"]) > 9000), np.nan, flux_df["GPP_DT_VUT_REF"])
        flux_df["GPP_DT_VUT_REF"] = np.where(flux_df["NEE_VUT_REF_QC"] >= 2, np.nan, flux_df["GPP_DT_VUT_REF"])
        flux_df.loc[FLAG_RAIN, "GPP_DT_VUT_REF"] = np.nan
        flux_df["NEE_VUT_REF"] = np.where(abs(flux_df["NEE_VUT_REF"]) > 9000, np.nan, flux_df["NEE_VUT_REF"])
        flux_df["NEE_VUT_REF"] = np.where(flux_df["NEE_VUT_REF_QC"] >= 2, np.nan, flux_df["NEE_VUT_REF"])
        flux_df.loc[FLAG_RAIN, "NEE_VUT_REF"] = np.nan

        flux_df["SW_IN_F"] = np.where(abs(flux_df["SW_IN_F"]) > 9000, np.nan, flux_df["SW_IN_F"])
        flux_df.loc[FLAG_RAIN, "SW_IN_F"] = np.nan
        flux_df["TA_F"] = np.where(abs(flux_df["TA_F"]) > 9000, np.nan, flux_df["TA_F"])
        flux_df.loc[FLAG_RAIN, "TA_F"] = np.nan
        flux_df["WS_F"] = np.where(abs(flux_df["WS_F"]) > 9000, np.nan, flux_df["WS_F"])
        flux_df.loc[FLAG_RAIN, "WS_F"] = np.nan
        flux_df["VPD_F"] = np.where(abs(flux_df["VPD_F"]) > 9000, np.nan, flux_df["VPD_F"])
        flux_df.loc[FLAG_RAIN, "VPD_F"] = np.nan
        flux_df["PA_F"] = np.where(abs(flux_df["PA_F"]) > 9000, np.nan, flux_df["PA_F"])
        flux_df.loc[FLAG_RAIN, "PA_F"] = np.nan
        flux_df["P_F"] = np.where(abs(flux_df["P_F"]) > 9000, np.nan, flux_df["P_F"])
        flux_df["NETRAD"] = np.where(abs(flux_df["NETRAD"]) > 9000, np.nan, flux_df["NETRAD"])
        flux_df.loc[FLAG_RAIN, "NETRAD"] = np.nan
        flux_df["USTAR"] = np.where(abs(flux_df["USTAR"]) > 9000, np.nan, flux_df["USTAR"])
        flux_df.loc[FLAG_RAIN, "USTAR"] = np.nan

        if FLAG_PAR:
            flux_df["PPFD_IN"] = np.where(abs(flux_df["PPFD_IN"]) > 9000, np.nan, flux_df["PPFD_IN"])
            flux_df.loc[FLAG_RAIN, "PPFD_IN"] = np.nan
            flux_df["PPFD_OUT"] = np.where(abs(flux_df["PPFD_OUT"]) > 9000, np.nan, flux_df["PPFD_OUT"])
            flux_df.loc[FLAG_RAIN, "PPFD_OUT"] = np.nan

        if FLAG_RH:
            flux_df["RH"] = np.where(abs(flux_df["RH"]) > 9000, np.nan, flux_df["RH"])
            flux_df.loc[FLAG_RAIN, "RH"] = np.nan

        if FLAG_SWC != 0:
            flux_df["SWC"] = np.where(abs(flux_df["SWC"]) > 9000, np.nan, flux_df["SWC"])
            if FLAG_SWC == 1:
              flux_df["SWC"] = np.where(flux_df["SWC_F_MDS_1_QC"] >= 2, np.nan, flux_df["SWC"])
            elif FLAG_SWC == 2:

              flux_df[SWC] = np.where(flux_df["SWC_F_MDS_2_QC"] >= 2, np.nan, flux_df["SWC"])

        if FLAG_CA != 0:
            if len(flux_df["CO2_F_MDS"].unique()) == 1 and flux_df["CO2_F_MDS"][0] == -9999:
                FLAG_CA = 0
            else:
                flux_df["CO2_F_MDS"] = np.where(abs(flux_df["CO2_F_MDS"]) > 9000, np.nan, flux_df["CO2_F_MDS"])
                flux_df["CO2_F_MDS"] = np.where(flux_df["CO2_F_MDS_QC"] >= 2, np.nan, flux_df["CO2_F_MDS"])
                flux_df.loc[FLAG_RAIN, "CO2_F_MDS"] = np.nan
                if flux_df["CO2_F_MDS"].isnull().all():
                    FLAG_CA = 0

        if FLAG_SW_IN_POT:
            flux_df["SW_IN_POT"] = np.where(abs(flux_df["SW_IN_POT"]) > 9000, np.nan, flux_df["SW_IN_POT"])
            flux_df.loc[FLAG_RAIN, "SW_IN_POT"] = np.nan
        
        ## Unit conversion
        flux_df["VPD_F"] = flux_df["VPD_F"]/10 # hPD -> kPa

        # Step 3: get rid of 95% percentile of soil moisture/RH as it might be right after rain
        flux_df["RH"] = np.where(flux_df["RH"] > np.quantile(flux_df["RH"].dropna(), 0.95), np.nan, flux_df["RH"])

        # Step 4: add flag information into the summary file
        #height_file.loc[height_file["Site"] == sitecode, "FLAG_HR"] = FLAG_FILE
        #height_file.loc[height_file["Site"] == sitecode, "FLAG_SWC"] = FLAG_SWC
        #height_file.loc[height_file["Site"] == sitecode, "FLAG_PAR"] = FLAG_PAR
        #height_file.loc[height_file["Site"] == sitecode, "FLAG_CA"] = FLAG_CA
        #height_file.loc[height_file["Site"] == sitecode, "FLAG_SW_IN_POT"] = FLAG_SW_IN_POT
        #height_file.loc[height_file["Site"] == sitecode, "FLAG_RH"] = FLAG_RH

        # Step 5: save outputs
        if not os.path.exists(output_dir): os.mkdir(output_dir)

        flux_df.to_csv(f"{output_dir}/{sitecode}.csv")
        #height_file.to_csv(f"./Data/Canopy_height_51_new.csv")
        shutil.rmtree(tempdir)

def extract_canopy(height_df, sitecode):
    site_height = height_df[height_df['Site'] == sitecode]
    return site_height["Measure_h"].values[0], site_height["Canopy_h"].values[0], site_height["Cover_type"].values[0], site_height["Latitude"].values[0], site_height["Longitude"].values[0], site_height["Altitude"].values[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read FLUXNET and process them')
    parser.add_argument('--fluxnet_dir', type=str, required=True, help='FLUXNET directory containing the zip files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory where the processed CSV files are located')

    args = parser.parse_args()

    read_data(
        data_dir= args.fluxnet_dir, #"../../Data/FLUXNET"
        output_dir= args.output_dir #"../data/fluxnet"
    )