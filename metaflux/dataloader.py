import os
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from .utils import *

class Fluxmetanet(Dataset):
    """
    This dataset generator will automatically randomize inputs at every iteration and takes into account temporality as key
    feature in climate data products. 
    """

    def __init__(self, root, mode, x_columns, y_column, context_columns=None, time_column="TIMESTAMP_START", time_agg=None, time_window=1):
        """
        Initializing the Fluxmetanet Dataset class
        Parameters:
        -----------
        root <str>: root path of dataset (before <mode>) in CSV, in the following structure: <class>/<mode>/<filenames>.csv
        mode <str>: train or test (ie. metatrain or metatest)
        x_columns <list>: list containing the column names of input features
        y_column <str>: the name of column for target variable
        context_columns <list>:  list containing the column names of contextual features
        time_column <str>: the name of column indicating time (must be of DateTime object)
        time_agg <str>: how to aggregate time across observations, defaults to 1H (must be compatible with DateTime object)
        time_window <int>: the size of a time window 
        """
        modes = ["train", "test"]
        assert mode in modes

        self.mode = mode
        self.x_columns =  x_columns
        if type(y_column) != list:
            self.y_column = [y_column]
        self.context_columns = context_columns
        self.time_column = time_column
        self.time_agg = time_agg
        self.time_window = time_window

        # Fit a one-hot encoder for PFT
        if self.context_columns != None and 'PFT' in self.context_columns:
            pft_list = list()
            for mode in modes:
                csv_files = glob(os.path.join(root, mode) + "/*.csv")
                for file in csv_files:
                    df = pd.read_csv(file, index_col=False, header=0)
                    for _, row in df.iterrows():
                        pft_list.append(row['PFT'])
            pft_list = np.array(list(set(pft_list)))
            self.enc = OneHotEncoder()
            self.enc.fit(pft_list.reshape(-1,1))

        # Generate series from CSV files
        self.data = self.create_data(os.path.join(root, mode))

    def create_data(self, csvf):
        """
        Return a list of timeseries data from CSV files

        Parameters:
        -----------
        csvf <str>: directory containining csv files
        
        
        Returns:
        --------
        timeseries <List[DataFrame]>: list of timeseries data
        """
        csv_files = glob(csvf + "/*.csv")
        for file in csv_files:
            df = pd.read_csv(file, index_col=False, header=0)
            df = df.loc[:,~df.columns.str.match("Unnamed")]

            # Preprocessing for timeseries dataset: aggregation, gap-filling, normalizing of inputs
            try:
                df = self._temporal_preprocessing(df)
                timeseries_d = self._generate_series(df, n=self.time_window)
            except:
                continue

            # Generate timeseries
            try:
                all_data = np.concatenate([all_data, timeseries_d])
            except:
                all_data = timeseries_d

        return all_data

    def _temporal_preprocessing(self, df):
        "Temporal aggregation, gap filling, and normalizing input variables"
        if self.time_agg != None:
            df = df.set_index(pd.DatetimeIndex(df[self.time_column])).sort_index().resample(self.time_agg).mean()

        if self.context_columns != None:
            if 'PFT' in self.context_columns:
                onehot_df = pd.DataFrame(self.enc.transform(df['PFT'].values.reshape(-1,1)).toarray())
                df = df[self.y_column + self.x_columns + [x for x in self.context_columns if x != 'PFT']]
                df = pd.concat([df, onehot_df], axis=1)
                df = df[df.columns.tolist()[1:] + df.columns.tolist()[:1]] # swap y to the last column
            else:
                df = df[self.x_columns + self.context_columns + self.y_column]

        else:
            df = df[self.x_columns + self.y_column]

        # Gap-filling and normalize data (except for the target variable)
        df = df.fillna(method="ffill") # forward fill
        df = df.fillna(method="bfill") # backward fill
        df.loc[:, df.columns.isin(self.x_columns)] = (df.loc[:, df.columns.isin(self.x_columns)] - df.loc[:, df.columns.isin(self.x_columns)].mean()) / df.loc[:, df.columns.isin(self.x_columns)].std()
        df.dropna(inplace=True)

        return df
            
    def _generate_series(self, df, n):
        """
        Generate time series data given the temporal window size

        Parameters:
        -----------
        df <DataFrame>: the dataframe where the time series is going to be generated
        n <int> the length of the temporal window (ie. sequence length)

        Returns:
        --------
        series <np.array>: collection of time series sequences of shape (n_rows, seq_len, n_features)
        """
        series = np.empty((len(df) - n, n, df.shape[1]))

        for i in range(len(df) - n):
            series[i] = df[i:i+n].values

        return series
    
    def __getitem__(self, index):
        """
        Helper function for the Dataset generator in PyTorch (shuffles and actively feeds data into our model during training)

        Parameters:
        -----------
        index: index of sets, 0 <= index <= batchsz-1,
        shuffles the sequence of observations for every sampling (shuffle in dataloader object instead)
        """

        x, y = self.data[:,:,:-1], self.data[:,:,-1]
        return torch.tensor(x[index]), torch.tensor(y[index])
        
    def __len__(self):
        return self.data.shape[0]