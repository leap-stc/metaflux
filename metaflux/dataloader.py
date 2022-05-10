import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Fluxmetanet(Dataset):
    """
    This dataset generator will automatically randomize inputs at every iteration and takes into account temporality as key
    feature in climate data products. 
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, x_columns, y_column, time_column="TIMESTAMP_START", time_agg="1H", seasonality=24):
        """
        Initializing the Fluxmetanet Dataset class
        Parameters:
        -----------
        root <str>: root path of dataset (before <mode>) in CSV, in the following structure: <class>/<mode>/<filenames>.csv
        mode <str>: train, val or test
        batchsz <int>: batch size of inputs
        n_way <int>: the number of classes under consideration
        k_shot <int>: the number of shots samples per class (ie. similar to a supervised training dataset)
        k_query <int>: the number of query samples per class (ie. similar to a supervised testing dataset)
        x_columns <list>: list containing the column names of input features
        y_column <str>: the name of column for target variable
        time_column <str>: the name of column indicating time (must be of DateTime object)
        time_agg <str>: how to aggregate time across observations, defaults to 1H (must be compatible with DateTime object)
        seasonality <int>: seasonality by which we are generating the data window 
        """

        self.mode = mode
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.setsz = self.n_way * self.k_shot
        self.querysz = self.n_way * self.k_query
        self.xcolumns =  x_columns
        self.ycolumns = [y_column]
        self.timecolumn = time_column
        self.time_agg = time_agg
        self.seasonality = seasonality
        
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query' % (mode, batchsz, n_way, k_shot, k_query))

        # Get and read CSV files
        csvdata = self.load_csv(os.path.join(root, mode))
        self.data = []
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)

        self.cls_num = len(self.data)

        # Dataset batching
        self.create_batch(self.batchsz)

    def load_csv(self, csvf):
        """
        Return a dict saving the information of our CSVs

        Parameters:
        -----------
        csvf <str>: directory containining csv files
        
        
        Returns:
        --------
        labels <dict>: {label1:[file1],...} since each CSV file is considered as one label, there exists only one file per label
        """
        csv_files = glob.glob(csvf + "/*.csv")
        dict_labels = {}
        for i, file in enumerate(csv_files):
            label = file.split("/")[-1].split(".csv")[0]

            # append filename to current label
            if label in dict_labels.keys():
                dict_labels[label].append(file)
            else:
                dict_labels[label] = [file]

        return dict_labels

    def create_batch(self, batchsz):
        """
        Create batch for meta-learning
        Parameters:
        -----------
        batchsz <int>: batch size

        Returns:
        --------
        support_x: (batchsz, cls_num, k_shot, seq_len, n_feature)
        support_y: (batchsz, cls_num, k_shot, seq_len, n_feature)
        query_x: (batchsz, cls_num, k_shot, seq_len, n_feature)
        query_y: (batchsz, cls_num, k_shot, seq_len, n_feature)
        """
        self.support_x_batch = []  # support set batch (x)
        self.support_y_batch = []  # support set batch (y)
        self.query_x_batch = []  # query set batch (x)
        self.query_y_batch = []  # query set batch (y)
        
        for b in range(batchsz):  # for each batch
            support_x = []
            support_y = []
            query_x = []
            query_y = []
            
            # 1.select n_way labels randomly
            if self.mode == "train":
                selected_cls = np.random.choice(self.cls_num, self.n_way, replace=False)
            else:
                selected_cls = np.random.choice(self.cls_num, max(2, self.cls_num - 1), replace=False) # take n-1 or at least 2 labels randomly
            
            for cls in tqdm(selected_cls):
                try:
                    cls_csv = self.data[cls][0]
                    cls_df = pd.read_csv(cls_csv, index_col=None, header=0)

                    # Aggregate time (generally to smooth out noisy observations)
                    cls_df = cls_df.set_index(pd.DatetimeIndex(cls_df["TIMESTAMP_START"])).resample(self.time_agg).mean()

                    # Gap-filling and normalize data (except for the target variable)
                    cls_df = cls_df.fillna(method="ffill")[self.xcolumns + self.ycolumns] # forward fill
                    cls_df = cls_df.fillna(method="bfill")[self.xcolumns + self.ycolumns] # backward fill
                    cls_df = cls_df.dropna()
                    cls_df.loc[:, cls_df.columns != self.ycolumns[0]] = (cls_df.loc[:, cls_df.columns != self.ycolumns[0]] - cls_df.loc[:, cls_df.columns != self.ycolumns[0]].mean()) / cls_df.loc[:, cls_df.columns != self.ycolumns[0]].std()

                    # Generate series
                    cls_df = self.generate_series(cls_df, n=self.seasonality) 
                    
                except:
                    print(f"Error in processing sites: {cls_csv}")
                    continue
                
                # Get x, y, and length of our timeseries
                cls_x, cls_y = cls_df[:,:,0:-1], cls_df[:,-1:,-1]
                cls_len = cls_df.shape[0]
                
                # Randomly choose the shots/support and query sets (training and testing data) from the randomly chosen labels above
                if self.mode == "train":
                    try:
                        selected_timeseries_idx = np.random.choice(cls_len, self.k_shot + self.k_query, False)
                        np.random.shuffle(selected_timeseries_idx)
                    except:
                        print(f"Error in processing sites: {cls_csv}")
                        continue
                else:
                    selected_timeseries_idx = np.arange(cls_len - self.k_shot - self.k_query, cls_len)
                        
                # 2. select k_shot + k_query for each class
                Dtrain_idx = np.array(selected_timeseries_idx[:self.k_shot])  # idx for Dtrain
                Dtest_idx = np.array(selected_timeseries_idx[self.k_shot:])  # idx for Dtest
        
                support_x.append(cls_x[Dtrain_idx]) # get selected timeseries for current Dtrain
                support_y.append(cls_y[Dtrain_idx])
                query_x.append(cls_x[Dtest_idx])
                query_y.append(cls_y[Dtest_idx])
            
            # Collect batches for support set
            self.support_x_batch.append(support_x)
            self.support_y_batch.append(support_y)
            
            # Collect batches for query set
            self.query_x_batch.append(query_x)
            self.query_y_batch.append(query_y)
            
    def generate_series(self, df, n=24):
        """
        Generate time series data given the temporal window size

        Parameters:
        -----------
        df <DataFrame>: the dataframe where the time series is going to be generated
        n <int> the length of the temporal window (ie. sequence length)

        Returns:
        --------
        df <DataFrame>: collection of time series sequences of shape (batch size, sequence length, number of features)
        """
        series_df = np.empty((len(df) - n, n, df.shape[1]))

        for i in range(len(df) - n):
            series_df[i] = df[i:i+n].values

        return series_df
    
    def __getitem__(self, index):
        """
        Helper function for the Dataset generator in PyTorch (shuffles and actively feeds data into our model during training)

        Parameters:
        -----------
        index: index of sets, 0 <= index <= batchsz-1,
        shuffles the sequence of observations for every sampling (shuffle in dataloader object instead)
        """
        flatten_support_x = torch.Tensor()
        flatten_support_y = torch.Tensor()
        flatten_query_x = torch.Tensor()
        flatten_query_y = torch.Tensor()
        
        # for support: shuffle along observation dimension
        for i in range(len(self.support_x_batch)):
            flatten_support_x = torch.cat((flatten_support_x, torch.tensor(self.support_x_batch[i])))
            flatten_support_y = torch.cat((flatten_support_y, torch.tensor(self.support_y_batch[i])))
            
        # for query: shuffle along observation dimension
        for i in range(len(self.query_x_batch)):
            flatten_query_x = torch.cat((flatten_query_x, torch.tensor(self.query_x_batch[i])))
            flatten_query_y = torch.cat((flatten_query_y, torch.tensor(self.query_y_batch[i])))
        
        return flatten_support_x, flatten_support_y, flatten_query_x, flatten_query_y
        
    def __len__(self):
        return self.batchsz