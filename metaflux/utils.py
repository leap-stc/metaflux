import os
from glob import glob
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import metaflux
import geopandas as gpd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import seaborn as sns

sns.set_context("paper")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(df, columns, new_col=None):
    # if its a list of variables to be normalized
    if type(columns) != str: 
        df.loc[:, df.columns.isin(columns)] = (df.loc[:, df.columns.isin(columns)] - df.loc[:, df.columns.isin(columns)].mean()) / df.loc[:, df.columns.isin(columns)].std()
    
    # otherwise a single string variable. also checks if we need a new column name
    else: 
        if new_col != None:
            df[new_col] = (df[columns] - df[columns].mean()) / df[columns].std()
        else:
            df[columns] = (df[columns] - df[columns].mean()) / df[columns].std()

    return df

def read_all_csv(data_dir, modes, norm_columns=None):
    station_csv = list()
    for mode in modes:
        csvf = os.path.join(data_dir, mode)
        station_csv.extend(glob(f"{csvf}/*.csv"))

    # Compute y_norm for all stations
    all_df = []
    for station in station_csv:
        df = pd.read_csv(station)
        df = df.fillna(method="ffill").fillna(method="bfill")
        if norm_columns != None:
            df = normalize(df, columns=norm_columns)
        all_df.append(df)

    all_df = pd.concat(all_df).reset_index(drop=True)

    return all_df

def robustness_check(maml_c, maml, encoder, base, fluxnet, hyper_args, factor):
    loss = torch.nn.MSELoss(reduce="mean")
    
    def _get_pred(learner, encoder, x):
        # latent = learner(x[:,:,:hyper_args["input_size"]])
        "Subroutine to perform prediction on input x"
        if encoder != None:
            encoding = encoder(x[:,:,hyper_args["input_size"]:])
            learner.module.update_encoding(encoding)

        return learner(x[:,:,:hyper_args["input_size"]])

    def _get_robustness_data(x, y, factor):
        "For shifted, noisy, and extreme tests"
        shifted_x = x + factor # Shifted
        noisy_x = x + torch.normal(factor, 1 + factor) # Noisy

        # Extremes
        norm_y = (y - y.mean()) / y.std()
        xtr_mask = norm_y > factor
        xtr_x = x[xtr_mask.squeeze(),:,:]
        xtr_y = y[xtr_mask.squeeze(),:]

        return shifted_x, noisy_x, xtr_x, xtr_y

    db_test = DataLoader(fluxnet, batch_size=hyper_args["batch_size"], shuffle=True)
    learner_c = maml_c.clone().double()
    learner = maml.clone().double()
    meta_c_loss_dict, meta_loss_dict, base_loss_dict = dict(), dict(), dict()
    meta_c_shift_losses, meta_shift_losses, base_shift_losses = list(), list(), list()
    meta_c_noise_losses, meta_noise_losses, base_noise_losses = list(), list(), list()
    meta_c_xtr_losses, meta_xtr_losses, base_xtr_losses = list(), list(), list()


    for f in factor:
        meta_c_shift_loss, meta_shift_loss, base_shift_loss = 0.0, 0.0, 0.0
        meta_c_noise_loss, meta_noise_loss, base_noise_loss = 0.0, 0.0, 0.0
        meta_c_xtr_loss, meta_xtr_loss, base_xtr_loss = 0.0, 0.0, 0.0

        with torch.no_grad():
            for step, (x, y) in enumerate(db_test):
                x, y = x.to(device), y.to(device)
                shifted_x, noisy_x, xtr_x, xtr_y = _get_robustness_data(x, y, f)

                # Shifted
                meta_c_shift_loss += loss(_get_pred(learner_c, encoder=encoder, x=shifted_x), y).item()
                meta_shift_loss += loss(_get_pred(learner, encoder=None, x=shifted_x), y).item()
                base_shift_loss += loss(base(shifted_x[:,:,:hyper_args["input_size"]]), y).item()

                # Noisy
                meta_c_noise_loss += loss(_get_pred(learner_c, encoder=encoder, x=noisy_x), y).item()
                meta_noise_loss += loss(_get_pred(learner, encoder=None, x=noisy_x), y).item()
                base_noise_loss += loss(base(noisy_x[:,:,:hyper_args["input_size"]]), y).item()

                # Extreme
                meta_c_xtr_loss += loss(_get_pred(learner_c, encoder=encoder, x=xtr_x), xtr_y).item()
                meta_xtr_loss += loss(_get_pred(learner, encoder=None, x=xtr_x), xtr_y).item()
                base_xtr_loss += loss(base(xtr_x[:,:,:hyper_args["input_size"]]), xtr_y).item()
            
        meta_c_shift_losses.append(meta_c_shift_loss / (step + 1))
        meta_c_noise_losses.append(meta_c_noise_loss / (step + 1))
        meta_c_xtr_losses.append(meta_c_xtr_loss / (step + 1))
        meta_shift_losses.append(meta_shift_loss / (step + 1))
        meta_noise_losses.append(meta_noise_loss / (step + 1))
        meta_xtr_losses.append(meta_xtr_loss / (step + 1))
        base_shift_losses.append(base_shift_loss / (step + 1))
        base_noise_losses.append(base_noise_loss / (step + 1))
        base_xtr_losses.append(base_xtr_loss / (step + 1))

    meta_c_loss_dict.update({
        "shift": np.sqrt(np.array(meta_c_shift_losses)),
        "noise": np.sqrt(np.array(meta_c_noise_losses)),
        "xtr": np.sqrt(np.array(meta_c_xtr_losses))
    })

    meta_loss_dict.update({
        "shift": np.sqrt(np.array(meta_shift_losses)),
        "noise": np.sqrt(np.array(meta_noise_losses)),
        "xtr": np.sqrt(np.array(meta_xtr_losses))
    })

    base_loss_dict.update({
        "shift": np.sqrt(np.array(base_shift_losses)),
        "noise": np.sqrt(np.array(base_noise_losses)),
        "xtr": np.sqrt(np.array(base_xtr_losses))
    })

    return meta_c_loss_dict, meta_loss_dict, base_loss_dict

def extreme_analysis(maml, base, hyper_args, data_dir, model_type="mlp", factor=1., is_plot=True):
    def _get_pred(learner, encoder, x):
        with_context = False if encoder == None else True
        "Subroutine to perform prediction on input x"
        if with_context:
            encoding = encoder(x)
            learner.module.update_encoding(encoding)

        return learner(x)

    def _get_extreme(x, y, y_norm, factor=factor):
        xtr_mask = y_norm > factor
        xtr_index = [index for index, value in enumerate(xtr_mask.squeeze()) if value]
        xtr_x = list()
        if model_type == "mlp":
            xtr_x = x[xtr_mask.squeeze()]
            xtr_y = y[xtr_mask.squeeze()]
        else:
            for xtr_idx in xtr_index[30:]:
                xtr_x.append(x[xtr_idx - 30 : xtr_idx, :])
            xtr_y = y[xtr_mask.squeeze()]
            xtr_y = xtr_y[30:]
        
        return torch.tensor(xtr_x).to(device), torch.tensor(xtr_y).to(device)

    learner = maml.clone().double()

    # Get all csv data
    modes = ["train", "test"]
    all_df = metaflux.utils.read_all_csv(data_dir, modes, norm_columns=hyper_args["xcolumns"])
    all_df = metaflux.utils.normalize(all_df, columns=f"{hyper_args['ycolumn'][0]}", new_col=f"{hyper_args['ycolumn'][0]}_norm")

    # Analyze individual station
    extreme_under_df, extreme_over_df = list(), list()
    for i, station in all_df.groupby("Site"):
        x, y, y_norm = station[hyper_args['xcolumns']].to_numpy(), station[hyper_args['ycolumn']].to_numpy(), station[[f"{hyper_args['ycolumn'][0]}_norm"]].to_numpy()
        x, y = _get_extreme(x, y, y_norm)
        try:
            maml_pred = _get_pred(learner, encoder=None, x=x)
            base_pred = base(x)
            if model_type != "mlp":
                maml_pred = maml_pred[:,-1:]
                base_pred = base_pred[:,-1:]

            extreme_under_d = {
                "climate": station["Climate"].iloc[0],
                "lon": station["Lon"].iloc[0],
                "lat": station["Lat"].iloc[0],
                "y": y.mean().item(),
                "base_under_prop": len(base_pred[base_pred < y]) / len(base_pred), 
                "maml_under_prop": len(maml_pred[maml_pred < y]) / len(maml_pred), 
                "base_under_mean": abs(base_pred[base_pred < y] - y[base_pred < y]).mean().item(),
                "maml_under_mean": abs(maml_pred[maml_pred < y] - y[maml_pred < y]).mean().item(),
            }

            extreme_over_d = {
                "climate": station["Climate"].iloc[0],
                "lon": station["Lon"].iloc[0],
                "lat": station["Lat"].iloc[0],
                "y": y.mean().item(),
                "base_over_prop": len(base_pred[base_pred > y]) / len(base_pred), 
                "base_over_mean": abs(base_pred[base_pred > y] - y[base_pred > y]).mean().item(),
                "maml_over_mean": abs(maml_pred[maml_pred > y] - y[maml_pred > y]).mean().item(),
            }

        except Exception as e:
            continue
        
        extreme_under_df.append(extreme_under_d)
        extreme_over_df.append(extreme_over_d)
    extreme_under_df = pd.DataFrame(extreme_under_df)
    extreme_over_df = pd.DataFrame(extreme_over_df)
    extreme_under_df.dropna(subset=("maml_under_mean", "base_under_mean"), inplace=True)
    extreme_over_df.dropna(subset=("maml_over_mean", "base_over_mean"), inplace=True)

    if is_plot:
        scaling_f = 1.7
        f, ax = plt.subplots(figsize=(16,12))
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        gdf = gpd.GeoDataFrame(extreme_under_df, geometry=gpd.points_from_xy(extreme_under_df.lon, extreme_under_df.lat))
        val_mean = int(gdf["base_under_mean"].mean())
        val_std = max(2*int(gdf["base_under_mean"].std()), 1)
        
        markersize_maml = np.exp(gdf['maml_under_mean']/scaling_f)
        markersize_base = np.exp(gdf['base_under_mean']/scaling_f)

        gdf.plot(markersize=markersize_maml, alpha=0.5, ax=ax, zorder=10)
        gdf.plot(markersize=markersize_base, alpha=0.5, ax=ax, zorder=5)
        world.boundary.plot(color="grey", linewidth=0.5, alpha=0.5, ax=ax, zorder=0)
        ax.set_ylim(-60,90)
        ax.axis("off")

        ax.scatter([], [], c='#1f77b4', alpha=0.5, s=100, label="Metalearning")
        ax.scatter([], [], c='orange', alpha=0.5, s=100, label="Non-Metalearning")
        ax.scatter([], [], c='xkcd:silver', alpha=1, s=np.exp((val_mean - val_std)/scaling_f), label=f'MAE < {val_mean - val_std}', edgecolor='black')
        ax.scatter([], [], c='xkcd:silver', alpha=1, s=np.exp((val_mean)/scaling_f), label=f'MAE {val_mean - val_std} - {val_mean + val_std}', edgecolor='black')
        ax.scatter([], [], c='xkcd:silver', alpha=1, s=np.exp((val_mean + val_std)/scaling_f), label=f'MAE > {val_mean + val_std}', edgecolor='black')

        ax.legend(title="MAE for underestimating\n extremes ($gCm^{-2}d^{-1}$)", labelspacing=1.5, borderpad=1, fontsize=8, title_fontsize=10, loc="lower left");

    return extreme_under_df, extreme_over_df

def analyze_embedding(data_dir, model, hyper_args, ohe, var="PFT", is_plot=True):
    modes = ["train", "test"]
    all_df = metaflux.utils.read_all_csv(data_dir=data_dir, modes=modes, norm_columns=hyper_args["xcolumns"])

    # Get embedding
    onehot_df = pd.DataFrame(ohe.transform(all_df[var].values.reshape(-1,1)).toarray())
    context_df = all_df[[x for x in hyper_args["contextcolumns"] if x != var]]
    context_df = pd.concat([context_df, onehot_df], axis=1)
    embedding = pd.DataFrame(model.encoder(torch.tensor(context_df.to_numpy()).to(device)).detach().cpu().numpy())

    # Get label encoder for PFT class
    le = LabelEncoder()
    embedding = pd.concat([embedding, pd.DataFrame({f"{var}_label": le.fit_transform(all_df[var])})], axis=1)

    # Analyze embedding
    tsne = TSNE(n_components=3, random_state=42)
    tsne_res = tsne.fit_transform(embedding)

    if is_plot:
        fig = px.scatter_3d(
            tsne_res, x=0, y=1, z=2,
            color=all_df.PFT, labels={"0": "Dimension 1", "1": "Dimension 2", "2": "Dimension 3", "color": "PFT"},
            width=800, height=800,
            
        )
        fig.update_traces(marker_size=4)
        fig.show()
        
    return tsne_res

def get_monthly_upscale_ds(year, src_dir, target_dir=None):
    """From daily to monthly scale for analysis"""

    data_path = os.path.join(src_dir, f'{year}')
    data_vars = ['GPP', 'GPP_std', 'RECO', 'RECO_std']
    upscale_files = glob(os.path.join(data_path, '*.nc'))
    upscale_files.sort()
    ds = list()
    units = '$gCm^{-2}d^{-1}$'
    description = "Upscaled global product of GPP and Reco (a combination of autotrophic and heterotrophic respiration) using meta-learning. 206 FLUXNET2015 stations are used to train each variable independently. We use the nighttime-partition for both GPP (GPP_NT) and Reco (RECO_NT). The inputs used are precipitation, vapor pressure deficit (VPD), 2-m air temperature from ERA5 reanalysis data, and LAI + EVI from MODIS. The global product is currently available at 0.25-degree resolution for the year 2001-2022 at a monthly scale"

    for month, upscale_file in enumerate(upscale_files):
        upscale_d = xr.open_dataset(upscale_file)
        lat, lon = upscale_d.lat.data, upscale_d.lon.data
        time = upscale_d.time.data[0]
        y_all = list()
        for data_var in data_vars:
            y_all.append(np.expand_dims(upscale_d[data_var].mean(dim='time').data, axis=0))

    
        temp = xr.Dataset(
            data_vars=dict(
                GPP=(
                    ["time", "lat", "lon"], 
                    y_all[0],
                    {'units': units, 
                    'long_name':'Gross Primary Production (GPP) ensemble mean'}
                ),
                GPP_std=(
                    ["time", "lat", "lon"], 
                    y_all[1],
                    {'units': units, 
                    'long_name':'Gross Primary Production (GPP) ensemble uncertainty'}
                ),
                RECO=(
                    ["time", "lat", "lon"], 
                    y_all[2],
                    {'units': units, 
                    'long_name':'Ecosystem Respiration (Reco) ensemble mean'}
                ),
                RECO_std=(
                    ["time", "lat", "lon"], 
                    y_all[3],
                    {'units': units, 
                    'long_name':'Ecosystem Respiration (Reco) ensemble uncertainty'}
                ),
            ),
                
            coords=dict(
                time=(["time"], [time]),
                lat=(["lat"], lat),
                lon=(["lon"], lon)
            ),
            attrs=dict(
                description=description)
        )

        ds.append(temp)
    ds = xr.merge(ds)

    if target_dir != None:
        target_dir = os.path.join(target_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(os.path.join(target_dir, f'METAFLUX_GPP_RECO_monthly_{year}.nc'), encoding=encoding)
        
    return ds

def get_all_upscale_data(upscale_dir, years):
    # Load all upscale products to an array
    ds_arr = list()
    upscale_files = glob(os.path.join(upscale_dir, "*.nc"))
    valid_files = list()
    for year in years:
        valid_files.extend([valid_file for valid_file in upscale_files if str(year) in valid_file])

    valid_files.sort()
    for upscale_f in valid_files:
        ds = xr.open_dataset(upscale_f)
        try:
            ds = ds.isel(lat=slice(None,None,-1))
        except:
            ds = ds.isel(latitude=slice(None,None,-1))
        ds_arr.append(ds)

    return ds_arr

def get_all_fluxcom_data(upscale_dir, years, var):
    # Load all upscale products to an array
    ds_arr = list()
    upscale_files = glob(os.path.join(upscale_dir, f"{var}*.nc"))
    valid_files = list()
    for year in years:
        valid_files.extend([valid_file for valid_file in upscale_files if str(year) in valid_file])

    valid_files.sort()
    for upscale_f in valid_files:
        ds_arr.append(xr.open_dataset(upscale_f))

    return ds_arr