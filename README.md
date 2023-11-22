# Metaflux
This repository contains the code that accompanies the paper `Nathaniel, J., Liu, J. & Gentine, P. MetaFlux: Meta-learning global carbon fluxes from sparse spatiotemporal observations. Sci Data 10, 440 (2023). https://doi.org/10.1038/s41597-023-02349-y`. 

The resulting dataset of global, high-resolution (0.25-degree), long-term (daily/monthly 2001-2021) gross primary production and ecosystem respiration are available in Zenodo at https://doi.org/10.5281/zenodo.7761881.

## Introduction
Meta-learning framework for climate sciences. Currently supports the following features:
- Takes as input timeseries data (eg. FLUXNET eddy covariance stations)
- Customizable hyperparameters (create your own and place them under the `configs` directory)
- Sample training script with sample data that can be adapted to your own use case

## Quickstart
1. Clone this repository into your private workspace:
```
git clone https://github.com/juannat7/metaflux.git
```

2. Install dependencies using `pip` or `conda`
```
pip install -r requirements.txt
```

## Sample notebooks
These sample notebooks attempt to demonstrate the applications of meta-learning for spatiotemporal domain adaptation. In particular, we tried to infer gross primary production (GPP) from key meteorological and remote sensing data points
by learning key features in data-abundant regions and adapt them to fluxes in data-sparse areas. We demonstrate the use of meta-learning in non-temporal, temporal, and with spatial context situations. Feel free to apply the algorithm presented in the notebook for your specific use cases: 

1. `01a_non_temporal_pipeline`: for non-temporal dataset and model (eg. MLP)
2. `01b_temporal_pipeline`: for temporal dataset and model (eg. LSTM, BiLSTM)

![Meta inference](https://github.com/juannat7/metaflux/blob/main/docs/gpp_infer.jpeg)

3. (experimental) `01c_with_encoder_pipeline`: adding context encoder to current classic metalearning model

![Encoder workflow](https://github.com/juannat7/metaflux/blob/main/docs/encoder_workflow.png)
![Meta inference with context encoder](https://github.com/juannat7/metaflux/blob/main/docs/gpp_encoder_infer.jpeg)

## Custom Docker Image
We publish a Docker Image with every release of the package. The images are hosted on [quay.io]() You can find the image tags on [quay.io](https://quay.io/repository/leap-stc/metaflux?tab=tags). 

### Use custom image on LEAP-Pangeo Hub
To use this image on the [LEAP-Pangeo Jupyterhub]() you can provide a [custom image url](https://leap-stc.github.io/leap-pangeo/jupyterhub.html#custom-images) as `quay.io/leap-stc/metaflux:<tag>`
