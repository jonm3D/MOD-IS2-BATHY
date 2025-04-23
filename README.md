# MOD-ICESat2 Bathymetry
Tools for accessing and processing satellite-lidar bathymetry (ICESat-2).

## Setup
```bash
mamba env create -f environment.yaml
```

## General Workflow

### Land Mask Creation
1. Go to NASA Earthdata Search
2. Search for "DSWx-HLS"
3. Download tiles as needed
4. Combine / clip DSWx data

```bash
python dswx/hls-mosaic.py <input_dir> <output_path> [--bbox <bbox_path>]
```

### Bathymetry Processing
1. Data Access. Configure request parameters at the top of `get_is2.py` and run to download ATL03 data. This may take several minutes depending on your AOI size and internet connection. 
2. Bathymetry Classification. Set up `config.yaml` as needed, `run_mod.py` analyzes each ICESat-2 granule and returns classified data + processing summaries. 

### Example Output