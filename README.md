# template-for-or-op

A Jupyter notebook project for MCMS Operations Research and Optimization problems.

## Getting Started

Initialize a new uv environment and install the required packages:

```bash
uv venv --python 3.10.19
uv pip install -r requirements.txt
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
uv pip install osmnx geopandas
```

## Project Structure
- `notebooks/`: Contains Jupyter notebooks for various OR and optimization problems.
- `data/`: Directory for datasets used in the notebooks.
- `scripts/`: Python scripts for data processing and analysis.
- `README.md`: Project documentation.
## Usage