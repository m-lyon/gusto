# gusto
GUSTO Data

## Install

To use this package, navigate to the root directory, and install the conda environment from the environment file:

```bash
conda env create -f environment.yml
```

Then install the package

```bash
pip install -e .
```

Check the data required to run this project is present by running
```bash
gusto_check_data.py
```