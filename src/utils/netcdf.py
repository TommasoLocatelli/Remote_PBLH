import pandas as pd
import xarray as xr

class NETCDF:
    """
    General Data Model for NETCDF.
    """
    
    def __init__(self, global_attrs=None, data=None, variables_attrs=None):
        self.global_attrs = global_attrs
        self.data = data
        self.variables_attrs = variables_attrs


def read_netcdf(file_path):
    """
    Read a generic NetCDF file into a NETCDF object.
    """
    content = xr.open_dataset(file_path)

    global_attrs = pd.DataFrame(
        content.attrs.items(),
        columns=["Attribute", "Value"]
    )

    data = content.to_dataframe().reset_index()

    variables_attrs = pd.DataFrame([
        {**var.attrs, "variable": var_name}
        for var_name, var in content.data_vars.items()
    ])

    return NETCDF(global_attrs, data, variables_attrs)
