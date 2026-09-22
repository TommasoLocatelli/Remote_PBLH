from utils.netcdf import read_netcdf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from doppler_lidar_pblh import describe_data, filter_data

def plots_wind(data):
    """
    Create a single figure with two time–height pcolormesh subplots
    for u-wind and v-wind.
    """

    # --- Pivot to time × height grids ---
    uwind = data.pivot(index="time", columns="height", values="uwind")
    vwind = data.pivot(index="time", columns="height", values="vwind")

    # Convert time to datetime
    uwind.index = pd.to_datetime(uwind.index)
    vwind.index = pd.to_datetime(vwind.index)

    times = uwind.index.values
    heights = uwind.columns.values

    # --- Create figure with two subplots ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # --- U-wind subplot ---
    pcm1 = axes[0].pcolormesh(times, heights, uwind.T,
                              shading="auto", cmap="RdBu_r")
    axes[0].set_ylabel("Height [m]")
    axes[0].set_title("U-wind component")
    cbar1 = fig.colorbar(pcm1, ax=axes[0])
    cbar1.set_label("u [m/s]")

    # --- V-wind subplot ---
    pcm2 = axes[1].pcolormesh(times, heights, vwind.T,
                              shading="auto", cmap="RdBu_r")
    axes[1].set_ylabel("Height [m]")
    axes[1].set_xlabel("Time")
    axes[1].set_title("V-wind component")
    cbar2 = fig.colorbar(pcm2, ax=axes[1])
    cbar2.set_label("v [m/s]")

    plt.tight_layout()
    plt.show()

def compute_wind_shear(data):
    """
    Compute vertical wind shear (du/dz and dv/dz) for each time and height.
    Returns a DataFrame with added columns: shear_u, shear_v.
    """

    # Sort by time and height to ensure correct gradient computation
    data = data.sort_values(["time", "height"]).copy()

    # Compute shear using groupby on time
    data["shear_u"] = data.groupby("time")["uwind"].diff() / data.groupby("time")["height"].diff()
    data["shear_v"] = data.groupby("time")["vwind"].diff() / data.groupby("time")["height"].diff()

    return data

def compute_wind_shear_magnitude(data):
    """
    Compute vertical wind shear magnitude sqrt((du/dz)^2 + (dv/dz)^2).
    """

    data = compute_wind_shear(data)
    data["shear_mag"] = np.sqrt(data["shear_u"]**2 + data["shear_v"]**2)

    return data

def plots_wind_with_shear(data, pblh=None):
    """
    Plot u-wind, v-wind, and wind shear magnitude in one figure.
    """

    # Compute shear
    data = compute_wind_shear_magnitude(data)

    # Pivot
    uwind = data.pivot(index="time", columns="height", values="uwind")
    vwind = data.pivot(index="time", columns="height", values="vwind")
    shear = data.pivot(index="time", columns="height", values="shear_mag")

    # Convert time
    uwind.index = pd.to_datetime(uwind.index)
    vwind.index = pd.to_datetime(vwind.index)
    shear.index = pd.to_datetime(shear.index)

    times = uwind.index.values
    heights = uwind.columns.values

    # Figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    # U-wind
    pcm1 = axes[0].pcolormesh(times, heights, uwind.T, shading="auto", cmap="RdBu_r")
    axes[0].set_ylabel("Height [m]")
    axes[0].set_title("U-wind")
    fig.colorbar(pcm1, ax=axes[0]).set_label("u [m/s]")

    # V-wind
    pcm2 = axes[1].pcolormesh(times, heights, vwind.T, shading="auto", cmap="RdBu_r")
    axes[1].set_ylabel("Height [m]")
    axes[1].set_title("V-wind")
    fig.colorbar(pcm2, ax=axes[1]).set_label("v [m/s]")

    # Shear magnitude
    pcm3 = axes[2].pcolormesh(times, heights, shear.T, shading="auto", cmap="viridis")
    axes[2].set_ylabel("Height [m]")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Wind Shear Magnitude")
    fig.colorbar(pcm3, ax=axes[2]).set_label("Shear [1/s]")

    if pblh is not None:
        overlay_pblh(axes[2], pblh)   # on shear magnitude

    plt.tight_layout()
    plt.show()

def compute_pblh_from_shear(data):
    """
    Compute PBL height as the height where wind shear magnitude is minimum
    for each time step.
    Returns a DataFrame with columns: time, pblh.
    """

    if "shear_mag" not in data.columns:
        raise ValueError("shear_mag not found. Run compute_wind_shear_magnitude(data) first.")

    # Sort to ensure correct grouping
    data = data.sort_values(["time", "height"])

    # For each time, find height where shear magnitude is minimum
    pblh = (
        data.groupby("time")
            .apply(lambda df: df.loc[df["shear_mag"].idxmin(), "height"])
            .reset_index(name="pblh")
    )

    return pblh

def overlay_pblh(ax, pblh):
    """
    Overlay PBL height on an existing time–height plot.
    """
    ax.plot(pblh["time"], pblh["pblh"], color="yellow", linewidth=2, label="PBLH")
    ax.legend()

if __name__=='__main__':
    path=r'data\cloudnet-examples\20260816_cabauw_wls200s-wind_dca88604.nc'
    netcdf=read_netcdf(path)
    data=filter_data(netcdf.data, 
                start_hour=0,
                end_hour=23,
                upper_bound=1200,
                cols=["time", "height", "altitude", "uwind", "vwind"])
    #describe_data(data)
    #plots_wind(data)
    data=compute_wind_shear(data)
    data=compute_wind_shear_magnitude(data)
    #plots_wind_with_shear(data)
    pblh=compute_pblh_from_shear(data)
    plots_wind_with_shear(data, pblh=pblh)
