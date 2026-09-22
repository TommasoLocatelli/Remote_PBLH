from utils.netcdf import read_netcdf
import pandas as pd
import matplotlib.pyplot as plt
from utils.preprocessing import describe_data, filter_data
import numpy as np
from scipy.signal import savgol_filter

def plot_beta(data, beta_clmn="beta"):
    """
    Time–height pcolormesh plot of attenuated backscatter (beta).
    """

    # Pivot to time × height grid
    beta = data.pivot(index="time", columns="height", values=beta_clmn)

    # Convert time to datetime
    beta.index = pd.to_datetime(beta.index)

    times = beta.index.values
    heights = beta.columns.values

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot beta
    pcm = ax.pcolormesh(times, heights, beta.T,
                        shading="auto",
                        cmap="viridis",
                        vmin=0,
                        vmax=1e-6)

    # Labels and title
    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Time")
    ax.set_title("Attenuated Backscatter (beta)")

    # Colorbar
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("beta [1/(m·sr)]")

    plt.tight_layout()
    plt.show()

def compute_beta_derivatives(data, beta_clmn="beta", savgol_filter_params=None):
    """
    Compute first and second vertical derivatives of beta_clmn.

    If savgol_filter_params is provided, apply Savitzky–Golay smoothing:
        savgol_filter_params = dict(window_length=11, polyorder=3)

    Adds columns:
        d_beta_dz   = ∂β/∂z
        d2_beta_dz2 = ∂²β/∂z²
    """

    data = data.sort_values(["time", "height"]).copy()

    # --- Savitzky–Golay derivative ---
    if savgol_filter_params is not None:

        def compute_savgol(df):
            h = df["height"].values
            b = df[beta_clmn].values

            # Height spacing (mean spacing is fine for ceilometer)
            dz = np.nanmean(np.diff(h))

            # Apply Savitzky–Golay
            d1 = savgol_filter(
                b,
                deriv=1,
                delta=dz,
                **savgol_filter_params
            )
            d2 = savgol_filter(
                b,
                deriv=2,
                delta=dz,
                **savgol_filter_params
            )

            df["d_beta_dz"] = d1
            df["d2_beta_dz2"] = d2
            return df

        data = data.groupby("time", group_keys=False).apply(compute_savgol)

    # --- Finite difference fallback ---
    else:
        data["d_beta_dz"] = (
            data.groupby("time")[beta_clmn].diff() /
            data.groupby("time")["height"].diff()
        )
        data["d2_beta_dz2"] = (
            data.groupby("time")["d_beta_dz"].diff() /
            data.groupby("time")["height"].diff()
        )

    return data

def plot_beta_derivatives(data, beta_clmn="beta", pblh=None):
    """
    Plot beta, first derivative, and second derivative of beta_clmn.
    """

    # Pivot beta
    beta = data.pivot(index="time", columns="height", values=beta_clmn)

    # Pivot derivatives
    d1 = data.pivot(index="time", columns="height", values="d_beta_dz")
    d2 = data.pivot(index="time", columns="height", values="d2_beta_dz2")

    # Convert time
    beta.index = pd.to_datetime(beta.index)
    d1.index = pd.to_datetime(d1.index)
    d2.index = pd.to_datetime(d2.index)

    times = beta.index.values
    heights = beta.columns.values

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # --- Beta ---
    pcm0 = axes[0].pcolormesh(times, heights, beta.T,
                              shading="auto", cmap="viridis",
                              vmin=0, vmax=1e-6)
    axes[0].set_ylabel("Height [m]")
    axes[0].set_title(f"{beta_clmn} (Attenuated Backscatter)")
    fig.colorbar(pcm0, ax=axes[0]).set_label("beta [1/(m·sr)]")

    # --- First derivative ---
    pcm1 = axes[1].pcolormesh(times, heights, d1.T,
                              shading="auto", cmap="RdBu_r")
    axes[1].set_ylabel("Height [m]")
    axes[1].set_title(f"First derivative ∂({beta_clmn})/∂z")
    fig.colorbar(pcm1, ax=axes[1]).set_label("dβ/dz")

    # --- Second derivative ---
    pcm2 = axes[2].pcolormesh(times, heights, d2.T,
                              shading="auto", cmap="RdBu_r")
    axes[2].set_ylabel("Height [m]")
    axes[2].set_xlabel("Time")
    axes[2].set_title(f"Second derivative ∂²({beta_clmn})/∂z²")
    fig.colorbar(pcm2, ax=axes[2]).set_label("d²β/dz²")

    if pblh is not None:
        overlay_pblh_beta(axes[0], pblh, label="pblh_first", color='yellow')
        overlay_pblh_beta(axes[0], pblh, label="pblh_second", color='red')

    plt.tight_layout()
    plt.show()

def compute_pblh_from_first_derivative(data, max_height=2100, border_bound=200):
    """
    Compute PBLH as the height of the largest negative peak in d_beta_dz,
    excluding candidates too close to the lower or upper boundaries.
    """

    if "d_beta_dz" not in data.columns:
        raise ValueError("d_beta_dz not found. Run compute_beta_derivatives() first.")

    data = data.sort_values(["time", "height"]).copy()

    def get_min_height(df):
        valid = df.dropna(subset=["d_beta_dz"])

        # Exclude candidates near boundaries
        valid = valid[
            (valid["height"] > border_bound) &
            (valid["height"] < (max_height - border_bound))
        ]

        if valid.empty:
            return float("nan")

        idx = valid["d_beta_dz"].idxmin()
        return valid.loc[idx, "height"]

    return (
        data.groupby("time")
            .apply(get_min_height)
            .reset_index(name="pblh_first")
    )

def compute_pblh_from_second_derivative(data, max_height=2100, border_bound=200):
    """
    Compute PBLH as the height of the minimum in d2_beta_dz2,
    excluding candidates too close to the lower or upper boundaries.
    """

    if "d2_beta_dz2" not in data.columns:
        raise ValueError("d2_beta_dz2 not found. Run compute_beta_derivatives() first.")

    data = data.sort_values(["time", "height"]).copy()

    def get_min_height(df):
        valid = df.dropna(subset=["d2_beta_dz2"])

        # Exclude candidates near boundaries
        valid = valid[
            (valid["height"] > border_bound) &
            (valid["height"] < (max_height - border_bound))
        ]

        if valid.empty:
            return float("nan")

        idx = valid["d2_beta_dz2"].idxmin()
        return valid.loc[idx, "height"]

    return (
        data.groupby("time")
            .apply(get_min_height)
            .reset_index(name="pblh_second")
    )

def compute_pblh_beta(data, beta_clmn="beta"):
    """
    Compute two PBLH estimates from beta derivatives:
      - pblh_first  = largest negative peak of first derivative
      - pblh_second = minimum of second derivative

    Returns a DataFrame with columns: time, pblh_first, pblh_second.
    """

    # Compute derivatives if missing
    if "d_beta_dz" not in data.columns or "d2_beta_dz2" not in data.columns:
        data = compute_beta_derivatives(data, beta_clmn)

    # Compute both estimates
    pblh_first = compute_pblh_from_first_derivative(data)
    pblh_second = compute_pblh_from_second_derivative(data)

    # Merge
    pblh = pblh_first.merge(pblh_second, on="time")

    return pblh

def overlay_pblh_beta(ax, pblh, label, color):
    ax.plot(
        pblh["time"],
        pblh[label],
        color=color,
        linewidth=2,
        label=label
    )
    ax.legend(loc="upper right")   # fast + stable

if __name__=='__main__':
    path=r'data\cloudnet-lidar-ceilometer\20260816_cabauw_chm15k_c07f533c.nc'
    netcdf=read_netcdf(path)
    data=netcdf.data
    data=filter_data(netcdf.data, 
                    start_hour=2,
                    end_hour=23,
                    upper_bound=2000,
                    lower_bound=50,
                    cols=["time", "height", "altitude", "beta", "beta_raw", "beta_smooth"])
    BETA_CLMN='beta_smooth'
    #describe_data(data)
    #plot_beta(data, beta_clmn='beta_smooth')
    data = compute_beta_derivatives(
        data,
        beta_clmn=BETA_CLMN,
        savgol_filter_params=dict(window_length=11, polyorder=3)
        )
    #plot_beta_derivatives(data, beta_clmn="beta_smooth")
    pblh=compute_pblh_beta(data, beta_clmn=BETA_CLMN)
    plot_beta_derivatives(data, beta_clmn=BETA_CLMN, pblh=pblh)