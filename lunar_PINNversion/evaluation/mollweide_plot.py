import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.optimize import root
import cmcrameri.cm as cmc
from matplotlib.gridspec import GridSpec

def geographic_to_Mollweide_point(
        points_geographic: np.ndarray
) -> np.ndarray:
    """
    Transform points on the unit sphere from geographic coordinates (ra,dec)
    to Mollweide projection coordiantes (x,y).

    INPUTS
    ------
    points_geographic: numpy array
        The geographic coords (ra,dec).
        Either a single point [shape=(2,)], or
        a list of points [shape=(Npoints,2)].

    RETURNS
    -------
    points_Mollweide: numpy array
        The Mollweide projection coords (x,y).
        Either a single point [shape=(2,)], or
        a list of points [shape=(Npoints,2)].
    """
    final_shape_Mollweide = list(points_geographic.shape)

    points_geographic = points_geographic.reshape(-1, points_geographic.shape[-1])

    points_Mollweide = np.zeros(shape=points_geographic.shape,
                                dtype=points_geographic.dtype)

    alpha_tol = 1.e-6

    def alpha_eq(x):
        return np.where(np.pi / 2 - np.abs(points_geographic[..., 1]) < alpha_tol, points_geographic[..., 1],
                        2 * x + np.sin(2 * x) - np.pi * np.sin(points_geographic[..., 1]))

    alpha = root(fun=alpha_eq, x0=points_geographic[..., 1], method='krylov', tol=1.e-10)

    points_Mollweide[..., 0] = 2 * np.sqrt(2) * (points_geographic[..., 0] - np.pi) * np.cos(alpha.x) / np.pi
    points_Mollweide[..., 1] = np.sqrt(2) * np.sin(alpha.x)

    points_Mollweide = points_Mollweide.reshape(final_shape_Mollweide)

    return points_Mollweide


def geographic_to_Cartesian_vector(points, dpoints):
    """
    Transform vectors in the tangent plane of the unit sphere from
    geographic coords (d_ra,d_dec) to Cartesian coords (d_x,d_y,d_z).

    INPUTS
    ------
    points: numpy array
        The geographic coords (r, theta, phi).
        Either a single point [shape=(3,)], or
        a list of points [shape=(Npoints, 3)].
    dpoints: numpy array
        The geographic coords (dr, d_ra, d_dec).
        Either a single point or many with shape
        matching points.

    RETURNS
    -------
    tangent_vector: numpy array
        The coords (d_x,d_y,d_z).
        Either a single point [shape=(3,)], or
        a list of points [shape=(Npoints,3)].
    """
    if points.ndim == 1:
        tangent_vector = np.zeros((3), dtype=dpoints.dtype)
    else:
        tangent_vector = np.zeros((len(points), 3), dtype=dpoints.dtype)

    r = points[..., 0]
    theta = np.pi / 2 - points[..., 1]
    phi = points[..., 2]

    dr = dpoints[..., 0]
    dtheta = - dpoints[..., 1]
    dphi = dpoints[..., 2]

    tangent_vector[..., 0] = (
            dr * np.sin(theta) * np.cos(phi)
            + r * np.cos(theta) * np.cos(phi) * dtheta
            - r * np.sin(theta) * np.sin(phi) * dphi
    )

    tangent_vector[..., 1] = (
            dr * np.sin(theta) * np.sin(phi)
            + r * np.cos(theta) * np.sin(phi) * dtheta
            + r * np.sin(theta) * np.cos(phi) * dphi
    )

    tangent_vector[..., 2] = (
            dr * np.cos(theta) - r * np.sin(theta) * dtheta
    )

    return tangent_vector

def deg_to_rad(X):
    return np.deg2rad(X)

def plot_mollweide_map(lon, lat, B_field, ax=None,
                       output_file=None, cnorm=None,
                       add_colorbar=True, cbar_label="B field [nT]"):
    """
    Plot Mollweide projection map.

    Parameters:
    -----------
    lon, lat, B_field : array-like
        Data to plot
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure.
    output_file : str, optional
        If provided, saves figure to this file
    cnorm : str, optional
        'symlog' for symmetric log normalization
    add_colorbar : bool
        Whether to add colorbar (default True)
    cbar_label : str
        Label for colorbar

    Returns:
    --------
    ax : matplotlib axis
        The axis that was plotted on
    im : scatter plot object
        The scatter plot (useful for adding colorbars later)
    """

    # ✅ Create figure/axis only if not provided
    if ax is None:
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection="mollweide")
        created_fig = True
    else:
        created_fig = False
        fig = ax.figure

    # Plot data
    if cnorm == 'symlog':
        im1 = ax.scatter(
            lon,
            lat,
            c=B_field,
            s=1,
            cmap=cmc.vik,
            rasterized=True,
            norm=colors.SymLogNorm(
                linthresh=0.03,
                linscale=0.03,
                vmin=np.nanquantile(B_field, 0.05),
                vmax=np.nanquantile(B_field, 0.95),
                base=10
            )
        )
    else:
        im1 = ax.scatter(
            lon,
            lat,
            c=B_field,
            s=1,
            cmap=cmc.oslo,
            rasterized=True,
            vmin=np.nanquantile(B_field, 0.05),
            vmax=np.nanquantile(B_field, 0.95),
        )

    ax.grid(True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # ✅ Only add colorbar if requested
    if add_colorbar:
        plt.colorbar(im1, ax=ax, orientation="horizontal",
                     pad=0.05, label=cbar_label)

    # ✅ Only save if output_file provided
    if output_file is not None:
        plt.savefig(output_file, dpi=150)

    # ✅ Return axis and image for further customization
    return ax, im1


def plot_three_component_mollweide(lon, lat, B_x, B_y, B_z,
                                   output_file='three_component_map.png',
                                   cnorm='symlog',
                                   figsize=(16, 24),
                                   titles=None,
                                   cbar_label="B field [micro Tesla]",
                                   share_colorbar=False):
    """
    Plot three magnetic field components on Mollweide projection in 3x1 grid.

    Parameters:
    -----------
    lon : array-like
        Longitude values in [-pi, pi]
    lat : array-like
        Latitude values
    B_x, B_y, B_z : array-like
        Three magnetic field components
    output_file : str
        Output filename
    cnorm : str, optional
        'symlog' for symmetric log normalization
    figsize : tuple
        Figure size (width, height)
    titles : list of str, optional
        Custom titles for each subplot. Default: ['$B_x$', '$B_y$', '$B_z$']
    cbar_label : str
        Label for colorbar
    share_colorbar : bool
        If True, use one colorbar for all three plots with shared scale.
        If False, each plot gets its own colorbar.

    Returns:
    --------
    fig : matplotlib figure
    axes : list of axes
    ims : list of scatter plot objects
    """

    # Default titles
    if titles is None:
        titles = ['$B_x$', '$B_y$', '$B_z$']

    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, figure=fig, hspace=0.3)

    # Create axes with Mollweide projection
    axes = []
    for i in range(3):
        ax = fig.add_subplot(gs[i, 0], projection="mollweide")
        axes.append(ax)

    # Data to plot
    B_fields = [B_x, B_y, B_z]
    ims = []

    # Determine shared color limits if needed
    if share_colorbar:
        all_data = np.concatenate([B_x, B_y, B_z])
        vmin = np.nanquantile(all_data, 0.05)
        vmax = np.nanquantile(all_data, 0.95)

    # Plot each component
    for ax, B_field, title in zip(axes, B_fields, titles):

        # Determine color limits
        if not share_colorbar:
            vmin = np.nanquantile(B_field, 0.05)
            vmax = np.nanquantile(B_field, 0.95)

        # Plot with appropriate normalization
        if cnorm == 'symlog':
            im = ax.scatter(
                lon,
                lat,
                c=B_field,
                s=1,
                cmap=cmc.vik,
                rasterized=True,
                norm=colors.SymLogNorm(
                    linthresh=0.03,
                    linscale=0.03,
                    vmin=vmin,
                    vmax=vmax,
                    base=10
                )
            )
        else:
            im = ax.scatter(
                lon,
                lat,
                c=B_field,
                s=1,
                cmap=cmc.oslo,
                rasterized=True,
                vmin=vmin,
                vmax=vmax,
            )

        ax.grid(True)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title, fontsize=16, pad=10)

        ims.append(im)

        # Add individual colorbar if not sharing
        if not share_colorbar:
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                                pad=0.05, label=cbar_label)

    # Add shared colorbar if requested
    if share_colorbar:
        # Add colorbar spanning all three subplots
        cbar = fig.colorbar(ims[0], ax=axes, orientation="horizontal",
                            pad=0.05, label=cbar_label,
                            fraction=0.046, aspect=40)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')

    return fig, axes, ims

def plot_four_component_mollweide(lon, lat, B_x, B_y, B_z, B_w,
                                  output_file='four_component_map.png',
                                  cnorm='symlog',
                                  figsize=(16, 32),
                                  titles=None,
                                  cbar_label="B field [nTesla]",
                                  share_colorbar=False):
    """
    Plot four magnetic field components on Mollweide projection in 4x1 grid.

    Parameters:
    -----------
    lon : array-like
        Longitude values in [-pi, pi]
    lat : array-like
        Latitude values
    B_x, B_y, B_z, B_w : array-like
        Four magnetic field components
    output_file : str
        Output filename
    cnorm : str, optional
        'symlog' for symmetric log normalization
    figsize : tuple
        Figure size (width, height)
    titles : list of str, optional
        Custom titles for each subplot. Default: ['$B_x$', '$B_y$', '$B_z$', '$B_w$']
    cbar_label : str
        Label for colorbar
    share_colorbar : bool
        If True, use one colorbar for all four plots with shared scale.
        If False, each plot gets its own colorbar.

    Returns:
    --------
    fig : matplotlib figure
    axes : list of axes
    ims : list of scatter plot objects
    """

    # Default titles
    if titles is None:
        titles = ['$B_x$', '$B_y$', '$B_z$', '$B_{total}$']

    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(100, 100, figure=fig, hspace=0.3,
                  wspace=0.1)

    # Create axes with Mollweide projection
    axes = []
    for i in range(4):
        ax = fig.add_subplot(gs[(i //2)*50:(i //2)*50 + 45,
                                 i%2*50:(i%2)*50+45],
                             projection="mollweide")
        axes.append(ax)

    # Data to plot
    B_fields = [B_x, B_y, B_z, B_w]
    ims = []

    # Determine shared color limits if needed
    if share_colorbar:
        all_data = np.concatenate([B_x, B_y, B_z, B_w])
        vmin = -1 # np.nanquantile(all_data, 0.05)
        vmax = 1 # np.nanquantile(all_data, 0.95)

    # Plot each component
    for ax, B_field, title in zip(axes, B_fields, titles):

        # Determine color limits
        if not share_colorbar:
            vmin = -5  # np.nanquantile(all_data, 0.05)
            vmax = 5  # np.nanquantile(all_data, 0.95)

        # Plot with appropriate normalization
        if ax == axes[-1]:
            im = ax.scatter(
                lon,
                lat,
                c=B_field,
                s=1,
                cmap=cmc.oslo,
                rasterized=True,

                norm=colors.LogNorm(
                    vmin=1e-2,
                    vmax=5)
            )
        elif cnorm == 'symlog':
            im = ax.scatter(
                lon,
                lat,
                c=B_field,
                s=1,
                cmap=cmc.vik,
                rasterized=True,
                norm=colors.SymLogNorm(
                    linthresh=1e-2,
                    linscale=1e-2,
                    vmin=vmin,
                    vmax=vmax,
                    base=10
                )
            )


        ax.grid(True)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title, fontsize=16, pad=10)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
        ims.append(im)

        # Add individual colorbar if not sharing
        if not share_colorbar:
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                                pad=0.15, label=cbar_label)

    # Add shared colorbar if requested
    if share_colorbar:
        # Add colorbar spanning all four subplots
        cbar = fig.colorbar(ims[0], ax=axes, orientation="horizontal",
                            pad=0.05, label=cbar_label,
                            fraction=0.046, aspect=40)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return fig, axes, ims

if __name__ == "__main__":
    lunar_data = np.genfromtxt("../../data/Moon_Mag_100km.txt", delimiter=' ', skip_header=True)

    latitude = deg_to_rad(lunar_data[:, 0])
    longitude = deg_to_rad(lunar_data[:, 1]) + np.pi

    lunar_coords = geographic_to_Mollweide_point(np.vstack((longitude, latitude)).T)

    lunar_pts = np.vstack((np.ones_like(latitude) * 1837e3, latitude, deg_to_rad(lunar_data[:, 1]))).T

    B_r = lunar_data[:, 4]
    B_theta = lunar_data[:, 3]
    B_phi = lunar_data[:, 2]

    B_vec = np.vstack((B_r, B_theta, B_phi)).T

    B_vec_cart = geographic_to_Cartesian_vector(lunar_pts, B_vec)

    B_field = np.linalg.norm(B_vec_cart, axis=1)
    # B_field = (B_field - B_field.min()) / (B_field.max() - B_field.min())

    plt.figure(figsize=(16, 10))
    ax = plt.subplot(111, projection="mollweide")

    # longitude must be in [-pi, pi]
    lon = deg_to_rad(lunar_data[:, 1])
    lat = deg_to_rad(lunar_data[:, 0])

    plot_mollweide_map(lon, lat, B_field,
                       output_file='../../data/moon_Btotal.png')

    plot_mollweide_map(lon, lat, B_r,
                       output_file='../../data/moon_B_r.png', cnorm='symlog')

    plot_mollweide_map(lon, lat, B_theta,
                       output_file='../../data/moon_B_theta.png', cnorm='symlog')

    plot_mollweide_map(lon, lat, B_phi,
                       output_file='../../data/moon_B_phi.png', cnorm='symlog')