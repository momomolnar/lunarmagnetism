import numpy as np

def spherical_to_cartesian(r, theta_rad, phi_rad):
    """
    Transform a single point from spherical coordinates to Cartesian coordinates.

    Parameters:
    r (float): Radius.
    theta_rad (float): Latitude angle in radians (range: -π/2 to π/2).
    phi_rad (float): Azimuthal angle (longitude) in radians (range: -π to π).

    Returns:
    tuple: The Cartesian coordinates (x, y, z).

    Note:
    - theta_rad = 0 is the equator
    - theta_rad = π/2 is the north pole
    - theta_rad = -π/2 is the south pole
    """

    x = r * np.cos(theta_rad) * np.cos(phi_rad)
    y = r * np.cos(theta_rad) * np.sin(phi_rad)
    z = r * np.sin(theta_rad)

    return (x, y, z)


def spherical_vector_to_cartesian(V_r, V_theta, V_phi, r, theta, phi, degrees=False):
    """
    Transform a vector field from spherical (V_r, V_theta, V_phi) to Cartesian (V_x, V_y, V_z).

    Parameters
    ----------
    V_r : array_like
        Radial component of the vector field.
    V_theta : array_like
        Latitudinal component of the vector field (positive northward).
    V_phi : array_like
        Azimuthal component of the vector field (positive eastward).
    r : array_like
        Radius coordinate.
    theta : array_like
        Latitude angle in radians (range: -π/2 to π/2), or degrees if degrees=True.
    phi : array_like
        Azimuthal angle (longitude) in radians (range: -π to π), or degrees if degrees=True.
    degrees : bool, optional
        If True, input angles are given in degrees.

    Returns
    -------
    V_x, V_y, V_z : ndarray
        Cartesian components of the vector field.

    Note:
    - theta = 0 is the equator
    - theta = π/2 (or 90°) is the north pole
    - theta = -π/2 (or -90°) is the south pole
    """
    # Convert to radians if needed
    if degrees:
        theta = np.radians(theta)
        phi = np.radians(phi)

    # Transformation formulas for latitude-based coordinates
    V_x = (
            V_r * np.cos(theta) * np.cos(phi)
            - V_theta * np.sin(theta) * np.cos(phi)
            - V_phi * np.sin(phi)
    )
    V_y = (
            V_r * np.cos(theta) * np.sin(phi)
            - V_theta * np.sin(theta) * np.sin(phi)
            + V_phi * np.cos(phi)
    )
    V_z = V_r * np.sin(theta) + V_theta * np.cos(theta)

    return V_x, V_y, V_z