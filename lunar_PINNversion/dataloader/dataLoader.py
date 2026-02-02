import numpy as np
from lunar_PINNversion.dataloader.util import spherical_vector_to_cartesian, spherical_to_cartesian

class Generic_data_loader():

    def __init__(self, filename):
        self.filename = filename
        self.R_lunar = 1737e3
        self.height_measurement = 1e5

    def __load_data__(self):
        self.data = np.loadtxt(self.filename).T
        self.theta, self.phi = self.data[0] / 180 * np.pi, self.data[1] / 180 * np.pi
        self.b_phi, self.b_theta, self.b_r =self.data[2], self.data[3], self.data[4]

    def __load_surface_data__(self):
        self.data = np.loadtxt(self.filename).T
        self.theta, self.phi = self.data[0] / 180 * np.pi, self.data[1] / 180 * np.pi
        self.B = self.data[2]

class Lunar_data_loader(Generic_data_loader):

    def __init__(self,filename):
        super().__init__(filename)
        self.__load_data__()
        self.transform_spherical_to_cartesian()

    def transform_spherical_to_cartesian(self):
        self.x_coord, self.y_coord, self.z_coord = spherical_to_cartesian((np.ones_like(self.theta) * self.R_lunar
                                                                           +self.height_measurement),
                                                                           self.theta, self.phi)
        self.b_x, self.b_y, self.b_z = spherical_vector_to_cartesian(self.b_r, self.b_theta,
                                                                    self.b_phi,
                                                                    (np.ones_like(self.theta) * self.R_lunar
                                                                    + self.height_measurement),
                                                                     self.theta, self.phi,
                                                                     degrees=True)


class Lunar_surface_data_loader(Generic_data_loader):

    def __init__(self, filename):
        super().__init__(filename)
        self.__load_surface_data__()
        self.transform_spherical_to_cartesian()

    def transform_spherical_to_cartesian(self):
        self.x_coord, self.y_coord, self.z_coord = spherical_to_cartesian((np.ones_like(self.theta) * self.R_lunar),
                                                                          self.theta, self.phi)

class Lunar_surface_ER_data_loader(Lunar_surface_data_loader):
    def __init__(self, filename):
        self.filename = filename
        self.__read_LRO_data__()
        self.R_lunar = 1737e3
        self.transform_spherical_to_cartesian()


    def __read_LRO_data__(self):
        """
        Reads a .csv file and extracts the first, second, third, and fourth columns.

        Args:
            None (uses `self.filename` as the file path)

        Sets:
            self.theta (numpy array): Converted spherical coordinate theta.
            self.phi (numpy array): Converted spherical coordinate phi.
            self.B_sc (numpy array): Magnetic field strength.
            self.alpha_c (numpy array): Angle alpha, in radians.
            self.B (numpy array): A computed magnetic field quantity.
        """
        result = []  # Initialize an empty list to store processed rows

        with open(self.filename, 'r') as file:
            for line in file:
                # Split the line into columns based on whitespace
                columns = line.split()

                if len(columns) >= 4:  # Ensure there are at least 4 columns
                    # Convert columns[1] to columns[4] to floats and store
                    selected_columns = [float(columns[1]), float(columns[2]),
                                        float(columns[3]), float(columns[4])]
                    result.append(selected_columns)

        # Convert the list to a NumPy array for easier mathematical operations
        result = np.array(result)

        # Extract the desired columns and process
        self.phi, self.theta = result[:, 0] / 180 * np.pi, result[:, 1] / 180 * np.pi
        self.phi =  (self.phi + np.pi) % (2*np.pi) - np.pi
        self.B_sc = result[:, 2]
        self.alpha_c = result[:, 3] / 180 * np.pi

        # Compute magnetic field quantity B from B_sc and alpha_c
        self.B = self.B_sc / np.sin(self.alpha_c) ** 2

        # Debug information to verify shapes and contents
        print(f"Loaded data: theta.shape={self.theta.shape}, phi.shape={self.phi.shape}")
        print(f"B_sc.shape={self.B_sc.shape}, alpha_c.shape={self.alpha_c.shape}")