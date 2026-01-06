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


