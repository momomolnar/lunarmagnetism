import numpy as np
import torch
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

def load_orbital_data(file_path, R_lunar, device):
    # Load your specific orbital data here; this is schematic!
    loader = Lunar_data_loader(filename=file_path)
    pts = np.stack([loader.x_coord, loader.y_coord, loader.z_coord], axis=-1) / R_lunar
    B = np.stack([loader.b_x, loader.b_y, loader.b_z], axis=-1)
    return torch.tensor(pts, dtype=torch.float32).to(device), torch.tensor(B, dtype=torch.float32).to(device)

def load_surface_amp_data(file_path, R_lunar, device):
    # Adapt if your format is different!
    loader = Lunar_surface_data_loader(filename=file_path)
    pts = np.stack([loader.x_coord, loader.y_coord, loader.z_coord], axis=-1) / R_lunar
    amp = torch.tensor(loader.B, dtype=torch.float32).to(device)
    return torch.tensor(pts, dtype=torch.float32).to(device), amp

def load_surface_vector_data(file_path, R_lunar, device):
    loader = Lunar_surface_data_loader(filename=file_path)
    pts = np.stack([loader.x_coord, loader.y_coord, loader.z_coord], axis=-1) / R_lunar
    B = np.stack([loader.b_x, loader.b_y, loader.b_z], axis=-1)
    return torch.tensor(pts, dtype=torch.float32).to(device), torch.tensor(B, dtype=torch.float32).to(device)

def make_loader_from_points(points, targets, batch_size, shuffle=True):
    dataset = TensorDataset(points, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def build_all_data_loaders(config, device):
    R_lunar = config['R_lunar']
    batch_size = config['batch_size']
    data_spec = config['data']

    # Orbital data (vector measurements on orbit)
    orbital_loaders = []
    for file in data_spec.get('orbital', []):
        pts, vecs = load_orbital_data(file, R_lunar, device)
        orbital_loaders.append(make_loader_from_points(pts, vecs, batch_size, shuffle=True))

    # Surface data: amplitude only
    surface_amp_loaders = []
    for file in data_spec.get('surface_amplitude', []):
        pts, amps = load_surface_amp_data(file, R_lunar, device)
        surface_amp_loaders.append(make_loader_from_points(pts, amps, batch_size, shuffle=True))

    # Surface data: vectors (Bx, By, Bz)
    surface_vec_loaders = []
    for file in data_spec.get('surface_vector', []):
        pts, vecs = load_surface_vector_data(file, R_lunar, device)
        surface_vec_loaders.append(make_loader_from_points(pts, vecs, batch_size, shuffle=True))

    return {
        'orbital': orbital_loaders,
        'surface_amplitude': surface_amp_loaders,
        'surface_vector': surface_vec_loaders
    }