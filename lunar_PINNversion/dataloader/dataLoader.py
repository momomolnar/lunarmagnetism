import numpy as np
import torch
import yaml
from lunar_PINNversion.dataloader.util import spherical_vector_to_cartesian, spherical_to_cartesian
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

## Define classes for all data types here

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

def load_orbital_data(path, source, R_lunar, device):
    """Adjust dispatch logic as you extend sources."""
    if source == "LRO":
        loader = Lunar_data_loader(filename=path)
    elif source == "Kaguya":
        loader = Lunar_surface_data_loader(filename=path)
    else:
        raise ValueError(f"Unknown orbital source '{source}' in {path}")

    x = np.array(loader.x_coord, dtype=np.float64)
    y = np.array(loader.y_coord, dtype=np.float64)
    z = np.array(loader.z_coord, dtype=np.float64)
    pts = np.stack([x, y, z], axis=-1) / float(R_lunar)
    Bx = np.array(loader.b_x, dtype=np.float64)
    By = np.array(loader.b_y, dtype=np.float64)
    Bz = np.array(loader.b_z, dtype=np.float64)
    B = np.stack([Bx, By, Bz], axis=-1)
    return torch.tensor(pts, dtype=torch.float32).to(device), torch.tensor(B, dtype=torch.float32).to(device)

def load_surface_amp_data(path, source, R_lunar, device):
    if source == "Apollo":
        loader = Lunar_surface_data_loader(filename=path)
    elif source == "LRO-ER":
        loader = Lunar_surface_ER_data_loader(filename=path)
    else:
        raise ValueError(f"Unknown surface amplitude source '{source}'")
    pts = np.stack([loader.x_coord, loader.y_coord, loader.z_coord], axis=-1) / float(R_lunar)
    amp = torch.tensor(loader.B, dtype=torch.float32).to(device)
    return torch.tensor(pts, dtype=torch.float32).to(device), amp

def load_surface_vector_data(path, source, R_lunar, device):
    if source == "Apollo":
        loader = Lunar_surface_data_loader(filename=path)
    elif source == "MagROVER":
        loader = Lunar_surface_data_loader(filename=path) # Or a custom one
    else:
        loader = Lunar_surface_data_loader(filename=path)
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

def build_data_loaders(data_section, R_lunar, device, batch_size):
    """Build DataLoaders for all file entries, using source-specific logic."""
    orbital_dl, surf_amp_dl, surf_vec_dl = [], [], []

    # Orbital vector data
    for entry in data_section.get('orbital', []):
        pts, B = load_orbital_data(entry['path'], entry['source'], R_lunar, device)
        orbital_dl.append(DataLoader(TensorDataset(pts, B), batch_size=batch_size, shuffle=True))

    # Surface amplitude data
    for entry in data_section.get('surface_amplitude', []):
        pts, amp = load_surface_amp_data(entry['path'], entry['source'], R_lunar, device)
        surf_amp_dl.append(DataLoader(TensorDataset(pts, amp), batch_size=batch_size, shuffle=True))

    # Surface vector data
    for entry in data_section.get('surface_vector', []):
        pts, B = load_surface_vector_data(entry['path'], entry['source'], R_lunar, device)
        surf_vec_dl.append(DataLoader(TensorDataset(pts, B), batch_size=batch_size, shuffle=True))

    return orbital_dl, surf_amp_dl, surf_vec_dl

def generate_collocation_loader(n_points, R_lunar, spherical_to_cartesian, device, batch_size, r_offset=0):
    domain = np.random.rand(n_points, 3)
    domain[:, 0] = domain[:, 0] * 1e5 + float(R_lunar) + r_offset
    domain[:, 1] = domain[:, 1] * np.pi - np.pi / 2
    domain[:, 2] = domain[:, 2] * 2 * np.pi - np.pi
    domain_xyz = np.array([spherical_to_cartesian(el[0] / float(R_lunar), el[1], el[2]) for el in domain])
    domain_xyz = torch.tensor(domain_xyz, dtype=torch.float32).to(device)
    return DataLoader(TensorDataset(domain_xyz), batch_size=batch_size, shuffle=True)

def concat_data_loaders(loader_list, batch_size):
    if not loader_list:
        return None
    dataset = ConcatDataset([l.dataset for l in loader_list])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
