import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class lc_dataset(Dataset):
    def __init__(self, N, curve_generator, seed=42, min_period=0.01, max_period=5):
        # Set seed
        self.seed = seed
        np.random.seed(self.seed)

        # Configure generators
        self.N = N
        self.curve_generator = curve_generator
        self.min_period = min_period
        self.max_period = max_period

        
        # Generate curves
        self.period_list = self.generate_periods()
        self.mjd_list, self.mag_list = self.generate_curves()

    def generate_periods(self):
        random_period_list = []
        for idx in range(self.N):
            random_period = np.random.uniform(self.min_period, self.max_period)
            random_period_list.append(random_period)
        return random_period_list
        
    def generate_curves(self):
        mjd_list = []
        mag_list = []
        for period in self.period_list:
            mjd, mag = self.curve_generator(period, N=self.N)
            mjd_list.append(mjd)
            mag_list.append(mag)
        return mjd_list, mag_list

    def to_df(self):
        dataset_df = pd.DataFrame({'mjd':self.mjd_list,
                                    'mag':self.mag_list,
                                    'period':self.period_list})
        return dataset_df

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        curve_dict = {'mjd':self.mjd_list[idx],
                    'mag':self.mag_list[idx],
                    'period':self.period_list[idx]}
        return curve_dict