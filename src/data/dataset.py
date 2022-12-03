import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src.data.curve_generator import *
import torch
import joblib
from src.utils import get_project_root, fix_seq_length
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class lc_dataset(Dataset):
    def __init__(self, seed=42, df=None):
        # Set seed
        self.seed = seed
        np.random.seed(self.seed)

        # Generators
        self.curve_generators = {'square': random_periodic_square_signal,
                                'sawtooth': random_periodic_sawtooth_signal,
                                'sinmix': random_periodic_sin_mix,
                                'gauss':random_gauss_signal}

        # Curve data
        self.period_list = []
        self.mjd_list = []
        self.mag_list = []
        self.labels = []

        if df is not None:
            self.period_list = df.period.values
            self.mjd_list = df.mjd.values
            self.mag_list = df.mag.values
            self.labels = df.label.values

    
    def clean_macho(self, data, normalize=True, encode_labels=True):
        if normalize:
            norm = lambda x: (x - x.mean()) / x.std()
            mag = np.array([norm(lc.measurements) for lc in data], dtype=object)
        else:  
            mag = np.array([lc.measurements for lc in data], dtype=object)
        
        labels = np.array([lc.label for lc in data], dtype=object)
        mask = (labels != 'RRL E') & (labels != 'RRL + GB')
        
        mjd = np.array([lc.times for lc in data], dtype=object)[mask]
        mag = mag[mask]
        labels = labels[mask]

        labels[(labels == 'RRL AB') | (labels == 'RRL C')] = 'RRL'
        labels[(labels == 'LPV WoodA') | (labels == 'LPV WoodB') | (labels == 'LPV WoodC') | (labels == 'LPV WoodD')] = 'LPV'
        labels[(labels == 'Ceph Fund') | (labels == 'Ceph 1st')] = 'Ceph'

        if encode_labels:
            le = preprocessing.LabelEncoder()
            labels = le.fit_transform(labels)

        return mag, mjd, labels

    def add_dataset(self, dataset_name, seq_len, normalize=True, folded=False, encode_labels=True, N=-1):
        full = joblib.load(get_project_root() / 'data' / dataset_name / 'full.pkl')
        if folded:
            for lc in full:
                lc.period_fold()
        if dataset_name == 'macho':
            mag, mjd, labels = self.clean_macho(full, normalize, encode_labels)
        if N != -1:
            idx = np.random.choice(len(mag), N, replace=False)
            mag = mag[idx]
            mjd = mjd[idx]
            labels = labels[idx]
        mag, mjd = fix_seq_length(mag, mjd, seq_len)
        self.mag_list += mag.tolist()
        self.mjd_list += mjd.tolist()
        self.labels += labels.tolist()
        self.period_list += np.zeros_like(labels).tolist()

    def train_test_split(self, test_size, random_state=42):
        df = self.to_df()
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
        return lc_dataset(df = df_train), lc_dataset(df = df_test)


    def generate_periods(self, N, min_period, max_period):
        """Genera una lista de períodos random.

        Parameters
        ----------
        N : int
            Número int que define la cantidad de períodos a generar.
        min_period : float
            float que define el mínimo valor posible a generar.
        max_period : float
            float que define el máximo valor posible a generar.

        Returns
        -------
        np.ndarray
            np.ndarray con el arreglo de períodos generados.            
        """
        random_period_list = []
        for idx in range(N):
            random_period = np.random.uniform(min_period, max_period)
            random_period_list.append(random_period)
        return random_period_list
        
    def add_curves(self, curve_type, N, seq_len, min_period, max_period, label, irregular=True, folded=False, normalize=True):
        """Añade N curvas de un determinado tipo al dataset.

        Parameters
        ----------
        curve_type : str
            str que define el tipo de curvas a generar.
        N : int
            Número int que define la cantidad de períodos a generar.            
        min_period : float
            float que define el mínimo valor posible a generar.
        max_period : float
            float que define el máximo valor posible a generar.
        label : int
            int que define el label que tendrán las curvas generadas.            
        """
        period_list = self.generate_periods(N, min_period, max_period)
        self.period_list += period_list
        for period in period_list:
            mjd, mag = self.curve_generators[curve_type](period, seq_len=seq_len, irregular=irregular)
            if folded:
                mjd = np.remainder(mjd, period) / period # phi
                mag = mag[mjd.argsort()] # sorted mag
                mjd.sort() # sorted phi
            if normalize:
                mag = (mag - mag.mean()) / mag.std()
            self.mjd_list.append(mjd)
            self.mag_list.append(mag)
            self.labels.append(label)

    def to_df(self):
        """Transforma los datos generados en un Pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            pd.DataFrame con los datos generados.            
        """        
        dataset_df = pd.DataFrame({'mjd':self.mjd_list,
                                    'mag':self.mag_list,
                                    'period':self.period_list,
                                    'label':self.labels})
        return dataset_df

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        curve_dict = {'mjd':torch.tensor(self.mjd_list[idx]),
                    'mag':torch.tensor(self.mag_list[idx]),
                    'period':torch.tensor(self.period_list[idx]),
                    'label':torch.tensor(self.labels[idx])}
        return curve_dict