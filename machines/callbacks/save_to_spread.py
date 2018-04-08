from time import time

import pandas as pd
from gspread_pandas import Spread

from keras.callbacks import Callback


class SaveToSpread(Callback):
    def __init__(self, dataset, machine, additional_parameters, file_weights):
        super().__init__()

        self.dataset = dataset
        self.machine = machine
        self.additional_parameters = additional_parameters
        self.file_weights = file_weights

        self._start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self._start_time = time()

    def on_epoch_end(self, epoch, logs=None):
        spread = Spread('maciek.szpak27@gmail.com',
                        'https://docs.google.com/spreadsheets/d/1CksJQdyochF1M6RB4XfFgewjYGak38ixOhoAMySWTM8/edit')
        df_results = spread.sheet_to_df(sheet='Quantitative')
        df_results = df_results.reset_index()

        df_results = df_results.append({'Dataset': self.dataset,
                                        'Machine': self.machine,
                                        'Training time': time() - self._start_time,
                                        'Epochs': epoch,
                                        'Loss': logs.get('loss'),
                                        'Accuracy': logs.get('acc'),
                                        'Validation loss': logs.get('val_loss'),
                                        'Validation accuracy': logs.get('val_acc'),
                                        'Weights file': self.file_weights.format(epoch=epoch, **logs),
                                        'Additional parameters': self.additional_parameters}, ignore_index=True)
        spread.df_to_sheet(df_results, sheet='Quantitative', index=False, replace=True)
