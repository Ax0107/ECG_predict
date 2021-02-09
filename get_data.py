import pandas as pd
import wfdb
from wfdb import processing
import numpy as np
import tensorflow as tf


DATA_DIR = 'data/lobachevsky-university-electrocardiography-database-1.0.1'

df = pd.read_csv(f'{DATA_DIR}/ludb.csv')
df['Age'] = df['Age'].apply(lambda x: x.replace('\n', ''))
df['Age'].replace('>89', 90, inplace=True)
df['Age'] = df['Age'].apply(pd.to_numeric)

data_10_to_21 = df.loc[(16 < df['Age']) & (df['Age'] < 21)]

DIAGNOSES = np.unique(df['Rhythms'])
print('Count of DIAGNOSES:', len(DIAGNOSES))


def get_class_of_diagnose(diagnose):
    global DIAGNOSES
    diagnose = diagnose.replace('Rhythm: ', '').replace('.', '')
    index = np.where(DIAGNOSES == diagnose.replace('Rhythm: ', '')) or -1
    return index[0][0]


def calculate_heart_rate(fs, r_peaks=None, diff=None):
    """
    Рассчитывает пульс в уд./мин.
    """

    if diff is None and r_peaks is None:
        return 0

    if r_peaks is not None:
        # Разница между ударами в ед. индекса
        diff = np.diff(r_peaks)

    if diff is not None:
        heart_rate = np.mean(fs / diff * 60)
        return heart_rate


def feature_selection(signal, fs, r_peaks_ix):
    RR_middle = np.mean(fs / r_peaks_ix)
    HR = calculate_heart_rate(fields['fs'], r_peaks=r_peaks_ix)
    return {'RPM': RR_middle, 'HR': HR}

def main():
    for i in range(0, len(data_10_to_21)):
        fid = data_10_to_21.iloc[i]['ID']

        signal, fields = wfdb.rdsamp(f'{DATA_DIR}/data/{fid}')
        fs = fields['fs']
        r_peaks_ix = processing.gqrs_detect(signal[:, 0], fs=fs, threshold=1.0)

        print(feature_selection(signal, fs, r_peaks_ix), get_class_of_diagnose(fields['comments'][3]))

        # # Проходимся по данным между пиками R
        # for i in range(0, len(r_peaks_ix), 2):
        #     data = signal[r_peaks_ix[i]:r_peaks_ix[i+1]]
        #     data = data.T[0]
        #     print(len(data))
        #
        #     # print("xa, ya = {}, {}".format(maximums[0].tolist(), data[maximums].tolist()), end='\n\n')
        #     break
        # break
#
#
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(2,)),
#   tf.keras.layers.Dense(10, activation='relu'),
#   tf.keras.layers.Dense(10)
# ])
