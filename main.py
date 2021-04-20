import pandas as pd
import wfdb
from wfdb import processing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as Model
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from os.path import exists

DATA_DIR = 'data/lobachevsky-university-electrocardiography-database-1.0.1'
DIAGNOSES = None
PARAMS = ['RPM', 'HR']


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
    HR = calculate_heart_rate(fs, r_peaks=r_peaks_ix)
    return {'RPM': RR_middle, 'HR': HR}


def process_data(df):
    global DIAGNOSES
    df['Age'] = df['Age'].apply(lambda x: x.replace('\n', ''))
    df['Age'].replace('>89', 90, inplace=True)
    df['Age'] = df['Age'].apply(pd.to_numeric)
    df['diagnose_class'] = df.apply(lambda x: get_class_of_diagnose(x['Rhythms']), axis=1)
    df['is_healthy'] = df.apply(lambda x: 0 if x['diagnose_class'] != 8 else 1, axis=1)

    df['data'] = df.apply(lambda row: get_data_for_patient(row['ID']), axis=1)
    for param in PARAMS:
        df[param] = df.apply(lambda row: row['data'][param], axis=1)

    df = df.drop(['data'], axis=1)

    return df


def get_data_for_patient(patient):
    signal, fields = wfdb.rdsamp(f'{DATA_DIR}/data/{patient}')
    fs = fields['fs']
    r_peaks_ix = processing.gqrs_detect(signal[:, 0], fs=fs, threshold=1.0)

    features = feature_selection(signal, fs, r_peaks_ix)
    return features


def get_trained_model(df):
    target_data = df.loc[(10 < df['Age']) & (df['Age'] < 20)]
    # target_data = df

    print('Got {} samples'.format(target_data.shape[0]))

    train = target_data.iloc[:].replace(np.nan, 0)

    model = Model(n_neighbors=2)
    model.fit(X=train[['RPM', "HR"]], y=train['is_healthy'])
    print('Model fit: done')
    return model


def start():
    global DIAGNOSES
    if not exists('saved_info_df.csv'):
        print('Reading data file.')
        df = pd.read_csv(f'{DATA_DIR}/ludb.csv')
        DIAGNOSES = np.unique(df['Rhythms'])

        print('Process data:', end='')
        df = process_data(df)
        print('OK!')
        df.to_csv('saved_info_df.csv')
        print('File saved.')
    else:
        df = pd.read_csv('saved_info_df.csv')
        DIAGNOSES = np.unique(df['Rhythms'])

    print('Data processed.')
    return df


def predict(model, data):

    # TESTING PREDICTION

    prediction = model.predict(data[['RPM', "HR"]])
    df_result = pd.DataFrame({'pred': prediction, 'actual': data['is_healthy']})
    print(df_result.head(15))

    return prediction

# if __name__ == "__main__":
    # main()
