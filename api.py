from flask import Flask, request
import pandas as pd
import wfdb
from wfdb import processing
import json

DATA_DIR = 'data/lobachevsky-university-electrocardiography-database-1.0.1'

# Preprocessing
df = pd.read_csv(f'{DATA_DIR}/ludb.csv')
df['Age'] = df['Age'].apply(lambda x: x.replace('\n', ''))
df['Age'].replace('>89', 90, inplace=True)
df['Age'] = df['Age'].apply(pd.to_numeric)

app = Flask(__name__)


def get_data(age_start, age_stop, limit=1):
    """
    Получение данных пациентов в промежутке возраста от age_start до age_stop (limit по количеству пациентов)
    :return: dict
    """
    data_for_age_range = df.loc[(age_start < df['Age']) & (df['Age'] < age_stop)]

    result = {'data': []}

    for i in range(0, limit):
        try:
            fid = data_for_age_range.iloc[i]['ID']

            signal, fields = wfdb.rdsamp(f'{DATA_DIR}/data/{fid}')
            qrs_locs = processing.gqrs_detect(signal[:, 0], fs=fields['fs'])
            result['data'].append({'ecg_data': signal.tolist(), 'qrs_ix': qrs_locs.tolist(), 'fields': fields})
        except Exception as e:
            print('Error (getting ecg data): {}'.format(e))
    print(result)
    result = json.dumps(result)
    return result


@app.route('/', methods=['GET'])
def api():

    data = '{}'
    if request.method == 'GET':
        age_start = request.args.get('age_start', 21)
        age_stop = request.args.get('age_stop', 35)
        limit = request.args.get('limit', 1)
        data = get_data(age_start, age_stop, limit)

    return data


if __name__ == "__main__":
    app.run('localhost', 9000)
