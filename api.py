from flask import Flask, request
import pandas as pd
import wfdb
from wfdb import processing
import json
from main import start, get_trained_model
from random import randint, choice

DATA_DIR = 'data/lobachevsky-university-electrocardiography-database-1.0.1'

df = start()

app = Flask(__name__)

MODEL = None


def get_data(age_start, age_stop, limit=1):
    """
    Получение данных пациентов в промежутке возраста от age_start до age_stop (limit по количеству пациентов)
    :return: dict
    """
    data_for_age_range = df.loc[(age_start < df['Age']) & (df['Age'] < age_stop)]

    result = {'data': []}

    for i in range(0, limit):
        try:
            pdf = data_for_age_range.iloc[i]
            fid = pdf['ID']

            signal, fields = wfdb.rdsamp(f'{DATA_DIR}/data/{fid}')
            qrs_locs = processing.gqrs_detect(signal[:, 0], fs=fields['fs'])
            # result['data'].append({'ecg_data': signal.tolist(), 'qrs_ix': qrs_locs.tolist(), 'fields': fields})

            patient_data = {
                    'avatar': f'require("~/assets/images/users/avatar-{randint(1, 4)}.jpg")',
                    'name': choice(["Смирнов", "Кузнецов", "Попов", "Васильев", "Петров", "Соколов", "Михайлов",
                                    "Фролов", "Журавлёв", "Николаев", "Крылов", "Максимов", "Сидоров", "Осипов", "Белоусов",
                                    "Федотов", "Дорофеев", "Егоров", "Матвеев", "Бобров", "Дмитриев","Калинин", 'Петров', "Васильев", "Тылык", "Иванов", "Пушкин", "Лермонтов", "Данилин", "Ванилин", "Герц", "Тесла", "Улец", "Кершин", "Ким"])+' '+choice(['И.', "А.", "В.", "Д.", "К."])+choice(['И.', "А.", "В.", "Д."]),
                    'age': str(pdf['Age']),
                    'id': str(pdf['ID']),
                    'diagnose': pdf['Rhythms'],
                    'state': 'Отклонение от нормы' if pdf['Rhythms'] != 'Sinus rhythm' else 'В норме',
                    'last_visit': f"{randint(1, 31)}/{randint(1, 12)}/2020",
                    'last_data_update': f"{randint(1, 31)}/{randint(1, 12)}/2021",
                    'tmt_online': choice(['Online', 'Offline']),
                    'link': '/dashboard/diagnosis/index2' if pdf['Rhythms'] != 'Sinus rhythm' else '/dashboard/diagnosis',
                  },
            print('{', end='')
            for key, value in patient_data[0].items():

                if key == 'avatar':
                    print(f"{key}: {value}", end=', ')

                else:
                    print(f"{key}: '{value}'", end=', ')
            print('},')
            result['data'].append(patient_data)

        except Exception as e:
            print('Error (getting ecg data): {}'.format(e))
    print(result)
    result = json.dumps(result)
    return result


@app.route('/data', methods=['GET'])
def data():
    data = '{}'
    if request.method == 'GET':
        age_start = request.args.get('age_start', 0)
        age_stop = request.args.get('age_stop', 100)
        limit = request.args.get('limit', 200)
        data = get_data(age_start, age_stop, limit)

    return data


@app.route('/predict_for_id', methods=['GET'])
def predict():
    data = '{}'
    if request.method == 'GET':
        pid = int(request.args.get('id', 0))
        data = df.iloc[pid]
        if MODEL is not None:
            prediction = MODEL.predict([data[['RPM', "HR"]]])
            prediction_text = f'PREDICTION: {"Пациент здоров" if prediction else "Пациент нездоров"}\nACTUAL: {"Пациент здоров" if data["is_healthy"] else "Пациент нездоров"}'
            return prediction_text


def get_model():
    global MODEL
    MODEL = get_trained_model(df)


if __name__ == "__main__":
    get_model()
    app.run('localhost', 9000)
