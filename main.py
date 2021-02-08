from flask import Flask, request
from model import predict_for_ECG

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():

    text = "Отправьте показатели в GET-запросе. Список показателей: RR, ST, QT, P-tooth, T-tooth"
    if request.method == 'GET':
        pr = request.args.get('pr', 0)
        st = request.args.get('st', 0)
        qt = request.args.get('qr', 0)
        p_tooth = request.args.get('p_tooth', 0)
        t_tooth = request.args.get('t_tooth', 0)

        data = predict_for_ECG(pr=int(pr), st=st, qt=qt, p_tooth=p_tooth, t_tooth=t_tooth)
        text = 'Result: {}'.format(data)

    return text


if __name__ == "__main__":
    app.run('localhost', 9000)
