from flask import Flask, render_template, request, jsonify, session, g
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .tsfeature_ana import tsfDataAnalysis

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 设置一个密钥用于会话

# 示例数据集
root_path = '/nas/datasets/Tensor-Time-Series-Dataset/Processed_Data/'

datasets = {
    'JONAS_NYC_bike': 'JONAS_NYC_bike',
    'JONAS_NYC_taxi': 'JONAS_NYC_taxi',
    'Metr-LA': 'Metr-LA',
    'METRO_HZ': 'METRO_HZ',
    'METRO_SH': 'METRO_SH',
    'PEMSBAY': 'PEMSBAY',
    'COVID_CHI': 'COVID_CHI',
    'COVID_US': 'COVID_US',
    'COVID_DEATHS': 'COVID_DEATHS',
    'ETT_hour': 'ETT_hour',
    'electricity': 'electricity',
    'weather': 'weather',
    'Jena_climate': 'Jena_climate',
    'nasdaq100': 'nasdaq100',
    'stocknet': 'stocknet',
    'crypto12': 'crypto12'
}

# 加载数据
def load_data(path):
    return pd.read_csv(path)

# 初始化数据
def initialize_ts_analysis(dataset_name):
    pkl_path = os.path.join(root_path, datasets[dataset_name], datasets[dataset_name] + '.pkl')
    return tsfDataAnalysis(pkl_path)

@app.before_request
def before_request():
    if 'dataset' in session:
        g.ts_analysis = initialize_ts_analysis(session['dataset'])

@app.route('/')
def index():
    return render_template('index.html', datasets=datasets)

@app.route('/select_dataset', methods=['POST'])
def select_dataset():
    dataset = request.form['dataset']
    session['dataset'] = dataset
    ts_analysis = initialize_ts_analysis(dataset)
    length, N, M = ts_analysis.data_shape
    return jsonify({'status': 'Dataset selected','shape': (length, N, M)})

@app.route('/get_features', methods=['POST'])
def get_features():
    if 'ts_analysis' not in g:
        return jsonify({'error': 'Dataset not selected'}), 400

    ts_analysis = g.ts_analysis
    N = int(request.form['N'])
    M = int(request.form['M'])

    features = ts_analysis.avaliable_features(N, M)
    feature_list = list(features.keys())

    return jsonify({'features': feature_list})

@app.route('/features', methods=['POST'])
def features():
    if 'ts_analysis' not in g:
        return jsonify({'error': 'Dataset not selected'}), 400

    ts_analysis = g.ts_analysis
    N = int(request.form['N'])
    M = int(request.form['M'])
    selected_features = request.form.getlist('selected_features')

    features = ts_analysis.avaliable_features(N, M)
    selected_features_data = {feature: features[feature] for feature in selected_features}

    return jsonify(selected_features_data)

@app.route('/plot', methods=['POST'])
def plot():
    if 'ts_analysis' not in g:
        return jsonify({'error': 'Dataset not selected'}), 400

    ts_analysis = g.ts_analysis
    N = int(request.form['N'])
    M = int(request.form['M'])
    start = int(request.form['start'])
    end = request.form['end']
    end = int(end) if end else None

    selected_features = request.form.getlist('selected_features')

    x, series = ts_analysis.plot_all_modality(N, M, start, end)
    T, num_modalities = series.shape  # 使用不同的变量名来表示子图的数量
    
    plots = []
    for i in range(num_modalities):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x, series[:, i])
        ax.set(xlabel='Time', ylabel='Value', title=f'{ts_analysis.dataset}, Node {N}, Modality {i}')
        ax.grid()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

        features = {feature: ts_analysis.avaliable_features(N, i).get(feature) for feature in selected_features}
        plots.append({'plot_url': f'data:image/png;base64,{plot_url}', 'features': features})

    return jsonify({'plots': plots})

# @app.route('/plot', methods=['POST'])
# def plot():
#     if 'ts_analysis' not in g:
#         return jsonify({'error': 'Dataset not selected'}), 400

#     ts_analysis = g.ts_analysis
#     N = int(request.form['N'])
#     M = int(request.form['M'])
#     start = int(request.form['start'])
#     end = request.form['end']
#     end = int(end) if end else None

#     x, series = ts_analysis.plot_single_ts(N, M, start, end)
#     # T,M = series.shape
    
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.plot(x, series)
#     ax.set(xlabel='Time', ylabel='Value', title=f'{ts_analysis.dataset}, Node {N}, Modality {M}')
#     ax.grid()

#     img = io.BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     plot_url = base64.b64encode(img.getvalue()).decode()

#     return jsonify({'plot_url': f'data:image/png;base64,{plot_url}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)