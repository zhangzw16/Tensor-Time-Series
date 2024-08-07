import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv('/home/zhuangjiaxin/workspace/TensorTSL/Tensor-Time-Series/output/csv/96-12-merge-v2.csv')

# 过滤掉包含 'metric_errpr' 的行
df = df[df['mae'] != 'metric_error']

# 将数值列转换为浮点型
df[['mae', 'mape', 'rmse', 'smape']] = df[['mae', 'mape', 'rmse', 'smape']].astype(float)

# 设置绘图风格
sns.set_theme(style="whitegrid")

# 获取唯一的 his_pred 和 dataset
his_preds = df['his_pred'].unique()
datasets = df['dataset'].unique()
metircs = ['mae', 'mape', 'rmse', 'smape']

# model_order = ['DCRNN', 'NET3', 'GraphWaveNet', 'AGCRN', 'MTGNN', 'TTS_Norm', 'ST_Norm', 'GMRL', 
#                'TimesNet', 'STGCN', 'AutoFormer', 'CrossFormer', 'PatchTST', 
#                'HM',]
model_order = ['DCRNN-0', 'DCRNN-2', 'DCRNN-3', 'NET3-0', 'NET3-2', 'AGCRN-0', 'AGCRN-2', 'MTGNN-0', 'MTGNN-2', 'ST_Norm-0', 'ST_Norm-2', 'TimesNet-0', 'StemGNN-0','AutoFormer-0', 'CrossFormer-0', 'PatchTST-0']
# 遍历每个 his_pred 和 dataset 组合
for his_pred in his_preds:
    for dataset in datasets:
        for metric in metircs:
            # 筛选出当前组合的数据
            subset = df[(df['his_pred'] == his_pred) & (df['dataset'] == dataset)].copy()
            
            subset['model'] = pd.Categorical(subset['model'], categories=model_order, ordered=True)
            subset.sort_values(by='model')

            if subset.empty:
                continue
            
            # 创建一个新的图表
            plt.figure(figsize=(20, 8))
            
            # 绘制 MAE 曲线
            # data = subset[metric]
            # sns.lineplot(data=subset, x='model', y=metric, marker='o', label=f'{metric}')
            # 准备数据，将数据从长格式转换为宽格式，以便每个模型的每个指标都能绘制在条形图上
            # melted_subset = subset.melt(id_vars=['model'], value_vars=['mae', 'mape', 'rmse', 'smape'], var_name='Metric', value_name='Value')

            # 绘制条形图
            ax = sns.barplot(data=subset, x='model', y=metric)
            color_map = {}
            for model in model_order:
                color_map[model] = 'b' if '-0' in model else 'g'
            x_labels = [t.get_text() for t in ax.get_xticklabels()]
            for p in ax.patches:
                # # 获取条形图对应的model名称
                # model_name = subset['model'][int(p.get_x() + p.get_width() / 2)]
                # # 设置颜色
                # p.set_color('g')
                x = p.get_x()
                width = p.get_width()
                model_name = x_labels[int(x + width / 2)]
                p.set_color(color_map[model_name])
                ax.annotate(format(p.get_height(), '.2f'),  # 格式化文本
                            (p.get_x() + p.get_width() / 2., p.get_height()),  # 定位文本的位置
                            ha = 'center', va = 'center',  # 水平居中，垂直居中
                            xytext = (0, 9),  # 文本的偏移量
                            textcoords = 'offset points')
            # # 绘制 MAPE 曲线
            # sns.lineplot(data=data, x='model', y='mape', marker='o', label='MAPE')
            
            # # 绘制 RMSE 曲线
            # sns.lineplot(data=data, x='model', y='rmse', marker='o', label='RMSE')
            
            # # 绘制 SMAPE 曲线
            # sns.lineplot(data=data, x='model', y='smape', marker='o', label='SMAPE')
            
            # 设置图表标题和标签
            plt.title(f'Performance Metrics for his_pred={his_pred}, dataset={dataset}, Metric={metric}')
            plt.xlabel('Model')
            plt.ylabel('Metric Value')
            # plt.legend()
            plt.xticks(rotation=45)
            
            # 显示图表
            plt.savefig(f'/home/zhuangjiaxin/workspace/TensorTSL/Tensor-Time-Series/output/imgs/merge-2/{metric}_{his_pred}_{dataset}.png', dpi=300)
            plt.close()