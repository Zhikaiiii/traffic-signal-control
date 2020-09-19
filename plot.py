# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

test_avg_reward = []
IQL_csv = pd.read_csv('./Logs/test46/Grid9_IQL_metric.csv')
test_avg_reward.append(np.asarray(IQL_csv['test_avg_reward']))

IQL_attention_csv = pd.read_csv('./Logs/test47/Grid9_IQL_Attention_metric.csv')
test_avg_reward.append(np.asarray(IQL_attention_csv['test_avg_reward'])[0: 46])

IQL_LSTM_attention_csv = pd.read_csv('./Logs/test48/Grid9_IQL_LSTM_Attention_metric.csv')
test_avg_reward.append(np.asarray(IQL_LSTM_attention_csv['test_avg_reward']))

IQL_double_attention_csv = pd.read_csv('./Logs/test54/Grid9_heavy_IQL_Double_Attention_metric.csv')
test_avg_reward.append(np.asarray(IQL_double_attention_csv['test_avg_reward']))
idx = np.argmax(test_avg_reward[-1])
plt.figure()
labels = ['IQL', 'IQL_Attention', 'IQL_LSTM_Attention', 'IQL_Double_Attention']
for i in range(4):
    plt.plot(test_avg_reward[i], label=labels[i])
plt.legend()
plt.title('test_avg_reward')
plt.show()