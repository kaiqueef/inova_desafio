import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
colors = np.array(['red','blue']) #cor 01 e cor 02
legend_elements = [Line2D([0], [0], color=colors[1], label='Regulares'),
                    Line2D([0], [0], color=colors[0], label='Anomalias')]                 
from sklearn.ensemble import IsolationForest

################## ENTRADAS PARA O MODELO ###################
n_estimators = 100
max_samples = 3000
contamination = 0.02
isolation = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination)

############################ DATA SET ##############################
import pandas as pd
DataSet = pd.read_csv('DataSet_Inova.csv')
DataSet = DataSet.drop(['TimeStamp'], axis=1)
maximo_x = max(DataSet.iloc[1:5000].to_numpy()[:,0])
maximo_y = max(DataSet.iloc[1:5000].to_numpy()[:,1])
DataSetTreinamento = pd.DataFrame(DataSet.iloc[0:5000]).to_numpy()
DataSet = DataSet.to_numpy()
####################################################################

##################### MOSTRAR TREINO ISOLATION ############################
y_pred = isolation.fit(DataSetTreinamento).predict(DataSetTreinamento)
plt.subplot(121)
plt.title('Dados treinados', size=12)
plt.scatter(DataSetTreinamento[:, 0], DataSetTreinamento[:, 1], s=10, color=colors[(y_pred + 1) // 2], alpha=0.2)
plt.xlim(0, maximo_x*1.05)
plt.ylim(0, maximo_y*1.05)
plt.legend(handles=legend_elements)
######################## DATASET INTEIRO ################################
y_pred = isolation.fit(DataSetTreinamento).predict(DataSet)
plt.subplot(122)
plt.scatter(DataSet[:, 0], DataSet[:, 1], s=10, color=colors[(y_pred + 1) // 2], alpha=0.2)
plt.xlim(0, maximo_x*1.05)
plt.ylim(0, maximo_y*1.05)
plt.title('DataSet Inteiro', size=12)
plt.legend(handles=legend_elements)

plt.suptitle('Isolation Forest',fontsize=18)
plt.show()