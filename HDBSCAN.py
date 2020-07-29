import hdbscan
import numpy as np
from hdbscan import approximate_predict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

################### CONFIGURAÇÕES DE PLOT/CORES #######################
plot_kwds={'alpha':0.25, 's':60, 'linewidths':0}
pal = sns.color_palette('bright', 16)
legend_elements = [Line2D([0], [0], color='gray', label='Regulares'),
                    Line2D([0], [0], color='b', label='Maquina Parada'),
                    Line2D([0], [0], color='orange', label='Pouca Potência')]
########################### DATA SET ################################
DataSet = pd.read_csv('DataSet_Inova.csv')

#---------------------- convertendo TimeStamp ------------------------
DataSet['TimeStamp'] = pd.to_datetime(DataSet['TimeStamp'])
from datetime import datetime
def datetime_to_float(d):
    epoch = datetime.utcfromtimestamp(0)
    total_seconds =  (d - epoch).total_seconds()
    return total_seconds

for i in range(DataSet.shape[0]):
    DataSet['TimeStamp'][i] = datetime_to_float(DataSet['TimeStamp'][i])
#----------------------- normalizando --------------------------------
DataSet['TimeStamp'] = DataSet['TimeStamp'] - DataSet['TimeStamp'].min()
DataSet['TimeStamp'] = DataSet['TimeStamp']/DataSet['TimeStamp'].max()
DataSet['cetr.uefl.aeg01.velocidadevento'] = DataSet['cetr.uefl.aeg01.velocidadevento']- DataSet['cetr.uefl.aeg01.velocidadevento'].min()
DataSet['cetr.uefl.aeg01.velocidadevento'] = DataSet['cetr.uefl.aeg01.velocidadevento']/DataSet['cetr.uefl.aeg01.velocidadevento'].max()
DataSet['cetr.uefl.aeg01.potenciaativa'] = DataSet['cetr.uefl.aeg01.potenciaativa']-DataSet['cetr.uefl.aeg01.potenciaativa'].min()
DataSet['cetr.uefl.aeg01.potenciaativa'] = DataSet['cetr.uefl.aeg01.potenciaativa']/DataSet['cetr.uefl.aeg01.potenciaativa'].max()

################ SELECIONANDO DADOS PARA TREINAMENTO ###################
Regular = DataSet.loc[lambda DataSet: DataSet['TimeStamp']<0.2,:]
DataSet = DataSet.drop(['TimeStamp'], axis=1)
Regular = Regular.drop(['TimeStamp'], axis=1)
Ruido1 = DataSet.loc[lambda DataSet: DataSet['cetr.uefl.aeg01.potenciaativa']>0.41,:]
Ruido1 = Ruido1.loc[lambda DataSet: Ruido1['cetr.uefl.aeg01.potenciaativa']<0.44,:]
Ruido1 = Ruido1.loc[lambda DataSet: Ruido1['cetr.uefl.aeg01.velocidadevento']>0.47,:]
Ruido2 = DataSet.loc[lambda DataSet: DataSet['cetr.uefl.aeg01.potenciaativa']<0.1,:]
Ruido2 = Ruido2.loc[lambda DataSet: Ruido2['cetr.uefl.aeg01.velocidadevento']>0.2,:]

DataSetTreinamento = np.concatenate((Ruido1, Ruido2, Regular), axis=0)
########################################################################

DataSet = pd.DataFrame(DataSet).to_numpy()
DataSetTreinamento = pd.DataFrame(DataSetTreinamento).to_numpy()

######################## ENTRADA DO HDBSCAN ############################
min_cluster_size = 40 
min_samples=2
######################## TREINANDO HDBSCAN #############################
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data = True).fit(DataSetTreinamento)
colors = [sns.desaturate(pal[col], sat) for col, sat in zip(clusterer.labels_,clusterer.probabilities_)]
plt.subplot(121)
plt.suptitle('HDBSCAN - TREINAMENTO',fontsize=18)
plt.title('Grupos identificados', fontsize=12)
plt.scatter(DataSetTreinamento[:,0], DataSetTreinamento[:,1], c=colors, **plot_kwds)
#--------------------- ISOLANDO OS CLUSTERS 0 e 1 ----------------------
isolado = 1 # 
for i in range(np.size(clusterer.labels_)):
    if clusterer.labels_[i] > isolado:
        clusterer.labels_[i] = 0
        clusterer.probabilities_[i] = 0

plt.subplot(122)
plt.title('Selecionando Grupos desejados', fontsize=12)
colors = [sns.desaturate(pal[col], sat) for col, sat in zip(clusterer.labels_,clusterer.probabilities_)]
plt.scatter(DataSetTreinamento[:,0], DataSetTreinamento[:,1], c=colors, **plot_kwds)

######################## DATASET INTEIRO ################################
test_labels, strengths = hdbscan.approximate_predict(clusterer, DataSet)
test_colors = [pal[col] if col >= 0 and col <=isolado else (0.5, 0.5, 0.5) for col in test_labels]

plt.figure()
plt.title('HDBSCAN - FINAL', fontsize=18)
plt.legend(handles=legend_elements)
plt.scatter(DataSet[:,0], DataSet[:,1], c=test_colors, **plot_kwds)
plt.show()