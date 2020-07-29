import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('DataSet_Inova.csv')

X = 'cetr.uefl.aeg01.velocidadevento'
Y = 'cetr.uefl.aeg01.potenciaativa'
plt.plot(df[X],df[Y], 'bo', alpha = 0.2)
plt.xlabel(X + ' [m/s]')
plt.ylabel(Y + ' [kW]')
print(df.shape)
plt.show()