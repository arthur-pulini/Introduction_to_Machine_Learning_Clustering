import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import k_means

datas = pd.read_csv("/home/arthur-pulini/Documentos/Programação/Machine learning Alura/Alura_Credit_Card/CC GENERAL.csv")
datas.drop(columns=['CUST_ID', 'TENURE'], inplace=True) #Retirando algumas colunas que não fazem sentido ao cluster
print(datas.head())

#Procurando dados faltantes 
missing = datas.isna().sum()
print(missing)

#Substituindo os dados faltantes pela média de suas respectivas colunas
datas.fillna(datas.median(), inplace=True)
missing = datas.isna().sum()
print(missing)

#Fazendo a normalização dos dados
values = Normalizer().fit_transform(datas.values)
print(values)