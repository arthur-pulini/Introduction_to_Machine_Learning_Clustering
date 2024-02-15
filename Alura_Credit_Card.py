import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn import metrics

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

#Para este algorito será usado o modelo de clusterização Kmeans, modelo de tipo centróide
#n_init = 10 informa que o kmeans deve rodar 10 vezes seguidas e retornar 10x o mesmo valor, isso para termos confiança no resultado
#max_iter informa o número de iterações
kmeans = KMeans(n_clusters = 5, n_init = 10, max_iter = 300)
yPred = kmeans.fit_predict(values)

#Aplicando a métrica de validação com o coeficiente de silhouette
labels = kmeans.labels_
silhouette = metrics.silhouette_score(values, labels, metric='euclidean')
print(silhouette)

#Aplicando Índice de Davies-Bouldin
dbs = metrics.davies_bouldin_score(values, labels)
print(dbs)

#Aplicando Índice Calinski
calinski = metrics.calinski_harabasz_score(values, labels)
print(calinski)