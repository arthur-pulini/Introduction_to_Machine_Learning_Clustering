import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

#Validação relativa dos métodos
def clusteringAlgorithm(n_clusters, datas):
    kmeans = KMeans(n_clusters = n_clusters, n_init = 10, max_iter = 300)
    labels = kmeans.fit_predict(datas)
    s = metrics.silhouette_score(datas, labels, metric='euclidean')
    dbs = metrics.davies_bouldin_score(datas, labels)
    calinski = metrics.calinski_harabasz_score(datas, labels)
    return s, dbs, calinski

s1, dbs1, calinski1 = clusteringAlgorithm(3, values)
print(s1, dbs1, calinski1)

s2, dbs2, calinski2 = clusteringAlgorithm(5, values)
print(s2, dbs2, calinski2)

s3, dbs3, calinski3 = clusteringAlgorithm(10, values)
print(s3, dbs3, calinski3)

s4, dbs4, calinski4 = clusteringAlgorithm(20, values)
print(s4, dbs4, calinski4)

s5, dbs5, calinski5 = clusteringAlgorithm(50, values)
print(s5, dbs5, calinski5)

#A partir das comparações será escolhida a configurção com 5 clusters, ela se mostrou mais eficaz com os resultados de silhouette e davies_bouldin,
#já o calinski_harabasz não teve uma variação significativa se comparado com a de 3 clusters

#Comparando os dados com dados aleatórios
randomData = np.random.rand(8950, 16)
s6, dbs6, calinski6 = clusteringAlgorithm(5, randomData)
print(s6, dbs6, calinski6)

print('.')

#validando a estabilidade do cluster
set1, set2, set3, = np.array_split(values, 3)
sSet1, dbsSet1, calinskiSet1 = clusteringAlgorithm(5, set1)
print(sSet1, dbsSet1, calinskiSet1)

sSet2, dbsSet2, calinskiSet2 = clusteringAlgorithm(5, set2)
print(sSet2, dbsSet2, calinskiSet2)

sSet3, dbsSet3, calinskiSet3 = clusteringAlgorithm(5, set3)
print(sSet3, dbsSet3, calinskiSet3)

#Como os resultados são parecidos, evidencia que o cluster é estável

#Representando graficamente os clustes pelos atributos PURCHASES(valor gasto) e PAYMENTS(Valor pago)
plt.scatter(datas['PURCHASES'], datas['PAYMENTS'], c=labels, s=5, cmap='rainbow')
plt.xlabel("PURCHASES")
plt.ylabel("PAYMENTS")
#plt.show()

#Representando graficamente os outros atributos, pois apenas com os anteriores não é possível tirar alguma conclusão
datas["cluster"] = labels
#sns.pairplot(datas[0:], hue="cluster") #As cores do gráfico serão diferenciadas pela coluna cluster
#Neste caso por se tratar de um dataframe com muitas caracteriísticas, não fica prática a visualização gráfica, 
#pois, fazendo a plotagem par a par teremos um conjunto de 240 gráficos

#Aplicando outra estratégia para a interpretação dos clusters
print(datas.groupby("cluster").describe())

centroids = kmeans.cluster_centers_
print(centroids)

#A partir do cálculos dos centróides, será avaliada a variâcia dos atributos. Serão escolhidos os atributos que apresentam
#os maiores valores distintos, pois, eles possuem uma maior chance de revelar as particularidades de cada um dos clusters
max = len(centroids[0])
for i in range(max):
    print(datas.columns.values[i], "\n{:.4f}".format(centroids[:, i].var()))
#Escolhendo as variáveis:
#BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS
# 0.0224,    0.0196,       0.0225,       0.0360,   0.0280

description = datas.groupby("cluster")[["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]]
nClients = description.size()
description = description.mean()
description['n_clients'] = nClients
print(description)