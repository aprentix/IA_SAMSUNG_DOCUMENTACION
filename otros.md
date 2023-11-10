## Kmeans
Se usa cuando tenemos un montón de datos sin etiquetar. (sin la columna de resultado), pues no es supervisado.
Su objetivo es encontrar K grupos (clusters) entre los datos crudos.
La Y no se usa, se ignora completamente. En lugar de la X se pasa una figura
El fit y el predict se usan en este caso para hacer las dos cosas, con el mismo conjunto, porque no se pasa la Y.
https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
(Para hallar el numero optimo de clusters)
## Clustering Jerárquico
Se hacen agrupaciones cercanas por el metodo del centroide. Se expande.
 (Medidas de similaridad [Ward, vinculacion maxima, media, unica])
 Etiquetas halladas por el clustering identificativa (representativa)
## Medidas de rendimiento de clustering
No se puede evaluar los modelos de clustering (si están correctos o no)
Podemos ver que la distancia entre los puntos sea pequeña, pero la distancia entre los cluster sea grande (cómo de bonita es la clusterización)
The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters. Mas cercano a uno es más bonito, mejor ha funcionado el algoritmo.
Lo dificil es ver por qué el modelo ha decidido que dos muestras se parecen.
Interpretacion para ver si funciona o no, son reglas negocio (se usa para comprobar, eso es un sesgo)

## Dimensionality reduction
RSA => (objetivo: reducir el numero de columnas del dataset)
Cuando estan correlacionadas muchas variable, merece la pena hacer Reduccion dimensional. Cuando no, se corre el riesgo de perder muchos datos.
Para ver si merece la pena, tenemos heuristicos especiales
pca.explained_variance_ratio_
se queda con las columnas que tienen menos varianza.
Los modelos van más alla de lo que se piensa, evita la paradoja del ganador (que cuenta solo cuando va bien y no cuando va mal)
Lo interesante es la capacidad de descubrimiento de los alg.
Para sistemas de recomendacion, el modelo ha de reentrenarse a cada rato.
LDA => (LatentDirichletAllocation)

## PCA comprobacion si beneficia
pca = PCA(n_components=13).fit(X)

plt.figure(figsize=(6,4))

xx = np.arange(1, 14, step=1)
yy = np.cumsum(pca.explained_variance_ratio_)

plt.plot(xx, yy)
plt.xlabel('Número de componentes principales')
plt.ylabel('Varianza explicada')
## Regresion, regresion y PCA, comparacion
## si aparece una recta |__ entonces, no merece la pena hacer PCA
## nos quedamos con el minimo de columnas
