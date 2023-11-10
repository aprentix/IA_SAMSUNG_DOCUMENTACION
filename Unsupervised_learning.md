## Clustering (trabaja con filas)
sklearn.cluster.KMeans(n_clusters=2, init=centroids[:2], n_init=1, max_iter=3, random_state=23)
sklearn.cluster.MiniBatchKMeans (se reduce el overfitting y para grandes conjuntos de datos, funciona mejor +componente aleatorio)
sklearn.cluster.DBSCAN(eps=7.0, min_samples=75, metric='manhattan') (para no forzar el numero de clusters) (el inicio es aleatorio)
sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters[j], linkage=linkage[i]) (Clustering Jerárquico) (aglomerativo) (se comienza a crear el cluster con los dos puntos mas cercanos)
sklearn.cluster.SpectralClustering(n_clusters=n_clusters[i], affinity=affinities[j]) (grafos)
## Medidas de rendimiento/calidad de clustering
### SIN ground truth
sklearn.metrics.silhouette_score(X, kmeans_model.labels_, metric='euclidean') (SILHOUETTE)
### CON ground truth (con salida esperada, la y)
sklearn.metrics.cluster.adjusted_rand_score(labels_true, labels_pred) (ARI)
## Dimensionality reduction (trabaja con columnas) (NPR metrica de calidad, distancia a vecinos originales)
sklearn.decomposition.PCA(n_components=2, random_state=42) (Principal Component Reduction) (PCA) (lineal)
sklearn.decomposition.TruncatedSVD(n_components=20, random_state=42) (lineal)
sklearn.decomposition.LatentDirichletAllocation(n_components=n_components,max_iter=5,learning_method='online',learning_offset=50.,random_state=1337)(LDA/Latent Dirichlet Allocation)
sklearn.manifold.Isomap(n_neighbors=30, n_components=2) (manifold learning/variedad) (en el peor de los casos igual que PCA, PCA es rápido)
sklearn.manifold.TSNE(n_components=2, perplexity=perplexity) (para la mejor graficacion, muchos fallos)
## Pasar figuras al fit
sklearn.datasets.make_moons(n_samples=500, noise=.05, random_state=23)
axs[1].scatter(moons[:,0], moons[:,1])
sklearn.datasets.make_blobs(n_samples=300, n_features=2, centers=[[-3,-3], [3,3]], cluster_std=2.5, random_state=23) (generate blobs for clustering)
## graficar clusters 
plot_dbscan(X=X, eps=[0.5, 1.0, 1.5], min_samples=[5, 10, 20]) (no es aleatorio el inicio)
scipy.cluster.hierarchy.linkage(X, 'single')
scipy.cluster.hierarchy.dendrogram(linked,orientation='top',labels=labelList,distance_sort='descending',show_leaf_counts=True)

