### PREPROCESAMIENTO
- X.drop("column_name") (borrar las columnas que no sean relevantes(mas nan))
- X1.drop_duplicates(inplace=True) (eliminar los duplicados)
- X1.dropna(inplace=True)
- sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean') (meter valores en los nan) (es preferible perder columnas a inventar informacion, en validacion puede generalizar mal)
- (ver las "correlaciones" de las categoricas)
#### Datos continuos
sklearn.preprocessing.StandardScaler() (si hay outliers)
sklearn.preprocessing.MinMaxScaler()
sklearn.preprocessing.RobustScaler() (sin outliers)
sklearn.preprocessing.PowerTransformer()
sklearn.preprocessing.QuantileTransformer(output_distribution='uniform')
##### Discretización
sklearn.preprocessing.Binarizer(threshold=0.0) (categoriza los datos de una columna)
sklearn.preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal')
##### Normalización
sklearn.preprocessing.normalize(X, norm='l1')
scipy.stats.kstest(X, 'norm')
#### Datos discretos
sklearn.preprocessing.OneHotEncoder()
sklearn.preprocessing.OrdinalEncoder()
sklearn.preprocessing.LabelEncoder()
#### Transformacion de columnas
column_transformer = sklearn.compose.ColumnTransformer(transformers=[
    ("drop", "drop", [0]),
    ("passthrough", "passthrough", [1]),
    ("scale", sklearn.preprocessing.StandardScaler(), [2]),
    ("min-max", sklearn.preprocessing.MinMaxScaler(), [3]),
    ("one-hot", sklearn.preprocessing.OneHotEncoder(), [4])
]);
### REGRESION
sklearn.linear_model.LinearRegression() (Regresion lineal)
sklearn.preprocessing.PolynomialFeatures(degree=3, include_bias=False) (Regresion polinomica)  (Usamos el LinearRegression)
sklearn.model_selection.cross_val_score(ridge, X, y, cv=5, scoring='neg_mean_squared_error')
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split()
##### Regularización en Regresion lineal
sklearn.linear_model.Ridge(alpha=alpha) (Ridge regression)
sklearn.linear_model.Lasso(alpha=alpha) (Lasso)
sklearn.linear_model.ElasticNet(alpha=alpha, l1_ratio=r) (Elastic-Net)
sklearn.linear_model.SGDRegressor(max_iter=max_iter, random_state=43) (sgd)
#### Metricas de evaluacion para Regresion
sklearn.metrics.mean_absolute_error(y, y_pred) (MAE)
sklearn.metrics.mean_squared_error(y, y_pred) (MSE)
sklearn.metrics.mean_squared_error(y, y_pred, squared=False) (RMSE)
sklearn.metrics.ridge.score(X,y) (R2)
sklearn.linear_model.LogisticRegression(random_state=42) (lr)(clasificacion)
### CLASIFICACION (clf)
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1) (knn)
sklearn.neighbors.NearestCentroid(shrink_threshold=8) (nc)
sklearn.svm.LinearSVC() (linear_smv)
sklearn.svm.SVC(kernel='rbf', gamma=gamma, C=C, probability=True) (svc)
sklearn.ensemble.RandomForestClassifier(random_state=42) (rf) (random forest) (siempre gana, funciona muy bien) (tambien se puede usar como regresor) (modelos iguales)
sklearn.ensemble.VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='hard', weights=[2, 1, 2]) (todos los ensemble tienen votacion) (cuando es soft no le ponemos pesos) (modelos diferentes)
sklearn.ensemble.BaggingClassifier(clf, n_estimators=8, max_samples=0.05, random_state=1)  (meter en bolsa) (metamodelo formado por otros modelos individuales) (se da prioridad a los valores con los que se equivoca)
sklearn.ensemble.AdaBoostClassifier(n_estimators=n_estimators, random_state=42) (boosting)
sklearn.ensemble.GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=1.0, criterion='mse', max_depth=1, random_state=42) (boosting) (se da retroalimentacion en una cadena) (el modelo ha de ser muy simple ej: LogisticRegresion)
**XGBoost es más eficiente que rf y el gradientBoosting**
sklearn.ensemble.StackingClassifier(estimators=estimators, passthrough=False) (apilar los modelos y retroalimentar los errores) (entrenar un modelo a partir de X_modelos)(no usar)
sklearn.naive_bayes.CategoricalNB() (nb)(naïve bayes)
sklearn.naive_bayes.GaussianNB() (nb)(naïve bayes)
sklearn.naive_bayes.MultinomialNB() (nb)(naïve bayes)
sklearn.naive_bayes.BernoulliNB() (nb)(naïve bayes)
sklearn.naive_bayes.ComplementNB() (nb)(naïve bayes)
sklearn.tree.DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=3, random_state=43)
#### Metricas de evaluacion para Clasificacion
sklearn.metrics.roc_auc_score()
sklearn.metrics.confusion_matrix|ConfusionMatrixDisplay
sklearn.metrics.recall_score()
sklearn.metrics.precision_score()
sklearn.metrics.f1_score (f1)
sklearn.metrics.classification_report(y_true, y_pred, target_names=target_names)
### AJUSTE DE HIPERPARAMETROS
sklearn.model_selection.GridSearchCV(RandomForestClassifier(random_state=42), parameters, verbose=1, n_jobs=-1, cv=3) (para hallar los mejores hiperparametros)
sklearn.model_selection.RandomizedSearchCV(rf, parameters, n_iter=10, verbose=1, n_jobs=-1, cv=3) 
## NUMPY
axs.contourf(xx, yy, Z, cmap='rainbow', alpha=0.7, antialiased=True) ###(draw contour lines and filled contours)
axs.scatter(X[:,0], X[:,1], c=y, cmap='rainbow', edgecolor='black')

https://plotly.com/python/3d-scatter-plots/ # pintar graficos en 3d
https://facebookresearch.github.io/nevergrad/ # para buscar los hiperparametros de forma inteligente
## Buscar la mejor generalizacion (el modelo que generalice más)
