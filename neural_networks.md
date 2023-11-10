## Keras basics
API - KERAS
FRAMEWORK - TENSORFLOW
LINEAR ALGEBRA - CUDA, BLAS, EIGEN
HARDWARE - GPU, CPU

## CAPAS
https://keras.io/api/layers/
tf.keras.layers.Dense(activation)
tf.keras.layers.ReLU
tf.keras.layers.Dropout ## para controlar el overfitting

## Mirar el overfitting con las graficas
Se mejoran los resultados con el modelo más complejo.

## Entrenamiento por lotes para reducir overfitting
En cada Epoch se usa el dataset completo para ajustar los pesos, dando lugar al overfitting. El tamaño del lote afecta directamente al sobreajuste del modelo
(cuando la validacion tiene tendencia ascendente se produce overfitting, con más capas)
(batch_size = tamaño de lote), subiendo el tamaño, la red aprende más rápido y corre riesgo de overfitting.
