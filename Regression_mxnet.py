import mxnet as mx 
import logging
import time
from keras.datasets import boston_housing
import numpy as np
import graphviz
from graphviz import Digraph
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score

#(X_train, y_train), (X_valid, y_valid) = boston_housing.load_data()
boston= load_boston()

X_train, X_valid, y_train, y_valid= train_test_split(boston.data, boston.target, test_size=0.2) 

batch_size = 8				#Tamaño de los lotes

train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True, label_name='lin_reg_label')#La base de datos se va a dividir de manera aleatoria en lotes de 128 tanto las imagenes como las y's

val_iter = mx.io.NDArrayIter(X_valid, y_valid, batch_size)


#------------------------CONSTRUCCION DE RED NEURONAL-------------------------------

X = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label') #marcadores de posicion para datos futuros, se va a llenar con datos del train y etiquetas o Y's

fc1  = mx.sym.FullyConnected(data=X,name='fc1' ,num_hidden=32) #data es lo que entra(cuando X se llene en el training)
act1 = mx.sym.Activation(data=fc1, act_type="relu")	
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 16)
act2 = mx.sym.Activation(data=fc2, act_type="relu")
fc3  = mx.sym.FullyConnected(data=act2, num_hidden = 1) #capa de salida con una neurona 
lro = mx.sym.LinearRegressionOutput(data=fc3, label=Y, name="lro") #Label son las Y's qque compararemos

#------------------------CONFIGURACION DEL MODELO-------------------------------
logging.getLogger().setLevel(logging.DEBUG)

model = mx.mod.Module(
    symbol = lro ,
    context=mx.cpu(),
    data_names=['data'],
    label_names = ['lin_reg_label']# network structure
)


#Digraph=mx.viz.plot_network(symbol=lro)
#Digraph.view()

#------------------------Entrenamiento-------------------------------
inicio = time.time()
model.fit(train_iter, eval_data= val_iter,
            #optimizer_params={'learning_rate':0.005, 'momentum': 0.9},
            optimizer= 'adam', 
            num_epoch=32,
            eval_metric='mse',
            batch_end_callback = mx.callback.Speedometer(batch_size, 100))

final = time.time()

tiempo = final-inicio

print(str(tiempo)+ " segundos")

metric = mx.metric.MSE()
model.score(val_iter, metric)

#acc = mx.metric.Accuracy()
#print(acc)
print("Esta es la metrica: " + str(metric))

print(X_valid[11])
print(y_valid[11])
	

#comentario 