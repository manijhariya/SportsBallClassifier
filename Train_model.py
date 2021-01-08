import pandas as pd
import numpy as np
from sys import exit
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv("ImageCSV.csv").values

# 0 for golf
# 1 for soccer
# 2 for basketball
X = data[:,:-1]
Y = data[:,-1]

X = X.reshape((X.shape[0],100,100,3))
Y = to_categorical(Y)

print ("X shape:",X[0].shape)
print ("Y shape:",Y[0].shape)

class Training_Models:
	def __init__(self,X,Y):
		self.X = X
		self.Y = Y

	def CNN_Model(self,save=False,graph=False):
		model = Sequential()
		model.add(Input(shape=(100,100,3)))
		model.add(Conv2D(32,(3,3), activation = 'relu'))
		model.add(MaxPooling2D(2,2))
		model.add(Conv2D(64,(3,3), activation = 'relu'))
		#model.add(Conv2D(64,(2,2), activation = 'relu'))
		model.add(Conv2D(64,(3,3), activation = 'relu'))
		model.add(MaxPooling2D(2,2))
		model.add(Flatten())
		model.add(Dense(3, activation='softmax'))

		opt = Adam(lr = 0.001)

		callback = EarlyStopping(
				monitor="val_loss",
				mode="auto",
				restore_best_weights=False,
				baseline=None,
				#min_delta=0,patience=0,verbose=0,
				)

		model.compile(optimizer=opt,  # Optimizer
			# Loss function to minimize
    			loss='categorical_crossentropy',
			# List of metrics to monitor
			metrics=['accuracy'])

		history = model.fit(self.X,self.Y,batch_size = 64, epochs = 50,validation_split = 0.2,callbacks=[callback])
		if graph:
			plt.plot(history.history['val_loss'],label='val_loss')
			plt.plot(history.history['loss'],label='loss')
			plt.show()

			plt.plot(history.history['accuracy'],label='accuracy')
			plt.plot(history.history['val_accuracy'],label='val_accuracy')
			plt.show()
		if save:
			model.save('CNN_model.h5')
			model.save_weights('CNN_model_weights.h5')

train = Training_Models(X,Y)
train.CNN_Model(True,True)
