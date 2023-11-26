import numpy as np
import pandas as pd
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

TEST_SIZE = 0.3
EPOCHS = 10
BATCH_SIZE = 250
ACTV_FUNC = 'relu'


def loadData():
    train = pd.read_csv("train.csv")

    y = train["label"]
    x = train.drop(labels=["label"], axis=1)

    x = x / 255.0

    x = x.values.reshape(-1, 28, 28, 1)

    y = to_categorical(y, num_classes=10)

    return x, y


X_train, Y_train = loadData()

# Separar dataset em sets de treinamento e teste
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=TEST_SIZE, random_state=2)
print("x_train shape", X_train.shape)
print("x_test shape", X_val.shape)
print("y_train shape", Y_train.shape)
print("y_test shape", Y_val.shape)

model = Sequential()
#
model.add(Conv2D(filters=8, kernel_size=(5, 5),
                 activation=ACTV_FUNC, input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
#
model.add(Conv2D(filters=16, kernel_size=(3, 3),
                 activation=ACTV_FUNC))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation=ACTV_FUNC))
model.add(Dense(10, activation="softmax"))

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Aumento de dados
datagen = ImageDataGenerator(
    featurewise_center=False,  # define a média de entrada como 0 sobre o conjunto de dados
    samplewise_center=False,  # define a média de cada amostra como 0
    featurewise_std_normalization=False,  # divide as entradas pelo desvio padrão do conjunto de dados
    samplewise_std_normalization=False,  # divide cada entrada pelo seu desvio padrão
    zca_whitening=False,  # redução de dimensão ZCA
    rotation_range=5,  # rotaciona aleatoriamente as imagens no intervalo de 5 graus
    zoom_range=0.1,  # Ampliação aleatória da imagem em 10%
    width_shift_range=0.1,  # desloca aleatoriamente as imagens horizontalmente em 10%
    height_shift_range=0.1,  # desloca aleatoriamente as imagens verticalmente em 10%
    horizontal_flip=False,  # inverte aleatoriamente as imagens horizontalmente
    vertical_flip=False)  # inverte aleatoriamente as imagens verticalmente

datagen.fit(X_train)

# Fit do modelo
history = model.fit(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                    epochs=EPOCHS, validation_data=(X_val, Y_val),
                    steps_per_epoch=X_train.shape[0] // BATCH_SIZE)

# Desenha as curvas de perda e precisão para treinamento e validação
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Preve os valores do conjunto de dados de validação
Y_pred = model.predict(X_val)
# Converte classes de predições em um vetor
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Converte observações de validação em um vetor
Y_true = np.argmax(Y_val, axis=1)
# computa a matriz de confusao
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# Mostra a matriz de confusao
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.0f', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
