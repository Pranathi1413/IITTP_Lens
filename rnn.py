import numpy as np
import os
import tensorflow as tf 
import tensorflow.keras.layers as KL 

def change_dim(x_train):
    n = x_train.shape[0]
    n_x_train = np.zeros((n, 224, 224))
    for i in range(n):
        for j in range(224):
            for k in range(224):
                n_x_train[i][j][k] = x_train[i][j][k][0] + x_train[i][j][k][1] + x_train[i][j][k][2]
                n_x_train[i][j][k] /= 765.0
    return n_x_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
x_train = np.load(os.path.join('datasets','x_train.npy'))
x_test = np.load(os.path.join('datasets','x_test.npy'))
y_train = np.load(os.path.join('datasets','y_train.npy'))
y_test = np.load(os.path.join('datasets','y_test.npy'))
x_train = change_dim(x_train)
x_test = change_dim(x_test)

inputs = KL.Input(shape=(224, 224))
rnn = KL.LSTM(64, activation='relu')(inputs)
outputs = KL.Dense(27, activation='softmax')(rnn)

model = tf.keras.models.Model(inputs, outputs)
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=20)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Loss: {0} - Acc: {1}'.format(test_loss, test_acc))