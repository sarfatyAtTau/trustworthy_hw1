"""
Code for training a fully connected neural network with 2 hidden
layers, each with 50 neurons, to predict digit parity.
"""
import utils
import consts
import numpy as np

from neural_networks import Model, DenseLayer, SigmoidActLayer, ReLUActLayer

np.random.seed(consts.SEED)

# load data reshape it
x_train, y_train, x_val, y_val, x_test, y_test = utils.read_hw_data()
x_train = x_train.reshape((len(x_train), -1))
x_val = x_val.reshape((len(x_val), -1))
x_test = x_test.reshape((len(x_test), -1))
y_train = y_train%2
y_val = y_val%2
y_test = y_test%2

# in/out dims
input_dim = x_train[0].size
n_classes = len(np.unique(y_train))

# build layers
layers = []
layers.append( DenseLayer(input_dim, 50) )
layers.append( ReLUActLayer() )
layers.append( DenseLayer(50, 50) )
layers.append( ReLUActLayer() )
layers.append( DenseLayer(50, 1) )
layers.append( SigmoidActLayer() )

# init model
model = Model(layers, lr=consts.LR, loss='mse')

# train
for e in range(3*consts.EPOCHS):
    # shuffle
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]
    # fit/update
    for i in range(0, len(x_train), consts.BATCH_SZ):
        model.fit_step(x_train[i:i+consts.BATCH_SZ], \
                       y_train[i:i+consts.BATCH_SZ])
    # validate
    preds = []
    for i in range(0, len(x_val), consts.BATCH_SZ):
        preds.append( model.predict(x_val[i:i+consts.BATCH_SZ]) )
    preds = np.concatenate(preds, axis=0).squeeze()
    val_acc = np.sum((preds>0.5)==y_val)/len(y_val)
    print(f'Val accuracy @ end of epoch {e}: {val_acc:0.4f}')

# test
preds = []
for i in range(0, len(x_test), consts.BATCH_SZ):
    preds.append( model.predict(x_test[i:i+consts.BATCH_SZ]) )
preds = np.concatenate(preds, axis=0).squeeze()
test_acc = np.sum((preds>0.5)==y_test)/len(y_test)
print(f'Test accuracy: {test_acc:0.4f}')

# save weights
model.save_weights('models/main_d.npz')
