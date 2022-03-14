"""
Evaluate the FGSM attack on the model trained in main_e.py
"""
import numpy as np
import utils
import consts

from attacks import FGSM
from neural_networks import Model, DenseLayer, ReLUActLayer, SoftmaxActLayer

np.random.seed(consts.SEED)

# load data reshape it
x_train, y_train, x_val, y_val, x_test, y_test = utils.read_hw_data()
x_train = x_train.reshape((len(x_train), -1))
x_val = x_val.reshape((len(x_val), -1))
x_test = x_test.reshape((len(x_test), -1))

# in/out dims
input_dim = x_train[0].size
n_classes = BLANK

# build layers
layers = []
layers.append( DenseLayer(input_dim, 128) )
layers.append( ReLUActLayer() )
layers.append( DenseLayer(128, 128) )
layers.append( ReLUActLayer() )
layers.append( DenseLayer(128, n_classes) )
layers.append( SoftmaxActLayer() )

# load model
model = Model(layers, lr=consts.LR, loss='cce')
model.load_weights('models/main_e.npz')

# run untargeted FGSM
attack = FGSM(consts.ADV_EPS, loss='cce')
preds = []
for i in range(0, len(x_test), consts.BATCH_SZ):
    x_adv = attack.execute(model, \
                           x_test[i:i+consts.BATCH_SZ], \
                           y_test[i:i+consts.BATCH_SZ])
    preds.append( model.predict(x_adv) )
preds = np.concatenate(preds, axis=0).squeeze()
attack_sr = BLANK # the attack success rate (in [0, 1])
print(f'The success rate of untargeted FGSM: {attack_sr:0.4f}')

# run targeted FGSM
preds = []
y_target = (y_test + np.random.randint(1, n_classes, y_test.shape))%n_classes
for i in range(0, len(x_test), consts.BATCH_SZ):
    x_adv = attack.execute(model, \
                           x_test[i:i+consts.BATCH_SZ], \
                           y_target[i:i+consts.BATCH_SZ], \
                           targeted=True)
    preds.append( model.predict(x_adv) )
preds = np.concatenate(preds, axis=0).squeeze()
attack_sr = BLANK # the attack success rate (in [0, 1])
print(f'The success rate of targeted FGSM: {attack_sr:0.4f}')
