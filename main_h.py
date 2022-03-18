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
n_classes = len(np.unique(y_train))

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
model.load_weights('models/pretrained.npz')

# run untargeted FGSM
tempered_epsilon = consts.ADV_EPS*1.1
attack = FGSM(tempered_epsilon, loss='cce')
preds = []
original_preds = []
x_test_adv_untargeted = []

for i in range(0, len(x_test), consts.BATCH_SZ):
    x_adv = attack.execute(model, \
                           x_test[i:i+consts.BATCH_SZ], \
                           y_test[i:i+consts.BATCH_SZ])
    preds.append(model.predict(x_adv))
    original_preds.append(model.predict(x_test[i:i+consts.BATCH_SZ]))
    x_test_adv_untargeted.append(x_adv)

preds = np.concatenate(preds, axis=0).squeeze()
original_preds = np.concatenate(original_preds, axis=0).squeeze()
x_test_adv_untargeted = np.concatenate(x_test_adv_untargeted, axis=0).squeeze()

categorical_preds = np.argmax(preds, axis=1)
categorical_original_preds = np.argmax(original_preds, axis=1)
original_success = categorical_original_preds == y_test

binary_untargeted_success = np.bitwise_and(categorical_preds != y_test, original_success)
attack_sr = np.sum(binary_untargeted_success)/np.sum(original_success)  # the attack success rate (in [0, 1])
print(f'The success rate of untargeted FGSM: {attack_sr:0.4f}')

for original_tag in np.unique(y_train)[1:2]:
    first_occurrence = np.argwhere(np.bitwise_and(binary_untargeted_success, y_test == original_tag))[0]
    utils.save_as_im(x_test[first_occurrence].reshape(28, 28), f'Images/improve_FGSM/benign_[{original_tag}]_epsilon_{tempered_epsilon}.jpeg')
    utils.save_as_im(x_test_adv_untargeted[first_occurrence].reshape(28, 28), f'Images/improve_FGSM/untargeted_adversarial_[{original_tag}]_epsilon_{tempered_epsilon}.jpeg')