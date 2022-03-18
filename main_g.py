"""
Evaluate the PGD attack on the model trained in main_e.py
"""
import numpy as np
import utils
import consts

from attacks import PGD
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
model.load_weights('models/main_e.npz')

# run untargeted PGD
attack = PGD(consts.ADV_EPS,
             consts.PGD_ALPHA,
             consts.PGD_ITERS,
             loss='cce')
preds = []
original_preds = []
x_test_adv_untargeted = []

for i in range(0, len(x_test), consts.BATCH_SZ):
    x_adv = attack.execute(model, \
                           x_test[i:i+consts.BATCH_SZ], \
                           y_test[i:i+consts.BATCH_SZ])
    preds.append( model.predict(x_adv) )
    original_preds.append(model.predict(x_test[i:i + consts.BATCH_SZ]))
    x_test_adv_untargeted.append(x_adv)

preds = np.concatenate(preds, axis=0).squeeze()
original_preds = np.concatenate(original_preds, axis=0).squeeze()
x_test_adv_untargeted = np.concatenate(x_test_adv_untargeted, axis=0).squeeze()

categorical_preds = np.argmax(preds, axis=1)
categorical_original_preds = np.argmax(original_preds, axis=1)
original_success = categorical_original_preds == y_test

binary_untargeted_success = np.bitwise_and(categorical_preds != y_test, original_success)
attack_sr = np.sum(binary_untargeted_success)/np.sum(original_success)  # the attack success rate (in [0, 1])
print(f'The success rate of untargeted PGD: {attack_sr:0.4f}')

# run targeted PGD
preds = []
original_preds = []
y_target = (y_test + np.random.randint(1, n_classes, y_test.shape))%n_classes
x_test_adv_targeted = []

for i in range(0, len(x_test), consts.BATCH_SZ):
    x_adv = attack.execute(model, \
                           x_test[i:i+consts.BATCH_SZ], \
                           y_target[i:i+consts.BATCH_SZ], \
                           targeted=True)
    preds.append( model.predict(x_adv) )
    original_preds.append(model.predict(x_test[i:i + consts.BATCH_SZ]))
    x_test_adv_targeted.append(x_adv)

preds = np.concatenate(preds, axis=0).squeeze()
original_preds = np.concatenate(original_preds, axis=0).squeeze()
x_test_adv_targeted = np.concatenate(x_test_adv_targeted, axis=0).squeeze()

categorical_preds = np.argmax(preds, axis=1)
categorical_original_preds = np.argmax(original_preds, axis=1)
original_success = categorical_original_preds == y_test
effective_success = np.bitwise_and(y_target != y_test, original_success)

binary_targeted_success = np.bitwise_and(categorical_preds == y_target, effective_success)
attack_sr = np.sum(binary_targeted_success)/np.sum(effective_success)  # the attack success rate (in [0, 1])
print(f'The success rate of targeted PGD: {attack_sr:0.4f}')

targeted_and_untargeted_success = np.bitwise_and(binary_targeted_success, binary_untargeted_success)
for original_tag in np.unique(y_train):
    first_occurrence = np.argwhere(np.bitwise_and(targeted_and_untargeted_success, y_test == original_tag))[0]
    utils.save_as_im(x_test[first_occurrence].reshape(28, 28), f'Images/PGD/benign_[{original_tag}].jpeg')
    utils.save_as_im(x_test_adv_targeted[first_occurrence].reshape(28, 28), f'Images/PGD/targeted_adversarial_[{original_tag}]_to_{y_target[first_occurrence]}.jpeg')
    utils.save_as_im(x_test_adv_untargeted[first_occurrence].reshape(28, 28), f'Images/PGD/untargeted_adversarial_[{original_tag}]_to_{y_target[first_occurrence]}.jpeg')