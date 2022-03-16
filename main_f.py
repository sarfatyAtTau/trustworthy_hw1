"""
Evaluate the FGSM attack on the model trained in main_e.py
"""
import numpy
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
model.load_weights('models/main_e.npz')

adversarial_examples_indices_to_visualize = []
for unique_class in np.unique(y_test):
    adversarial_examples_indices_to_visualize.append(np.argwhere(y_test == unique_class)[0])


# run untargeted FGSM

attack = FGSM(consts.ADV_EPS, loss='cce')
preds = []
original_preds = []

# visualization
benign_examples_to_visualize = []
adversarial_examples_to_visualize = []
original_examples_class_to_visualize = []

for i in range(0, len(x_test), consts.BATCH_SZ):
    x_adv = attack.execute(model, \
                           x_test[i:i+consts.BATCH_SZ], \
                           y_test[i:i+consts.BATCH_SZ])
    preds.append(model.predict(x_adv))
    original_preds.append(model.predict(x_test[i:i+consts.BATCH_SZ]))

    for example_index in adversarial_examples_indices_to_visualize:
        if i <= example_index < i + consts.BATCH_SZ:
            benign_examples_to_visualize.append(x_test[example_index])
            adversarial_examples_to_visualize.append(x_adv[example_index - i])
            original_examples_class_to_visualize.append(y_test[example_index])

preds = np.concatenate(preds, axis=0).squeeze()
original_preds = np.concatenate(original_preds, axis=0).squeeze()

categorical_preds = np.argmax(preds, axis=1)
categorical_original_preds = np.argmax(original_preds, axis=1)
original_success = categorical_original_preds == y_test

binary_untargeted_success = numpy.bitwise_and(categorical_preds != y_test, original_success)
attack_sr = np.sum(binary_untargeted_success)/np.sum(original_success)  # the attack success rate (in [0, 1])
print(f'The success rate of untargeted FGSM: {attack_sr:0.4f}')

for benign, adversarial, class_tag in zip(benign_examples_to_visualize, adversarial_examples_to_visualize, original_examples_class_to_visualize):
    utils.save_as_im(benign.reshape(28, 28), f'benign_{class_tag}.jpeg')
    utils.save_as_im(adversarial.reshape(28, 28), f'untargeted_adversarial{class_tag}.jpeg')

# run targeted FGSM
preds = []
original_preds = []
y_target = (y_test + np.random.randint(1, n_classes, y_test.shape))%n_classes

# visualization
benign_examples_to_visualize = []
adversarial_examples_to_visualize = []
original_examples_class_to_visualize = []
target_examples_class_to_visualize = []

for i in range(0, len(x_test), consts.BATCH_SZ):
    x_adv = attack.execute(model, \
                           x_test[i:i+consts.BATCH_SZ], \
                           y_target[i:i+consts.BATCH_SZ], \
                           targeted=True)
    preds.append( model.predict(x_adv) )
    original_preds.append(model.predict(x_test[i:i + consts.BATCH_SZ]))

    for example_index in adversarial_examples_indices_to_visualize:
        if i <= example_index < i + consts.BATCH_SZ:
            benign_examples_to_visualize.append(x_test[example_index])
            adversarial_examples_to_visualize.append(x_adv[example_index - i])
            original_examples_class_to_visualize.append(y_test[example_index])
            target_examples_class_to_visualize.append(y_target[example_index])

preds = np.concatenate(preds, axis=0).squeeze()
original_preds = np.concatenate(original_preds, axis=0).squeeze()

categorical_preds = np.argmax(preds, axis=1)
categorical_original_preds = np.argmax(original_preds, axis=1)
original_success = categorical_original_preds == y_test
effective_success = numpy.bitwise_and(y_target != y_test, original_success)

binary_targeted_success = numpy.bitwise_and(categorical_preds == y_target, effective_success)
attack_sr = np.sum(binary_targeted_success)/np.sum(effective_success)  # the attack success rate (in [0, 1])
print(f'The success rate of targeted FGSM: {attack_sr:0.4f}')

for benign, adversarial, class_tag, target_tag in zip(benign_examples_to_visualize, adversarial_examples_to_visualize, original_examples_class_to_visualize, target_examples_class_to_visualize):
    # utils.save_as_im(benign.reshape(28, 28), f'benign_{class_tag}.jpeg')
    utils.save_as_im(adversarial.reshape(28, 28), f'targeted_adversarial_from_{class_tag}_to_{target_tag}.jpeg')