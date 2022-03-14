"""
Code for training a binary logistic-regression model to predict
digit parity.
"""
import numpy as np
import utils
import consts

from logistic_regression import BinaryLogisticRegression, eval_acc

np.random.seed(consts.SEED)

# load data and update labels
x_train, y_train, x_val, y_val, x_test, y_test = utils.read_hw_data()
y_train = y_train%2
y_val = y_val%2
y_test = y_test%2

# load and train 
model = BinaryLogisticRegression(x_train[0].size, consts.LR)

# train
best_val_acc = -1.
for e in range(consts.EPOCHS):
    # shuffle
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]
    # fit/update
    for i in range(len(x_train)):
        model.fit_step(x_train[i].reshape((-1,)), y_train[i])
    # validate
    val_acc = eval_acc(model, x_val, y_val)
    if val_acc>best_val_acc:
        best_val_acc = val_acc
        best_W = model.W
        best_b = model.b

# pick best model
model.W = best_W
model.b = best_b
print(f'Top validation accuracy: {best_val_acc:0.4f}')

# test
test_acc = eval_acc(model, x_test, y_test)
print(f'Test accuracy: {test_acc:0.4f}')

utils.save_as_im(model.W.reshape(28, 28), "W_even_or_odd.jpeg")
