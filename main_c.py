"""
Code for training a multi-class logistic-regression 
model to classify digits
"""
import numpy as np
import utils
import consts
from logistic_regression import MulticlassLogisticRegression, eval_acc

np.random.seed(consts.SEED)

# load data and update
x_train, y_train, x_val, y_val, x_test, y_test = utils.read_hw_data()

# load and train
n_classes = 4
model = MulticlassLogisticRegression(x_train[0].size,
                                     consts.LR,
                                     n_classes)

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

utils.save_as_im(model.W[0].reshape(28, 28), "W_0_class.jpeg")
utils.save_as_im(model.W[1].reshape(28, 28), "W_1_class.jpeg")
utils.save_as_im(model.W[2].reshape(28, 28), "W_2_class.jpeg")
utils.save_as_im(model.W[3].reshape(28, 28), "W_3_class.jpeg")
