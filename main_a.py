# This is a sample Python script.
import numpy as np
import os
from PIL import Image
import utils

x_train, y_train, x_val, y_val, x_test, y_test = utils.read_hw_data()
number_of_train_samples = np.shape(x_train)[0]
number_of_val_samples = np.shape(x_val)[0]
number_of_test_samples = np.shape(x_test)[0]

# How many samples, overall, are included in the dataset?
total_number_of_samples_in_dataset = number_of_train_samples + number_of_val_samples + number_of_test_samples
print(f'Number of samples overall in the dataset: {total_number_of_samples_in_dataset}')

# How many classes do the data belong to? What are these classes?
set_of_classes = set(y_train)
print(f'Number of classes the data belong to: {len(set_of_classes)}')
for data_class in set_of_classes:
    index_of_first_occurrence = np.where(y_train == data_class)[0][0]
    image_file_path = "class_example_"+str(data_class)+"_example.jpeg"
    if not os.path.isfile(image_file_path):
        utils.save_as_im(x_train[index_of_first_occurrence], image_file_path)
    Image.open(image_file_path).show()

# What is the dimensionality of the data?
print(f'Data dimensionality is: {np.shape(x_train)[1:]}')

# What are the sizes of the training, validation, and test sets?
print(f'Size of training set: {number_of_train_samples}')
print(f'Size of validation set: {number_of_val_samples}')
print(f'Size of test set: {number_of_test_samples}')

# How many samples of each class are included in each set?
train_unique, train_counts = np.unique(y_train, return_counts=True)
train_class_occurrences = dict(zip(train_unique, train_counts))
print(f'Classes occurrences in training set: {train_class_occurrences}')

val_unique, val_counts = np.unique(y_val, return_counts=True)
val_class_occurrences = dict(zip(val_unique, val_counts))
print(f'Classes occurrences in training set: {val_class_occurrences}')

test_unique, test_counts = np.unique(y_test, return_counts=True)
test_class_occurrences = dict(zip(test_unique, test_counts))
print(f'Classes occurrences in training set: {test_class_occurrences}')
