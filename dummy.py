import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential, losses, optimizers, utils
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model, save_model
from matplotlib import pyplot as plt
import random


def create_model():
    if K.image_data_format() == 'channels_first':
        input_shape = (1, 28, 28)  # (1, img_rows, img_cols)
    else:
        input_shape = (28, 28, 1)  # (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def prepare_X_data(x_unprepared_data, printing=True):
    if printing:
        print("unprepared X_shape:" + str(x_unprepared_data.shape))

    img_rows = x_unprepared_data.shape[1]
    img_cols = x_unprepared_data.shape[2]

    if K.image_data_format() == 'channels_first':
        x_data = x_unprepared_data.reshape(x_unprepared_data.shape[0], 1, img_rows, img_cols)
    else:
        x_data = x_unprepared_data.reshape(x_unprepared_data.shape[0], img_rows, img_cols, 1)

    x_data = x_data.astype('float32')
    x_data /= 255
    if printing:
        print('prepared X_data shape:', x_data.shape)
        print("")
    return x_data


def prepare_Y_data(y_unpreprared_data, printing=True):
    if printing:
        print("unprepared Y_shape:" + str(y_unpreprared_data.shape))
    y_data = utils.to_categorical(y_unpreprared_data, 10)
    if printing:
        print('prepared Y_data shape:', y_data.shape)
        print("")
    return y_data


def random_predict(model):
    (_, _), (x_test, y_test) = mnist.load_data()
    index = random.randint(0, len(x_test) - 1)
    X = x_test[index]
    X = prepare_X_data(x_unprepared_data=np.array([X]))
    prediction = model.predict(X)
    print("index: " + str(index))
    print("Prediction array: " + str(prediction))
    print("Prediction: " + str(np.argmax(prediction)))
    print("Ground truth: " + str(y_test[index]))
    print("")
    return np.argmax(prediction), y_test[index]


def predict_fault(model):
    while True:
        x, y = random_predict(model)
        if x != y:
            break


def train_model(model):
    (x_train, y_train), (_, _) = mnist.load_data()
    x_prep_train = prepare_X_data(x_unprepared_data=x_train)
    y_prep_train = prepare_Y_data(y_unpreprared_data=y_train)
    model.fit(x_prep_train, y_prep_train, batch_size=128, epochs=12, verbose=1)
    print("save model")
    print("")
    save_model(model=model, filepath="example_model.h5")


def eval_model(model):
    (_, _), (x_test, y_test) = mnist.load_data()
    x_prep_test = prepare_X_data(x_unprepared_data=x_test)
    y_prep_test = prepare_Y_data(y_unpreprared_data=y_test)
    score = model.evaluate(x_prep_test, y_prep_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("")


def get_part_of_list(lst, start, items):
    end = len(lst)
    if start + items > end:
        missing = start + items - end
        result = lst[start:end]
        result = np.concatenate((result, lst[:missing]))
        return result, missing
    else:
        return lst[start:start + items], start + items


def train_generator(iterations=500):
    (x_train, y_train), (_, _) = mnist.load_data()  # Do not load all data in memory at once!
    iteration = 0
    start = 0
    batch_size = 128
    while True:
        if iterations == iteration:
            break
        x_unprep_subset, _ = get_part_of_list(lst=x_train, start=start, items=batch_size)
        y_unprep_subset, next_start = get_part_of_list(lst=y_train, start=start, items=batch_size)
        start = next_start
        x_prep_subset = prepare_X_data(x_unprepared_data=x_unprep_subset)
        y_prep_subset = prepare_Y_data(y_unpreprared_data=y_unprep_subset)
        yield x_prep_subset, y_prep_subset
        iteration += 1


def test_generator(iterations=100):
    (_, _), (x_test, y_test) = mnist.load_data()  # Do not load all data in memory at once!
    iteration = 0
    start = 0
    batch_size = 128
    while True:
        if iterations == iteration:
            break
        x_unprep_subset, _ = get_part_of_list(lst=x_test, start=start, items=batch_size)
        y_unprep_subset, next_start = get_part_of_list(lst=y_test, start=start, items=batch_size)
        start = next_start
        x_prep_subset = prepare_X_data(x_unprepared_data=x_unprep_subset)
        y_prep_subset = prepare_Y_data(y_unpreprared_data=y_unprep_subset)
        yield x_prep_subset, y_prep_subset
        iteration += 1


def train_model_with_generator(model):
    train_gen = train_generator(iterations=500)
    model.fit(x=train_gen)
    print("save model")
    print("")
    save_model(model=model, filepath="example_model.h5")


def eval_model_with_generator(model):
    test_gen = test_generator(iterations=500)
    score = model.evaluate_generator(generator=test_gen)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# # model = create_model()
# # train_model(model=model)
# model = load_model("example_model.h5")
# # eval_model(model)
# # for _ in range(10):
# #     random_predict(model)
# predict_fault(model)

(_, _), (x_test, y_test) = mnist.load_data()
first_image = x_test[591]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
