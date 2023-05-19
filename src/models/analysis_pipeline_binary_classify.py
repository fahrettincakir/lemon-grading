from src.data.COCOBinaryClassificationGenerator import COCOBinaryClassificationGenerator
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Input,
    UpSampling2D,
    concatenate,
)
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt

from keras.applications import EfficientNetB0
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dropout
import pandas as pd
from sklearn.model_selection import ParameterGrid
from numpy.random import seed


# CONSTANTS
show_plot = False

data_dir = r".\lemon_grading\data\processed\coco_dataset"

classes = [
    "pedicel",
    "blemish",
    "dark_style_remains",
    "artifact",
    "mould",
    "gangrene",
    "illness",
    "name",
]  # List of classes for instance segmentation
batch_size = 4
target_size = (256, 256)
IMG_HEIGHT = target_size[0]
IMG_WIDTH = target_size[1]

pos_classes = ["mould", "gangrene", "illness"]


# Set seed for Keras models
seed_value = 42
seed(seed_value)
tf.random.set_seed(seed_value)

seed = 123
np.random.seed(seed)

# Define augmenter here and then pass it to generator
augmenter = iaa.Sequential(
    [
        iaa.Resize(target_size, interpolation="cubic"),
    ]
)

train_generator = COCOBinaryClassificationGenerator(
    data_dir,
    classes,
    positive_classes=pos_classes,
    negative_classes=list(set(classes) - set(pos_classes)),
    batch_size=batch_size,
    target_size=target_size,
    data_type="train",
    shuffle=True,
    augmenter=augmenter,
)
validation_generator = COCOBinaryClassificationGenerator(
    data_dir,
    classes,
    positive_classes=pos_classes,
    negative_classes=list(set(classes) - set(pos_classes)),
    batch_size=batch_size,
    target_size=target_size,
    data_type="val",
    shuffle=False,
    augmenter=augmenter,
)
test_generator = COCOBinaryClassificationGenerator(
    data_dir,
    classes,
    positive_classes=pos_classes,
    negative_classes=list(set(classes) - set(pos_classes)),
    batch_size=batch_size,
    target_size=target_size,
    data_type="test",
    shuffle=False,
    augmenter=augmenter,
)

batch_index = 1  # Index of the batch to retrieve
batch_X_images, batch_y_targets = train_generator[batch_index]

# Define the hyperparameter ranges to explore
param_grid = {
    "unfreeze_top_num_layers": [0, 10, 20, 40],  # Number of hidden layers
    "dropout_rate": [0.0, 0.2, 0.4],  # Dropout rate
    "learning_rate": [0.001, 0.01, 0.05],  # Learning rate
}

# Generate all possible combinations of hyperparameters
param_combinations = ParameterGrid(param_grid)

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(
    columns=[
        "unfreeze_top_num_layers",
        "dropout_rate",
        "learning_rate",
        "training_loss",
        "validation_loss",
        "training_accuracy",
        "validation_accuracy",
        "best_epoch",
        "best_val_accuracy",
        "test_loss",
        "test_acc",
        "history",
    ]
)

## load base model
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Iterate over each combination of hyperparameters
for params in param_combinations:
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(params["dropout_rate"])(x)  # Add dropout layer
    predictions = Dense(1, activation="sigmoid")(x)  # binary output

    model = Model(inputs=base_model.input, outputs=predictions)
    # Unfreeze the top layers of the base model_2 for fine-tuning
    for layer in base_model.layers[-params["unfreeze_top_num_layers"] :]:
        layer.trainable = True

    # Adjust learning rate
    optimizer = Adam(learning_rate=params["learning_rate"])

    # Compile the model_2
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model_2
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),  # number of batches per epoch
        epochs=10,  # number of epochs
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
    )

    # Loss over epochs
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    # Accuracy over epochs
    training_accuracy = history.history["accuracy"]
    validation_accuracy = history.history["val_accuracy"]

    # If you want to get the best validation accuracy and the corresponding epoch
    best_epoch = np.argmax(validation_accuracy)
    best_val_accuracy = validation_accuracy[best_epoch]

    test_loss, test_acc = model.evaluate(x=test_generator, steps=len(test_generator))

    # Add the results, hyperparameters, and history to the DataFrame
    new_row = pd.Series(
        {
            "unfreeze_top_num_layers": params["unfreeze_top_num_layers"],
            "dropout_rate": params["dropout_rate"],
            "learning_rate": params["learning_rate"],
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "history": history,
        }
    )

    results_df = pd.concat([results_df, new_row.to_frame().T], ignore_index=True)

    if show_plot:
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(training_accuracy)
        plt.plot(validation_accuracy)
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(training_loss)
        plt.plot(validation_loss)
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")

        plt.show()
