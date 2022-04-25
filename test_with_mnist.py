import tensorflow as tf, traceback

from model import weighted_categorical_crossentropy, class_weights, generate_class_weights

# TEST THE MODEL AND WEIGHTS
########################################################################################################################

# Compile loss function to be used for both model outputs
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
class_weights_mnist = generate_class_weights(train_labels, one_hot_encoded=False)  # Previously imported
n_classes = len(class_weights_mnist)
print(str(class_weights_mnist))
print(str(n_classes))
loss = weighted_categorical_crossentropy(list(class_weights_mnist.values()))
# Method by Morten Grøftehauge. # Source: https://github.com/keras-team/keras/issues/11735#issuecomment-641775516

# Build model with Keras Functional API
def build_model():
    input_layer = tf.keras.layers.Input(shape=(28, 28))
    dense_layer = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    dense_layer = tf.keras.layers.Flatten()(dense_layer)
    output_layer_1 = tf.keras.layers.Dense(n_classes, name="output_1")(dense_layer)
    output_layer_2 = tf.keras.layers.Dense(n_classes, name="output_2")(dense_layer)
    model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer_1, output_layer_2])
    return model

def test():
    # Build and compile model specifying custom loss and metrics for both outputs
    model = build_model()
    model.compile(optimizer="adam", loss={"output_1": loss, "output_2": loss},
                  metrics={"output_1": "accuracy", "output_2": "accuracy"})
    # Just for demo purposes, add the same label as second output
    train_labels = (train_labels, train_labels)
    test_labels = (test_labels, test_labels)
    # Train and evaluate the model
    model.fit(train_images, train_labels, epochs=2)
    result = model.evaluate(test_images, test_labels, verbose=2)
    print("\nTest accuracy_output 1:", result[3])
    print("\nTest accuracy output 2:", result[4])

    age_weights = {0: 1, 1: 1.8446, 2: 0.5949, 3: 0.3593, 4: 0.61249, 5: 1.2032, 6: 1.2199, 7: 3.39511, 8: 3.3142,
                   9: 8.4695}
    gen_weights = {0: 0.955, 1: 1.049}

    loss_gender = weighted_categorical_crossentropy(list(gen_weights.values()))
    loss_age = weighted_categorical_crossentropy(list(age_weights.values()))
    print("FIN")

########################################################################################################################