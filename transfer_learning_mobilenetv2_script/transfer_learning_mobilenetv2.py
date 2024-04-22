from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Variables
image_width, image_height = 224, 224
batch_size = 32
epochs = 20

# Dataset: add dataset path (example: "/users/joban/desktop/dataset")
dataset_directory = ""

# Data augmentation properties and splitting data in 80% training and 20% validation
image_augmentation_properties = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Augmentation generator on training dataset
training_augmentation_generator = image_augmentation_properties.flow_from_directory(
    dataset_directory,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

# Augmentation generator on validation dataset
validation_augmentation_generator = image_augmentation_properties.flow_from_directory(
    dataset_directory,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
    subset='validation'
)

# Early stopping when over-fitting
callback = EarlyStopping(monitor='loss',
                         patience=3)

# Load MobileNetV2 with pre-trained weights on ImageNet
model_foundation = MobileNetV2(input_shape=(image_width, image_height, 3),
                               include_top=False,
                               weights='imagenet')

# Freeze the layers in InceptionV3
for layer in model_foundation.layers:
    layer.trainable = False

# CNN model
model = Sequential()
model.add(model_foundation)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    training_augmentation_generator,
    validation_data=validation_augmentation_generator,
    epochs=epochs,
    callbacks=[callback]
)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('OtoFrame Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('OtoFrame Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model
results = model.evaluate(validation_augmentation_generator)
print("Test Loss, Test Accuracy:", results)

# Predictions on validation data
class_predictions = model.predict(validation_augmentation_generator)
predicted_classes = (class_predictions > 0.5).astype(int)

# True labels of the validation set
true_classes = validation_augmentation_generator.classes

# Calculate and print additional evaluation metrics
print("OtoFrame Report:")
print(classification_report(true_classes, predicted_classes))

# Confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="g", xticklabels=['Suitable', 'Not Suitable'],
            yticklabels=['Suitable', 'Not Suitable'], cbar=False)
plt.ylabel('True Labels', fontsize=10)
plt.xlabel('Predicted Labels', fontsize=10)
plt.title('Confusion Matrix', fontsize=12)
plt.show()

# Save the trained model
model.save('otoframe_mobilenetv2_model.h5')
