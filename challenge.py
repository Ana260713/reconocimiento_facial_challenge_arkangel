import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input

## CARGAR LA DATA

data = pd.read_csv("C:/Users/Ana/Documents/ARKANGEL/reconocimiento/base/facial.csv") # DataFrame

# Realizar un muestreo aleatorio de 1000 datos
short_data = data.sample(n=15000, random_state=42)

## ANALIZAR LA DATA

print(short_data.shape) # Cantidad total de filas y columnas en el DataFrame
print(short_data.isnull().sum()) #Cantidad de valores nulos por columna 
print(short_data.head()) #Muestra las primeras filas del DataFrame


## PREPROCESAR LA DATA

CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"] #Definición etiquetas de clase

emotion_counts = {emotion: list(short_data['emotion']).count(emotion) for emotion in np.unique(short_data['emotion'])} #Crear un diccionario para contar el número de imágenes por emoción

plt.figure(figsize=(10, 6)) # Crear un gráfico de barras utilizando matplotlib
sns.barplot(x=CLASS_LABELS, y=list(emotion_counts.values()), palette="viridis")
plt.xlabel("Emotions")
plt.ylabel("Number of Images")
plt.title("Train Data Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Preprocesamiento de los píxeles y las etiquetas
train_pixels = short_data["pixels"].astype(str).str.split(" ").tolist()
train_pixels = np.array(train_pixels, dtype='float32')
train_pixels = train_pixels.reshape((-1, 48, 48, 1))  # Ajustar la forma de las imágenes

# Cambiar la forma de las imágenes para tener 3 canales (RGB)
train_pixels_rgb = np.repeat(train_pixels, 3, axis=-1)

labels = to_categorical(short_data['emotion'], num_classes=7)

# Definir el diccionario de etiquetas
label_dict = {i: label for i, label in enumerate(CLASS_LABELS)}

# División de los Datos en Conjuntos de Entrenamiento, Prueba y Validación
X_train, X_test, y_train, y_test = train_test_split(train_pixels_rgb, labels, test_size=0.1, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False) 

X_train_preprocessed = preprocess_input(X_train)
X_val_preprocessed = preprocess_input(X_val)

input_shape = X_train_preprocessed.shape[1:]  # Obtener la forma de las imágenes

## MODELO

## VGG16 (Arquitectura consta de 16 capas, incluyendo 13 capas convolucionales y 3 capas totalmente conectadas)

# Definir el modelo VGG16 con las capas adicionales
def vgg16_model(input_shape, num_classes):
    base_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    return model

num_classes = 7

model_vgg16 = vgg16_model(input_shape, num_classes)

model_vgg16.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_vgg16.summary()

# Definir las transformaciones de aumento de datos
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)
valgen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)

# Ajustar los generadores a los conjuntos de entrenamiento y validación
datagen.fit(X_train_preprocessed)
valgen.fit(X_val_preprocessed)

# Crear generadores de imágenes aumentadas
batch_size = 64

train_generator = datagen.flow(X_train_preprocessed, y_train, batch_size=batch_size)
val_generator = valgen.flow(X_val_preprocessed, y_val, batch_size=batch_size)

checkpointer = [EarlyStopping(monitor='val_accuracy', verbose=1, 
                              restore_best_weights=True, mode="max", patience=5),
                ModelCheckpoint('best_model_vgg16.h5', monitor="val_accuracy", verbose=1,
                                save_best_only=True, mode="max")]

history_vgg16 = model_vgg16.fit(train_generator,
                                 epochs=30,
                                 batch_size=64,   
                                 verbose=1,
                                 callbacks=checkpointer,
                                 validation_data=val_generator)

# Visualizar métricas de entrenamiento y validación
plt.figure(figsize=(12, 4))

# Entrenar el modelo
history_vgg16 = model_vgg16.fit(train_generator,
                                 epochs=30,
                                 batch_size=64,   
                                 verbose=1,
                                 callbacks=checkpointer,
                                 validation_data=val_generator)

# Visualizar métricas de entrenamiento y validación
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_vgg16.history["loss"], label="Training Loss")
plt.plot(history_vgg16.history["val_loss"], label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.subplot(1, 2, 2)
plt.plot(history_vgg16.history["accuracy"], label="Training Accuracy")
plt.plot(history_vgg16.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.tight_layout()
plt.show()

# Evaluar el modelo en el conjunto de pruebas utilizando `evaluate` method
loss_vgg16, accuracy_vgg16 = model_vgg16.evaluate(X_test, y_test)
print("VGG16 Test Loss:", loss_vgg16)
print("VGG16 Test Accuracy:", accuracy_vgg16)

# Realizar predicciones y visualizar resultados
preds_vgg16 = model_vgg16.predict(X_test)
y_pred_vgg16 = np.argmax(preds_vgg16, axis=1)

# Generar matriz de confusión para VGG16
cm_vgg16 = confusion_matrix(np.argmax(y_test, axis=1), y_pred_vgg16, labels=[0, 1, 2, 3, 4, 5, 6])
cm_vgg16_norm = cm_vgg16.astype('float') / cm_vgg16.sum(axis=1)[:, np.newaxis]  # Normalizar la matriz

plt.figure(figsize=(10, 8))
sns.heatmap(cm_vgg16_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_dict.values(), yticklabels=label_dict.values())
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Normalized Confusion Matrix - VGG16")
plt.tight_layout()
plt.show()

# Imprimir el informe de clasificación
classification_rep = classification_report(np.argmax(y_test, axis=1), y_pred_vgg16,
                                           target_names=CLASS_LABELS, digits=3)
print("Classification Report:\n", classification_rep)

## VGG19 (Extensión de VGG16 y tiene 19 capas en total)

# Definir el modelo VGG19 con las capas adicionales
def vgg19_model(input_shape, num_classes):
    base_model = VGG19(input_shape=input_shape, include_top=False, weights='imagenet')
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    return model

num_classes = 7

model_vgg19 = vgg19_model(input_shape, num_classes)

model_vgg19.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_vgg19.summary()

# Definir las transformaciones de aumento de datos
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)
valgen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)

# Ajustar los generadores a los conjuntos de entrenamiento y validación
datagen.fit(X_train_preprocessed)
valgen.fit(X_val_preprocessed)

# Crear generadores de imágenes aumentadas
batch_size = 64

train_generator = datagen.flow(X_train_preprocessed, y_train, batch_size=batch_size)
val_generator = valgen.flow(X_val_preprocessed, y_val, batch_size=batch_size)

checkpointer = [EarlyStopping(monitor='val_accuracy', verbose=1, 
                              restore_best_weights=True, mode="max", patience=5),
                ModelCheckpoint('best_model_vgg19.h5', monitor="val_accuracy", verbose=1,
                                save_best_only=True, mode="max")]

history_vgg19 = model_vgg19.fit(train_generator,
                                 epochs=30,
                                 batch_size=64,   
                                 verbose=1,
                                 callbacks=checkpointer,
                                 validation_data=val_generator)

# Visualizar métricas de entrenamiento y validación
plt.figure(figsize=(12, 4))

# Entrenar el modelo
history_vgg19 = model_vgg19.fit(train_generator,
                                 epochs=30,
                                 batch_size=64,   
                                 verbose=1,
                                 callbacks=checkpointer,
                                 validation_data=val_generator)

# Visualizar métricas de entrenamiento y validación
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_vgg19.history["loss"], label="Training Loss")
plt.plot(history_vgg19.history["val_loss"], label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.subplot(1, 2, 2)
plt.plot(history_vgg19.history["accuracy"], label="Training Accuracy")
plt.plot(history_vgg19.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.tight_layout()
plt.show()

# Evaluar el modelo en el conjunto de pruebas utilizando `evaluate` method
loss_vgg19, accuracy_vgg19 = model_vgg19.evaluate(X_test, y_test)
print("VGG19 Test Loss:", loss_vgg19)
print("VGG19 Test Accuracy:", accuracy_vgg19)

# Realizar predicciones y visualizar resultados
preds_vgg19 = model_vgg19.predict(X_test)
y_pred_vgg19 = np.argmax(preds_vgg19, axis=1)

# Generar matriz de confusión para VGG19
cm_vgg19 = confusion_matrix(np.argmax(y_test, axis=1), y_pred_vgg19, labels=[0, 1, 2, 3, 4, 5, 6])
cm_vgg19_norm = cm_vgg19.astype('float') / cm_vgg19.sum(axis=1)[:, np.newaxis]  # Normalizar la matriz

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))  # Ajusta el tamaño de la figura
sns.heatmap(cm_vgg19_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Normalized Confusion Matrix - VGG19")
plt.tight_layout()
plt.show()

# Imprimir el informe de clasificación
classification_rep = classification_report(np.argmax(y_test, axis=1), y_pred_vgg19,
                                           target_names=CLASS_LABELS, digits=3)
print("Classification Report:\n", classification_rep)

# Guardar el modelo VGG16 en un archivo
model_vgg16.save("model_vgg16.h5")

# Guardar el modelo VGG19 en un archivo
model_vgg19.save("model_vgg19.h5")


