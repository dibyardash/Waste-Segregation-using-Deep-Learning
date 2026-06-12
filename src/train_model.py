import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True, zoom_range=0.2, rotation_range=25)
 
train_data = datagen.flow_from_directory(
    "dataset/",
    target_size=(224,224),
    batch_size=32,
    subset='training'
)
 
val_data = datagen.flow_from_directory(
    "dataset/",
    target_size=(224,224),
    batch_size=32,
    subset='validation'
)
 
# Save class mapping

with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)
 
base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
base.trainable = False
 
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
out = tf.keras.layers.Dense(train_data.num_classes, activation='softmax')(x)
 
model = tf.keras.Model(inputs=base.input, outputs=out)
 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
model.fit(train_data, validation_data=val_data, epochs=15)
 
model.save("waste_model.h5")
 
print("Model trained!")