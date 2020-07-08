import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

IMG_SIZE = 160
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
SAVE_MODELS_DIR = 'saved_models'

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

print(raw_train)
print(raw_validation)
print(raw_test)

get_label_name = metadata.features['label'].int2str
for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
tet_batches = test.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
    ...
print(image_batch.shape, label_batch.shape)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False
base_model.summary()
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['acc']
)
model.summary()
print(len(model.trainable_variables))

initial_epochs = 10
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)
print(f'inital loss: {loss0:.2f}')
print(f'inital acc: {accuracy0:.2f}')

history = model.fit(train_batches, epochs=initial_epochs, validation_data=validation_batches)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.legend(loc='lower right')
plt.ylabel('Acc')
plt.ylim([min(plt.ylim()), 1])
plt.title('Train and Val acc')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Train loss')
plt.plot(val_loss, label='Val loss')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.ylim([0, 1.0])
plt.title('Train and val loss')
plt.xlabel('epoch')
plt.show()

# 微调: 顶层通用, 微调只调后几层专业层
base_model.trainable = True
print('Number of layers in the base model: ', len(base_model.layers))
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    metrics=['acc']
)
model.summary()
print(len(model.trainable_variables))

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
history_fine = model.fit(
    train_batches,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_batches
)

acc += history_fine.history['acc']
val_acc += history_fine.history['val_acc']
loss += history_fine.history['loss']
val_loss = history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Train acc')
plt.plot(val_acc, label='Val acc')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='start fine Tuning')
plt.legend(loc='lower right')
plt.title('train and val acc')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Train loss')
plt.plot(val_loss, label='Val loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='start fine Truning')
plt.legend(loc='upper right')
plt.title('train and val loss')
plt.xlabel('epoch')
plt.show()


t = time.time()
export_path = f'{SAVE_MODELS_DIR}/{int(t)}'
if not os.path.exists(export_path):
    os.makedirs(export_path)

model.save(export_path, save_format='tf')
print(export_path)


reloaded = tf.keras.models.load_model(export_path)
reloaded_result_batch = reloaded.predict(image_batch)
print(abs(reloaded_result_batch - label_batch).max())
print()