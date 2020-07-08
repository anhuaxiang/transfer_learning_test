import os
import time
import numpy as np
import tensorflow as tf
import PIL.Image as Image
import tensorflow_hub as hub
import matplotlib.pylab as plt

from tensorflow.keras import layers

IMAGE_SHAPE = (224, 224)
SAVE_MODELS_DIR = 'saved_models'
classifier_url = "https://hub.tensorflow.google.cn/google/tf2-preview/mobilenet_v2/classification/2"
classifier = tf.keras.Sequential(
    [hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE + (3,))]
)

grace_hopper = tf.keras.utils.get_file(
    'image.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'
)

labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
)
imagenet_labels = np.array(open(labels_path).read().splitlines())

grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper = np.array(grace_hopper) / 255.0
print(grace_hopper.shape)

result = classifier.predict(grace_hopper[np.newaxis, ...])
print(result.shape)
predicted_class = np.argmax(result[0], axis=-1)
print(predicted_class)

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("prediction: " + predicted_class_name.title())
plt.show()

data_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True
)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)
for image_batch, label_batch in image_data:
    print("image batch shape: ", image_batch.shape)
    print("label batch shape: ", label_batch.shape)
    break

result_batch = classifier.predict(image_batch)
print(result_batch.shape)
predicted_class_name = imagenet_labels[np.argmax(result_batch, axis=-1)]
print(predicted_class_name)

plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.imshow(image_batch[n])
    plt.title(predicted_class_name[n])
    plt.axis('off')
    plt.suptitle('imagenet predictions')
plt.show()

feature_extractor_url = "https://hub.tensorflow.google.cn/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=IMAGE_SHAPE + (3,))
feature_batch = feature_extractor_layer(image_batch)

feature_extractor_layer.trainable = False
model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(image_data.num_classes)
])
model.summary()

predictions = model(image_batch)
print(predictions.shape)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['acc']
)


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batch_loss = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_loss.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()


steps_per_epoch = np.ceil(image_data.samples / image_data.batch_size)
batch_stats_callback = CollectBatchStats()

history = model.fit_generator(image_data, epochs=2, steps_per_epoch=steps_per_epoch, callbacks=[batch_stats_callback])
print(history)

plt.figure()
plt.ylabel('loss')
plt.xlabel('train steps')
plt.ylim([0, 2])
plt.plot(batch_stats_callback.batch_loss)
plt.show()

plt.figure()
plt.ylabel('acc')
plt.xlabel('train steps')
plt.ylim([0, 1])
plt.plot(batch_stats_callback.batch_acc)
plt.show()

class_names = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
class_names = np.array([key.title() for key, value in class_names])
print(class_names)

predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]

label_id = np.argmax(label_batch, axis=-1)


plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6, 5, n+1)
    plt.imshow(image_batch[n])
    color = "green" if predicted_id[n] == label_id[n] else "red"
    plt.title(predicted_label_batch[n].title(), color=color)
    plt.axis('off')
    plt.suptitle('model predictions (green: correct, red: incorrect)')
plt.show()

t = time.time()
export_path = f'{SAVE_MODELS_DIR}/{int(t)}'
if not os.path.exists(export_path):
    os.makedirs(export_path)

model.save(export_path, save_format='tf')
print(export_path)


reloaded = tf.keras.models.load_model(export_path)
reloaded_result_batch = reloaded.predict(image_batch)
print(abs(reloaded_result_batch - predicted_batch).max())