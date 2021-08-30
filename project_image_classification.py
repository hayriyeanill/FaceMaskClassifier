import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# In every epoch, you can callback to a code function, having checked the metrics.
# If they're what you want to say, then you can cancel the training at that point.
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('accuracy') > 0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

original_src = "C:\PycharmProjects\CMP5133-Project"

src_with_mask = os.path.join(original_src, 'with_mask')
with_mask_dir = os.listdir(src_with_mask)

src_without_mask = os.path.join(original_src, 'without_mask')
without_mask_dir = os.listdir(src_without_mask)


def rename_image_labels(new_name, file, src):
  for filename in file:
    old_name_dir = os.path.join(src, filename)
    new_name_dir = os.path.join(src, new_name + filename)
    print(old_name_dir, new_name_dir)
    os.rename(old_name_dir, new_name_dir)

#rename_image_labels("with_mask_", with_mask_dir, src_with_mask)
#rename_image_labels("without_mask_", without_mask_dir, src_without_mask)

# Create train and validation folder
train = os.path.join(original_src, 'train')
#os.mkdir(train)
train_with_mask= os.path.join(train, 'with_mask')
#os.mkdir(train_with_mask)
train_without_mask= os.path.join(train, 'without_mask')
#os.mkdir(train_without_mask)

validation = os.path.join(original_src, 'validation')
#os.mkdir(validation)
val_with_mask= os.path.join(validation, 'with_mask')
#os.mkdir(val_with_mask)
val_without_mask = os.path.join(validation, 'without_mask')
#os.mkdir(val_without_mask)


def upload_train_file(file_name, src_dir, dst_dir):
  fname = [file_name +'{}.png'.format(i) for i in range(1, 1994)]
  for f in fname:
    src = os.path.join(src_dir, f)
    dst = os.path.join(dst_dir, f)
    print("src",src)
    print("dst",dst)
    shutil.copyfile(src,dst)


def upload_validation_file(file_name, src_dir, dst_dir):
  fname = [file_name +'{}.png'.format(i) for i in range(1994, 2994)]
  for f in fname:
    src = os.path.join(src_dir, f)
    dst = os.path.join(dst_dir, f)
    print("src",src)
    print("dst",dst)
    shutil.copyfile(src,dst)


upload_train_file("with_mask_", src_with_mask, train_with_mask)
upload_train_file("without_mask_", src_without_mask, train_without_mask)
upload_validation_file("with_mask_", src_with_mask, val_with_mask)
upload_validation_file("without_mask_", src_without_mask, val_without_mask)

print(len(os.listdir(train_with_mask)), len(os.listdir(train_without_mask)))
print(len(os.listdir(val_with_mask)), len(os.listdir(val_without_mask)))


train_with_mask_dir = os.listdir(train_with_mask)
train_without_mask_dir = os.listdir(train_without_mask)
validation_with_mask_dir = os.listdir(val_with_mask)
validation_without_mask_dir = os.listdir(val_without_mask)

print(len(train_with_mask_dir), len(train_without_mask_dir))
print(len(validation_with_mask_dir), len(validation_without_mask_dir))


# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(4 * 4, 4 * 4)

pic_index += 8
next_with_mask_pix = [os.path.join(train_with_mask, fname) for fname in train_with_mask_dir[pic_index-8:pic_index]]
next_without_mask_pix = [os.path.join(train_without_mask, fname) for fname in train_without_mask_dir[pic_index-8:pic_index]]


for i, img_path in enumerate(next_with_mask_pix+next_without_mask_pix):
  sp = plt.subplot(4, 4, i + 1)
  sp.axis('Off') # Don't show axes

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 100 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train,
        target_size=(150, 150),
        batch_size=100,
        class_mode='binary')

# Flow training images in batches of 20 using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        validation,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')


history = model.fit(
      train_generator,
      steps_per_epoch=10,
      epochs=50,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=10, callbacks=[callbacks])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#https://www.kaggle.com/vijaykumar1799/face-mask-detection
#https://github.com/lmoroney/dlaicourse/blob/master/Horse-or-Human-WithDropouts.ipynb
