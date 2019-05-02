from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier = Sequential()
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

 training_set = train_datagen.flow_from_directory('train',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')

 test_set = test_datagen.flow_from_directory('test',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')

classifier.fit_generator(training_set,
steps_per_epoch = 17000,
epochs = 2,
validation_data = test_set,
validation_steps = 3000)

import numpy as np
from keras.preprocessing import image

individual_image=image.load_img('second.png', target_size = (64, 64))
individual_image = image.img_to_array(individual_image)
individual_image = np.expand_dims(individual_image, axis = 0)
result = classifier.predict(individual_image)
result

training_set.class_indices

individual_image=image.load_img('image.jpg', target_size = (64, 64))
individual_image = image.img_to_array(individual_image)
individual_image = np.expand_dims(individual_image, axis = 0)
result = classifier.predict(individual_image)
result