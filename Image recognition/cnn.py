
#convoluted neural network

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier = Sequential()

#Convolution
classifier.add(Convolution2D( 32 , 3, 3 , input_shape = (64,64,3), activation = 'relu'))

#Pooling 
classifier.add(MaxPooling2D(pool_size = (2,2)))

#flattening
classifier.add(Flatten())

#full connection
classifier.add(Dense(output_dim = 128 , activation = 'relu'))
#output layer
classifier.add(Dense(output_dim = 1 , activation = 'sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam' , loss ='binary_crossentropy' , metrics = ['accuracy'] )

#fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                        'training_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
classifier.fit(
                training_set,
                steps_per_epoch=8000,
                epochs=20,
                validation_data=test_set,
                validation_steps=2000)