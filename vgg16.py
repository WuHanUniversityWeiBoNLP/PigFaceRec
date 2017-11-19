from keras.applications import VGG16
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
#using data generator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train_data',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'data/test_data',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
#pretrained model vgg16
model = VGG16(input_shape=(224,224,3),include_top=False)

model_base = Sequential()
for i  in model.layers[:]:
    model_base.add(i)

model_base.add(Flatten())
model_base.add(Dense(1024,activation='relu'))
model_base.add(Dropout(0.5))
model_base.add(Dense(1024,activation='relu'))
model_base.add(Dropout(0.5))
model_base.add(Dense(30,activation='sigmoid'))

model_base.compile(optimizer=Adam(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])

model_base.fit_generator(generator=train_generator,steps_per_epoch=2000,epochs=20,validation_data=validation_generator,validation_steps=20)

#print model_base.summary()