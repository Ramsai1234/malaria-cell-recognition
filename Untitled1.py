#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install seaborn


# In[45]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
# to share the GPU resources for multiple sessions
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


my_data_dir = 'dataset/cell_images'


# In[48]:


os.listdir(my_data_dir)


# In[49]:


test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'


# In[50]:


os.listdir(train_path)


# In[51]:


len(os.listdir(train_path+'/uninfected/'))


# In[52]:


len(os.listdir(train_path+'/parasitized/'))


# In[54]:


os.listdir(train_path+'/parasitized')[0]


# In[55]:


para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])


# In[56]:


plt.imshow(para_img)


# In[57]:


dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)


# In[58]:


sns.jointplot(x=dim1,y=dim2)


# In[59]:


image_shape = (130,130,3)


# In[61]:


help(ImageDataGenerator)


# In[62]:


image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )


# In[63]:


image_gen.flow_from_directory(train_path)


# In[64]:


image_gen.flow_from_directory(test_path)


# In[66]:


model = models.Sequential()
model.add(keras.Input(shape=(image_shape)))
model.add(layers.Conv2D(filters=64,kernel_size=(5,5),activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(4, 2)))

model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(5,5), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(4, 4)))

model.add(layers.Flatten())

model.add(layers.Dense(128))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[68]:


batch_size = 16


# In[69]:


help(image_gen.flow_from_directory)


# In[70]:


train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')


# In[71]:


train_image_gen.batch_size


# In[72]:


len(train_image_gen.classes)


# In[73]:


train_image_gen.total_batches_seen


# In[74]:


test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)


# In[75]:


train_image_gen.class_indices


# In[76]:


results = model.fit(train_image_gen,epochs=10,
                              validation_data=test_image_gen
                             )


# In[77]:


model.save('cell_model.h5')


# In[78]:


losses = pd.DataFrame(model.history.history)


# In[79]:


losses[['loss','val_loss']].plot()


# In[80]:


model.metrics_names


# In[81]:


model.evaluate(test_image_gen)


# In[82]:


pred_probabilities = model.predict(test_image_gen)


# In[83]:


test_image_gen.classes


# In[84]:


predictions = pred_probabilities > 0.5


# In[85]:


print(classification_report(test_image_gen.classes,predictions))


# In[86]:


confusion_matrix(test_image_gen.classes,predictions)


# In[87]:


import random


# In[89]:


list_dir=["Un Infected","parasitized"]
dir_=(random.choice(list_dir))
p_img=imread(train_path+'/'+dir_+'/'+[random.randint(0,100)])
img  = tf.convert_to_tensor(np.asarray(p_img))
img = tf.image.resize(img,(130,130))
img=img.numpy()
pred=bool(model.predict(img.reshape(1,130,130,3))<0.5 )
plt.title("Model prediction: "+("Parasitized" if pred  else "Un Infected")+"\nActual Value: "+str(dir_))
plt.axis("off")
plt.imshow(img)
plt.show()


# In[ ]:




