import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform
from tensorflow import keras

def delete_memes(FOLDER,THRESHOLD,see_pred = False):
  '''
  Predict wheter an image is a meme or not, then delete it.

  FOLDER = str , images folder
  THRESHOLD = float, probability to decide if is a meme
  see_pred = boolean, if it set to True, shows 15 predictions
                      and probabilities in order to decide the THRESHOLD (then you have to run the function again)
  '''

  model_transfer = keras.models.load_model('inceptionv3')

  images = os.listdir(FOLDER)
  images = [FOLDER + i for i in images]

  if see_pred:
    if len(images) >= 15:
      sample = random.sample(images,20)
      f = plt.figure(figsize=(25, 20))
      rows = 5
      cols = 4

      for i in range(20):
          img = sample[i]
          f.add_subplot(rows, cols, i+1)
          img_np = mpimg.imread(img)
          img_np = transform.resize(img_np, (150, 150, 3))
          plt.imshow(img_np)
          img_np = np.expand_dims(img_np, axis=0)
          p = model_transfer.predict(img_np)[0][0]
          plt.title('Probability: '+ str(round(p,2)))
          plt.axis('off')

      plt.tight_layout()
      plt.show()

    else:
      sample = random.sample(images,len(images))
      f = plt.figure(figsize=(25, 20))
      rows = len(images)//2
      cols = len(images)//2

      for i in range(len(images)):
          img = sample[i]
          f.add_subplot(rows, cols, i+1)
          img_np = mpimg.imread(img)
          img_np = transform.resize(img_np, (150, 150, 3))
          plt.imshow(img_np)
          img_np = np.expand_dims(img_np, axis=0)
          p = model_transfer.predict(img_np)[0][0]
          plt.title('Probability: '+ str(round(p,2)))
          plt.axis('off')

      plt.tight_layout()
      plt.show()

  else:
    i = 0
    prob = []

    for image in images:
      img = mpimg.imread(image)
      img = transform.resize(img, (150, 150, 3))
      img = np.expand_dims(img, axis=0)
      p = model_transfer.predict(img)[0][0]

      if p < THRESHOLD:
        os.remove(image)
        i += 1

    print(f'{i} images removed')
