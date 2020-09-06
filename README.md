# MemeDetector
This is my first computer vision deep learning project. It is motivated by the increasing amount of memes that I receive every day on my WhatsApp groups.
The idea is to train a CNN that will be able to predict wheter an image is a meme or not in order to deleted from my device.

# Give a try to MemeDetector
If you want to try this model, you have to follow this steps:
1. Download delete_memes.py and inceptionv3 folder
2. Run the script delete_memes.py (open command and: python3 -i delete_memes.py):
     - Type delete_memes(
     - You have to specify the images folder, threshold and a see_pred argument.
     - If you want to look the predictions before you delete the images in order to decide the threshold to use, you have to pass the last argument to True.
     - 0 --> meme and 1 --> normal photo.
     - Then, you will look some images and the probability assigned. 
     - Once you decide the threshold, put it in the script and it will delete all memes!

## 1. Data
The images were collected with the script called ```main.py```. I took it from this video: https://www.youtube.com/watch?v=cImRC-AZs48

### 1.1 Memes
Collected by the following searches:
- Memes from Argentina
- Memes from 2020
- Old memes
- New memes

*Colleting 1003 images.*

### 1.2 Normal photos
Collected by the following searches:
- bar amigos
- mendoza argentina
- gimnasio
- family photo
- selfie real
- university classroom
- party photos
- landscape photos
- used cars

*Collecting 1565 images.*

After collecting the data I made some cleaning because there were a lot of misslabels in normal photos.

### 1.4 Preprocessing
I used ImageDataGeneration to do Data Augmentation (create new data from the existing images by rotating, shifting and doing other transformations to original photos) due the lack of data.

## 2. Model
### 2.1 Learning Rate search
I used callbacks to search a good Learning rate.

### 2.2 Baseline model
Model with two Conv + MaxPooling layers

### 2.3 Transfer learning
I donwloaded the weights of a InceptionV3 model, then I trained a few layers with my data.
The model reached 99.8% accuracy.

In this project, I consider that it is more important to keep photos than delete them. So, I was looking for a 100% recall. (better to keep some memes than delete important photos)
