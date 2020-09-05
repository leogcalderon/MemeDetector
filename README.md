# MemeDetector
This is my first computer vision deep learning project. It is motivated by the increasing amount of memes that I receive every day on my WhatsApp groups.
The idea is to train a CNN that will be able to predict if an image is a meme in order to deleted from my device.

## 1. Data
Some data was collected with the script called ```main.py```. I took it from this video: https://www.youtube.com/watch?v=cImRC-AZs48
-**Memes:** the searches were 'memes 2020', 'memes argentina' and 'old memes', 'new memes'. Collecting 1003 images.
-**Normal photos:** google searches were 'bar amigos', 'mendoza argentina', 'gimnasio', 'family photo', 'selfie real', 'university classroom', 'party photos', 'landscape photos', 'used cars'. Colleting 1565 images.

After collecting the data I made some cleaning because there were a lot of misslabels in normal photos.

## 2. Approach
- The data ingestion and processing will be made with TensorFlow.
- Then, a trained model will be used, using transfer learning.
