# image-reshuffling
Partially implementation of paper "Unsupervised Representation Learning by Sorting Sequences" (2017ICCV)

<p align="center">
<img src="https://i.imgur.com/mu4SEud.png">
</p>

Only implement the model architecture. For dataset, I use "HMDB-51" video cropped in 1 image per 0.1 second duration without specific processing described in paper

You can download the data I have preprocessed [here](https://drive.google.com/drive/folders/11S1jRM_YZJ2NAnJ1TE_uefga-Taxp57X?usp=sharing), or do it by yourself

Cropped file name would be "{video_name}_{second}.jpg"

## Requirement
- python 3.6+
- pytorch 0.4+
- numpy
- matplotlib
- PIL

## Performance
<p align="center">
<img src="https://i.imgur.com/tsU1afw.png">
</p>
