---
title: "Machine Learning Project: Moving Car Detection Using GMM and SVMs"
date: 2018-10-07
tags: [machine learning, images processing, SVMs, GMM]
header:
   image: "/images/moving_car/car_moving_logo.jpeg"
excerpt: "Machine Learning, Support Vector Machines, Images Processing"
mathjax: "true"
---

## Introduction

This project develops an application to detect moving car on high way, and thus computing car's speed basing on an input video. In particular, this work applies GMM model [1] to detect moving objects that could consist moving cars or objects likely trees, motors. Next, Support Vector Machine (SVM) is applied to classify between observed car and the others, and eventually the speed of car is computed. This project uses Python and OpenCV library to count cars, trained a classifier with the set of vehicle and non-vehicle images. The source video file section used for this project is from [Highway 01 - Free Video Footage - Background HD Free](https://www.youtube.com/watch?v=dTdsjKRyMuU).

To accomplish this, I have done the following:

* Training the Linear Support Vector Machine to classify the vehicle and non-vehicle objects using Histogram of Oriented Gradients (HOG).
* Opening the video file and applying Gaussian Mixture Model to isolate the moving objects from a static background.
* Classifying the vehicle object by using the trained SVMs.
* Tracking the cars and computing the speed of the car.

## Histogram of Oriented Gradients:

A class of object such as a vehicle vary so much in color. Structural cues like shape give a more robust representation. Gradients of specific regions directions captures some notion of shape. The idea of HOG is instead of using each individual gradient direction of each individual pixel of an image, the pixels are grouped into small cells. For each cell, all the gradient directions are computed and grouped into a number of orientation bins. The gradient magnitude is then summed up in each sample. So stronger gradients contribute more weight to their bins, and effects of small random orientations due to noise is reduced. This histogram gives a picture of the dominant orientation of that cell. Doing this for all cells gives a representation of the structure of the image. The HOG features keep the representation of an object distinct but also allow for some variations in shape.

<img src="{{ site.url }}{{ site.baseurl }}/images/moving_car/HOG_1.png" alt="">

<img src="{{ site.url }}{{ site.baseurl }}/images/moving_car/HOG_2.png" alt="">

## Feature Extraction:

The number of *orientations*, *pixels_per_cell*, and *cells_per_block* for computing the HOG features of a single channel of an image can be specified. The number of *orientations* is the number of orientation bins that the gradients of the pixels of each cell will be split up in the histogram. The *pixels_per_cell* is the number of pixels of each row and column per cell over each gradient the histogram is computed. The *cells_per_block* specifies the local area over which the histogram counts in a given cell will be normalized. Having this parameters is said to generally lead to a more robust feature set.

<img src="{{ site.url }}{{ site.baseurl }}/images/moving_car/HOG_3.png" alt="">

```python
    feature_params = {
      'color_model': 'yuv',                # hls, hsv, yuv, ycrcb
      'bounding_box_size': 64,             # 64 pixels x 64 pixel image
      'number_of_orientations': 11,        # 6 - 12
      'pixels_per_cell': 8,                # 8, 16
      'cells_per_block': 2,                # 1, 2
      'do_transform_sqrt': True
    }
    source = FeatureSourcer(feature_params, vehicle_image)
    vehicle_features = source.features(vehicle_image)
    rgb_img, y_img, u_img, v_img = source.visualize()
```

## Classifier Training:

The Support Vector Machine model is trained by using 8792 samples of vehicle images and 8968 samples of non-image. This dataset is preselected by [Udacity](https://www.udacity.com/) with images from the [GTI vehicle image database](www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KTTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/). As a safety measure, a *scaler* is used to transform the raw features before feeding them to the classifier for training, reducing the chance of the classifier to behave badly.

```python
    # Feature Extraction...
    for img in vehicle_imgs:
      vehicles_features.append(source.features(img))
    for img in nonvehicle_imgs:
      nonvehicles_features.append(source.features(img))

    # Scaling Features...
    unscaled_x = np.vstack((vehicles_features, nonvehicles_features)).astype(np.float64)
    scaler = StandardScaler().fit(unscaled_x)
    x = scaler.transform(unscaled_x)
    y = np.hstack((np.ones(total_vehicles), np.zeros(total_nonvehicles)))

    # Training Features...
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = rand.randint(1, 100))
    svc = LinearSVC()
    svc.fit(x_train, y_train)
    accuracy = svc.score(x_test, y_test)
```

## Importing the input video:







Python code block:
```python
    import numpy as np
    def test_function(x, y):
      z = np.sum(x,y)
      return z
