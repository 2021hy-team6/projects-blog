---
layout: post
title:  "SE Brief Introduction"
date:   2021-10-21 00:00:00 +0900
author: Kim Soohyun
category: SE
---

> Recycling Assistant    
> SE, OpenCV, TACO, WasteNet


## Objectives

The purpose of our project is to design an artificial intelligence model that helps people separate recycling in real time. If you show the camera the trash to recycle in real time, it tells you how to recycle it.

## Applications

This model can be placed in recycling grounds used by many people, including apartments, share houses, and companies. If many people throw away trash, it can cause confusion because the garbage is not properly classified.
Our model can prevent those confusion and present an accurate recycling method for the environment.

## Simplified Procedures 

|Input|Modules that help Classification|Using AI Technology|Output|
|:-------------------:|:-------------------:|:----------------:|:--------------------:|
|An object to the camera in real-time|**OpenCV, TACO, WasteNet**|**Recycling Statistics**|Feedback to the user which category the object has to be recycled|

## Primary Components

### Recycling Classification 

The Recycling Classification will be designed using a module like TACO and WasteNet. Also using OpenCV, we will use Computer Vision technology.

### Recycling Statistics

It provides recycling statistics, such as recycled objects, which are the most discarded in the user group. By providing the statistics, we can enhance the recycling awareness of the user group.