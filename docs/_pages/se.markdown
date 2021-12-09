---
layout: page
title: Recycling Assistant
permalink: /se/
---

> Identifying and Categorizing Objects based on Recyclability

### Table of Content

* [Introduction](#introduction)
* [Development Envirionment](#development-envirionment)
* [Specifications](#specifications)
* [Architecture Implementation](#architecture-implementation)
* [Conclusion](#conclusion)

### Repositories

- C++ Application Repository : [RecycleM8 Github Repository](https://github.com/2021hy-team6/recyclem8) 

### Paper

- View Paper : [Direct PDF Viewing Link](https://github.com/2021hy-team6/se-paper/blob/main/recycling-assistant-paper-WIP.pdf)
- Download Paper : [Direct PDF Download Link](https://raw.githubusercontent.com/2021hy-team6/se-paper/main/recycling-assistant-paper-WIP.pdf)

### Members

* Doo Woong Chung, Dept. Information Systems, dwchung@hanyang.ac.kr
* Kim Soohyun, Dept. Information Systems, soowithwoo@gmail.com
* Lim Hongrok, Dept. Information Systems, hongrr123@gmail.com

### Abstract

|Input|Modules that help Classification|Using AI Technology|Output|
|:-------------------:|:-------------------:|:----------------:|:--------------------:|
|An object to the camera in real-time|**OpenCV, TACO**|**Recycling Statistics**|Feedback to the user which category the object has to be recycled|


## Introduction

### Motivation

The idea for this project stems from the observation that recycling, while seemingly a simple topic from an external point of view, can get confusing with certain types of objects. For example, as a mixture of different materials are often used in products these days, it is sometimes confusing to tell if an object is really recyclable or not.

### Objectives

The purpose of our project is to design an artificial intelligence application that can help people separate recycling in real-time. If you present the application's linked camera with an object, it should return some feedback on how to recycle it.
As a result, we hope that this application will help reduce the problem of incorrectly recycled objects occurring, or recyclable objects being sent to the landfill from a normal consumer user standpoint.

### Applications

An example of an application for this would be to place it in a recycling area, to help reduce confusion on certain recyclable objects, as well as presenting some educational benefit. In addition, statistics would be supported by the application, allowing users to see the overall recycling load.


## Development Environment

### OS

#### Linux

We will mainly be looking to use the Linux platform as the "base" platform, as it is more convenient to develop for, but we will also look to maximize compatibility with other operating systems so that it may be easily ported, if required.


### Languages

#### C++
We are targeting usage of the language C++ due to developer familiarity, as well familiarity with libraries that may be leveraged for this project. 
As mentioned previously, C++ has a fair amount of available libraries and frameworks that would help conserve time when implementing features into the product.
#### Python
As the project has a focus on using models, and the detection and classification of objects, Python will be used in order to process the dataset, as it has a myriad of tools that can handle that task easily, such as Pandas.

#### SQL-PostgreSQL
To allow the possibility of expansion and to also leave the data accessible, while offloading the data to the disk rather than the memory, PostgreSQL will be used in order to store data from the detections.


### Modules

#### SSD MobileNet v2
SSD MobileNet v2 is an object recognition algorithm that is fast - using a "single shot" to detect multiple objects. This will fit our necessity for speediness, especially once frozen for inference, as we are going for real-time detection. 
#### OpenCV
OpenCV is a software library that focuses on computer vision. It contains support for ML model execution, as well as various image manipulation functions.

A lot of the internal UI features will be implemented via OpenCV (ie. bounding boxes, displaying of results, etc), and the model will be fed to OpenCV's native model handling to return these results. It is available on many platforms and languages, supporting Linux, C++ and Python, amongst many others.

### Dataset

#### TACO
We are mainly looking to use the TACO dataset, as it is a specialized dataset on the topic of waste/litter, with segmentation support, and a large set of COCO annotated images.
However, we will be altering the training data slightly, due to the TACO dataset also containing categories and images that are not completely applicable to our targeted usage - such as cigarette detection.
The TACO dataset also contains enough categories that we feel would cover general use, such as plastic film, plastic bottle, cans, glass, etc - which is commonly seen waste that falls into recycling categories.
In addition, the TACO dataset seems to be centered around the topic of identifying different types of objects. While other datasets seem to go for more of a "litter detection dataset" binary classification (litter or not litter), the TACO dataset goes for a more specific classification approach. This will allow us to provide more specific tips to each category of recycling.

* < http://tacodataset.org/ >
* <https://github.com/pedropro/TACO > (MIT License)
* < https://arxiv.org/abs/2003.06975 >


## Specifications

### Initial Setting & Connection

![Initial_Page] (/Users/kimsoohyun/Documents/GitHub/projects-blog/docs/files/images/start.jpg)

When the user first runs the application, their settings/config file will be read, and then they will be directed an initial starting screen. As the application will be run locally, there will be no login or registration required. The window size should be fixed. 
Once the "Start" button is clicked, they will be brought to the next menu.

![Connection] (/Users/kimsoohyun/Documents/GitHub/projects-blog/docs/files/images/nothing_detected.jpg)

The default connection value is camera index 0 (the device index - 0 refers to the default camera for the system), however it can be assigned to another index, or to a video (ie. mp4 or webcam stream) through the settings menu. For a webcam stream, a valid webcam stream link must be provided.

![Setting1] (/Users/kimsoohyun/Documents/GitHub/projects-blog/docs/files/images/settings_1.jpg)
![Setting2] (/Users/kimsoohyun/Documents/GitHub/projects-blog/docs/files/images/settings_2.jpg)

In the setting menu, the user can change the setting at any time.

### Object Detection

![Detection] (/Users/kimsoohyun/Documents/GitHub/projects-blog/docs/files/images/successful_detection.jpg)

When an object is detected by the model, a bounding box will be drawn on the object, alongside what it recognized, and the confidence level. In addition, the recycling category and any special instructions will be displayed.

![aerosol_detected](https://user-images.githubusercontent.com/53211603/145336442-30741ea8-daa5-4353-b50c-9ca14aa153ff.jpg)

When an object class as a tip attached in the classification file, it will be shown alongside the detection in the text. For example, the tip to not pierce an aerosol canister, or not to crush an aluminum drink can. However, if the object does not have a tip attached, such as the metal bottle cap, then no tip will be shown - only the detected object name, and the recycling class (Metal).

### Database & Statistics

#### Database Settings

![Database] (/Users/kimsoohyun/Documents/GitHub/projects-blog/docs/files/images/settings_3.jpg)

This tab will contain settings related to the database. In this, the user may simply decide to choose to disable, or enable, the usage of database with regards to scans. The user may choose to do this if they would prefer not to have a database instance running, have no need for a database, or for privacy reasons or convenience reasons (ie. choosing not to run the dashboard)

Specific settings such as the database connection string will only be accessible through direct editing of the config file, as the values are automatically set in relation to the separate setup stage for the database.

#### Statistics

![Statistics] (/Users/kimsoohyun/Documents/GitHub/projects-blog/docs/files/images/statistics_overall.jpg)
![Session_Statistics] (/Users/kimsoohyun/Documents/GitHub/projects-blog/docs/files/images/statistics_session.jpg)

When the user clicks on "View Statistics" in the Statistics Context Menu Button, they will be presented with a Statistics Window. The window should be non-modal - as in, the user should still be able to interact with the other window at the same time while viewing statistics. In the window, they will see two tabs - one for session stats, and one for overall stats.

#### Dashboard
![Dashboard] (/Users/kimsoohyun/Documents/GitHub/projects-blog/docs/files/images/stats_overall.jpg)

In the dashboard window, it shows the number of detected objects for each hour and each super categories. The graph contains not all super categories, but displays top five most recognized super categories in a day.

## Architecture Implementation

### System Diagram

![App_diagram] (/Users/kimsoohyun/Documents/GitHub/projects-blog/docs/files/images/app_diagram.jpg)

The following diagram shows our overall architecture and application ecosystem for this project.

### Database Architecture

![Database_diagram] (/Users/kimsoohyun/Documents/GitHub/projects-blog/docs/files/images/db_diagram.jpg)

The following diagram shows our architecture for the database, and the source of the database data (insertion sources). Note that the dashboard is not showcased here, as the dashboard mainly only does selections from the database, and is not a source of data.

## Conclusion

#### Meaning
Recycling Assistant is available on all devices equipped with webcams. If it is an item that requires attention when recycling, Recycling assistant explains precautions to the user along with an appropriate method of recycling. By providing statistics, users can know the type of garbage that is most discharged, and encourage them to reduce disposable consumption by raising awareness.

#### Further Research Direction

If home appliances or large products equipped with recycling assistant are released in the future, products that can be automatically separated can be developed by adding automatic recycling functions. If designed with cute characters, it can be used as educational software even for young children.