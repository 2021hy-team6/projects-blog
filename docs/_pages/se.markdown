---
layout: page
title: Recycling Assistant
permalink: /se/
---

> Identifying and Categorizing Objects based on Recyclability

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

### Table of Content

* [Introduction](#introduction)
* [Development Envirionment](#development-envirionment)
* [Specifications](#specifications)
* [Architecture Implementation](#architecture-implementation)
* [Conclusion](#conclusion)

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
We are mainly looking to use the TACO dataset, as it is a specialized dataset on the topic of waste/litter, with segmentation support, and a large set of COCO annotated images. However, we will be altering the training data slightly, due to the TACO dataset also containing categories and images that are not completely applicable to our targeted usage - such as cigarette detection.

The TACO dataset also contains enough categories that we feel would cover general use, such as plastic film, plastic bottle, cans, glass, etc - which is commonly seen waste that falls into recycling categories. In addition, the TACO dataset seems to be centered around the topic of identifying different types of objects. While other datasets seem to go for more of a "litter detection dataset" binary classification (litter or not litter), the TACO dataset goes for a more specific classification approach. This will allow us to provide more specific tips to each category of recycling.

* < http://tacodataset.org/ >
* <https://github.com/pedropro/TACO > (MIT License)
* < https://arxiv.org/abs/2003.06975 >


## Specifications

### Initial Setting & Connection

![start](https://user-images.githubusercontent.com/53211603/145336503-a7b09de6-5088-4655-b906-af883394dd2f.jpg)

When the user first runs the application, their settings/config file will be read, and then they will be directed an initial starting screen. As the application will be run locally, there will be no login or registration required. The window size should be fixed. 

Once the "Start" button is clicked, they will be brought to the next menu.

![nothing_detected](https://user-images.githubusercontent.com/53211603/145336494-cf2fdd3f-7036-49bd-923d-4a87a623b7bb.jpg)

The default connection value is camera index 0 (the device index - 0 refers to the default camera for the system), however it can be assigned to another index, or to a video (ie. mp4 or webcam stream) through the settings menu. For a webcam stream, a valid webcam stream link must be provided.

![settings_1](https://user-images.githubusercontent.com/53211603/145336497-6b58d305-ff4b-468f-9201-086debdb7d68.jpg)
![app_settings](https://user-images.githubusercontent.com/53211603/145336472-7a5580b4-b637-4d76-a65e-1c9748da67bb.jpg)

When the user clicks on "Adjust Settings" in the Settings Context Menu Button, they will be presented with a modal Settings Menu Window. In the window, they will see three tabs - one for OpenCV Settings, one for Application Settings, and one for Database Settings.

### Object Detection

![successful_detection](https://user-images.githubusercontent.com/53211603/145336536-197c2166-c912-4fae-8f41-58fd8e1865ff.jpg)

When an object is detected by the model, a bounding box will be drawn on the object, alongside what it recognized, and the confidence level. In addition, the recycling category and any special instructions will be displayed.

![aerosol_detected](https://user-images.githubusercontent.com/53211603/145336442-30741ea8-daa5-4353-b50c-9ca14aa153ff.jpg)

When an object class as a tip attached in the classification file, it will be shown alongside the detection in the text. For example, the tip to not pierce an aerosol canister, or not to crush an aluminum drink can. However, if the object does not have a tip attached, such as the metal bottle cap, then no tip will be shown - only the detected object name, and the recycling class (Metal).

### Database & Statistics

#### Database Settings

![settings_3](https://user-images.githubusercontent.com/53211603/145336500-c3f0478f-56b5-4320-bbe3-2cbedf310b90.jpg)

This tab will contain settings related to the database. In this, the user may simply decide to choose to disable, or enable, the usage of database with regards to scans. The user may choose to do this if they would prefer not to have a database instance running, have no need for a database, or for privacy reasons or convenience reasons (ie. choosing not to run the dashboard)

Specific settings such as the database connection string will only be accessible through direct editing of the config file, as the values are automatically set in relation to the separate setup stage for the database.

#### Statistics

When the user clicks on "View Statistics" in the Statistics Context Menu Button, they will be presented with a Statistics Window. In the window, they will see two tabs - one for session stats, and one for overall stats.

![statistics_session](https://user-images.githubusercontent.com/53211603/145336526-5bfb9d3e-267a-4d4d-b6b4-c488723a9a95.jpg)

In the session stats window, they will see an "Overall Scan Count", and a "Categorical Scan Count". Overall scan count is self-explanatory - it is the number of times any object was scanned.

Categorical scan count is the number of times an object from a category was scanned, divided by the overall scan count. The resulting percentage number is then rounded, and then fed into the progress bar. The categories will be divided into the following:

* Plastic
* Aluminium
* Glass
* Others

![stats_overall](https://user-images.githubusercontent.com/53211603/145336532-db09194c-538e-4914-bcb6-fa019aa0ad95.jpg)

This section is only functional if database is enabled and set up, as it requires database data in order to operate. It shows a button that will automatically launch the user's browser and direct them to the set dashboard for more in-depth information.

#### Dashboard

![chart_hourly_detected](https://user-images.githubusercontent.com/53211603/145336479-f3aacef9-f71a-49cd-b9fa-f4d2e8881d37.jpg)

In the dashboard window, it shows the number of detected objects for each hour and each super categories. The graph contains not all super categories, but displays top five most recognized super categories in a day.

![chart_recyclable_rate](https://user-images.githubusercontent.com/53211603/145336490-0306851d-2880-448f-88c1-7fa7d33ee7c4.jpg)

Among the detected objects today, recyclable rates shows the percentage of recyclable objects for each hour. The number of objects which is marked as a recyclable ones is divided by the overall detection count. If an object has no label whether it is recyclable or not, those kinds of counts will be discarded in this statistics.

![chart_detection_time](https://user-images.githubusercontent.com/53211603/145336476-67cf9c1d-4655-45e3-9633-e97750d213ac.jpg)

Detection time chart tracks how long it takes to make an prediction for a scanned image in the precision of a millisecond. This statistics contains the average of these times for each hour of today. If there is no detection in some hours, the graph won't contain those hours in x axis.

![chart_usage_counts](https://user-images.githubusercontent.com/53211603/145338528-3d3d7fa9-b1d2-44cb-a2ea-6d5596767769.jpg)

Usage counts collects the overall number of detected images for a week. The images without any detection will be excluded in this statistics. It also contains today's records until the time of stats.

![chart_monthly_comp](https://user-images.githubusercontent.com/53211603/145336483-d38feb34-923c-47d9-916c-aabde680f43d.jpg)

Monthly comparison table compares this month's scanned counts for each super category with the last month's one. This charts shows five super categories with the most counts at this month. The differences of numbers between two months will be also shown.

![chart_most_detected](https://user-images.githubusercontent.com/53211603/145336487-8bd7bdc1-f3ce-4e1c-af3e-652f39cf874e.jpg)

Most detected objects table shows the cumulative counts of recyclable objects in this year. This list will have top 10 most detected objects' names with their super category. It also calculates the proportions of scanned objects, divided by the number of overall scanned images.

## Architecture Implementation

### System Diagram

![app_diagram](https://user-images.githubusercontent.com/53211603/145336468-d8642519-eb56-49d8-a7f7-0ee2365c45a7.jpg)

This diagram shows our overall architecture and application ecosystem for this project.

### Database Architecture

![db_diagram](https://user-images.githubusercontent.com/53211603/145336492-e35d368d-0dc8-44e6-a74e-2b3fede5926c.jpg)

The diagram shows our architecture for the database, and the source of the database data (insertion sources).

## Conclusion

#### Meaning
Recycling Assistant is available on all devices equipped with webcams. If it is an item that requires attention when recycling, Recycling assistant explains precautions to the user along with an appropriate method of recycling.

By providing statistics, users can know the type of garbage that is most discharged, and encourage them to reduce disposable consumption by raising awareness.

#### Further Research Direction

If home appliances or large products equipped with recycling assistant are released in the future, products that can be automatically separated can be developed by adding automatic recycling functions. In case of this software is designed with cute characters, it can be used as educational software even for young children.


### Related Work

