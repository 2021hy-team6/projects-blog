---
layout: post
title:  "AI Brief Introduction"
date:   2021-10-19 00:00:00 +0900
category: AI
---

> Sentiment Analysis on Live Transcription  
> AI, NUGU SDK

![NUGU Inside](https://mblogthumb-phinf.pstatic.net/MjAxOTA5MDJfMTI2/MDAxNTY3Mzg5OTYyNDE4.gIwNGV1LFw2Bolziw8dR4VNvzi-gnuFLWrGTHb9Fad4g.dOTFjhzYlp2yjzRS2c1zUOUSRg88_V3BrBs8mpstGlkg.PNG.nuguai/NUGU_inside_primary.png?type=w800)

## Objectives

The objective of the project is to portray the sentiment of a conversation,
while providing a transcription of the audio - while the software transcribes the conversation, 
the application will estimate the speakers' underlying emotion, or sentiment.

## Applications

An example of where it could be used is a group meeting. The converstion would be transcribed,
and the sentiment classified per the speaker's spoken words. The transcribed texts and result
of sentiment analysis would be available in a text format after the meeting is over, but also
available on the screen in real-time.

## Simplified Procedures 

|Conversation in Audio|Transcription in Text|Sentiment Analysis|Text & Analysis Result|
|:-------------------:|:-------------------:|:----------------:|:--------------------:|
|Input                |**NUGU SDK**         |**AI**            |Output                |

## Primary Components

### Sentiment Analysis

The Sentiment Analysis model used in this application will be trained by us, in order to gauge the
sentiment given textual content.

### Live Transcription

The Transcription aspect will be handled by an external API (ie. NUGU). In order to provide
results to the screen, as well as input for the Sentiment Analysis portion in real-time,
appropriate streaming I/O must be used (ie. HTTP/2 or Websocket Usage).
