---
layout: post
title:  "AI Project Proposal"
date:   2021-10-13 00:00:00 +0900
category: AI
---

**Live Transcription w/ Sentiment Analysis**

This project is aiming to be able to transcribe audio and determine the sentiment of the conversation. Utilizing the NUGU SDK, speech recognition will be leveraged in order to transcribe audio (speech-to-text) - as the NUGU SDK already provides an ASR/STT function. Sentiment Analysis will be utilized in order to portray the sentiment of the transcribed audio.

The NUGU SDK provides plugins such as gstreamer and portaudio, which can be used in order to receive audio input. Additional external APIs may also be leveraged in order to increase accuracy of the transcribed audio; however, full accuracy is not necessarily our goal – it only needs to be as accurate as to provide the same general gist and sentiment. A database may also be maintained in order to provide transcription and sentiment logs.

We envision the final product to be more focused on the features provided by the NUGU SDK rather than the NUGU Candle – however, if the NUGU Candle can be leveraged in a way to assist in matters such as sentiment portrayal, or other relevant features, we hope to be able to incorporate those additional features into the product.

