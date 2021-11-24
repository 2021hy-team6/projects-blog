---
layout: page
title: Sentiment Transcriber
permalink: /ai/
---

> Sentiment Analysis on Live Transcription

GitHub Repository : [Notebook](https://github.com/2021hy-team6/sentiment_analysis_nb/blob/main/Sentiment_Analysis.ipynb)

**Members**

* Doo Woong Chung, Dept. Information Systems, dwchung@hanyang.ac.kr
* Kim Soohyun, Dept. Information Systems, soowithwoo@gmail.com
* Lim Hongrok, Dept. Information Systems, hongrr123@gmail.com

**Introduction**

{% comment %}
Motivation: Why are you doing this?
What do you want to see at the end
{% endcomment %}

The objective of the project is to portray the sentiment of a conversation,
while providing a transcription of the audio - while the software transcribes the conversation, 
the application will estimate the speakers' underlying emotion, or sentiment.

An example of where it could be used is a group meeting. The converstion would be transcribed,
and the sentiment classified per the speaker's spoken words. The transcribed texts and result
of sentiment analysis would be available in a text format after the meeting is over, but also
ebe available on the screen in real-time.

**Datasets**

For our dataset, we will be using a data from Amazon's reviews, due to their variance in vocabulary,
and phrasing. 

The dataset is described as following by dataset authors:

"The Amazon reviews dataset consists of reviews from Amazon. The data spans a period of 18 years, including ~35 million reviews up to March 2013."

"The Amazon reviews full score dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from the above dataset. 
It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. 
Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).

"The Amazon reviews full score dataset is constructed by randomly taking 600,000 training samples and 130,000 testing samples for each review score from 1 to 5. 
In total there are 3,000,000 trainig samples and 650,000 testing samples."


{% comment %}
Describing your dataset
{% endcomment %}

**Methodology**

GitHub Repository : [Notebook](https://github.com/2021hy-team6/sentiment_analysis_nb/blob/main/Sentiment_Analysis.ipynb)

{% comment %}
Explaining your choice of algorithms (methods)\
Explaining features or code (if any)
{% endcomment %}

Standardization of the text was necessary, so we lower-cased all text, and then merged the title and text columns, as we were only interested in how certain vocabulary
could affect the sentiment of a sentence.

Then, they were tokenized (broken down into individual words), and lemmatized, which can be described as reverting words to their base form (ie. sampling -> sample),
which would reduce the focus on the tense/form of the word, rather than the word itself.

Stopwords (ie. the, a, an) were also removed, as we do not require the stopwords for our usage (stopwords don't seem to really affect sentiment), and some frequently
appearing words were filtered (ie. "book", "film").

A max vocabulary count was then implemented, with only the top N words being kept, so as to minimize niche vocabulary affecting the training.

After this initial processing, our wordcloud looked like the following: 
![wordcloud](https://i.imgur.com/Uxoz9m0.png)

Then came the decision on whether to go with a multiclass classification system, or with a binary system. We tried both, but to explain - 
Training a multiclass classification system with all 5 classes (1, 2, 3, 4, 5 star reviews) resulted in fairly low validation accuracy (>40% or much lower).
As a result, they were translated into 3 - Negative, Neutral, Positive, where 1, 2 star reviews fall into the negative category, 3 star reviews fall into the neutral category,
and 4, 5 star reviews fall into the positive category.

Then, the text was converted into a list so that tensorflow could sequence the text. In this part, words get mapped into numbers (you can basically think of the most_frequent_vocab),
with niche words being converted into an "<unk>"(unknown) token. It was then padded (cut or truncated based on length) to standardize review length, which is also required in order 
to convert it into a tensor. 

The data was then split 70/30 for training/validation.

For multiclass training, the values of the labels/categories need to be "hot encoded" - that is, it needs to be turned into a type of array value. For example, if it is a positive review,
it becomes [0, 0, 1], negative reviews being [1, 0, 0] and neutral being [0, 0, 0].


**Evaluation & Analysis**

![training](https://i.imgur.com/15qK9Rb.png)

Not the best in terms of accuracy, but validation accuracy is keeping up with training accuracy, and does not seem to be overfitting (which would be noticeable if training accuracy 
went up significantly more compared to validation accuracy)/

TODO

{% comment %}
Graphs, tables, any statistics (if any)
{% endcomment %}

**Related Work**

TODO

{% comment %}
(e.g., existing studies)
Tools, libraries, blogs, or any documentation that you have used to do this project.
{% endcomment %}

**Conclusion**

TODO

{% comment %}
Discussion
{% endcomment %}
