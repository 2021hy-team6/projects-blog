---
layout: page
title: Sentiment Transcriber
permalink: /ai/
---

> Sentiment Analysis on Live Transcription

Notebook Repository : [Notebook](https://github.com/2021hy-team6/sentiment_analysis_nb/blob/main/Sentiment_Analysis.ipynb)

**Table of Content**
<!--
TODO
-->

**Members**

* Doo Woong Chung, Dept. Information Systems, dwchung@hanyang.ac.kr
* Kim Soohyun, Dept. Information Systems, soowithwoo@gmail.com
* Lim Hongrok, Dept. Information Systems, hongrr123@gmail.com

**Introduction**

<!--
Motivation: Why are you doing this?
What do you want to see at the end
-->

The objective of the project is to portray the sentiment of a conversation,
while providing a transcription of the audio. While the software transcribes the conversation, 
the application will estimate the speakers' underlying emotion, or sentiment.

An example of where it could be used is a group meeting. The conversation would be transcribed,
and sentiment classified per the speaker's spoken words. The transcribed texts and result
of sentiment analysis would be available in a text format after the meeting is over, but also
be available on the screen in real-time.

When a meeting is held on video in a business situation, it may be difficult to accurately convey opinions. This can help you understand each other's intentions accurately by recording your remarks and help convey your opinions even in a video communication environment where communication is not smooth.

Sentiment transcriber can also help those people's lives in that they can deliver accurate opinions and emotions at the same time to people with hearing impairment or speech impairment who have difficulty in communicating. In addition, sentimental transceiver can be helpful for those who have difficulty empathizing or recognizing other people's emotions. By recognizing positive and negative emotions, they can lead their communication in a better direction.

**Dataset**

For our dataset, we will be using a data from Amazon's reviews, due to their variance in vocabulary,
and phrasing. 

The dataset is described as the following by the dataset author (Xiang Zhang (xiang.zhang@nyu.edu)):

"The Amazon reviews dataset consists of reviews from Amazon. The data spans a period of 18 years, including ~35 million reviews up to March 2013."

"The Amazon reviews full score dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from the above dataset. 
It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. 
Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).

"The Amazon reviews full score dataset is constructed by randomly taking 600,000 training samples and 130,000 testing samples for each review score from 1 to 5. 
In total there are 3,000,000 trainig samples and 650,000 testing samples."

**Dataset Inspection**

GitHub Repository : [Notebook](https://github.com/2021hy-team6/sentiment_analysis_nb/blob/main/Data_Inspection.ipynb)

In this dataset, after parsing text into words with categorized labels,
positive, neutral and negative labeling ratio are calculated per word.
For example, the word 'absurd' yields the following ratios: negative(0.66), neutral(0.14), positive(0.20))
While the distribution of neutral ratio's median hovers around 0.21, the others' distribution are 0.39,
which indicates that most words are classified in negative and positive polarity, rather than neutral sentiment.

![density_of_words](https://user-images.githubusercontent.com/59322692/143865891-7c6e3142-7aed-4cf6-b291-bf9542ca9b7d.png)

To be specific, while the most frequently used words can represent both positive/negative meanings,
these words clearly lean to the 1 or 5 scores - which might be a huge clue to distinguish sentiment.

![freq_words_pos](https://user-images.githubusercontent.com/59322692/143865898-7f677caa-02c2-4b71-be77-882294cc26a5.png)
![freq_words_neg](https://user-images.githubusercontent.com/59322692/143865896-3b6bffcb-1c0e-42fa-ad1c-dff7ff1e5268.png)

<!--
Describing your dataset
-->

**Methodology**

GitHub Repository : [Notebook](https://github.com/2021hy-team6/sentiment_analysis_nb/blob/main/Sentiment_Analysis.ipynb)

<!--
Explaining your choice of algorithms (methods)\
Explaining features or code (if any)
-->

First, we have to have Pandas read from our dataset. It's in a CSV format, so we can just use the read_csv function, and then check how the data looks with
data.head(). Here, we only pull 100,000 rows.

![reading-dataset](https://i.imgur.com/U5u5rg5.jpg)

Then, we have to assign column names so that we can manipulate the dataset easier in the future, and then use Pandas to change the text to lower case.

![processing-pt1](https://i.imgur.com/jQLnS73.jpg)

Standardization of the text is necessary, which is why we lower-cased the text.

Then, the text is tokenized (broken down into individual words), and lemmatized, which can be described as reverting words to their base form (ie. sampling -> sample),
which would reduce the focus on the tense/form of the word, rather than the word itself. In order to do this, we define a function, so that we can use it in conjunction
with Pandas' apply.

We use spaCy here, which is an open-source software library that specializes in natural language processing, in order to handle tokenization and lemmatization, as well as 
handling the filtering of stopwords.
Stopwords (ie. the, a, an) were removed, as stopwords are not required for our usage (stopwords don't *really* signal sentiment), alongside some frequently
appearing words were filtered (ie. "book", "film"), that did not seem appropriate for our usage.

![tokenize](https://i.imgur.com/Fv0T3WK.jpg)

We also merge the title and text columns, as we are only interested in how certain vocabulary and sentence structure can affect the sentiment of a sentence, before finally
applying the standardization function we defined above.

![convert-into-tokens](https://i.imgur.com/jROK6MJ.jpg)

A max vocabulary count was then implemented, with only the top N words being kept, so as to minimize niche vocabulary affecting the training.
Then, the text was converted into a list so that tensorflow could sequence the text. In this part, words get mapped into numbers (you can basically think of the most_frequent_vocab 
being mapped),
with niche words being converted into an "<unk>"(unknown) token. It was then padded (cut or truncated based on length) to standardize review length, which is also required in order 
to convert it into a tensor. 

We create a separate column to see the post-processed text, re-merged into a single string, for further data handling down the line, as well as using it for some easier 
graphing.

After this processing, our wordcloud looked like the following: 

![wordcloud](https://i.imgur.com/Uxoz9m0.png)


![multiclass-classification](https://i.imgur.com/h0IkRXX.jpg)

Then came the decision on whether to go with a multiclass classification system, or with a binary system. 
Training a multiclass classification system with all 5 classes (1, 2, 3, 4, 5 star reviews) resulted in fairly low validation accuracy (>40% or lower).
As a result, they are converted into 3 - Negative, Neutral, Positive, where 1, 2 star reviews fall into the negative category, 3 star reviews fall into the neutral category,
and 4, 5 star reviews fall into the positive category.

![pre-tf-processing-one](https://i.imgur.com/bkrnO6G.jpg)


![pre-tf-processing-two](https://i.imgur.com/03JcQVY.jpg)

Before we shove this data into the Tensorflow model, we need to process it into a compatible format. We convert our data into a numpy list format, and then set a max length.
This max length represents the length of a sentence - anything exceeding that limit will be truncated, anything under, will be padded. The length needs to be uniform, 
in order to be able to be converted into a Tensor.

The data is then split 70/30 for training/validation datasets respectively. This will allow us to monitor the fitting of the models as the training occurs. What we want to see is
the loss for both to go down, with the accuracy for both going up, and what we want to avoid is seeing the accuracy for the training dataset going up, while the validation accuracy
falls or stays stagnant, as that means that overfitting is occurring.

![hot-encoding](https://i.imgur.com/qzt1NLb.jpg)

For multiclass training, the values of the labels/categories need to be "hot encoded" - that is, it needs to be turned into a type of array value. For example, if it is a positive review,
it becomes [0, 0, 1], negative reviews being [1, 0, 0] and neutral being [0, 0, 0].

![define-model](https://i.imgur.com/WeiZBJD.jpg)

Then we define our model. The first layer is an Keras Embedding layer. It requires that our data is integer encoded, which we did previously. This layer does something similar to hot-encoding, 
but the values are weighted, and the value changes during training. We define our input dimension as our max word count we defined earlier, and set our output dimension here at 16.
We can increase our output dimension in the future for a more fine-tuned data, but that does take more data, and time, to train.

We add in a Dropout layer in order to try and prevent overfitting from happening early in our data. What this does is it randomly sets input units to 0 at the rate defined. It essentially 
helps to balance out certain inputs that the model may be starting to perceive a bias from.

Then, an Average Pooling Layer is added to down-scale the dimensions. Essentially, the data is averaged down in a manner that it *can* be represented in a 1D format. This is so that we can 
pass the data into subsequent 1D layers easily.

Finally, we add a couple of ReLU activated layers. What occurs in ReLU layers is that until the sum of an input hits a certain point, the output value remains constant, after which it increases 
in a linear fashion.

The final layer is a softmax activated layer. Softmax is identical to Sigmoid, except it is used for non-binary applications. Essentially, it establishes a weight value for its predictions: 
If it sees a value that is unexpected, the area on the graph hits close to 0, where the slope is more exaggerated, which tells model that it needs to adjust the weights a fair amount. On the 
other hand, if it sees a value that is expected, then the slope is relatively flat, meaning that the weights do not need to be adjusted very much.

**Evaluation & Analysis**

![training](https://i.imgur.com/fRsb7nO.jpg)

~65% is definitely not the best in terms of accuracy, but validation accuracy is keeping up with training accuracy, and does not seem to be overfitting (which would be noticeable if training accuracy 
went up significantly more compared to validation accuracy).

An easy way to increase our accuracy would be to drop it from multi-class to binary classification - the reduction in classes, from testing, yields a noticeable increase in accuracy (something like 
15 to 20 percent).

<!--
Graphs, tables, any statistics (if any)
-->

**Summary**

By developing a model using Amazon's dataset, it was possible to conduct sentiment analysis of words in various sentences as much as possible. After that sentences in the Amazon data set were analyzed to increase the accuracy of emotional analysis by excluding unnecessary words from each sentence.

For sentiment analysis, the degree of positive/negative of each word was divided by score. To organize the data, we read the csv file by using pandas module. For the model training, the word was returned to the most basic unit and tokenized. Training a multiclass classification system with all 5 classes resulted in low validation accuracy. Therefore we devide the result into three classses; Negative, Positive, and Neutral.


To use multiclass training we converted the data into array value by using 'hot encoding'. It is a vector representation method of a word that uses the size of a set of words as a vector, gives the index of the word you want to represent, and gives the other index zero.

Out model consists of several layers. Keras Embedding layer, Dropout layer, Avarage Pooling layer, ReLU activated layers, softwmax activated layer. Each layer balances data, removes deflection, and adusts weights during training. 

As a result, we have developed a model that has 65% accuracy. To increase the accuracy, we can shift the multi-classification to a binary classification which can increase the accuracy; 15-20% percent.


**Limitation & Further Research Direction**

After the efficacy of sentiment transcriber has been proven, models can be developed for smooth communication of people from different cultures. Since our model was trained by using Amazon's review data set in English, there are some limitations to state other country people's sentiment. Other sentiment transcriber models can be developed by using the most popular website's data set in that country.

When negotiating, operating, or trading with a business partner company, the percentage of positive or negative emotions can be analyzed to investigate the probability of closing a transaction after the meeting and develop it as a model for generating profits.

<!--
Discussion
-->


**Related Work**

We referred to several blogs, datasets and libraries in order to create the model and its training notebook.

https://kaggle.com/bittlingmayer/amazonreviews

https://kaggle.com/paoloripamonti/twitter-sentiment-analysis

https://github.com/bentrevett/pytorch-sentiment-analysis

https://towardsdatascience.com/how-to-train-a-deep-learning-sentiment-analysis-model-4716c946c2ea

https://towardsdatascience.com/a-complete-step-by-step-tutorial-on-sentiment-analysis-in-keras-and-tensorflow-ea420cc8913f

https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6


https://github.com/tensorflow/tensorflow

https://tensorflow.org/text/guide/word_embeddings

<!--
(e.g., existing studies)
Tools, libraries, blogs, or any documentation that you have used to do this project.
-->
