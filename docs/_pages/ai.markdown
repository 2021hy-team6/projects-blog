---
layout: page
title: Sentiment Transcriber
permalink: /ai/
---

> Sentiment Analysis on Live Transcription

##### Notebook Repository : [Notebook](https://github.com/2021hy-team6/sentiment_analysis_nb/blob/main/Sentiment_Analysis.ipynb)
##### Transcription Application Repository : [Sentiment Transcriber](https://github.com/2021hy-team6/sentiment-transcriber-app)

### Table of Content

* [Members](#member)
* [Demo Video](#demo-video)
* [Introduction](#introduction)
* [Preamble](#preamble)
* [Dataset](#dataset)
* [Dataset Inspection](#dataset-inspection)
* [Methodology](#methodology)
* [Decision of the parameters](#decision-of-the-parameters)
* [Evaluation & Analysis](#evaluation--analysis)
* [Performance measurement](#performance-measurement)
* [Summary](#summary)
* [Limitation & Further Research Direction](#limitation--further-research-direction)
* [Related Work](#related-work)

### Members

* Doo Woong Chung, Dept. Information Systems, dwchung@hanyang.ac.kr
* Kim Soohyun, Dept. Information Systems, soowithwoo@gmail.com
* Lim Hongrok, Dept. Information Systems, hongrr123@gmail.com

### Demo Video

Direct Link: [Sentiment Transcriber Demo](https://streamable.com/2113kz)

<iframe width="720" height="480" src="https://streamable.com/e/2113kz" frameborder="0"> </iframe>

##### The analyzed sentiment is displayed as a grey (neutral), green (positive), or red (negative) dot next to the sentence.

### Introduction

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

### Preamble

Initially, we were planning to use NUGU's SDK and provided models for transcription. However, upon getting access to the SDK and publicly available models, we found that the only models available were models such as an endpoint detection model, a keyword detection model, and a NUGU activation model - none of which would be suitable for our use. As such, we looked to use other available models, such as Mozilla's DeepSpeech. While testing it out with the provided model, we found that it was not accurate enough for our use-case. While our goal was never to have 100% transcription accuracy, we wanted to at least be able to look back and get the gist of what was said earlier, and the sentiment of that sentence. 

In essence, since the accuracy of the transcription also directly impacts the accuracy of the sentiment analyzer, it would also result in both an inaccurate transcription of the conversation, but also a completely incorrect sentiment reading. For example, this is the result we received - which is incomprehensible.

![deepspeech_result](https://i.imgur.com/PYyHI9t.png)

As such, we decided to use Amazon Web Service's Transcription API, due to significantly higher accuracy, and availability.

### Dataset

For our dataset, we will be using a data from Amazon's reviews, due to their variance in vocabulary,
and phrasing. 

```sh
$ wget https://s3.amazonaws.com/fast-ai-nlp/amazon_review_full_csv.tgz
$ tar -xvzf amazon_review_full_csv.tgz
$ head -n 5 amazon_review_full_csv/train.csv # score, title, text
```
<pre>"3","more like funchuck","Gave this to my dad for a gag gift after directing ""Nunsense,"" he got a reall kick out of it!"
"5","Inspiring","I hope a lot of people hear this cd. We need more strong and positive vibes like this. Great vocals, fresh tunes, cross-cultural happiness. Her blues is from the gut. The pop sounds are catchy and mature."
"5","The best soundtrack ever to anything.","I'm reading a lot of reviews saying that this is the best 'game soundtrack' and I figured that I'd write a review to disagree a bit. This in my opinino is Yasunori Mitsuda's ultimate masterpiece. The music is timeless and I'm been listening to it for years now and its beauty simply refuses to fade.The price tag on this is pretty staggering I must say, but if you are going to buy any cd for this much money, this is the only one that I feel would be worth every penny."
"4","Chrono Cross OST","The music of Yasunori Misuda is without question my close second below the great Nobuo Uematsu.Chrono Cross OST is a wonderful creation filled with rich orchestra and synthesized sounds. While ambiance is one of the music's major factors, yet at times it's very uplifting and vigorous. Some of my favourite tracks include; ""Scars Left by Time, The Girl who Stole the Stars, and Another World""."
"5","Too good to be true","Probably the greatest soundtrack in history! Usually it's better to have played the game first but this is so enjoyable anyway! I worked so hard getting this soundtrack and after spending [money] to get it it was really worth every penny!! Get this OST! it's amazing! The first few tracks will have you dancing around with delight (especially Scars Left by Time)!! BUY IT NOW!!"</pre>

The dataset is described as the following by the dataset author (Xiang Zhang (xiang.zhang@nyu.edu)):

"The Amazon reviews dataset consists of reviews from Amazon. The data spans a period of 18 years, including ~35 million reviews up to March 2013."

"The Amazon reviews full score dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from the above dataset. 
It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. 
Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).

"The Amazon reviews full score dataset is constructed by randomly taking 600,000 training samples and 130,000 testing samples for each review score from 1 to 5. 
In total there are 3,000,000 trainig samples and 650,000 testing samples."

### Dataset Inspection

![pos_and_neg_words](https://user-images.githubusercontent.com/59322692/145327478-31e6de82-e125-439a-af3c-c0390ac00b52.png)

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

### Methodology

GitHub Repository : [Notebook](https://github.com/2021hy-team6/sentiment_analysis_nb/blob/main/Sentiment_Analysis.ipynb)

<!--
Explaining your choice of algorithms (methods)\
Explaining features or code (if any)
-->

First, we have to have Pandas read from our dataset. It's in a CSV format, so we can just use the read_csv function, and then check how the data looks with
data.head(). Here, we pull 2,000,000 rows.

<!--![reading-dataset](https://i.imgur.com/U5u5rg5.jpg)-->
```python
import pandas

# Read Dataset
data = pandas.read_csv('amazon_review_full_csv/train.csv')

# Assigning Column Names
data.columns = ["score", "title", "text"]

data.head()
```
<table class="dataframe" border="1"> <thead> <tr> <th></th> <th>score</th> <th>title</th> <th>text</th> </tr> </thead> <tbody> <tr> <th>0</th> <td>5</td> <td>Inspiring</td> <td>I hope a lot of people hear this cd. We need m...</td> </tr> <tr> <th>1</th> <td>5</td> <td>The best soundtrack ever to anything.</td> <td>I'm reading a lot of reviews saying that this ...</td> </tr> <tr> <th>2</th> <td>4</td> <td>Chrono Cross OST</td> <td>The music of Yasunori Misuda is without questi...</td> </tr> <tr> <th>3</th> <td>5</td> <td>Too good to be true</td> <td>Probably the greatest soundtrack in history! U...</td> </tr> <tr> <th>4</th> <td>5</td> <td>There's a reason for the price</td> <td>There's a reason this CD is so expensive, even...</td> </tr> </tbody></table>

Then, we have to assign column names so that we can manipulate the dataset easier in the future, and then use Pandas to change the text to lower case.

<!--![processing-pt1](https://i.imgur.com/jQLnS73.jpg)-->
```python
# Lower Casing
data["title"] = data["title"].str.lower()
data["text"] = data["text"].str.lower()

data.head()
```
<table class="dataframe" border="1"> <thead> <tr> <th></th> <th>score</th> <th>title</th> <th>text</th> </tr> </thead> <tbody> <tr> <th>0</th> <td>5</td> <td>inspiring</td> <td>i hope a lot of people hear this cd. we need m...</td> </tr> <tr> <th>1</th> <td>5</td> <td>the best soundtrack ever to anything.</td> <td>i'm reading a lot of reviews saying that this ...</td> </tr> <tr> <th>2</th> <td>4</td> <td>chrono cross ost</td> <td>the music of yasunori misuda is without questi...</td> </tr> <tr> <th>3</th> <td>5</td> <td>too good to be true</td> <td>probably the greatest soundtrack in history! u...</td> </tr> <tr> <th>4</th> <td>5</td> <td>there's a reason for the price</td> <td>there's a reason this cd is so expensive, even...</td> </tr> </tbody></table>

Standardization of the text is necessary, which is why we lower-cased the text.

Then, the text is tokenized (broken down into individual words), and lemmatized, which can be described as reverting words to their base form (ie. sampling -> sample),
which would reduce the focus on the tense/form of the word, rather than the word itself. In order to do this, we define a function, so that we can use it in conjunction
with Pandas' apply.

We use spaCy here, which is an open-source software library that specializes in natural language processing, in order to handle tokenization and lemmatization, as well as 
handling the filtering of stopwords.
Stopwords (ie. the, a, an) were removed, as stopwords are not required for our usage (stopwords don't *really* signal sentiment), alongside some frequently
appearing words were filtered (ie. "book", "film"), that did not seem appropriate for our usage.

<!--![tokenize](https://i.imgur.com/Fv0T3WK.jpg)-->
```python
# Define Tokenize & Lemma-ize Function & Basic Filtering + Remove Stopwords
import re
import spacy

nlp = spacy.load("en_core_web_sm")
tokenizer = nlp.tokenizer
nlp.Defaults.stop_words |= { "book", "movie", "film" }

def tokenize(text):
  text = re.sub(r'\b(\d+)\b', '', str(text))
  tokenized_text = tokenizer(text)
  based_text = [token.lemma_ for token in tokenized_text if not token.is_punct and not token.is_stop]

  return based_text
```

```python
# Merge Title & Text Columns, Tokenize/Lemma-ize
data.text = data.title + " " + data.text
data.drop(columns='title', inplace=True)

data.text = data.text.apply(tokenize)
data.head()
```
<table class="dataframe" border="1"> <thead> <tr> <th></th> <th>score</th> <th>text</th> </tr> </thead> <tbody> <tr> <th>0</th> <td>5</td> <td>[inspire, hope, lot, people, hear, cd, need, s...</td> </tr> <tr> <th>1</th> <td>5</td> <td>[well, soundtrack, read, lot, review, say, wel...</td> </tr> <tr> <th>2</th> <td>4</td> <td>[chrono, cross, ost, music, yasunori, misuda, ...</td> </tr> <tr> <th>3</th> <td>5</td> <td>[good, true, probably, great, soundtrack, hist...</td> </tr> <tr> <th>4</th> <td>5</td> <td>[reason, price, reason, cd, expensive, version...</td> </tr> </tbody></table>

We also merge the title and text columns, as we are only interested in how certain vocabulary and sentence structure can affect the sentiment of a sentence, before finally
applying the standardization function we defined above.

<!--![convert-into-tokens](https://i.imgur.com/jROK6MJ.jpg)-->
```python
# Tokenize the texts
from tensorflow.keras.preprocessing.text import Tokenizer

MAX_VOCAB_COUNT = 700

data["liststring"] = data.text.apply(lambda x: ' '.join(map(str, x)))
tf_tokenizer = Tokenizer(num_words=MAX_VOCAB_COUNT, oov_token="<unk>")
tf_tokenizer.fit_on_texts(data.liststring.to_list())
word_index = tf_tokenizer.word_index

print(list(word_index)[:10])
```
<pre>['&lt;unk&gt;', 'like', 'good', 'read', 'great', 'well', 'time', 'buy', 'think', 'love']</pre>

A max vocabulary count was then implemented, with only the top N words being kept, so as to minimize niche vocabulary affecting the training.
Then, the text was converted into a list so that tensorflow could sequence the text. In this part, words get mapped into numbers (you can basically think of the most_frequent_vocab 
being mapped),
with niche words being converted into an "<unk>"(unknown) token. It was then padded (cut or truncated based on length) to standardize review length, which is also required in order 
to convert it into a tensor. 

```python
# Get most frequent vocabularies
most_frequent_vocab = pandas.Series(" ".join(data.liststring).split()).value_counts()[:MAX_VOCAB_COUNT]
print(most_frequent_vocab[:5].to_dict())
```
<pre>{'like': 36091, 'good': 35665, 'read': 34349, 'great': 30606, 'well': 25881}</pre>

We create a separate column to see the post-processed text, re-merged into a single string, for further data handling down the line, as well as using it for some easier 
graphing.

After this processing, our wordcloud looked like the following: 

![word_cloud_common](https://user-images.githubusercontent.com/59322692/144555947-77a52c16-18e0-4ffe-9447-d7a261e502f9.png)

<!--![multiclass-classification](https://i.imgur.com/h0IkRXX.jpg)-->
```python
# Labeling
data['score'] = data.score.apply(lambda x: 0 if x in [1, 2] else x)
data['score'] = data.score.apply(lambda x: 1 if x in [3] else x)
data['score'] = data.score.apply(lambda x: 2 if x in [4, 5] else x)
```

Then came the decision on whether to go with a multiclass classification system, or with a binary system. 
Training a multiclass classification system with all 5 classes (1, 2, 3, 4, 5 star reviews) resulted in fairly low validation accuracy (>40% or lower).
As a result, they are converted into 3 - Negative, Neutral, Positive, where 1, 2 star reviews fall into the negative category, 3 star reviews fall into the neutral category,
and 4, 5 star reviews fall into the positive category.

<!--![pre-tf-processing-one](https://i.imgur.com/bkrnO6G.jpg)-->
<!--![pre-tf-processing-two](https://i.imgur.com/03JcQVY.jpg)-->
```python
# parse dataset into list type
import numpy

text_list = data.text.to_list()
label_list = numpy.asarray(data.score.to_list())

print('sample text :', text_list[:1][0])
print('sample label :', label_list[:1][0])
```

<pre>
sample text : ['inspire', 'hope', 'lot', 'people', 'hear', 'cd', 'need', 'strong', 'positive', 'vibes', 'like', 'great', 'vocal', 'fresh', 'tune', 'cross', 'cultural', 'happiness', 'blue', 'gut', 'pop', 'sound', 'catchy', 'mature']
sample label : 1</pre>

```python
# Generate arrays of sequences
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORD_LENGTH = 50

indexed_text = tf_tokenizer.texts_to_sequences(text_list)
rectangularized = pad_sequences(indexed_text, maxlen=MAX_WORD_LENGTH, truncating='post')

tf_text = tensorflow.convert_to_tensor(rectangularized)
print(tf_text)
```
<pre>tf.Tensor(
[[  0   0   0 ...  45   1   1]
 [  0   0   0 ...  38  67   1]
 [  0   0   0 ...   1  49 101]
 ...
 [  0   0   0 ... 208 688   1]
 [ 44   1 320 ...   1   1   1]
 [  0   0   0 ... 186 116   1]], shape=(100000, 50), dtype=int32)</pre>

Before we shove this data into the Tensorflow model, we need to process it into a compatible format. We convert our data into a numpy list format, and then set a max length.
This max length represents the length of a sentence - anything exceeding that limit will be truncated, anything under, will be padded. The length needs to be uniform, 
in order to be able to be converted into a Tensor.

```python
# Split the dataset
split = round(len(data)*0.7)

training_texts = tf_text[:split]
training_labels = label_list[:split]

validation_texts = tf_text[split:]
validation_labels = label_list[split:]

print('test:', len(training_texts), 'validation :', len(validation_texts)
```
<pre>test: 70000 validation: 30000</pre>

The data is then split 70/30 for training/validation datasets respectively. This will allow us to monitor the fitting of the models as the training occurs. What we want to see is
the loss for both to go down, with the accuracy for both going up, and what we want to avoid is seeing the accuracy for the training dataset going up, while the validation accuracy
falls or stays stagnant, as that means that overfitting is occurring.

<!--![hot-encoding](https://i.imgur.com/qzt1NLb.jpg)-->
```python
# One Hot Encoding
hot_encoded = tensorflow.keras.utils.to_categorical(training_labels)
validation_hot_encoded = tensorflow.keras.utils.to_categorical(validation_labels)

print(hot_encoded[:5])
```
<pre>[[0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]]</pre>
 
For multiclass training, the values of the labels/categories need to be "hot encoded" - that is, it needs to be turned into a type of array value. For example, if it is a positive review,
it becomes [0, 0, 1], negative reviews being [1, 0, 0] and neutral being [0, 0, 0].

<!--![define-model](https://i.imgur.com/WeiZBJD.jpg)-->
```python
# Create an model
tf.keras.backend.clear_session()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(MAX_VOCAB_COUNT, 16, input_length=MAX_WORD_LENGTH))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```
<pre>Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 50, 16)            12800     
                                                                 
 dropout (Dropout)           (None, 50, 16)            0         
                                                                 
 global_average_pooling1d (G  (None, 16)               0         
 lobalAveragePooling1D)                                          
                                                                 
 dropout_1 (Dropout)         (None, 16)                0         
                                                                 
 dense (Dense)               (None, 64)                1088      
                                                                 
 dense_1 (Dense)             (None, 32)                2080      
                                                                 
 dense_2 (Dense)             (None, 3)                 99        
                                                                 
=================================================================
Total params: 16,067
Trainable params: 16,067
Non-trainable params: 0
_________________________________________________________________</pre>

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

<!--![training](https://i.imgur.com/fRsb7nO.jpg)-->
```python
# Execute Training
epoch_count = 100
history = model.fit(training_texts, hot_encoded, validation_data=(validation_texts, validation_hot_encoded), epochs=epoch_count)
```

### Decision of the parameters

When we preprocess the input layer, we have used two constants; `MAX_VOCAB_COUNT` and `MAX_WORD_COUNT`.
We used arbitrary numbers for those constants until we built the model. Before the actual training,
we have tried two types of inspections; varying either `MAX_VOCAB_COUNT` or `MAX_WORD_COUNT`, without changing others.
The test training results are followings.

<div style="display: inline-block">
    <img src="https://user-images.githubusercontent.com/59322692/144569268-51d1c0bc-7a6f-474f-9fe3-d02a64f47297.png"
            alt="word_count_opt" style="width: calc(50% - 10px)" />
    <img src="https://user-images.githubusercontent.com/59322692/144569245-9807b6e3-44b7-446b-aed8-ecddefb5504d.png"
            alt="vocab_count_opt" style="width: calc(50% - 10px)" />
</div>

On the left side graph that inspected `MAX_WORD_COUNT`, it suggested that the maximum word count should be more than 80.
However, the another graph (`MAX_VOCAB_COUNT`) had no clear regressions of the loss and accuracy, so we couldn't figure out the meaningful
number of `MAX_VOCAB_COUNT` within our trials.

### Evaluation & Analysis

<!--
Graphs, tables, any statistics (if any)
-->

![training_statistics](https://user-images.githubusercontent.com/59322692/144564041-df476be1-be3d-4360-a293-9d94c0c8e29d.png)

During the 100 epochs of training, the minimum validation loss (`0.76578`, red line) was found at 46 epochs, and the maximum validation accuracy (`0.66642`, blue line) appeared at 93 epochs. As considering what the training history shows above, setting the epochs around 40 ~ 50 might be enough to fit a model.

~66% is definitely not the best in terms of accuracy, but validation accuracy is keeping up with training accuracy, and does not seem to be overfitting (which would be noticeable if training accuracy 
went up significantly more compared to validation accuracy).

### Confusion Matrix

![Confusion_Matrix_of_Validation_dataset](https://user-images.githubusercontent.com/59322692/144565938-7adeca78-c675-41bb-ad12-8c795e88735c.png)
![Confusion_Matrix_of_Test_dataset](https://user-images.githubusercontent.com/59322692/144566625-b9ed3e60-fc6c-4dba-b92a-912069e36126.png)

To evaluate the model, we made an confusion matrix for the validation dataset. Furthermore, to confirm the validity of the evaludation,
we applied this type of confusion matrix with the pure dataset which haven't be used during the model training. According to those matrices,
the precision of `Negative - Negative` and `Positive - Positive` classification is clearly higher than other combinations. Although it is disappointing
that the number of classified neutral sentences correctly is quite low, this phenomenon made us to remind the fact that the ratio of neutral words is
lower than negative and positive words' ratio. (see the above Dataset Inspection section)

### Performance measurement

We calculated the precision and recall scores for each target labels.

```python
# Analyze the result
from sklearn.metrics import classification_report
print(classification_report(val_y, val_p, target_names=['Negative', 'Neutral', 'Positive']))
```
<table class="dataframe" style="text-align: right;">
    <thead>
        <tr><th></th><th>precision</th><th>&emsp;recall</th><th>f1-score</th><th>&nbsp;support</th></tr>
    </thead>
    <tbody>
        <tr><th>Negative</th><td>0.68</td><td>0.78</td><td>0.73</td><td>159078</td></tr>
        <tr><th>Neutral</th><td>0.51</td><td>0.16</td><td>0.24</td><td>80495</td></tr>
        <tr><th>Positive</th><td>0.68</td><td>0.81</td><td>0.74</td><td>160427</td></tr>
        <tr><th>accuracy</th><td></td><td></td><td>0.67</td><td>400000</td></tr>
        <!--<tr><th>macro avg</th><td>0.62</td><td>0.58</td><td>0.57</td><td>400000</td></tr>
        <tr><th>weighted avg</th><td>0.64</td><td>0.67</td><td>0.63</td><td>400000</td></tr>-->
    </tbody>
</table>

Inside of the overall 67% accuracy, the Negative and Positive f1-scores are close to ~74%. It also revealed the fact that
the number of samples labeled with Neutral was almost half of other labels.

As we have seen, an easy way to increase our accuracy would be to drop it from multi-class to binary classification - the reduction in classes, from testing, yields a noticeable increase in accuracy (something like 
15 to 20 percent).

### Summary

By developing a model using Amazon's dataset, it was possible to conduct sentiment analysis of words in various sentences as much as possible. After that sentences in the Amazon data set were analyzed to increase the accuracy of emotional analysis by excluding unnecessary words from each sentence.

For sentiment analysis, the degree of positive/negative of each word was divided by score. To organize the data, we read the csv file by using pandas module. For the model training, the word was returned to the most basic unit and tokenized. Training a multiclass classification system with all 5 classes resulted in low validation accuracy. Therefore we devide the result into three classses; Negative, Positive, and Neutral.

To use multiclass training we converted the data into array value by using 'hot encoding'. It is a vector representation method of a word that uses the size of a set of words as a vector, gives the index of the word you want to represent, and gives the other index zero.

Out model consists of several layers. Keras Embedding layer, Dropout layer, Avarage Pooling layer, ReLU activated layers, softmax activated layer. Each layer balances data, removes deflection, and adusts weights during training. 

As a result, we have developed a model that has 65% accuracy. To increase the accuracy, we can shift the multi-classification to a binary classification which can increase the accuracy about 15-20% percent.

### Limitation & Further Research Direction

A limitation with our model is that it is not fully intended for our type of usage, which causes further inaccuracies in our results. However, during our dataset evaluation/processing, we noticed that despite the dataset being sourced from Amazon Reviews, it is more applicable for our use-case than we initially considered. For example, our wordcloud shows words that are clearly negative or positive in any context. 

In addition, some of the limitations we encounter with the format of this model is that the input must be formatted in a certain way. One of the further research directions would be to try and simplify this so that it would be less computationally expensive on the host application, so that the processing is done as a layer within the model.

We'd also like to see more accuracy/a better fit, as it is a bit awkward to see sentiment misclassification occurring (eg. an actual positive sentiment sentence being misconstrued as negative). Though there is always likely to be a mismatch, it would be nicer to see a higher overall accuracy, especially for our purposes.

### Related Work

We referred to several blogs, datasets and libraries in order to create the model and its training notebook. This is because the below work is heavily related to what we're trying to do (analyze textual sentiment).

[https://kaggle.com/bittlingmayer/amazonreviews](https://kaggle.com/bittlingmayer/amazonreviews)

[https://kaggle.com/paoloripamonti/twitter-sentiment-analysis](https://kaggle.com/paoloripamonti/twitter-sentiment-analysis)

[https://github.com/bentrevett/pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis)

[https://towardsdatascience.com/how-to-train-a-deep-learning-sentiment-analysis-model-4716c946c2ea](https://towardsdatascience.com/how-to-train-a-deep-learning-sentiment-analysis-model-4716c946c2ea)

[https://towardsdatascience.com/a-complete-step-by-step-tutorial-on-sentiment-analysis-in-keras-and-tensorflow-ea420cc8913f](https://towardsdatascience.com/a-complete-step-by-step-tutorial-on-sentiment-analysis-in-keras-and-tensorflow-ea420cc8913f)

[https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

[https://github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)

[https://tensorflow.org/text/guide/word_embeddings](https://tensorflow.org/text/guide/word_embeddings)

<!--
(e.g., existing studies)
Tools, libraries, blogs, or any documentation that you have used to do this project.
-->

### Conclusion

The above model was then saved, and then exported to the C++ application for use, alongside with a JSON of the word index. The application consists of a simple server, serving a webpage with basic HTML/CSS/JS to output the results of the transcription and sentiment analysis.

The process that happens is that AWS Transcription SDK uses PortAudio to forward audio to the AWS Transcription API in order to begin the transcription process. As the audio is transcribed, there are two types - partial, and full. Partial can be described as the "incomplete" transcription, while Full is described as the "complete" transcription - essentially the final prediction of what the audio said.

The final transcribed text is forwarded to the inference class. Internally, a similar text pre-processing occurs, where Amazon's Transcriptions results are then vectorized, processed, and then inserted into the model for inference. Partial text is not analyzed in order to cut down on unnecessary computation/inference on a non-final result, that will often be inaccurate. Partial text is still sent to the server however, to be displayed. The vectorized text was then passed through the word indexing map, to produce a vector of word indices, for inference. For inference, a Tensorflow wrapper library called Cppflow was used. This vector of word indices were then turned into the Cppflow Tensor objects, and inserted into the model for inference.

A tensor is returned from this, looking like the following `[0 9 12]`. This is similar to our hot-encoding process, so it would be equivalent to `[0 0 1]`, or 2 - positive sentiment. The max index is computed and cast into a matching Sentiment Enum, and passed with the text to the server. The server then sends a websocket message to all subscribed clients, where the javascript on the page places a corresponding colored dot at the end of the final sentence based on the inferred sentiment (as is seen in the demo, or the picture below).

![sentiment-transcriber-result](https://i.imgur.com/jvu3Zmh.png)

Overall we noticed that there is some bias towards either positive or negative sentiment, as expected, with often neutral text being construed as negative or positive. In addition, there are times where it provides incorrect readings, which we suspect come down to overall model inaccuracies.
