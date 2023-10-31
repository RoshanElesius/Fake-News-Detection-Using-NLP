# Fake News Detection

## Overview  

Fake news is like a wildfire in the age of social media, especially during critical events like elections and pandemics such as Covid-19. It's spreading like a contagion, and it's becoming increasingly difficult to separate fact from fiction amidst the information overload. In today's world, the credibility of news is crucial. With the rapid spread of information on social media, distinguishing fake news from real news is a pressing concern. Natural Language Processing tools offer a solution by using historical data to classify news articles accurately.

## Problem Definition

The problem is to develop a fake news detection model using a Kaggle dataset.
The goal is to distinguish between genuine and fake news articles based on their titles and text.
This project involves using natural language processing (NLP) techniques to preprocess the text data, building a machine learning model for classification, and evaluating the model's performance. 

The intended application of the project is for use in applying visibility weights in social media.  Using weights produced by this model, social networks can make stories which are highly likely to be fake news less visible.

## Dataset Description


[https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

* train.csv: A full training dataset with the following attributes:
  * id: unique id for a news article
  * title: the title of a news article
  * author: author of the news article
  * text: the text of the article; could be incomplete
  * label: a label that marks the article as potentially unreliable
    * 1: unreliable
    * 0: reliable
* test.csv: A testing training dataset with all the same attributes at train.csv without the label.

## File Structure
The file structure is the following

```
.
|
+-- datasets
|   +-- train.csv
|   +-- test.csv
+-- images
|   +-- Logistic Regression with hyperparameter Tuning-cm.png
|   +-- lstm-cm.png
+-- *.py
```

## Try It Out

1. Clone the repository by following this command  

`> git clone git://github.com/RoshanElesius/Fake-News-Detection-Using-NLP.git`

`> cd Fake-News-Detection-Using-NLP`

2. Make sure you have all the dependencies installed-  

1. Basic Libraries:
* pandas
* numpy

2. Visualization Libraries:
* matplotlib
* seaborn

3. NLTK Libraries:
* re and string
* stopwords
* wordcloud

4. Machine Learning Libraries:
* GridSearchCV
* LogisticRegression
* DecisionTreeClassifier
* MultinomialNB
* KNeighborsClassifier
*  train_test_split

5. Deep Learning Libraries:
* tensorflow.keras.layers
    

## Comparing Accuracies of Models

| Model                                               |     Accuracy      |
|:---------------------------------------------------:|:----------------: |
| Logistic Regression Test                            | 0.9660040199274997|
| Decision Tree Test                                  | 0.9353049482414729|
| KNN Test                                            | 0.6119253084088696|
| Naive Bayes Test                                    | 0.9373328405462511|
| Logistic Regression with hyperparameter Tuning      | 0.9803065407235787|
| LSTM                                                | 0.998218262806236 |

## Data Visualization

* Count of News subject

![Alt text](<count of news subject-1.png>)

* Count of News subject based on True or Fake news

![Alt text](<count of news subject(2).png>)

## N-gram Analysis

* Top 20 words in news

![Alt text](<20 words.png>)

* Top 20 bigrams in news

![Alt text](<20 bigram.png>)

## Word Cloud Of Fake News

![Alt text](<Fake news.png>)

## Word Cloud Of True News

![Alt text](<True News.png>)

## Time series analysis -Fake/True News

![Alt text](<Time series analysis.png>)


## ROC-AUC Curve

![Alt text](<ROC-AUC Curve.png>)

## Confusion Matrices

* Logistic Regression with hyperparameter Tuning

![Alt text](<logistic regression.png>)

* LSTM

![Alt text](LSTM.png)


