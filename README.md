# Fake News Detection

## Overview  

Fake news is like a wildfire in the age of social media, especially during critical events like elections and pandemics such as Covid-19. It's spreading like a contagion, and it's becoming increasingly difficult to separate fact from fiction amidst the information overload. In today's world, the credibility of news is crucial. With the rapid spread of information on social media, distinguishing fake news from real news is a pressing concern. Natural Language Processing tools offer a solution by using historical data to classify news articles accurately.
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

i). Basic Libraries:
* pandas
* numpy

ii). Visualization Libraries:
* matplotlib
* seaborn

iii). NLTK Libraries:
* re and string
* stopwords
* wordcloud

iv). Machine Learning Libraries:
* GridSearchCV
* LogisticRegression
* DecisionTreeClassifier
* MultinomialNB
* KNeighborsClassifier
*  train_test_split

v). Deep Learning Libraries:
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

![count of news subject](https://github.com/RoshanElesius/Fake-News-Detection-Using-NLP/assets/138104926/9fe718fe-9e7f-4cd7-adcb-008b67964cf3)

* Count of News subject based on True or Fake news

![count of news subject(2)](https://github.com/RoshanElesius/Fake-News-Detection-Using-NLP/assets/138104926/865a052e-a490-45ab-8d95-d1a26eba2a69)


## N-gram Analysis

* Top 20 words in news

![20 words](https://github.com/RoshanElesius/Fake-News-Detection-Using-NLP/assets/138104926/367a1589-d0aa-4d2b-aad9-e6842dd7f7c9)


* Top 20 bigrams in news

![20 bigram](https://github.com/RoshanElesius/Fake-News-Detection-Using-NLP/assets/138104926/1cd2b88d-56c9-4cd9-8b26-99788a4e3fd3)


## Word Cloud Of Fake News

![Fake news](https://github.com/RoshanElesius/Fake-News-Detection-Using-NLP/assets/138104926/d6cc49b8-c906-4fe4-bfdb-de238a5e0b38)

## Word Cloud Of True News

![True News](https://github.com/RoshanElesius/Fake-News-Detection-Using-NLP/assets/138104926/da91bc8a-e02a-48ba-ac98-7405a857c62e)

## Time series analysis -Fake/True News

![Time series analysis](https://github.com/RoshanElesius/Fake-News-Detection-Using-NLP/assets/138104926/c1fb0b3d-64b8-4430-a641-e5bb19c05c7a)

## ROC-AUC Curve

![ROC-AUC Curve](https://github.com/RoshanElesius/Fake-News-Detection-Using-NLP/assets/138104926/a2d050f0-d298-4b00-899a-17a75056f150)

## Confusion Matrices

* Logistic Regression with hyperparameter Tuning

![logistic regression](https://github.com/RoshanElesius/Fake-News-Detection-Using-NLP/assets/138104926/27c49847-38f7-45f6-9316-1d5294273e74)

* LSTM

![LSTM](https://github.com/RoshanElesius/Fake-News-Detection-Using-NLP/assets/138104926/b2aeb462-eb31-4846-aafb-35a7d02beb06)

## Conclusion:
Natural Language Processing (NLP) is a pivotal weapon in our ongoing battle against the spread of fake news. Its effectiveness relies on intricate feature interactions, particularly in categorical features such as the 'subject' of news articles. Success in fake news detection requires comprehensive feature engineering, extracting valuable insights from text, and precise model optimization. By delving deep into these intricacies, NLP equips us with the tools to identify and combat the ever-evolving landscape of misinformation. It empowers us to gain a deeper understanding of deceptive content, reinforcing the foundation of trustworthy information and upholding the quality of public discourse. NLP's ability to unveil hidden associations and enhance predictive performance makes it a crucial asset in preserving the integrity of our information ecosystem.


