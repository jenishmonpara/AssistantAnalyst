# AssistantAnalyst
[Web App](https://assistantanalyst.herokuapp.com)

This project seeks to utilize Feed Forward Neural Networks, Long-Short Term Memory (LSTM) Neural Network algorithm, to predict stock prices.

## Contents
- [Contents](#contents)
- [Overview](#overview)
- [Data Used](#data-used)
- [Simple Models](#simple-models)
- [Neural Network Model](#neural-network-model)
- [LSTM](#lstm)
- [What next?](#what-next?)

## Overview
Investment firms, hedge funds and even individuals have been using financial models to better understand market behavior and make profitable investments and trades. A wealth of information is available in the form of historical stock prices and company performance data, suitable for machine learning algorithms to process.

Can we actually predict stock prices with machine learning? Investors make educated guesses by analyzing data. They'll read the news, study the company history, industry trends and other lots of data points that go into making a prediction. The prevailing theories is that stock prices are totally random and unpredictable but that raises the question why top firms like Morgan Stanley and Citigroup hire quantitative analysts to build predictive models. We have this idea of a trading floor being filled with adrenaline infuse men with loose ties running around yelling something into a phone but these days they're more likely to see rows of machine learning experts quietly sitting in front of computer screens. In fact about 70% of all orders on Wall Street are now placed by software, we're now living in the age of the algorithm.

In this project I have experimented with various Machine Learning algorithms and models to predict stock prices. For data with timeframes recurrent neural networks (RNNs) come in handy but recent researches have shown that LSTM, networks are the most popular and useful variants of RNNs.

I have used Keras to build a LSTM to predict stock prices using historical closing price and trading volume and visualize both the predicted price values over time and the optimal parameters for the model.

## Data Used
I have used Yahoo Finances api with the help of pandas-datareader to get up to date data on tickers. The DataReader function call returns a dataframe object of following type : 

![Data format](https://github.com/jenishmonpara/AssistantAnalyst/blob/main/Dataset%20head.png)


## Simple Models
Initally I tried linear regressor and it did not do very well. However if you try multi-variate linear regression by including the effects of "Volume" for each day, it will give you better results. 
SVM regressor with rbf kernel performed very bad both, with and without volumes. I tried many optimal and suboptimal parameter (C,gamma) tuning after grid search but it still was not able to perform well. 
I think I am making some mistake in formulation of problem before determining the input feature shape and I am working on it to correct this.

## Neural Network Model
This was a big improvement on simple models. I experimented with number of layers and optimizers. To summarize, 2 dense layers worked very well. I avoided adding more layers to eliminate overfitting. I tried SGD and Adam optimizers, among which Adam seemed to be working better.

* Summary
![Summary](https://github.com/jenishmonpara/AssistantAnalyst/blob/main/Neural%20Model.png)
* Training
![Training](https://github.com/jenishmonpara/AssistantAnalyst/blob/main/Neural%20Training.png)
* Forecast
![Forecast](https://github.com/jenishmonpara/AssistantAnalyst/blob/main/Neural%20Forecast.png)

## LSTM
I am still working on this part. I want to use LSTM for its memory retaining property.

## What next?
* Deploying the project.
* Predicting price as a weighted average of all models. I am considering tuning the weights using reinforcement learning.
* Incorporating more features such as sentiment analysis.
