# Machine_Learning
In this repository you can find all the homeworks that I completed for the Machine Learning course at the University of Padua

## 1. Classification on wine dataset
We will be working with a dataset on wines from the UCI machine learning repository
(http://archive.ics.uci.edu/ml/datasets/Wine). It contains data for 178 instances. 
The dataset is the results of a chemical analysis of wines grown in the same region
in Italy but derived from three different cultivars. The analysis determined the
quantities of 13 constituents found in each of the three types of wines. 

## 2. SVM 
In this notebook we are going to explore the use of Support Vector Machines (SVMs) for image classification. We are going to use the famous MNIST dataset, that is a dataset of handwritten digits. We get the data from mldata.org, that is a public repository for machine learning data.

## 3. NN and Clustering
In this notebook we are going to explore the use of Neural Networks for image classification. We are going to use a dataset of small images of clothes and accessories, the Fashion MNIST. You can find more information regarding the dataset here: https://pravarmahajan.github.io/fashion/.
Moreover we cluster 2000 images in the fashion MNIST dataset, and try to understand if the clusters we obtain correspond to the true labels.

## 4. NN, Linear Regression, kNN, CLustering
We consider the dataset containing house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.

https://www.kaggle.com/harlfoxem/housesalesprediction

For each house we know 18 house features (e.g., number of bedrooms, number of bathrooms, etc.) plus its price, that is what we would like to predict.

4.1) NN: we learn the best neural network with 1 hidden layer and between 1 and 9 hidden nodes, choosing the best number of hidden nodes with cross-validation.
<br>
4.2) Linear Regression: we learn the linear model on train and validation, and get error (1-R^2) on train and validation and on test data.
<br>
4.3) k-Nearest Neighbours: In this part we explore the k-Nearest Neighbours (kNN) method for regression. In order to do this, 
we load the scikit-learn package *neighbors.KNeighborsRegressor* 
<br>
4.4) Clustering: In this part we explore the use of clustering to identify groups of *similar* instances, and then learning models that are specific to each group.
Once you have clustered the data, and then learned a model for each cluster, the prediction for a new instance is obtained by using the model of the cluster that is the closest to the instance, where the distance of a cluster to the instance is defined as the distance of the *center* of the cluster to the instance.
