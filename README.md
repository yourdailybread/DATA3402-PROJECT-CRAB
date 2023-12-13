# DATA3402-PROJECT-CRAB


# Crab Age Regression Problem

This repository attempts to apply Regression Learning Algorithms to Crab characteristics such as their weight and height using data from "Regression with a Crab Age Dataset" Kaggle challenge https://www.kaggle.com/competitions/playground-series-s3e16

# Overview


**DATA**

The input of my data are the Crabs characteristics which contain their Length, Diameter, Height, Weight, Shucked Weight, Viscera Weight, and Shell Weight which in total was around 6 MB.

**Preprocessing and Cleanup**

Some of the data that I cleaned up include ID since it wasn't relevant to the problem. 

**Data Visualization** 

![download](https://github.com/yourdailybread/DATA3402-PROJECT-TEMPLE/assets/123412500/aa757f33-191e-45f7-985b-696d26f1f934)

![download](https://github.com/yourdailybread/DATA3402-PROJECT-TEMPLE/assets/123412500/38841e66-fc40-48c1-bef5-b114ba717fc0)

**Problem Formulation** 

The purpose of using these model algorithms 


**Training**

Some of the training models I used included 
- Decision Tree
- Linear Regression Model 

**Performance Comparison** 

For the linear regression model, I used multiple regression performance

![image](https://github.com/yourdailybread/DATA3402-PROJECT-TEMPLE/assets/123412500/81a7b949-9c68-49ed-bebe-f1e3cf8cfa2a)


    mean absolute error 1.5009822216279365

    mean squred error 4.61101351720401

    root mean squred error 2.1473270633985897

    r2_score 0.541095305741286

**Conclusions**

Above all, I found that the linear regression model worked best for my project, however, in order to enhance my work in the future I would try to take more time to train different x and y values. Additionally, I would also try to use different models such as Clustering and Gaussian. 


# How to reproduce results 

- Train.csv: this is the training model originally in the Kaggle challenege, you can download it and upload it to your Jupyter Notebook
- Data_visualization.ipynb: This jupyter notebook is where I visualized all of my data from the training set and tried to see any patterns 
- linear_regression_model.ipynb: This notebook contains the Linear Regression training model
- training_model.ipynb: This notebook contains the Decision Tree training model

**Software Setup** 

The setup I used for all of the training models included import numpy, pandas, matplotlib.pyplot, seaborn, sklearn.preprocessing, sklearn.model_selection, sklearn.linear_model ,sklearn.linear_model, statsmodels.api

**Training** 

My first model I used a Decision Tree, I separated my data into a training set (70%) and a testing set (30%) with the variables X, Y, X_train, X_test, y_train, y_test. From there I used the gini index and entropy 

Then with my linear regression model, I created my x and y labels that corresponded with length and age, respectively. Then I scaled my model using Scikit learn's preprocessing and model selection functions, after that I used the multiple regression performance method to measure the accuracy of my model. 


**Citations**
https://scikit-learn.org/stable/index.html
https://www.geeksforgeeks.org/decision-tree-implementation-python/
