## Data cleaning and preprocessing

We firstly clean the data and make the necessary changes needed to proceed with further calculations. 
For this we perform the following actions:
• Remove the column with id which would not be useful in our calculations
• Drop rows with NA values
• Change the response variable values to a binary value with 1 indicating malignant and 0 indicating
benign
• Convert the value of the variables from a factor to a quantitative variable
• Scale the matrix containing all the predictor variables
• Form training set and test sets for further usage


## Exploratory Data Analysis

To get an overview of the data in hand we will first perform exploratory analysis on the data in hand.
![image](https://github.com/paul2596/breast_cancer_classifier/assets/71576923/d1cdf713-b762-4d89-a6b9-c6f1cdc719ad)

From the data above we can see that the predictor variables contain information about the 9 characteristics of the tissues that was observed. And the final response variable holds the value distinguishes between a
benign and malignant to describe the extent of severity of the tumor. We see the proportion of Benign and Malignant observations in our data which is around 65% Benign and 35% Malignant, so we can confirm that the data is not imbalanced. 
![image](https://github.com/paul2596/breast_cancer_classifier/assets/71576923/bc5725e8-ec5c-4bd0-b461-6790dbb31d5d)
Further the correlation plot shows the relation between the variables, most of the variables demonstrate a good correlation except for Mitoses which has a week correlation with the
other variables.


Building Classifiers Using:
1) Best subset selection method
The goal here is to identify the best model by taking out the explanatory variables that can predict the
response more accurately. We use regsubsets to carry out this in R and use exhaustive method to consider
all subsets
![image](https://github.com/paul2596/breast_cancer_classifier/assets/71576923/b5382811-aecb-4483-8414-af93760d3bc0)

We see that the model with 8 predictor variables seems to be have more significance. And we can drop the
predictor variable ‘Mitoses’ and choose the model with Cl.thickness, Cell.size, Cell.shape, Marg.adhesion,
Epith.c.size, Bare.nuclei, Bl.cromatin, Normal.nucleoli. This is the best subset model as it suits all the three
criteria’s and is indifferent.

2) Logistic regression using Lasso penalty
We use LASSO to shrink the coefficient values we do this by assigning a penalty to the loss function which is
lambda. Since the same shrinkage is carried out on all the variables, scaled data has to be used for LASSO.
![image](https://github.com/paul2596/breast_cancer_classifier/assets/71576923/188269fc-eb89-4404-a32c-da0c4b0c1a97)
![image](https://github.com/paul2596/breast_cancer_classifier/assets/71576923/4616dec1-ca3b-4124-b076-dc9e013304c0)


For lambda we consider a range of values including the null model containing only the intercept value. From
the plot and the coefficient summary of the LASSO fit we see that all the explanatory variables are present
and the shrinkage effect for minimum lambda is low. We also see the shrinkage effect on different coefficients
above for beta_hat[,25] the coefficient value of mitoses shrinks to 0. The range of lambda might have to be
tuned further to effectively drop coefficients. From the logLambda vs coefficient plot we see that the value
of the coefficients drop to 0 and the first explanatory variable to drop out is Mitoses which was suggested
by the subset selection method we had seen earlier.


3) Discriminant Analysis method Linear Disciminant Analysis (LDA)
We use LDA to find the region in which our explanatory variable falls in depending on the value it holds
![image](https://github.com/paul2596/breast_cancer_classifier/assets/71576923/ea823f9b-6a1f-4fac-a71d-efdfb63ba9ce)

From the LDA model we can observe that the prior probabilities of each group i.e malignant or benign
having percentages 64.61% for benign and 35.38% for malignant. The group mean shows the mean value of
the different tissue characteristics in different groups. We can also see the coefficients of linear discriminant
returns only LD1 because we have just one variable defining a group.


## Compare performance of models using cross validation based on Test Error

![image](https://github.com/paul2596/breast_cancer_classifier/assets/71576923/37534612-9553-41ee-b2e9-44b5f4148b32)

The best classifier is the one obtained using Best subset selection and Logistic regression with lasso penalty
because they have a low error value for prediction on the test set. It includes 8 out of the 9 predictor variables
i.e Cl.thickness, Cell.size, Cell.shape, Marg.adhesion, Epith.c.size, Bare.nuclei, Bl.cromatin, Normal.nucleoli.
Mitoses was left out from the subset selection based on the 3 criterias that was considered earlier i.e adjusted
R square, AIC and BIC, we see that we still get a good accuracy after dropping the column. This Classifier
could be further used on an unknown dataset to predict whether or not the tumour is Malignant or Benign
