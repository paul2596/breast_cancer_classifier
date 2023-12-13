install.packages("mlbench")
install.packages("dplyr")
install.packages("tidyverse")
install.packages("leaps")
install.packages("glmnet")
install.packages("bestglm")
install.packages("MASS")
install.packages("corrplot")
install.packages("nclSLR")

library(mlbench)
library(dplyr)
library(tidyverse)
library(leaps)
library(glmnet)
library(bestglm)
library(MASS)
library(corrplot)
library(nclSLR)

data(BreastCancer)
?BreastCancer
#check size of dataset
dim(BreastCancer)
(p = ncol(BreastCancer) - 2)
#print first few rows
head(BreastCancer)

##Data cleaning
##drop NA values
breast_cancer_df <- BreastCancer %>% drop_na()
##malignant or benign as binary with value 1 and 0
breast_cancer_df <- breast_cancer_df %>% mutate(Class = ifelse(Class=='malignant', 1, 0))
##convert variables to a quantitative variable
for(i in 2:11){
  breast_cancer_df[,i] = as.numeric(breast_cancer_df[,i])
}

##dimension of the cleaned data set
dim(breast_cancer_df)

colMeans(data.matrix(breast_cancer_df[,2:10]))
apply(breast_cancer_df[,2:10], 2, sd)

cor(data.matrix(breast_cancer_df[2:11]))
corrplot(cor(data.matrix(breast_cancer_df[2:11])))




## Extract response variable
y <- breast_cancer_df[,11]

## Extract predictor variables to form model
x_raw <- breast_cancer_df[,2:10]
x_raw <- data.matrix(x_raw)

x_scaled <- scale(x_raw)
bc_data <- data.frame(x_scaled, y)

##For reproducibility of the results we set seed
set.seed(1234)
##Split data into training set and test set
sample <- sample(c(TRUE, FALSE), nrow(bc_data), replace=TRUE, prob=c(0.7,0.3))
train_set  <- bc_data[sample, ]
test_set   <- bc_data[!sample, ]
xtrain_set <- train_set[,1:9]
ytrain_set <- train_set[,10]

xtest_set <- test_set[,1:9]
ytest_set <- test_set[,10]

n = nrow(x_scaled)
p = ncol(x_scaled)

##Proportion of malignant and benign cases in the data
prop.table(table(bc_data$y))


corrplot(cor(bc_data), method = 'number')


########################
## best subset selection
########################
bss_fit = regsubsets(y~ ., data=bc_data, method="exhaustive", nvmax=9) 
## Summary of the results 
bss_summary = summary(bss_fit)



best_adjR = which.max(bss_summary$adjr2)

best_cp = which.min(bss_summary$cp)

best_bic = which.min(bss_summary$bic)

coef(bss_fit,6)
coef(bss_fit,8)

## dividing the  plot section
par(mfrow = c(1,3))

plot(1:9,bss_summary$adjr2, xlab="Number of predictors", ylab="Adjusted R^2", type="b")
points(best_adjR ,bss_summary$adjr2[best_adjR], col="red", pch=18 )



plot(1:9,bss_summary$cp, xlab="Number of predictors", ylab="CP", type="b")
points(best_cp ,bss_summary$cp[best_cp], col="red", pch=18 )


plot(1:9,bss_summary$bic, xlab="Number of predictors", ylab="BIC", type="b")
points(best_bic ,bss_summary$bic[best_bic], col="red", pch=18)



##We see that the model with 6 predictors gives a better fit
m8 <- lm(y ~ Cl.thickness + Cell.size+ Cell.shape+ Marg.adhesion +Epith.c.size+Bare.nuclei+ Bl.cromatin +Normal.nucleoli, data = bc_data) # 8 predictors
m6 <- lm(y ~ Cl.thickness +Cell.size +Cell.shape+Bare.nuclei+ Bl.cromatin +Normal.nucleoli , data = bc_data) #6 predictors
summary(m8)$coeff %>% round(4)
summary(m6)$coeff %>% round(4)

######################
## Logistic regression
######################
##regularized form of logistic regression, i.e. with a ridge or LASSO penalty;
## select values for the tuning parameter lambda

grid = 10^seq(-4, -1, length.out=100)
## Fit a model with LASSO penalty for each value corresponding to the tuning parameter
lasso_fit = glmnet(x_scaled, y, family="binomial", alpha=1, standardize=FALSE, lambda=grid)
beta_hat = coef(lasso_fit)
beta_hat[,1]

beta_hat[,25]

beta_hat[,100]

## split window for plots
par(mfrow=c(1,1))

##fit the scaled data using glmnet
lasso_cv_fit = cv.glmnet(x_scaled, y, family="binomial", alpha=1, standardize=FALSE, lambda=grid,
                         type.measure="class")

plot(lasso_fit, xvar="lambda", col=rainbow(9), label=TRUE)
plot(lasso_cv_fit)

# The optimal value for the tuning parameter
lambda_lasso_min = lasso_cv_fit$lambda.min

which_lambda_lasso = which(lasso_cv_fit$lambda == lambda_lasso_min)
## Find the parameter estimates associated with optimal value of the tuning parameter
coef(lasso_fit, s=lambda_lasso_min)

#####
##LDA 
#####
lnda_temp = linDA(variables=matrix(as.matrix(train_set[,c(1,2,3,4,5,6,7,8)]), ncol=8),
                  group=train_set[,10])


lda_train = lda(y~., data=train_set[,c(1,2,3,4,5,6,7,8,10)])
## Compute fitted values for the validation data
lda_test = predict(lda_train, test_set[,c(1,2,3,4,5,6,7,8,10)])
yhat_test_lda = lda_test$class
lda_train

##Coefficients
lnda_temp$functions
## Calculate (test) confusion matrix
(confusion_lda = table(Observed=test_set$y, Predicted=yhat_test_lda))

## Compare performance of models using cross validation based on Test Error
# Training error without LASSO penalty is 0.02928258
# Training error with LASSO penalty is 0.02369668
# Test error without LASSO penalty is 0.01
# Test error with LASSO penalty is 0.02369668
##For reproducibility of the results we set seed
set.seed(12345)
logreg1_train = glm(y ~ ., data=test_set[,c(1,2,3,4,5,6,7,8,10)], family="binomial")
bc_data_red_train = data.frame(train_set[,c(1,2,3,4,5,6,7,8,10)])
bc_data_red_test = data.frame(test_set[,c(1,2,3,4,5,6,7,8,10)])

##Best fit predictions
best_sub_logreg_fit = glm(y ~ ., data=bc_data_red_train, family="binomial")
phat_best_sub = predict(best_sub_logreg_fit,bc_data_red_test,type="response")
yhat_best_sub = ifelse(phat_best_sub>0.5,1,0)


##summarize fit of the model
summary(logreg_fit)


##Training error without LASSO penalty
phat_train = predict(log_reg_fit_bm, bc_data_red, type="response")
## Compute fitted (i.e. predicted) values:
yhat_train = ifelse(phat_train > 0.5, 1, 0)
## Calculate confusion matrix:
(confusion__t_wl = table(Observed=y, Predicted=yhat_train))

##Training error with LASSO penalty
## Compute predicted probabilities:
phat_train_l = predict(lasso_fit, x_scaled, s=lambda_lasso_min, type="response")
## Compute fitted (i.e. predicted) values:
yhat_train_l = ifelse(phat_train_l > 0.5, 1, 0)
## Calculate confusion matrix:
(confusion__t_l = table(Observed=y, Predicted=yhat_train_l))
1 - mean(y==yhat_train_l)

##Test error without LASSO penalty
wl_train = glm(y~.,data=bc_data_red_train, family="binomial")
phat_test_wl = predict(wl_train,bc_data_red_test,type="response")
yhat_test_wl = ifelse(phat_test_wl>0.5,1,0)




## Perform cross-validation over the training data to select tuning parameter for 10 folds
lasso_cv_train_10 = cv.glmnet(as.matrix(train_set[,c(1,2,3,4,5,6,7,8)]), as.matrix(train_set[,10]),
                              family="binomial", alpha=1, standardize=FALSE, lambda=grid,
                              type.measure="class")
## Identify the optimal value for the tuning parameter
(lambda_lasso_min_train_10 = lasso_cv_train_10$lambda.min)


##Test error with LASSO penalty
## Fit logistic regression model with LASSO penalty to training data:
lasso_train = glmnet(train_set[,c(1,2,3,4,5,6,7,8)], train_set[,10], family="binomial",
                     alpha=1, standardize=FALSE, lambda=lambda_lasso_min_train_10)
## Compute fitted values for the validation data:
phat_test_las_10 = predict(lasso_train, as.matrix(test_set[,c(1,2,3,4,5,6,7,8)]), s=lambda_lasso_min_train_10,
                           type="response")
yhat_test_las_10 = ifelse(phat_test_las_10 > 0.5, 1, 0)



## Perform cross-validation over the training data to select tuning parameter for 5 folds
lasso_cv_train_5 = cv.glmnet(as.matrix(train_set[,c(1,2,3,4,5,6,7,8)]), as.matrix(train_set[,10]),
                             family="binomial", alpha=1, standardize=FALSE, lambda=grid,
                             type.measure="class", nfolds = 5)
## Identify the optimal value for the tuning parameter
(lambda_lasso_min_train_5 = lasso_cv_train_5$lambda.min)


##Test error with LASSO penalty
## Fit logistic regression model with LASSO penalty to training data:
lasso_train = glmnet(train_set[,c(1,2,3,4,5,6,7,8)], train_set[,10], family="binomial",
                     alpha=1, standardize=FALSE, lambda=lambda_lasso_min_train_5)
## Compute fitted values for the validation data:
phat_test_las_5 = predict(lasso_train, as.matrix(test_set[,c(1,2,3,4,5,6,7,8)]),
                          s=lambda_lasso_min_train_5,
                          type="response")
yhat_test_las_5 = ifelse(phat_test_las_5 > 0.5, 1, 0)

##########################################
#Computing Errors for the classifiers built
##########################################
##Training error with LASSO penalty
1 - mean(y==yhat_train_l)

##Training error without LASSO penalty
1 - mean(y==yhat_train)

##Test error with LASSO penalty
1-mean(test_set$y == yhat_test_las_10)

##Test error without LASSO penalty
1-mean(test_set$y==yhat_test_wl)

## Compute test error for best subset selection model
1-mean(test_set$y==yhat_best_sub)

## Compute test error for lasso 10 fold model
1 - mean(test_set$y == yhat_test_las_10)

## Compute test error for lasso 5 fold
1 - mean(test_set$y == yhat_test_las_5)

##Test error for LDA
1 - mean(test_set$y == yhat_test_lda)

##Forming summary to identify best classifier
methods <-c("Best subset selection","LR with Lasso","LDA")

error<-c((1-mean(test_set$y==yhat_best_sub)),
         (1-mean(test_set$y == yhat_test_las_10)),
         (1-mean(test_set$y == yhat_test_lda)))

##Final summary of test errors
data.frame(Method = methods, Error=error)

