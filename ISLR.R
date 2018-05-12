
# Chapter 1: ISLR ---------------------------------------------------------




## Introduction to Statistical Learning with R
# https://rpubs.com/ppaquay/
          
## Chapter 1: Introduction
# Overview of Statistical Learning

# statistical learning refers to understanding data and the tools used to do it
# each tool can be classified as supervised or unsupervised

# supervised: predicting an outcome based on inputs
# unsupervised: inputs but no explicit output - we can learn relationships and strucutre of data

# we will work through real world data sets in this book
library("ISLR")
data(Wage)

# wage data:
# we want to understand the association between employee's age and education
# also calendar year on wage
# given an employees age we can predict age
# variation makes it unlikely that age is the only predictor for wage
# most accrurate prediction may include age, education, and the year

data("Smarket")
# stock market data
# wage data involves predicting a continous or quantitative output value
# this is often refered to as a regression problem - trying to predict an estimate
# we may want to predict a categorical or qualitative output
# the goal is to predict if stock will "increase" or "decrease"
# the statistical problem does not include predicting an exact number
# will we fall into "up" or "down" bucket?
# this is known as a classification problem

data("NCI60")
# gene expression data
# another important class of problems involves only observing input variables
# we may want to group different customers together - clustering problem
# we are not trying to predict an outcome variable (numeric or categorical)
# we want to determine if there are groups in our data
# principal components are an example of this method - finding the smallest dimensions that explain the most



# chapter 2: SLM ----------------------------------------------------------



## Chapter 2: Statistical Learning
# if we can determine relationship between advertisting and sales:
# we can adjust budget to put adversting in the best places to gain the most sales
# advertising budgets are the input variables
# sales is the output variable
# inputs go by many different names: predictors, dependent variable, features - INDEPENDANT VARIABLE
# sales is the repsonse or the DEPENDENT VARIABLE

# suppose we have a quantititive repsonse Y from predictors p
# we assume there is some relationship between Y and the x(set of predictors)
# Y = function(X) + error
# f is a fixed unknown function of X and has a random error term
# f represnts the systematic information which X provides about Y
# we want to estimate f based on obsevations of X and Y
# Statistical Learning: set of approaches for estimating f = the relationship between repsonse and independant variables


# why estimate f?
        # prediction
        # inference

## Prediction:
# in many situations we have the X but not the repsonse Y
# we can predict Y using Yhat = fhat(X)
# fhat is our estimate of the relationship between Y and X are
# predicting drug reaction based on blood samples
        # we can easily get blood samples
        # want to predict reaction so we don't give drugs to people who will have reactions
# the accuracy of our prediction Yhat depends on two quantities:
        # reducible error
        # irreducible error
# in general our estimate of f will not be perfect
# we can reduce some of this error by using the best statistical learning technique to estimate f
# however, even if we had the perfect f, our repsonse prediction would still have some error!!
# this is becuase Y is a function of the error term - and cannot be predicted by X
# variability of Y by e also affects our predictions = IRREDUCIBLE ERROR
# no matter how well we estinate f we cannot reduce the error of the error term e
# e may contain error from variable we did not include
# e may contain "natural" error that is unmeasurable
# error = the squared difference between the actual value and the predicted value
# this error can be reduced to "reducible" and "irreducible"
# variance of the error is the irreducible error
# the focus of this book is to estimate f with the aim of minimizing reducible error!
# irreducible error will always provide the upper bound of the prediction interval


## Inference:
# we are often intereted is understanding the way our response is affected as our features change
# our goal is to estimate F but not necessarily make predictions for Y
# we want to understand the relationship between X and Y
# we want to understand how Y changes as a function of our variables
        # which predictors are associated with the response? = identify IMPORTANT predictors
        # what is the relationship between the response and each predictor? = postive / negative predictors
        # can the relationship between Y and each predictor be summarized by linear equation?
# many problems are prediction, inference or a combo of both!
# how will variables affect the probability of purchase?
# modeling can be done for both prediction and inference
# different models "fit" for different problems i.e. prediction, inference
# simple linear models may be good for inference but not extrememly accurate prediction
# complicated models may be good for accurate prediction but not good for inference


## How do we estimate F?
# most models show similiar ways of estimating F - the function between repsonse and predictors
# training data = observations we used to train our models in estimating F
# our goal is to apply a statistical learning method to the training data in order to estimate F
# most models for estimating F can be parametric or non-parametric
        # parametric = modeling based on a underlying "known" distribution
        # non-parametric = modeling without using an underlying distribution

## Parametric Methods
        # first we make an assumption about the functional form of F
                # is the form linear? f(x) = Bo + B1x1 + ... Bnxn + error
        # use the training data to fit our model = TRAIN our model
# parametric modeling reduces the problem of estimating f down to estimating a set of model parameters
# this is much easier = i.e. fitting a linear model with estimates for F by each predictor
# the problem with parametric methods is we may not know the underlying form of the function F
# if our model does not match the form of the underlying data our model will be poor
# we can try to model on multiple different distributions but this adds complexity
# complexity can cause us to overfit our training data - predict too well and not have results generalize
# overfitting is following the noise or error to closely in predictions - this will not perform well in real life
        # linear form: income = intercept + slope(eduction) + slope(seniority)
# since we assumed a linear relationship between the response and two etimators: our problem is only estimating intercept, eduction slope and seniority slope
# results may not be in a linear form - regression may capture some of the information but not all
# there is a positive relationship between eduction and income but predicting off this may be not the best

## Non-Parametric Methods
# NPM  do not make assumptions about the underlying form of the data
# we want our F to get as close to the data points as possible 
# NPM have the potential to fit a wider range for possilble shapes of F
# parametric may be "wrong" about the underlying form and give us a bad fit
# NPM avoid this danger - no assumption of the form is made
# since NPM do not reduce the estimating F problem to a small amount of parameters a lot of data is needed
# there are trade-offs between parametric and non-parametric methods:
        # PM: assumes the underlying distribution; danger if assumption is wrong
        # NPM: no assumption but we can overfit; danger in selecting the best fit 


## Trade off between Prediction Accuracy and Model Interpretability
# some models are more flexible than others: ie linear regression vs. thin-line spline
# why would we ever choose a more restricted model instead of a flexible approach?
# restrictive models are better at inference = we can easily understand the relationship of the variables
# other models can give back highly complex estimates of f that is difficult to understand how any one predictor is associated with the repsonse
# least squares = inflexible but interpretable
# lasso = more inflexiible than linear regression = only best predictors are kept in the model
# generalized additive model: more flexible than linear regression (uses curve to estimate)
# non linear models: bagging, boosting, support vector machines with non-linear kernals are harder to intrepet but more flexible
# if inference is the goal= use simple and inflexible statistical learning methods
# if we want the most accurate prediction = more fleixble and complicated methods may be the best
# however = highly flexible methods have a high potential of overfitting the noise of the data (error)
# in some cases simple rigid models may better predict becuase they do not overfit as much as flexible methods


## Supervised vs. Unsupervised Learning
# most statitical learning problems fall into supervised or unsupervised learning
# supervised: for each predictor there is a repsonse measurement
        # we wish to fit a model to determine the function between our predictors and our response
        # we can predict and infer based on the predictors and the repsonse
        # linear regression, logistic regression, GAM, boosting, SVM are all supervised - they require a set of predictors and  repsonse variable
# unsupervised: we observed data but do not have a set repsonse = "working blind"
# we lack the response variable that can supervise our analysis
# we want to figure out the relationships between this data = between variables and observations
# cluster analysis = group observations together on the basis of similiar observations
# we are identifying underlying gorups within the data 
# semi-supervised learning: we know the predictors but not all of the repsonses: we can use learning methods to model on obs with known repsonse and those without known repsonse


## Regression vs. Classification
# variables can be quantitative or qualitative (categorical)
# quantitative = numbers, qualitative - classes of characters
# qualitative = classification problems
# quantitative = regression problems
# some models can be mixed with qualitiative and quantitiative variables
# we are generally concerned with the type of variable the response variable is 
# variable type of predictors does not matter too much - we can have quant / qual predictors in a regression or classification problem (repsonse)


## Assesing Model Accuracy
# there is no free lunch in statistics: no one method works for all data
# there are many statistical learning methods that we can apply to many problems
# different methods may work better on certaion data sets or with certain problems
# it is important to have a process to ascertain with method works the "best"

# Measuring the Quality of Fit
# we need to measure how well the predictions actually match the observed data
# quantify how the predicted repsonse is close to the actual values
# regression setting: mean squared error (MSE) is a common measure
# MSE will be small if the predicted values are close to the actual values
# MSE can be computed on the training dataset but we need to know how it will generalize!
# we want to estimate the MSE we will see when new "unseen" data comes into the model
# we want to predict into the future: need a measure to tell us performance on unknown future values of predictors
# we want the model that gives us the lowest TEST MSE as possible
# one way is to apply the model built on training to the test data and calculate MSE
# there is no gaurentee the model with lowest TRAIN MSE will have the lowest TEST MSE
# as we increase complexity we may fit the TRAIN data better, but may not generalize to TEST!
# TRAIN MSE and TEST MSE almost diverge as we add more complexity to our models
# as complexity increases - TRAIN MSE will DECREASE - but TEST MSE may NOT!!
# this is known as OVERFITTING the training data!!!
# we are working to hard to find patterns in the training data - patterns that will not generalize into the unseen testing data
# TRAIN MSE will always be lower than TEST MSE
# but - more complexity may result in higher TEST MSE than restrictive models 
# OVERFITTING = a less flexible model produces better results than a complex model
# we need to use cross validation to see how our TRAIN MSE will generalize to TEST MSE!!


## Bias - Variance Trade-Off
# TEST MSE  = decomposed into three parts: variance, bias, variarnce of the error
# expected test MSE - average test MSE we would get if repeatedly estimated f using a large number of training sets
# in order to minimize expected test error, we need to pick a stats model that gives us low variance and low bias
# what are variance and bias?
# variance is the amount F will change if we estimated it using a different training set
        # different training sets will give us different F
        # we do not want the F to vary alot in different training sets
        # complex models have a high variance - F will change based on training sets
# bias refers to the error of estimating a real problem with too simple a model
        # linear regression required a linear form - bias will be there if form is not exactly linear
        # if the form of the data is non-linear - no amount of training data will produce good results using linear models
        # more flexible models result in less bias
# as we use more flexible methods variance will increase but bias will decrease
# the rate at which these quantities go up or down determines whether TEST MSE goes down
# complex models - bias reduced faster than variance = TEST MSE will go down
# however at some point further complexity does not improve bias but we keep adding in more VARIANCE!
# this results in higher TEST MSE
# this relationship between bias and variance is the bias-variance tradeoff
# it is easy to get low variance but high bias will be there
# easy to get low bias but high variance will be there
# our best model will aim to minimize both at a certain "Sweet spot"
# this trade off is fundamental to deciding on the best model
# cross validation is a way to estimate the TEST MSE based on the TRAIN MSE

## Classification Setting
# how do we pick the best classification model?
# the bias variance trade off also applies to the classification setting!
# we have to modify our thinking since the response is no longer categorical
# most common approach for quantifying accuracy in our classification problem is ERROR RATE
# ERROR RATE = the proportion of mistakes that are made when we apply F to our training data
# if our actual repsonse matches the predicted repsonse we are accurate
# we keep the same experiement design by applying this metric to our training data first
# we will then apply our classifier to the test set to get our TEST ERROR
# we want a model that has very low TEST ERROR

## Bayes Classifier
# test error is minimized on average by a simple classifier that assigns each observation to the most likely class given its predictors
# we simply assign a test observation with predictor set  to the class with the highest probbability 
# this probability is a conditional probability where we want probability given a unqiue set of predictors
# This simple classifier is called the Bayes Classifier
# we can use the bayes classifier to assign observations to the class they will most likely be apart of given thier unique values
# bayes decision boundry is where observations fall exactly in between the classes based on thier unique values
# the Bayes classifier produces the lowest possible test error rate = BAYES ERROR RATE
# Bayes Error is exactly the same as the irreducible error 

## K Nearest Neighbors
# we can't always use Bayes classifier becuase we do not always have the conditional distribution of Y given predictors vector
# many approaches try to estimate the conditional distribution and mimic Bayes Classifier by assigning obs by the higest estimated probability
# K NEAREST NEIGHBORS classifier does this 
# first identifies the K points in the training data that are closest to the newest observation, then it estimates the conditional probability for assign the new observation its class
# if K = 3, KNN will find the three closest points by distance, then estimate and assign it a class based on highest estimated probabaility
# KNN can get close to the "gold standard" Bayes Classifier
# remember the true conditional distribution between a value and its predictors is not known in KNN
# the choice of K is manually selected and has a large effect on the classifier we obtain
# here is where our variance / bias tradeoff comes back in again
# higher K will give us less variance but higher bias
# lower K will give us more variance but lower bias
# we again want a classifier that minimizes both variance and bias!!!
# with K = 1, training error may be low but it may not generalize to the new test data!
# with K = 1000 training error may be low but we may overfit to the noise of the training data, test error may suffer

### IN BOTH REGRESSION AND CLASSIFICATION settings - CHOOSING THE CORRECT level of flexibility is crital to the success of the statistical learning model
# the U shape of highly flexible models in test metric can make this choice very difficult!
# we can use the training data to estimate the test error using cross validation!
# this will help us choose a statistical learning method that minimizes variance and bias = the "best" model with optimal flexibility without OVERFITTING!!





## Chapter 2: R LAB
# Conceptual Questions:

## 1. will performance be better or worse from a flexible model?

# large sample n and small number of predictors?
# more flexible fit will be better fitting to the large sample size

# small sample with very large amount of predictors?
# flexible will be worse - small amount of observations will overfit the model

# the relationship between predictors and response is highly non-linear?
# flexible will be better - able to model closer to the actual relationship

# the variance of the error terms is extremely high?
# flexible model will fit worse! - will fit the noise of the highly variable data


## 2. Is each scenario classificaiton problem or regression problem?

# profit, employees, industry and salary - want to understand salary for CEOs
# regression and inference

# will new product be success or failure? based on previous product launches and a set of predictors
# classification - or "classification" regression - finding the probability new product will be a success or failure

# want to predict % change in exchange rate in relation to weekly stock market - have predictors each with % of change
# regression - we want to predict the quantitative % of change


## 3. revist the bias-variance decomposition:

# provide the reasoning of bias, variance, training error, test error, irreducible error as we go from less flexible to more flexible
# bias: more flexible will have increasing bias = bias is the change of F based on new samples - flexible can overfit to the data it is looking at
# variance: more flexible will have decreasing variance = flexible models can better fit the data closer to the actual observed points
# training error: more flexible will have decreasing training error: as we fit better with fleixble models we reduce error because we fit better to the observed points
# testing error: classic "U" shape as we get more flexible: we will reduce testing error to a certain point, then overfit the training data and increase testing error as we get more complex
# bayes irreducible error: defines the lower limit that the test error is bounded: training error lower than irreducible error we have overfit


## 4. real-life applications of statistical learning

# what are applications of a classificaiton model?
# will customer churn yes or no: historical data on customers flagged with yes-churn and no-churn
# will customer conquest a Mazda, yes or no: historical data on customers flagged with yes-conquest, no-conquest
# will ad work with a certain target group, yes or no: historical stats on adds that achieved a certain customer group response rate, yes -above limit, no - below limit

# what are applications of a regression model?
# exactly how many sales will we sale this month: time, controlables, outside factors, incentives
# how many parts should we order for a new carline launch? histoircal data, planned sales, estimated sales, time
# what is the estimated amount of profit we should expect relative to the industry? profit, incentive spend, volume estimates, time

# what are the applications of a cluster analysis model?
# are there specific subgroups of our dealerships
# are there specific subgroups of our most loyal customers
# natural cluster of which customers repsonse to which accessories

## 5. What are the advantages and disadvantages of a flexible model in classification and regression?

# flexible models can give accurate predictors but are hard to draw inference from 
# flexible models tend to overfit the "noise" of the data it can "see" and will not generalize well
# these cases apply to both regression and classification problems
# less flexible models may not match the "form" of the underlying data and not predict as well as a flexible model (i.e linear model on non-linear data)
# less flexible models will have more variance in thier predictions


## 6. what are the differences between a parametric and non-parametric statistical learning approach?

# parametric = based on the underlying form of the data - if we know the underlying distribution (linear, poisson) - we can accurately model the data
# non-parametric = not concerned with the form of the data - aims to achieve best fit by resampling
# parametric is great if we know the distribution, not great if we do not know the distribution or if it is very unique
# non-parametric can overfit if there is a underlying distribution in the data
# non-parametric requires a ton of data to estimate the relationship, something we might not have available

## Applied in R

## 8. explore the college dataset

# load data
library(ISLR)
ISLR::College

str(College); dim(College); summary(College)
# 'data.frame':	777 obs. of  18 variables:
# $ Private    : Factor w/ 2 levels "No","Yes": 2 2 2 2 2 2 2 2 2 2 ...
# $ Apps       : num  1660 2186 1428 417 193 ...
# $ Accept     : num  1232 1924 1097 349 146 ...
# $ Enroll     : num  721 512 336 137 55 158 103 489 227 172 ...
# $ Top10perc  : num  23 16 22 60 16 38 17 37 30 21 ...
# $ Top25perc  : num  52 29 50 89 44 62 45 68 63 44 ...
# $ F.Undergrad: num  2885 2683 1036 510 249 ...
# $ P.Undergrad: num  537 1227 99 63 869 ...
# $ Outstate   : num  7440 12280 11250 12960 7560 ...
# $ Room.Board : num  3300 6450 3750 5450 4120 ...
# $ Books      : num  450 750 400 450 800 500 500 450 300 660 ...
# $ Personal   : num  2200 1500 1165 875 1500 ...
# $ PhD        : num  70 29 53 92 76 67 90 89 79 40 ...
# $ Terminal   : num  78 30 66 97 72 73 93 100 84 41 ...
# $ S.F.Ratio  : num  18.1 12.2 12.9 7.7 11.9 9.4 11.5 13.7 11.3 11.5 ...
# $ perc.alumni: num  12 16 30 37 2 11 26 37 23 15 ...
# $ Expend     : num  7041 10527 8735 19016 10922 ...
# $ Grad.Rate  : num  60 56 54 59 15 55 63 73 80 52 ...
# [1] 777  18

# Private        Apps           Accept          Enroll       Top10perc       Top25perc      F.Undergrad   
# No :212   Min.   :   81   Min.   :   72   Min.   :  35   Min.   : 1.00   Min.   :  9.0   Min.   :  139  
# Yes:565   1st Qu.:  776   1st Qu.:  604   1st Qu.: 242   1st Qu.:15.00   1st Qu.: 41.0   1st Qu.:  992  
# Median : 1558   Median : 1110   Median : 434   Median :23.00   Median : 54.0   Median : 1707  
# Mean   : 3002   Mean   : 2019   Mean   : 780   Mean   :27.56   Mean   : 55.8   Mean   : 3700  
# 3rd Qu.: 3624   3rd Qu.: 2424   3rd Qu.: 902   3rd Qu.:35.00   3rd Qu.: 69.0   3rd Qu.: 4005  
# Max.   :48094   Max.   :26330   Max.   :6392   Max.   :96.00   Max.   :100.0   Max.   :31643  
# P.Undergrad         Outstate       Room.Board       Books           Personal         PhD            Terminal    
# Min.   :    1.0   Min.   : 2340   Min.   :1780   Min.   :  96.0   Min.   : 250   Min.   :  8.00   Min.   : 24.0  
# 1st Qu.:   95.0   1st Qu.: 7320   1st Qu.:3597   1st Qu.: 470.0   1st Qu.: 850   1st Qu.: 62.00   1st Qu.: 71.0  
# Median :  353.0   Median : 9990   Median :4200   Median : 500.0   Median :1200   Median : 75.00   Median : 82.0  
# Mean   :  855.3   Mean   :10441   Mean   :4358   Mean   : 549.4   Mean   :1341   Mean   : 72.66   Mean   : 79.7  
# 3rd Qu.:  967.0   3rd Qu.:12925   3rd Qu.:5050   3rd Qu.: 600.0   3rd Qu.:1700   3rd Qu.: 85.00   3rd Qu.: 92.0  
# Max.   :21836.0   Max.   :21700   Max.   :8124   Max.   :2340.0   Max.   :6800   Max.   :103.00   Max.   :100.0  
# S.F.Ratio      perc.alumni        Expend        Grad.Rate     
# Min.   : 2.50   Min.   : 0.00   Min.   : 3186   Min.   : 10.00  
# 1st Qu.:11.50   1st Qu.:13.00   1st Qu.: 6751   1st Qu.: 53.00  
# Median :13.60   Median :21.00   Median : 8377   Median : 65.00  
# Mean   :14.09   Mean   :22.74   Mean   : 9660   Mean   : 65.46  
# 3rd Qu.:16.50   3rd Qu.:31.00   3rd Qu.:10830   3rd Qu.: 78.00  
# Max.   :39.80   Max.   :64.00   Max.   :56233   Max.   :118.00 


# pairs plot of the data:
ggpairs(College[,1:10])

# boxplot of select variables: Outstate and Private
ggplot(data = College) +
        geom_violin(aes(x = Private, y = Outstate, color = Private))

# create a new variable Elite by binning the Top10perc variable
# how many elite colleges are there - use summary - 78 total colleges
college <- College %>% 
        mutate(Elite = factor(ifelse(Top10perc > 50, "Yes", "No")))

summary(college)
# Private        Apps           Accept          Enroll       Top10perc       Top25perc    
# No :212   Min.   :   81   Min.   :   72   Min.   :  35   Min.   : 1.00   Min.   :  9.0  
# Yes:565   1st Qu.:  776   1st Qu.:  604   1st Qu.: 242   1st Qu.:15.00   1st Qu.: 41.0  
# Median : 1558   Median : 1110   Median : 434   Median :23.00   Median : 54.0  
# Mean   : 3002   Mean   : 2019   Mean   : 780   Mean   :27.56   Mean   : 55.8  
# 3rd Qu.: 3624   3rd Qu.: 2424   3rd Qu.: 902   3rd Qu.:35.00   3rd Qu.: 69.0  
# Max.   :48094   Max.   :26330   Max.   :6392   Max.   :96.00   Max.   :100.0  
# F.Undergrad     P.Undergrad         Outstate       Room.Board       Books           Personal   
# Min.   :  139   Min.   :    1.0   Min.   : 2340   Min.   :1780   Min.   :  96.0   Min.   : 250  
# 1st Qu.:  992   1st Qu.:   95.0   1st Qu.: 7320   1st Qu.:3597   1st Qu.: 470.0   1st Qu.: 850  
# Median : 1707   Median :  353.0   Median : 9990   Median :4200   Median : 500.0   Median :1200  
# Mean   : 3700   Mean   :  855.3   Mean   :10441   Mean   :4358   Mean   : 549.4   Mean   :1341  
# 3rd Qu.: 4005   3rd Qu.:  967.0   3rd Qu.:12925   3rd Qu.:5050   3rd Qu.: 600.0   3rd Qu.:1700  
# Max.   :31643   Max.   :21836.0   Max.   :21700   Max.   :8124   Max.   :2340.0   Max.   :6800  
# PhD            Terminal       S.F.Ratio      perc.alumni        Expend        Grad.Rate     
# Min.   :  8.00   Min.   : 24.0   Min.   : 2.50   Min.   : 0.00   Min.   : 3186   Min.   : 10.00  
# 1st Qu.: 62.00   1st Qu.: 71.0   1st Qu.:11.50   1st Qu.:13.00   1st Qu.: 6751   1st Qu.: 53.00  
# Median : 75.00   Median : 82.0   Median :13.60   Median :21.00   Median : 8377   Median : 65.00  
# Mean   : 72.66   Mean   : 79.7   Mean   :14.09   Mean   :22.74   Mean   : 9660   Mean   : 65.46  
# 3rd Qu.: 85.00   3rd Qu.: 92.0   3rd Qu.:16.50   3rd Qu.:31.00   3rd Qu.:10830   3rd Qu.: 78.00  
# Max.   :103.00   Max.   :100.0   Max.   :39.80   Max.   :64.00   Max.   :56233   Max.   :118.00  
# Elite    
# No :699  
# Yes: 78 


# use histogram to plot some of the key variables
par(mfrow = c(2,2))

# hist on apps
hist(college$Apps)

#hist on accept
hist(college$Accept)

#hist on enroll
hist(college$Enroll)

#hist on F.Undergrad
hist(college$F.Undergrad)




## 9. explore the auto data set

# load the data
# which variables are categorical and which are numerical = use structure!
data("Auto")
summary(Auto); str(Auto)
# mpg          cylinders      displacement     horsepower        weight      acceleration  
# Min.   : 9.00   Min.   :3.000   Min.   : 68.0   Min.   : 46.0   Min.   :1613   Min.   : 8.00  
# 1st Qu.:17.00   1st Qu.:4.000   1st Qu.:105.0   1st Qu.: 75.0   1st Qu.:2225   1st Qu.:13.78  
# Median :22.75   Median :4.000   Median :151.0   Median : 93.5   Median :2804   Median :15.50  
# Mean   :23.45   Mean   :5.472   Mean   :194.4   Mean   :104.5   Mean   :2978   Mean   :15.54  
# 3rd Qu.:29.00   3rd Qu.:8.000   3rd Qu.:275.8   3rd Qu.:126.0   3rd Qu.:3615   3rd Qu.:17.02  
# Max.   :46.60   Max.   :8.000   Max.   :455.0   Max.   :230.0   Max.   :5140   Max.   :24.80  
# 
# year           origin                      name    
# Min.   :70.00   Min.   :1.000   amc matador       :  5  
# 1st Qu.:73.00   1st Qu.:1.000   ford pinto        :  5  
# Median :76.00   Median :1.000   toyota corolla    :  5  
# Mean   :75.98   Mean   :1.577   amc gremlin       :  4  
# 3rd Qu.:79.00   3rd Qu.:2.000   amc hornet        :  4  
# Max.   :82.00   Max.   :3.000   chevrolet chevette:  4  
# (Other)           :365 

# 'data.frame':	392 obs. of  9 variables:
# $ mpg         : num  18 15 18 16 17 15 14 14 14 15 ...
# $ cylinders   : num  8 8 8 8 8 8 8 8 8 8 ...
# $ displacement: num  307 350 318 304 302 429 454 440 455 390 ...
# $ horsepower  : num  130 165 150 150 140 198 220 215 225 190 ...
# $ weight      : num  3504 3693 3436 3433 3449 ...
# $ acceleration: num  12 11.5 11 12 10.5 10 9 8.5 10 8.5 ...
# $ year        : num  70 70 70 70 70 70 70 70 70 70 ...
# $ origin      : num  1 1 1 1 1 1 1 1 1 1 ...
# $ name        : Factor w/ 304 levels "amc ambassador brougham",..: 49 36 231 14 161 141 54 223 241 2 ...

# what is the range of each predictor?
sapply(Auto[,1:8], range)
#        mpg cylinders displacement horsepower weight acceleration year origin
# [1,]  9.0         3           68         46   1613          8.0   70      1
# [2,] 46.6         8          455        230   5140         24.8   82      3

# what is the mean of each predictor?
sapply(Auto[,1:8], mean)
# mpg    cylinders displacement   horsepower       weight acceleration         year       origin 
# 23.445918     5.471939   194.411990   104.469388  2977.584184    15.541327    75.979592     1.576531

# what is the standard deviation of each predictor?
sapply(Auto[,1:8], sd)
# mpg    cylinders displacement   horsepower       weight acceleration         year       origin 
# 7.8050075    1.7057832  104.6440039   38.4911599  849.4025600    2.7588641    3.6837365    0.8055182 


# remove the 10th through 85th observations what is the mean, sd and range of the new data set?
auto.cut <- Auto[-10:-85,]

dim(auto.cut); dim(Auto)

sapply(auto.cut[,1:8], range)
# mpg cylinders displacement horsepower weight acceleration year origin
# [1,] 11.0         3           68         46   1649          8.5   70      1
# [2,] 46.6         8          455        230   4997         24.8   82      3

sapply(auto.cut[,1:8], mean)
# mpg    cylinders displacement   horsepower       weight acceleration         year       origin 
# 24.404430     5.373418   187.240506   100.721519  2935.971519    15.726899    77.145570     1.601266 

sapply(auto.cut[,1:8], sd)
# mpg    cylinders displacement   horsepower       weight acceleration         year       origin 
# 7.867283     1.654179    99.678367    35.708853   811.300208     2.693721     3.106217     0.819910 


# investigate relationships between each of the variables
# which variables seem useful to include in a linear regression model?
# cylinders, displacement, weight, horsepower could be useful predictors
ggpairs(Auto[,1:8])



## 10. explore the boston dataset

# read in the data and explore dimensions
library(MASS)
data("Boston")
?Boston


str(Boston); summary(Boston); dim(Boston)
# 'data.frame':	506 obs. of  14 variables:
#         $ crim   : num  0.00632 0.02731 0.02729 0.03237 0.06905 ...
# $ zn     : num  18 0 0 0 0 0 12.5 12.5 12.5 12.5 ...
# $ indus  : num  2.31 7.07 7.07 2.18 2.18 2.18 7.87 7.87 7.87 7.87 ...
# $ chas   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ nox    : num  0.538 0.469 0.469 0.458 0.458 0.458 0.524 0.524 0.524 0.524 ...
# $ rm     : num  6.58 6.42 7.18 7 7.15 ...
# $ age    : num  65.2 78.9 61.1 45.8 54.2 58.7 66.6 96.1 100 85.9 ...
# $ dis    : num  4.09 4.97 4.97 6.06 6.06 ...
# $ rad    : int  1 2 2 3 3 3 5 5 5 5 ...
# $ tax    : num  296 242 242 222 222 222 311 311 311 311 ...
# $ ptratio: num  15.3 17.8 17.8 18.7 18.7 18.7 15.2 15.2 15.2 15.2 ...
# $ black  : num  397 397 393 395 397 ...
# $ lstat  : num  4.98 9.14 4.03 2.94 5.33 ...
# $ medv   : num  24 21.6 34.7 33.4 36.2 28.7 22.9 27.1 16.5 18.9 ...
# crim                zn             indus            chas              nox               rm       
# Min.   : 0.00632   Min.   :  0.00   Min.   : 0.46   Min.   :0.00000   Min.   :0.3850   Min.   :3.561  
# 1st Qu.: 0.08204   1st Qu.:  0.00   1st Qu.: 5.19   1st Qu.:0.00000   1st Qu.:0.4490   1st Qu.:5.886  
# Median : 0.25651   Median :  0.00   Median : 9.69   Median :0.00000   Median :0.5380   Median :6.208  
# Mean   : 3.61352   Mean   : 11.36   Mean   :11.14   Mean   :0.06917   Mean   :0.5547   Mean   :6.285  
# 3rd Qu.: 3.67708   3rd Qu.: 12.50   3rd Qu.:18.10   3rd Qu.:0.00000   3rd Qu.:0.6240   3rd Qu.:6.623  
# Max.   :88.97620   Max.   :100.00   Max.   :27.74   Max.   :1.00000   Max.   :0.8710   Max.   :8.780  
# age              dis              rad              tax           ptratio          black       
# Min.   :  2.90   Min.   : 1.130   Min.   : 1.000   Min.   :187.0   Min.   :12.60   Min.   :  0.32  
# 1st Qu.: 45.02   1st Qu.: 2.100   1st Qu.: 4.000   1st Qu.:279.0   1st Qu.:17.40   1st Qu.:375.38  
# Median : 77.50   Median : 3.207   Median : 5.000   Median :330.0   Median :19.05   Median :391.44  
# Mean   : 68.57   Mean   : 3.795   Mean   : 9.549   Mean   :408.2   Mean   :18.46   Mean   :356.67  
# 3rd Qu.: 94.08   3rd Qu.: 5.188   3rd Qu.:24.000   3rd Qu.:666.0   3rd Qu.:20.20   3rd Qu.:396.23  
# Max.   :100.00   Max.   :12.127   Max.   :24.000   Max.   :711.0   Max.   :22.00   Max.   :396.90  
# lstat            medv      
# Min.   : 1.73   Min.   : 5.00  
# 1st Qu.: 6.95   1st Qu.:17.02  
# Median :11.36   Median :21.20  
# Mean   :12.65   Mean   :22.53  
# 3rd Qu.:16.95   3rd Qu.:25.00  
# Max.   :37.97   Max.   :50.00  
# [1] 506  14


# make some scatterplots on certain variables

# crime rate on black
ggplot(data = Boston) +
        geom_jitter(aes(x = black, y = crim))

# crime rate on taxes
ggplot(data = Boston) +
        geom_jitter(aes(x = tax, y = crim))

# crime rate on industry
ggplot(data = Boston) +
        geom_jitter(aes(x = indus, y = crim))

# crime rate on lstat
ggplot(data = Boston) +
        geom_jitter(aes(x = lstat, y = crim))

# how many suburbs are bound by the Charles river?
sum(Boston$chas)
# [1] 35

# what is the median pupil-teacher ratio amoung towns in this dataset?
median(Boston$ptratio)
# [1] 19.05

# which towns have the lowest median value of owner occupied homes?
Boston[Boston$medv==min(Boston$medv),]
#         crim zn indus chas   nox    rm age    dis rad tax ptratio  black lstat medv
# 399 38.3518  0  18.1    0 0.693 5.453 100 1.4896  24 666    20.2 396.90 30.59    5
# 406 67.9208  0  18.1    0 0.693 5.683 100 1.4254  24 666    20.2 384.97 22.98    5

sapply(Boston, range)
#          crim  zn indus chas   nox    rm   age     dis rad tax ptratio  black lstat medv
# [1,]  0.00632   0  0.46    0 0.385 3.561   2.9  1.1296   1 187    12.6   0.32  1.73    5
# [2,] 88.97620 100 27.74    1 0.871 8.780 100.0 12.1265  24 711    22.0 396.90 37.97   50


# how many rooms average more than 7 rooms per dwelling?
count(Boston[Boston$rm >= 7,])

# # A tibble: 1 x 1
# n
# <int>
#  1    64
Boston[Boston$rm >= 7,]
#         crim   zn indus chas    nox    rm   age    dis rad tax ptratio  black lstat medv
# 3    0.02729  0.0  7.07    0 0.4690 7.185  61.1 4.9671   2 242    17.8 392.83  4.03 34.7
# 5    0.06905  0.0  2.18    0 0.4580 7.147  54.2 6.0622   3 222    18.7 396.90  5.33 36.2
# 41   0.03359 75.0  2.95    0 0.4280 7.024  15.8 5.4011   3 252    18.3 395.62  1.98 34.9
# 56   0.01311 90.0  1.22    0 0.4030 7.249  21.9 8.6966   5 226    17.9 395.93  4.81 35.4
# 65   0.01951 17.5  1.38    0 0.4161 7.104  59.5 9.2229   3 216    18.6 393.24  8.05 33.0
# 89   0.05660  0.0  3.41    0 0.4890 7.007  86.3 3.4217   2 270    17.8 396.90  5.50 23.6
# 90   0.05302  0.0  3.41    0 0.4890 7.079  63.1 3.4145   2 270    17.8 396.06  5.70 28.7
# 98   0.12083  0.0  2.89    0 0.4450 8.069  76.0 3.4952   2 276    18.0 396.90  4.21 38.7
# 99   0.08187  0.0  2.89    0 0.4450 7.820  36.9 3.4952   2 276    18.0 393.53  3.57 43.8
# 100  0.06860  0.0  2.89    0 0.4450 7.416  62.5 3.4952   2 276    18.0 396.90  6.19 33.2
# 162  1.46336  0.0 19.58    0 0.6050 7.489  90.8 1.9709   5 403    14.7 374.43  1.73 50.0
# 163  1.83377  0.0 19.58    1 0.6050 7.802  98.2 2.0407   5 403    14.7 389.61  1.92 50.0
# 164  1.51902  0.0 19.58    1 0.6050 8.375  93.9 2.1620   5 403    14.7 388.45  3.32 50.0
# 167  2.01019  0.0 19.58    0 0.6050 7.929  96.2 2.0459   5 403    14.7 369.30  3.70 50.0
# 181  0.06588  0.0  2.46    0 0.4880 7.765  83.3 2.7410   3 193    17.8 395.56  7.56 39.8
# 183  0.09103  0.0  2.46    0 0.4880 7.155  92.2 2.7006   3 193    17.8 394.12  4.82 37.9
# 187  0.05602  0.0  2.46    0 0.4880 7.831  53.6 3.1992   3 193    17.8 392.63  4.45 50.0
# 190  0.08370 45.0  3.44    0 0.4370 7.185  38.9 4.5667   5 398    15.2 396.90  5.39 34.9
# 193  0.08664 45.0  3.44    0 0.4370 7.178  26.3 6.4798   5 398    15.2 390.49  2.87 36.4
# 196  0.01381 80.0  0.46    0 0.4220 7.875  32.0 5.6484   4 255    14.4 394.23  2.97 50.0
# 197  0.04011 80.0  1.52    0 0.4040 7.287  34.1 7.3090   2 329    12.6 396.90  4.08 33.3
# 198  0.04666 80.0  1.52    0 0.4040 7.107  36.6 7.3090   2 329    12.6 354.31  8.61 30.3
# 199  0.03768 80.0  1.52    0 0.4040 7.274  38.3 7.3090   2 329    12.6 392.20  6.62 34.6
# 201  0.01778 95.0  1.47    0 0.4030 7.135  13.9 7.6534   3 402    17.0 384.30  4.45 32.9
# 203  0.02177 82.5  2.03    0 0.4150 7.610  15.7 6.2700   2 348    14.7 395.38  3.11 42.3
# 204  0.03510 95.0  2.68    0 0.4161 7.853  33.2 5.1180   4 224    14.7 392.78  3.81 48.5
# 205  0.02009 95.0  2.68    0 0.4161 8.034  31.9 5.1180   4 224    14.7 390.55  2.88 50.0
# 225  0.31533  0.0  6.20    0 0.5040 8.266  78.3 2.8944   8 307    17.4 385.05  4.14 44.8
# 226  0.52693  0.0  6.20    0 0.5040 8.725  83.0 2.8944   8 307    17.4 382.00  4.63 50.0
# 227  0.38214  0.0  6.20    0 0.5040 8.040  86.5 3.2157   8 307    17.4 387.38  3.13 37.6
# 228  0.41238  0.0  6.20    0 0.5040 7.163  79.9 3.2157   8 307    17.4 372.08  6.36 31.6
# 229  0.29819  0.0  6.20    0 0.5040 7.686  17.0 3.3751   8 307    17.4 377.51  3.92 46.7
# 232  0.46296  0.0  6.20    0 0.5040 7.412  76.9 3.6715   8 307    17.4 376.14  5.25 31.7
# 233  0.57529  0.0  6.20    0 0.5070 8.337  73.3 3.8384   8 307    17.4 385.91  2.47 41.7
# 234  0.33147  0.0  6.20    0 0.5070 8.247  70.4 3.6519   8 307    17.4 378.95  3.95 48.3
# 238  0.51183  0.0  6.20    0 0.5070 7.358  71.6 4.1480   8 307    17.4 390.07  4.73 31.5
# 254  0.36894 22.0  5.86    0 0.4310 8.259   8.4 8.9067   7 330    19.1 396.90  3.54 42.8
# 257  0.01538 90.0  3.75    0 0.3940 7.454  34.2 6.3361   3 244    15.9 386.34  3.11 44.0
# 258  0.61154 20.0  3.97    0 0.6470 8.704  86.9 1.8010   5 264    13.0 389.70  5.12 50.0
# 259  0.66351 20.0  3.97    0 0.6470 7.333 100.0 1.8946   5 264    13.0 383.29  7.79 36.0
# 261  0.54011 20.0  3.97    0 0.6470 7.203  81.8 2.1121   5 264    13.0 392.80  9.59 33.8
# 262  0.53412 20.0  3.97    0 0.6470 7.520  89.4 2.1398   5 264    13.0 388.37  7.26 43.1
# 263  0.52014 20.0  3.97    0 0.6470 8.398  91.5 2.2885   5 264    13.0 386.86  5.91 48.8
# 264  0.82526 20.0  3.97    0 0.6470 7.327  94.5 2.0788   5 264    13.0 393.42 11.25 31.0
# 265  0.55007 20.0  3.97    0 0.6470 7.206  91.6 1.9301   5 264    13.0 387.89  8.10 36.5
# 267  0.78570 20.0  3.97    0 0.6470 7.014  84.6 2.1329   5 264    13.0 384.07 14.79 30.7
# 268  0.57834 20.0  3.97    0 0.5750 8.297  67.0 2.4216   5 264    13.0 384.54  7.44 50.0
# 269  0.54050 20.0  3.97    0 0.5750 7.470  52.6 2.8720   5 264    13.0 390.30  3.16 43.5
# 274  0.22188 20.0  6.96    1 0.4640 7.691  51.8 4.3665   3 223    18.6 390.77  6.58 35.2
# 277  0.10469 40.0  6.41    1 0.4470 7.267  49.0 4.7872   4 254    17.6 389.25  6.05 33.2
# 281  0.03578 20.0  3.33    0 0.4429 7.820  64.5 4.6947   5 216    14.9 387.31  3.76 45.4
# 283  0.06129 20.0  3.33    1 0.4429 7.645  49.7 5.2119   5 216    14.9 377.07  3.01 46.0
# 284  0.01501 90.0  1.21    1 0.4010 7.923  24.8 5.8850   1 198    13.6 395.52  3.16 50.0
# 285  0.00906 90.0  2.97    0 0.4000 7.088  20.8 7.3073   1 285    15.3 394.72  7.85 32.2
# 292  0.07886 80.0  4.95    0 0.4110 7.148  27.7 5.1167   4 245    19.2 396.90  3.56 37.3
# 300  0.05561 70.0  2.24    0 0.4000 7.041  10.0 7.8278   5 358    14.8 371.58  4.74 29.0
# 305  0.05515 33.0  2.18    0 0.4720 7.236  41.1 4.0220   7 222    18.4 393.68  6.93 36.1
# 307  0.07503 33.0  2.18    0 0.4720 7.420  71.9 3.0992   7 222    18.4 396.90  6.47 33.4
# 342  0.01301 35.0  1.52    0 0.4420 7.241  49.3 7.0379   1 284    15.5 394.74  5.49 32.7
# 365  3.47428  0.0 18.10    1 0.7180 8.780  82.9 1.9047  24 666    20.2 354.55  5.29 21.9
# 371  6.53876  0.0 18.10    1 0.6310 7.016  97.5 1.2024  24 666    20.2 392.05  2.96 50.0
# 376 19.60910  0.0 18.10    0 0.6710 7.313  97.9 1.3163  24 666    20.2 396.90 13.44 15.0
# 454  8.24809  0.0 18.10    0 0.7130 7.393  99.3 2.4527  24 666    20.2 375.87 16.74 17.8
# 483  5.73116  0.0 18.10    0 0.5320 7.061  77.0 3.4106  24 666    20.2 395.28  7.01 25.0


# how many average more than 8 rooms per dwelling?
count(Boston[Boston$rm >= 8,])
# # A tibble: 1 x 1
# n
# <int>
#         1    13

Boston[Boston$rm >= 8,]
#        crim zn indus chas    nox    rm  age    dis rad tax ptratio  black lstat medv
# 98  0.12083  0  2.89    0 0.4450 8.069 76.0 3.4952   2 276    18.0 396.90  4.21 38.7
# 164 1.51902  0 19.58    1 0.6050 8.375 93.9 2.1620   5 403    14.7 388.45  3.32 50.0
# 205 0.02009 95  2.68    0 0.4161 8.034 31.9 5.1180   4 224    14.7 390.55  2.88 50.0
# 225 0.31533  0  6.20    0 0.5040 8.266 78.3 2.8944   8 307    17.4 385.05  4.14 44.8
# 226 0.52693  0  6.20    0 0.5040 8.725 83.0 2.8944   8 307    17.4 382.00  4.63 50.0
# 227 0.38214  0  6.20    0 0.5040 8.040 86.5 3.2157   8 307    17.4 387.38  3.13 37.6
# 233 0.57529  0  6.20    0 0.5070 8.337 73.3 3.8384   8 307    17.4 385.91  2.47 41.7
# 234 0.33147  0  6.20    0 0.5070 8.247 70.4 3.6519   8 307    17.4 378.95  3.95 48.3
# 254 0.36894 22  5.86    0 0.4310 8.259  8.4 8.9067   7 330    19.1 396.90  3.54 42.8
# 258 0.61154 20  3.97    0 0.6470 8.704 86.9 1.8010   5 264    13.0 389.70  5.12 50.0
# 263 0.52014 20  3.97    0 0.6470 8.398 91.5 2.2885   5 264    13.0 386.86  5.91 48.8
# 268 0.57834 20  3.97    0 0.5750 8.297 67.0 2.4216   5 264    13.0 384.54  7.44 50.0
# 365 3.47428  0 18.10    1 0.7180 8.780 82.9 1.9047  24 666    20.2 354.55  5.29 21.9










# chapter 3: linear regression --------------------------------------------




## Chapter 3: Linear Regression
# linear regression is a simple approach to supervised learning
# linear regressiom is a tool for predicting a quantitiative repsonse

# motivating example:
# is there a relationship between advertisting budget and sales?
# how strong is the relationship between advertising budget and sales?
# which media contribute to sales?
# how accurately can we estimate the effect of each mediuam on sales?
# how accurately can we predict future sales?
# is the relationship between advertising and sales strictly linear?
# is there an interaction or syngery effect in combining the different marketing mediums?

# LINEAR REGRESSION CAN BE USED TO ANSWER ALL OF THESE QUESTIONS!!

## Simple Linear Regression:
# predict a quatitative repsonse Y on the basis of a SINGLE predictor variable X
# basic formula: Y = Bo + B1*x1 = simple linear regression!
# note: Y is approximately modeled as in textbook - wavy equals sign
# example: sales = Bo + B1 * TV
# equation B0 and B1 are two unknown constants that we would like to estimate
# Bo and B1 are known as the model coefficients or parameters
# once we have estimates of both Bo and B1 we can predict Y on new values of X!!
# this new value will be an estimate based on our model!!!

## Estimating the Coefficients
# in practice B0 and B1 are unknown - if we want to predict - we must use known data to estimate our model parameters
# for example: let's take many real life actual sales and advertsiting spend and model these estimates on this data
# our goal is to fit a linear model that fits the data well
# we want to find the B0 (intercept) and B1 (slope) to fit our line as close to the actual observations as possible

# there are many ways to measure this "CLOSENESS" of our model line to the actual data
# THE MOST COMMON is LEAST SQUARES criteria
# for each prediction we can calculate the RESIDUALS = the difference from the prediction minus the actual value!!
# RESDIUAL SUM OF SQUARE ERROR = residual1^2 + ... + residualN^2 (add up the squared residuals for each prediction and observation)
# THE LEAST SQUARES criteria chooses the BO and B1 that MINIMIZES THE RESUIDAL SUM OF SQUARED ERROR formula!!!
# the minimizers of the LEAST SQUARED ERROR (RSS) is the sample means of x and y
# example: B0 = 7.03, B1 = .0475 = an additional $1000 spent on TV is associated with selling approximately 48 additional sales!
# these estimates were chosen by mimimizing the RSS function - the values gives are the exact values that minimize the function

# Assessing the Accuracy of our Coefficient Estimates
# for an estimated model: Y = B0 + B1*X + error
# B0 is the intercept term = expected value of Y when X is zero
# B1 is the slope term = average increase in Y with a one-unit increase in X
# e is the error term = catch-all for what we may miss with this simple model: the TRUE relationship may not exactly be lienar
# THIS MODEL ESTIMATE IS THE POPULATION REGRESSION LINE: OUR BEST ESTIMATE OF THE TRUE RELATIONSHIP BETWEEN Y AND X
# THE LEAST SQUARES REGRESSION COEFFICIENT ESTIMATES (B0 and B1) ARE THE LEAST SQUARES LINE
# IN REAL WORLD APPLICATIONS WE CANNOT KNOW THE TRUE RELATIONSHIP AND MUST ESTIMATE IT FROM THE DATA
# WE WILL HAVE ACCESS TO A SET OF OBSERVATIONS FROM WHERE WE CAN COMPUTE THE LEAST SQUARES LINE
# population regression line is the "true" value of the function = WE DO NOT KNOW THIS!
# THE LEAST SQUARES LINE IS OUR BEST ESTIMATE OF THE TRUE FUNCTION = WE GET THIS BY MINIMIZING THE RSS OF THE SLOPE AND INTERCEPT TERM

# at first glance - how can multiple lines accruately hone in on the true relationship?
# this is a concept driven from INTRODUCTORY STATISTICS
# sample mean will not equal the population mean = BUT IT IS A GOOD ESTIMATE if we have lots of data points that we are trying to measure!
# in the same way - we do not know the intercept and slope terms but can estimate them for our best approxmiation of the "true" population regression relationship
# the estimates of the least squares line are our best estimates of the b0 (intercept) and b1 slope of the true population regression line
# the relationship between sample mean example and linear regression is based on the concept of BIAS
# if we use the sample mean to estimate the actual mean, the estimate is unbiased = on average we expect U-hat to equal U
# THIS MEANS ON THE BIAS OF ONE PARTICULAR SET OF OBSERVATIONS THAT WE KNOWN= u-hat is exactly U based on a large sample size!! we know all the observations and can directly calculate the U for the data available
# AN UNBIASED ESTIMATOR DOES NOT SYSTEMATICALLY OVER OR UNDER ESTIMATE THE TRUE PARAMETER = THIS REQUIRES A LARGE AMOUNT OF OBSERVATIONS !!
# this property of unbiasness also holds true for linear regression:
# if we estimate B0 and B1 on the bias of a particular dataset, then our estimates won't be 100% true to the ACTUAL function B1 and B0
# BUT IF WE COULD AVERAGE THESE ESTIMATES OUT OVER A HUGE AMOUNT OF OBSERVATIONS - THESE ESTIMATES WOULD GET BETTER AND BETTER!!
# THIS IS THE IDEA OF AVERAGING MANY REGRESSION LINES TOGETHER BASED ON DIFFERENT SAMPLES OF DATA TO GET AS CLOSE AS POSSIBLE TO THE TRUE REGRESSION LINE!!!

# can we guage how accurate our estimate is of U or of the relationship function?
# an average of U or the relationship function will be very accurate over many datasets: but a single estimate of U may be significantly over or under called!
# HOW FAR OFF WILL THAT SINGLE ESTIMATE OF U BE?
# we answer this question by computing the standard error = sigma ^2 divided by n = VARIANCE = SE^2
# the standard error tells us the average amount that our estimate of U differs from the actual value of U
# THIS EQUATION ALSO TELLS US THAT OUR STANDARD ERROR SHRINKS WHEN WE INCREASE N!!!!!!
# WE CAN TAKE THIS CASE AND APPLY IT TO LEAST SQUARES LINE!!!!

# we can compute standard errors for both B1 and B0 of our regression line
# notice that the standard error of B1 is smaller when we have more "Spread" of x values to estimate the model on
# WE HAVE MORE LEVERAGE TO ESTIMATE OUR LINE WHEN OUR Xs ARE MORE SPREAD OUT!
# notice that the standard error of B0 is the same standard error as if x where 0
# in general we do not know sigma^2 explicitly  - but we can estimate it from our data 
# THE ESTIMATE OF sigma squared is known as the RESIDUAL STANDARD ERROR and equals = RSE = sqrt(RESIDUAL SUM OF SQUARES) / (n - 2)

# standard errors can be used to compute condifence intervals
# we can determine for some level of confidence that the true value will fall within a range of values - we do not know the true value but we know it falls in the computed range
# the range is defined in terms of lower and upper limits computed from our observed data
# WE CAN USE CONFIDENCE INTERVALS ON REGRESSION COEFFICIENTS
# confidence interval of the regression slope == B1hat + / - 2 * SE(B1hat) = this range will contain the true value of B1 with some level of confidence!!
# confidence interval of the intercept term == B0hat +/- 2 * SE(B0hat)

# case example: advertising data: 
# 95% CI for B1 is [.042, .053], 95% CI for B0 is [6.13, 7.935]
# this means we conclude = without advertising = sales will on average fall somewhere between 6,130 and 7,940 units!!
# EACH $1000 increase in television advertising will give us on average increase of sales between 42 and 53 units!!

# standard errors can also be used for HYPOTHESIS TESTING ON OUR REGRESSION COEFFICIENTS!
# most common hypothesis test invloves testing the NULL HYPOTHESIS OF = IS THIS COEFFICIENT EQUAL TO ZERO? OR IS THERE NO RELATIONSHIP BETWEEN X AND Y?
# H0: there is no relationship between X and Y, Ha: There is some relationship between X and Y
# THIS CORRESPONDS TO TESTING: H0: B1 = 0 vs. HA: B1 != 0
# if B1 is zero - the term in our model will "drop out with bong in hand" and we conclude there is no relationship between X and Y
# WE NEED TO TEST HOW FAR AWAY OUR COEFFICIENT IS AWAY FROM ZERO = WE DO THIS DEPENDING ON THE STANDARD ERROR FOR OUR COEFFICIENT!
# IF WE HAVE SMALL STANDARD ERROR VALUES OF B1 = THERE IS STRONG EVIDENCE THAT B1 != 0
# IF STANDARD ERROR OF OUR COEFFICIENT IS LARGE, THEN B1's affect might not be strong enough to say it's value is not zero
# TO CALCULATE HOW FAR AWAY AN ESTIMATE IS FROM 0 WE COMPUTE A T-STATISTIC (MEASURES STANDARD DEVIATIONS AWAY FROM 0)
# P-VALUE: compute a certain number under a t distribution held at X1 = 0 == HOW PROBABLE IS IT THAT THE DIFFERENT BETWEEN COEFFICIENT AND 0 HAPPENED BY CHANCE?
# WHEN P VALUE IS SMALL = VERY LOW CHANCE THAT THE DIFFERENCE HAPPENEDED BY CHANCE = CLEAR EVIDENCE THAT THE COEFFICIENT IS NOT ZERO!!!
# WHEN P VALUE IS SMALL = we can INFER THAT THERE IS SOME TYPE OF RELATIONSHIP BETWEEN Y AND X
# THIS IS REJECTING THE NULL HYPOTHESIS THAT THE COEFFICIENT VALUE IS = 0!!!!!!
# IN THE ADVERTISING DATA WE CAN DEDUCE FROM THE P VALUE THAT B0 and B1 DO NOT EQUAL ZERO = THERE IS A RELATIONSHIP BETWEEN ADVERTISING AND SALES!!!


## Assesing the Accuracy of the Model
# once we have rejected the null hypothesis that the coefficients are zero we want to determine how well our model fits the data
# linear regression is assesed with two related quantities: residual standard error (RSE) and R^2 statistic

## Residual Standard Error
# recall from our model that associated with each observation is and error term E
# due to the presence of error terms - even if we knew the true regression line we would not be able to perfectly predict Y from X
# The RSE is an estimate of the standard deviation of our error term E
# it is the average amount that the response will deviate from the true regression line
# RSE uses the Resiudal Sum of Squared Error in its calculation
# in our advertising example - the RSE is 3.26 - what does this mean?
# actual sales in each market deviate from the true regression line by approximately 3,260 units on average
# even if the model were correct and the true values of our unknown coefficients (B0 and B1) were known exactly, any prediction of sales on will still be off 3260 units
# is this acceptable? it depends on the problem in context!!
# the actual percentage error is the RSE divided by the mean of sales per day = 3260 / 14000 in our dataset = 23% percentage error
# THE RSE IS CONSIDERED A MEASURE OF LACK OF FIT OF THE MODEL
# WE WANT RSE TO BE AS SMALL AS POSSIBLE! THIS MEANS OUR PREDICITONS WILL BE CLOSE TO THE TRUE REGRESSION LINE!!

## R Squared Statistic
# The RSE provides an absolute measure of lack of fit of the model to our data
# since it is measured in units of Y it is not always clear  what constitutes a good RSE
# r squared provided an alternate measure of fit 
# it takes the form of a proportion = THE PROPORTION OF VARIANCE EXPLAINED!
# simply R squared is the Residual Sum of Squares divided by the Total Sum of squares subtracted from 1
# total sum of squares measures the total variance in response Y and can be thought of the amount of variability inheret to the response before the model is fit
# Redisual Sum of Squares measures the amount of variability that is LEFT UNEXPLAINED AFTER FITTING THE MODEL!!!
# when we subtract these two - we get the amount of VARIABILITY THAT IS EXPLAINED BY THE REGRESSION MODEL!
# R squared measures the proportion of variability in Y than can be explained by our X variables
# R squared close to one means we have explained almost all of our variation by fitting our model
# in our example R^2 was .61 = just under two-thirds of the variation was explained by our model!
# r squared is always between 0 and 1 and offers better parsamonious intrepretation vs. RSE
# but it is still difficult to determine what a good r squared is...and will depend on the application
# some fields linear models are extreme approximations of the true data - a fit in the mid range may be very good performance
# in single variable regression - CORRELATION BETWEEN X AND Y CAN BE USED TO DETERMINE MODEL FIT - IT IS A MEASURE OF THE LINEAR RELATIONSHIP BETWEEN X AND Y - THIS IS THE SAME AS R2
# IN A SIMPLE REGRESSION SETTING - R^2 == r^2 == however this does not generalize to multiple regression
# R^2 allows use to see the fit of a model in multiple linear regression!!


## Multiple Linear Regression:
# we often have more than just one predictor that we want to include in our model
# one option is to run three separate regression models on each of the variables we want to model
# how do we make a single prediction across all three models?
# our seperate models ignore the effects of the other variables - we only have one coefficient per model
# if variables are correlated with each other - we can get bad estimates on the individual effects of the variables
# instead of fitting a simple linear regression on each individual variable - we should extend our model to multiple variables
# we do this by giving each predictor a seperate slope coefficient in our traditional linear regression model
# model is now: Y = B0 + B1X1 + ... + BnXn + e
# slope interpretation = the average effect on Y for a one unit increase in Xj HOLDING ALL OTHER PREDICTORS CONSTANT
# advertising example = sales = B0 + B1(TV) + B2(RADIO) + B3(NEWSPAPER) + e

## Estimating the Regression Coefficients
# in the same case as simple linear regression - our true model parameters are unknown and we need to estimate them!
# given we can find these estimates - prediction is: yhat = B0hat + B1hat(x1) + ... + Bnhat(xn) + e
# our parameters are estimated using the same least squares approach = WE CHOOSE our parameters to MINIMIZE THE SUM OF SQUARED RESIDUALS!
# THE EXACT VALUES OF OUR PARAMETERS THAN MINIMIZE THE SUM OF SQUARED RESIDUALS ARE THE SLOPE COEFFICIENTS!!
# multiple linear regression gives a least squared estiamte coefficient for each predictor holding the other variables in the model constant
# example: the newspaper regression coefficient estimate in the multiple linear regression now very close to 0 and no longer statistically significant
# simple and multiple linear regressio coefficients can be very different!
# this difference is because simple linear regression ignores other predictors such as TV and RADIO in its model
# in a multiple linear regression - the newspaper coefficient represents the average effect while keeping TV and RADIO fixed!!!
# does it make sense for simple regression to have newspaper "pop" but the multiple regression not?
# YES IT DOES - we need to check the correlation with newspaper and other variables
# in a simple linear regression only estimates sales vs. newspaper = newspaper gets the "credit" for higher sales
# in reality - newspaper and radio are closely linked and radio has more effect holding newspaper constant
# newspaper may be misattributing some of the radio effect on sales due to thier correlation
# CLASSIC REGRESSION ON SHARK ATTACKS VS ICE CREAM SALES!!!
# THERE IS A POSITIVE RELATIONSHIP BETWEEN ATTACKS AND SALES! WE ARE NOT INCLUDING THE WEATHER VARIABLE
# WHEN WE ADD WEATHER TO THE VARIABLE - ICE CREAM SALES ARE NO LONGER A SIGNIFICANT PREDICTOR - AFTER ADJUSTING FOR TEMPERATURE


## Important Questions:
# when we perform multiple linear regression we are concerned with some main questions:
        # is at least one of my predictors useful in predicting the response?
        # do all the predictors help explain Y or is only a subset of predictors useful?
        # How well does my model fit the data?
        # given a set of predictor values (xs) what repsonse value should we predict? How accurate is our prediction?


## Question 1: Is there a relationship between response and predictors?
# in simple linear regression we can test if our slope coefficient is = 0
# in multiple linear regression we first need to ask if all coeffecients togehter equal 0
# Null Hypothesis: H0 = B1 = B2 =...=BN == 0
# Alternative Hypothesis: Ha = at least one of the Bj is not zero!!!
# the hypothesis test is performed by computing the F statistic
# when there is no relationship between the repsonse and predictors = F value is close to 1
# if F statistic is way large than one - there is compelling evidence that there is a relationship between the repsonse and at least one predictor
# F statistics can be converted to p values based on n and p
# for any given n and p we can compute the p value using the F distribution
# p value lets us know if we can accept or reject the null hypothesis
# IF WE REJECT THEN AT LEAST ONE VARIABLE IS RELATED TO OUR RESPONSE!!
# indiviudal p values are also given in the output of the regression model - they are exactly equivalant to the F distribution
# we can use these to determine the relative effect of each variable when including it in the model!
# why do we need the overall F statistic if we have individual p values for each coefficient?
        # large numbers of predictors cause problems - some coefficients will be significant by chance
        # we expect to see five small p values even is the absence of any true relationship with Y
        # if we use individual t tests and p values there is a high chance we could incorrectly conclude some variables do have an association with Y
        # THE F STATISTIC ADJUSTS FOR THE NUMBER OF PREDICTORS TO COMBAT THIS PROBLEM
        # if H0 is true there is only a 5% chance that the F statistic will result in a pvalue below .05, regrardless of the number of predictors
# WHEN PREDICTORS IS GREATER THAN THE NUMBER OF OBSERVATIONS WE CANNOT USE LINEAR REGRESSION!!!!
        # cannot use F statistic
        # cannot compute least sum of squared residuals
        

## Question 2: Deciding on Important Variables
# first step is to check the overall F statistic = at least one variable is related to our response
# how do we know which variable is significant to include in the model?
# be careful of high dimensionality - the more predictors we have we are bound to get some variables as significant 
# it is most often the case that the repsonse is only related to a subset of the predictors
# the process of figuring out which variables are significant is called VARIABLE OR FEATURE SELECTION
# try out many different models each containing a different subset of predictors
# fit all iterations of these models and then choose the best on the basis of an statistical metric
        # Mallows Cp, Akaike Information Criteria (AIC), Baysian Information Crieria (BIC), adjusted R^2
# when we have high dimensionality we cannot test out all possible models and compare against the metric - too many models!!!
# How do we deal with this?
# Forward Selection - begin with a null model (intercept and no predictors) fit
        # iteratively fit models by adding the variables and add to the null model the variable than gives the lowest Residual Sum Squares
        # continue to test and add variables to our "main" model until some stopping criteria is met
# Backwards Selection - start with all variables in the model
        # iteratively remove the variable with the highest P VALUE - this is the variable that is the least significant
        # fit a new model on the p -1 set of predictors
        # continue until a stopping rule has been reached
# Mixed Selection - combination of backwards and forwards selection
        # start with null - add variables that provide the best fit
        # if at any point  the p value for one variable in the model increases over a certain significance we remove that variable
        # continue to perform the forward and backward steps until hitting a stopping rule
        # continue until all models in model have low p value - all models outside the model have large p value if added into the model
# backwards selection cannot be used is p > n 
# forward selection can always be used = forward selection is a greedy approach = might include variables early that later become redundant
# CHAPTER 6 will cover FEATURE AND VARIABLE SELECTION IN DEPTH!



## Question 3: How well does my model fit the data?
# two most common measures of model fit are RSE and R^2
# R^2 is the fraction of variance explained by the model
# these quantities are computed and interpretted in the same way as a simple linear regression
# R^2 in simple linear regression is the square of the correlation of the repsonse and single variable
# R^2 in multiple linear regression = the square of the correlation between the response and the fitted linear model = Cor(Y, Yhat)^2
# R^2 value close to 1 indicates the model explains a large portion of the variance in the repsonse variable
# example: the advertising example shows R^2 of .8972 using all three advertising media  to predict sales
# example: the model with only TV and radio has an R^2 of .89719
# there is only a small increase in variance explained by adding NEWSPAPER into the model
# but i thought that NEWPAPER was insignificant when added to the model from previous example?
# turns out - R^2 will always increase when more variables are added  = EVEN IF THOSE VARIABLES ARE WEAKLY CORRELATED WITH THE REPSONSE
# this happens because adding another variable to the least squares equations must allow us to fit the training data more accurately
# however - the amount of variance expained (R^2) that is added from adding NEWSPAPER to the model is minimal = even more evidence that we could drop this variable
# there is essentially no "real" improvement of model fit when adding NEWSPAPER to the model
# we calculate R^2 on the training data - including NEWSPAPER will likely lead to poor results on the testing data do to overfitting
# in constrast a model only regressing on TV only has an R^2 of .61
# adding radio to the model adds quite a fit more explanation of variance - a substantial improvement in R^2
# we will predict better with these two variables in our model rather than just with TV
# we could also examine the p values of the two variable model to furher deduce the effect radio has holding tv constant
# Two variable RSE = TV and RADIO = 1.686
# Three variable RSE = TV, RADIO, NEWSPAPER = 1.686
# One variable RSE = TV = 3.26
# this corroborates our previous conclusion that a model which uses TV and RADIO to predict sales is much more accurate than one that just uses TV
# further = there is no point in adding NEWSPAPER to the model - we actually get an increase in RSE
# wait, how is an increase in RSE possible if  possible if RSS must decrease with more variables added?
# RSE is calculated with an element of the number of predictors included in its formula: if the variance decrease is small relative to the inclusion of predictors (p)
# this is exactly the case - adding newspaper gives us three predictors but the reduction in variance is minimal from adding NEWSPAPER = our RSE actually increases when adding NEWSPAPER
# we must take care to plot the data = patterns in the data may appear that are not evident when producing numerical summaries
# in our example: we actually overestimate SALES when most of the spending was spent in exclusively in either TV or RADIO
# in our example: we actually underestimate SALES when our budget was split between TV and RADIO
# this pronounced non-linear pattern cannot be modeled accurately using linear regression = "form" for the data is not linear
# this suggests a synergy or and interaction effect between the different types of advertising media = we will cover interaction terms later in this chapter


## How do we predict using multiple linear regression?
# one we have fit our linear regression model it is straight forward to predict repsonse on new xs
# substitiute our new x values into our equation to produce predictions of the repsonse
# there are three sorts of uncertainty with this prediction
        # the coefficient estimates are estimates of the true model parameters = the least squares plane is only and estimate for the true population regression plane
        # the inaccuarcy of the coefficient estimates is related to the reducilble error - we can compute a confidence interval to estimate how close Y^hat will be to the true F(x)
        # in reality using a linear model for f(X) is almost always an approximation of reality - there is an additional source of reducible error called MODEL BIAS
        # even if we knew the true f(X) i.e. the accurate model parameters - the response value cannot be predicted perfectly - this is IRREDUCIBLE ERROR
        # how much will Y vary from Yhat? = we use prediction intervals to answer this question
        # prediction intervals are always wider than confidence intervals - they incorporate the error in the estimate for f(X) (reducible error) and the uncertainty of the noise (irreducible error)
# we use a confidence interval to quantify the uncertainty surrounding the average sales over a large number of cities
# the prediction interval is used to quantify the uncertainty surrounding sales FOR A PARTICULAR INDIVIDUAL CITY
# PREDICTION INTERVALS ARE ALWAYS WIDER THAN CONFIDENCE INTERVALS
# the PREDICTION INTERVAL ACTUALLY SWALLOWS UP THE CONFIDENCE INTERVALS i.e. 95% of intervals within the PREDICITION INVTERVAL RANGE will contain the true value of Y for the INDIVIDUAL CITY


## Other Considerations in the Regression Model:
# you can model on qualitative variables, qunatitative variables or a mix of the two
# this section will discuss modeling qualitative variables in the multiple linear regression settingß

## Predictors with Only Two Levels
# lets investigate the difference in credit card balance between males and females
# if a qualitative predictor or factor has only two levels - we incorporate it into our model as a dummy variable - 1 for one group, 0 for the other
# in this case = 1 is male, 0 is female
# here is the model with the two level factor: y = B0 + B1X1 + e
        # if x is male = our model is y = BO + B1X1 + e = x is one so we keep the B1 slope estimate
        # if x if female = the x term goes to zero = y = B0 + B1(x1 = 0) + e ===== y = B0 + e ==== the estimate for female is JUST THE INTERCEPT!!
# INTERPREATION OF THE MODEL combines both cases of our dummy variable = here is how we would interpret the normal "intercept" value of each group
# the female average estimate will be the intercept plus the male and female slope coefficient
# the male average estimate will be only the intercept 
# the slope B1 is NOW THE AVERAGE ESTIMATE BETWEEN MALES AND FEMALES!!!! our slope "combines" the effect of both male and female
# to look at the singular affect holding others constant = female = B0 (male average estimate) + B1 (male and female slope coefficient); male = B0 (male avaerge estimate)
# the p value in this example is very high for THE COMBINED SLOPE COEFFICIENT = this means that the difference between male and female is not significant
# we can flip the dummy variable coding - this does not change the regression fit but does "flip" the interpreation of the model
# the final predictions for the estimates of males and females will be indentical regardless of our coding scheme = we need to just figure the right way to interpret each group relative to the slope and intercept terms provided


## Qualitiative Predictors with More than Two Levels
# we cannot use a single dummy variable to represent all possible values in a factor with more than two levels
# however, we can create multiple dummy variables for each level in the factor
# example: ethincity
        # asian dummy: 1 if Asian, 0 if not
        # white dummy: 1 if white, 0 if not
# both of these dummy variables can be included in our regression fit
# y = B0 + B1x1 + B2x2 + e
# interpretation:
        # B0 + B1 + e if asian
        # B0 + B2 + e if white
        # B0 + ei if not asian or white
# B0 is the average estimate for other, B1 is the difference in estimate from Asian to other, B2 is the difference in estimate from white to other
# we will always have one fewer dummy variable than the number of levels in the factor variable
# the level with no dummy variable - OTHER - is known as the baseline
# check the pvalues for the slope coefficeint - is not significant - there is evidence that there is no effect between the asian and other and the white and other ethiniciites
# again - all these levels can be rearranges based on dummy variable selection - we can change the "baseline" and all results will be the same - but how we get there is a different interpretation
# dummy variable order will affect the p value of the coefficents based on which level they are set at - USE THE F TEST TO TEST OVERALL IF ETHNICITY HAS AND EFFECT !!!
# F TEST WILL NOT CARE HOW OUR DUMMY VARIABLES ARE CODED!
# an F Test of .96 = there is evidence that there is no difference between ethnicity and our predictions!! we cannot reject the null that there is no difference!
# we can mix quantittive and qualitative predictors using this dummy variable approach
# we include the dummy variables and the quantitative variable together in the same model: sales = intercept + ethinicity dummy1 + ethniicity dummy2 + quantitative INCOME + ethe d
# we can revels the factors to give different contrasts of our variable levels - answers will not change but the strict interpretation relative to the factor level scaling will




## Extensions of the Linear Model
# the standard linear regression model provides interpretable results and works quite well on many problems
# however the linear model takes serveral highly restricive assumptions that are often violated in practice
        # 1. the form of data between the repsonse and predictor must be additive and linear!!
                # this means that for a change in x on the response y is independent of all other variables!
                # features must not be correlated with each other!!
# the additive nature of the linear regression model is restrictive - and ASSUMES A ONE UNIT CHANGE IN X RESULTS IN A CORRESPONDING CHANGE IN Y THAT IS CONSTANT FOR ALL X!!
# there are ways that we can releax these assumptions - discussed below

## Removing the Additive Assumption
# in previous analysis of the advertising dataset we concluded that TV and radio are assoicaited with sales
# the linear models that confirmed this assumed that the effect of sales increasing vs. all other mediums held constant
# sales were independent of the spend on any other type of media
# our simple model may be incorrect
# spending on radio may actually increase the effecitivness of TV 
# the slope term of TV should increase as the spend on radio increases
# in this case - spending half on radio and half on Tv will actually result in more sales than 100% of budget allocated to each individually
# this is known as a synergy effect - or interaction effect
# notice that when either tv and radio spend is low - our model overestimates sales
# when advertising is split between tv and radio - our model will underestimate sales
# this gives us a clue that our model has some synergy effect that we are not capturing in our simple linear regression
# consider the model: Y = B0 + B1X1 + B2X2 + e
        # according to this model - a one unit increase in X1, Y will increase by an average of B1 units
        # the presence of X2 does not alter this statement, regardless of x2, increase in x1 will have the same effect on Y
        # this goes against our needs if we want to model X1 effect based on the what x2 is also
# we can extend our model by adding an interaction term as a third predictor
# new model: Y = B0 + B1x1 + B2x2 + B3(x1*x2) + e
# this new model will "relax" the additive assumption of our linear regression!!
# we can re-write this model as: Y = B0 + (B1 + B3x2)*x1 + B2x2 + e
# we now have a B1 slope estimate that changes with x2!!
# the effect of x1 on Y is no longer constant - it will depend on the value for x2!!

# example:
# lets study the productivity of a factory
# we want to predict the number of units produced based on number of production lines and workers
# it seems likely that the effect that the effect of increasing the number of lines will depend on the number of workers
# this relationship suggests we should include and interaction term in our model between lines and workers
# lets fit our interaction model:
        # units = 1.2 + 3.4*lines + .22*workers + 1.4(lines * workers)
        # units = 1.2 + (3.4 + 1.4 * workers) * lines + .22 * workers
# adding an additional line will increase the number of units produced by 3.4 + 1.4*workers
# the more workers we have the stronger the effect of lines with be on units!!

# advertising example:
# lets fit an interaction model on the advertising dataset
        # sales = B0 + B1*TV + B2*radio + B3 *(radio*TV) + e
        # sales = B0 + (B1 + B3*radio)*TV + B2*radio + e
# we can interpret B3 as the increase in the effectiveness of TV advertising for a one unit increase in radio
# the result strongly suggest that the interaction model is superior to the linear model with only the main effects!!
# the p value for the interaction term (TV*Radio) is extrememly low - meaning B3 is not 0!!!
# IT IS CLEAR THAT THE TRUE RELATIONSHIP IS NOT ADDITIVE!!
# the R2 (portion of variance explained) is 96.8% compared to only 89.7% in our main effect simple regression!
# 69% of the variability remains in sales excluding the interaction term! (96.8 - 89.7) / (100 - 89.7) = 69%
# an increase is TV advertising of $1000 is associaited with increased sales of (B1 +B3*RADIO) *1000 units
# increase is TV is worth 1000*(19+1.1*RADIO) sales!!!!
# we can flip this around for radio !!
# increase in RADIO will be associated with and increase of sales of (B2 + B3)*TV *1,000 = 29 + 1.1 * TV units
# in this exmaple TV, RADIO and INTERACTION are all significant - all three variables should be included in the model!

# it can be the case that the interaction term has small p value but the main effects do not!
# THE HEIRACHIAL PRINCIPAL = if we include an interaction we need to include the main effects
# rationale: if X1 * X2 is related to the repsonse we do not care if the effect or X1 or X2 is zero!!

# interaction can also be used with qualitiative variables
# interactions terms can also be used with mixed models - quantitative and qualitaitve variables
# consider the credit data set:
        # predict balance using income (quantitative) and student (qualitative) as variables
# model without interaction term becomes:
        # balance = B0 + B1*income + B2(if student = yes) + 0 (if student = no)
        # balance = B1*income + (B0+B2[if student = yes]) + B0[if student = no]
# notice this amounts to fitting two parallel lines to our data, one for students one for non-students
# the lines for students and non-students have different intercepts: B0 + B2 vs. just B0
# BUT THE LINES HAVE THE SAME SLOPES B1
# the fact that the lines are parallel means that the average effect on balance on INCOME does not matter what value of student is
# this is a huge limitation as we know balance may be extrememly different for a student or non-student

# this limitation can be addressed by adding an interaction term
# we create this interaction term by multiplying income with the dummy variable for student!!
# our model now becomes:
        # balance = B0 + B1*INCOME + [(B2 + B3*INCOME) if student = yes] + 0[if student = no]
        # (B0 + B2) + (B1 + B3) * INCOME IF STUDENT!!!!
        # B0 + B1 * INCOME IF NOT A STUDENT!!!!!
# once again we have two different regression lines for students and non-students
# now our lines have different intercepts and different slopes!!!
        # intercepts = B0 + B2 vs. B0
        # slopes = B1 + B3 vs. B1
# this allows for the possibility that changes in income may affect the credit card balance of students vs non-students differently
# we note that the non-student line is higher than the slope for students
# the suggests that increases in income are associated with smaller increases in balance in students vs non-students



## Non-Linear Relationships
# as discussed - linear regression model assumes the underlying form of the data is linear between response and predictors
# in some cases - the true relationship between repsonse and predictors is not LINEAR!
# we can extend the linear model to accomodate non-linear relationships - polynomical regression
# non-linear relationships are usually curved and a "straight line" fit will not pick up the relationship of the "tails"
# to try to fit some of these tails - we need to complete some transformations to our linear model by adding transformed predictors
# for example a quadratic relationship between repsonse and predictors
        # model example: mpg = B0 + B1*HORSEPOWER + B2*HORSEPOWER^2 + e
        # this may better fit our "quadratic looking model"
        # this is still a linear model !!
        # we can still interpret and predict using the same ideas from our simple linear model
# in this case a quadratic fit performs muc better than a simple "straight line" linear fit
# R2 for linear model is only .608, our quadratic linear model achieves .688!!
# the pvalue for the quadratic term is highly significant

# if including HORSEPOWER^2 led to such a big improvement - why nont HORSEPOWER ^3 or more polynomicals?
# THE IDEA IS WE WILL TAKE IN TOO MUCH CURVATURE and NOT MODEL THE DATA CORRECTLY
# we get unnessesary wiggle when adding higher order polynomials
# it is unclear if adding these terms actually helps us fit the data more!
# this approach of extending linear regression by adding transformed predictors (polynomial) is polynomical regression

## Potential Problems
# when we fit a linear regression model to a particular dataset many problems can occur:
        # non-linear "form" of the underlying repsonse-predictor relationships
        # correlation of error terms (errors terms need to be normally distributed and independent)
        # non- constant variance of our error terms 
        # outliers
        # high-leverage points
        # collinearity between a set of our predictors
# identifying and overcoming these problems is as much as art as a science...

## Non-Linear Data
# the linear regression model assumes a straight line relationship between response and predictor
# if the true relationship is far from linear - almost all the conclusions we draw are suspect - mostly likely wrong
# the prediction accuracy will be severely reduced
# Resiudal Plots are a useful graphical tool for indentifying non-linearality
# given a simple linear regression we can plot the residuals ei for each predictor xi
# in multiple linear regression we can plot the reisudals versus the PREDICTED VALUES
# IDEALLY THE RESIUDAL PLOT WILL SHOW NO DISCERNABLE PATTERN
# PRESENCE OF A PATTERN MAY INDICATE A PROBLEM WITH AN ASPECT OF OUR MODEL (possible that the form of the data is non-linear)
# residual plots with "u shapes" or any shape indicate our data is non-linear
# if residuals are evenly dispersed for all predictors and no clear pattern is seen - we may have a good fit of the data
# if the residual plot indicates that there are non-linear associations in the data
        # use the non-linear transformations of the predictors:
        # log(x), sqrt(x), x^2 in our regression model
        # to be clear we add these new predictors into the same model - our model now has a transformed term

## Correlation of Error Terms
# an important assumption of linear regression is that the error terms are uncorrelated
# what does this mean?
# if errors are uncorrelated the fact that a single ei is postive does not matter to the next ei sign
# the standard errors that are computed for the estimated regression coefficients are BASED ON THE ASSUMPTION OF UNCORRELATED ERRORS
# if there were a correlation between error terms - the estimate error terms will tend to underestimate the TRUE STANDARD ERRORS
# THIS MAKES OUR CONFIDENCE AND PREDICTION INTERVALS NARROWER THAN THEY SHOULD BE - BAD MODEL
# for example: a 95% confidence interval may in reality have a much lower % of containing the true value!!
# in addition pvalues associated with the model will be lower than they should be - giving us significance where there is actually none
# IF THE ERROR TERMS ARE CORRELATED WE MAY HAVE TOO MUCH CONFIDENCE IN OUR MODEL

# example: suppose we doubled our data - bservations and errors terms identical in pairs
# if we ignored this our standard error calculations would be calculated if we had 2n the sample size
# our confidence intervals would be narrower by a factor of sqrt(2) - we add in lots of correlated error - each sample with have another entry with the same error

# why might correlations amoung the error terms occur?
# correlation of error terms frequently occur in the context of time series forecasting
# time series = observations for which measurements are obtained at discrete points in time
# in many cases - observations that are obtained at adjacent time points will have positively correlated errors
# HOW DO WE DETECT CORRELATION OF ERRORS IN TIME SERIES DATA?
        # PLOT OUR RESIUDALS ACROSS TIME!!
# if errors are uncorrelated THERE SHOULD BE NO PATTERN 
# if errors are correlated then we may see TRACKING in the residuals - adjacent residuals may have similiar values

# correlation of the error terms can occur outside of time series data also
# consider a study where height is predicted by weight
# our errors will be correlated with any of our sample we of the same family, had the same diet etc.
# the assumption of of uncorrelated errors is extremely important for linear regression and other statisical models
# GOOD EXPERIMENTAL DESIGN IS HOW WE MITIGATE THE CORRELATED ERRORS

## Non-Constant Variance of Error Terms
# another important assumption of the linear regression model is that the error terms have a constant variance
# Variance(ei) = sigma^2
# standard errors, confidence intervals, hypothesis tests all depend on this assumption
# it is often the case that the variances of the error terms are non-constant
# ERROR MAY INCREASE AS THE VALUE OF THE RESPONSE INCREASES
# NON-CONSTANT VARIANCE IS KNOWN AS HETEROSCEDASTICITY
# we can detect this if we see a FUNNEL SHAPE in the residual plot!!
# a funnel shape will show the magnitude of the residuals tend to increase with the fitted values
# what do we do when we see a funnel shape?
        # one possible transformation is to transform THE RESPONSE Y using a concave function
        # we can use log Y or sqrt Y
        # this transformation results in a greater amount of shrinkage of the larger responses - REDUCES HETEROSKEDASTICITY
# after we transform Y we should see a constant variance!!
# sometimes we have a good idea of the variance of each response
# for example the ith repsonse could be an average of ni raw observations...
# if each of these observations is uncorrelated with variance sigma^2 then thier average variance is is simga^2 / n
# in this case a simple remedy is to fit our model by weighted least squares with weights proportional to the inverse of variances
# most linear regression software allows us to weight the observations using weighted least squares regression

## Outliers
# outliers are values where the RESPONSE Y is a EXTREMELY DIFFERENT VALUE
# an outlier is a point for which yi is far from the value predicted by the model
# outliers can be systematic within the data or the result of data entry error
# outliers can be present but not affect the actual fit of our regression model - but they may change the diagnostics
# it is actually typical for an outlier to have minimial effect on our least squares fit
# outliers can still cause other problems...
# example: with outlier we have Residual Squared Error of 1.07 - without we have RSE of .77
# becuase RSE is used to compute all confidence intervals and pvalues - an outlier can significantly alter our interpretation of our model!!
# in our example inclusion of outlier causes R^2 to decline from .892 to .805
# residual plots can be again used to identify outliers
        # we can clearly see data points "outside" the data and the model fit in the residual plots
        # we can plot the scaled or studentized residuals to help determine how large a residual needs to be to be classified as an outlier
        # observations with student residuals above +/- 3 are possible outliers
        # in our case we see the outlier point at a scale of 6 in the studentized residuals
# how do we handle data points that we think are residuals?
        # if we can determine the outlier is the result of entry error we can remove the observation
        # be careful outliers can exist in the model and may be a part of the actual data
        # if we throw outliers out we may be losing that information we need to model the data
        # outliers can actually show us that our model may not be picking up on something - we might need another predictor

## High Leverage Points
# Leverage points are values where the PREDICTOR X is an EXTREMELY DIFFERENT VALUE
# our predictor value can be far and away different from the rest of the x values
# removing our high leverage points can have substansive effect our our least squares fit vs. removing the outliers
# high leverage observations tend to have a sizable impact on the estimated regression line
# it is cause of concern if only a few data points affect the fit of our line greatly
# any problems with these high leverage points may invalidate our model fit
# to indetifiy high leverage points we need to look for x values that are outside the normal range of our normal xs
# multiple regression can complicate this - an x value may be outside the scale of normal "xs" but still a valid point in our model
# it is hard to visually determine high leverage points in multiple linear regression
# however we can calculate the leverage using the leverage statistic
        # a large value means the data point has high leverage
        # h = 1/n + (xi - x)^2 / sumof(xi - x)^2
        # in this equation h increases as the distance increases from xi - Xbar
        # average leverage of all data points: (p + 1) / n
        # if a given observation has high value than the average leverage of all data points = HIGH LEVERAGE POINT
# high leverage points can be outliers too! the are outside bounds of repsonse and predictors
# our previous outlier is within the bound of x - so it is not a high leverage point!! - low leverage means low effect on fit!!


## Collinearity
# collinearity refers to the situation in which two or more predictors are closely related to one another
# credit card example: limit and rating are very highly correlated with each other! COLLINEAR!!
# collinear variables are difficult to "split apart" and see the true effect on one of the predictors
# collinear variables will "move with eachother" increase and decrease together
# collinear variables allow for small changes in data to greatly effect our estimates - resulting in model error
# these related variables cause us to be unsure about the true value of our estimate - and increase model error
# collinearity also makes it difficult to accurately reject coefficenits as 0!!
        # if our Bj estimate is not very certain > effects our t test hypothesis > we can inaccurately conclude that a variable has important when it does not!

# to avoid situations where one variable's affect is masked by collinearity we need to identify these cases when making our model
# a simply way to address colinearity is to look at the correlation matrix or pairs plot of our data
# a large value for a correlation coefficient means two variables are closely related to each other
# however it is possible for collinearity to exist between three or more variables in our model!
        # the correlation matrix will have to be scanned for all possible combinations of variables and "counted"
        # this is called multicollinearity!!
        # a better way to inspect for this is using the Variance Inflation Factor (VIF)
# VIF = the ratio of variance of Bj when fitting the full model divided by Bj fit with just that variable in question
        # the smallest possible value for VIF is one - the complete absencse of colinearity
        # typically there is always a small amount of collinearity in our data
        # typically a VIF of 5 to 10 indicates a problem-some amount of collinearity
# we can compute the VIF for each variable in our model
        # VIF(Bj) = 1 / (1 - R^2(xj given x - j))
        # in this equation R^2 is from the regression of Xj onto all of the other predictors
        # if R^2 is 1 then collinearity is present and VIF will be large!!!
# example: credit data
        # VIF values: age = 1.01, rating = 160.67, limit = 160.59
        # there is a lot of collinearlit in our data!
# when faced with collinearity in our data there are two simple solutions...
# drop the problematic variables from our model
        # this won't effect fit too much becuase we know the informatoin provided by the variable onto the repsonse is REDUNDANT!!
        # in this case we can solve our collinearity problem without compromising the fit!!
# combined the collinear variables into one single predictor variable
        # in our example: take the average of standarized versions of limit and rating in order to create a new variable
        # this new variable will measure CREDIT WORTHINESS!!



## THE MARKETING PLAN
# lets review the chapter using the Advertising data example

## is there a relationship between advertising sales and budget?
# this question can be answered by fitting a multiple linear regression model
# we model: SALES = BO + B1*TV + B2*RADIO + B3*NEWSPAPER + e
# we can test each the overall coefficient models with the F-statisitc: we want to test if any of the coefficient values are 0
# with a pvalue adjusted to the F-statistic test: we have clear evidence between advertising and sales


## how strong is the relationship?
# we discussed measures of model accuracy
# RSE estimates the standard deviation of the repsonse from the population regression line
# for the advertising data = RSE = 1681 and the mean value of the repsonse is 14022
# this indicates a percentage error of 12%
# the R^2 statistic records the percentage of variability in the response that is explained by the predictors
# the predictors explain almost 90% of the variance in sales


## Which media contribute to sales?
# to address this question we can examine each predictors pvalues associated with thier t statistic
# the pvalues for TV and RADIO are low but the pvalue for NEWSPAPER is not
# this suggests that only TV and RADIO are related to SALES


## How large is the effect of each medium on sales?
# the standard error of each coefficient can be used to compute a confidence interval for each Bj
# for the Advertising data:
        # TV: (-.043, .049)
        # RADIO: (.172, .206)
        # NEWSPAPER: (-.013, .011)
# the confidence intervals for TV and RADIO are narrow and relatively far from zero
# this is more evidence that these media contribute and are related to sales
# the interval for NEWSPAPER includes 0! this indicates the variable is not statistically significant given TV and RADIO

# we saw that predictor collinearity can result in very wide standard errors
# could collinearity be the reason that the confidence interval of NEWSPAPER is so wide?
# VIFs are 1.005, 1.145 and 1.145 (all close to 1) = very low evidence for collinearity

# in order to access the association of each medium individually on sales...
# we can perform three seperate linear regression models - one for each variable
# we find strong association between TV and SALES, and between RADIO and SALES
# there is evidence of mild association between NEWSPAPER and SALES when TV AND RADIO ARE EXCLUDED!


## How accurately can we predict future sales?
# the response can be predicted using our "full" model
# the accuracy of our prediction depends on if we want to estimate one point for the average response
# to estimate an individual response we will use a PREDICTION INVTERVAL
# PREDICTION INVTERVALS are always wider than the confidence intervals because they account for irreducible error


## Is the relationship linear?
# we examine residual plots to determine if the form of our relationship is linear
# if the relationships are linear - then the residual plots will display NO PATTERN
# in the advertising example we observe a non-linear effect in our data
# we can use x and y transformations in our model to better predict with a non-linear form data

## Is there synergy amoung the advertising media?
# a standard linear regression model assumes an additive relationship between the predictors and repsonse
# an additive model is easy to interpret - each effect of a predictor on the repsonse is unrealted to the other predictors
# the additive assumption may be unrealistic for certain datasets
# we can add in interaction terms into our model to accomodate non-additiive relationships
# a small pvalue we see with the interaction term indicates the presense of a non-additiive relationship
# the advertising data may not be additive
# including an interaction term in the model increases R^2 from 90% to 97% 


## Comparison of Linear Regression with K-Nearest Neighbors
# linear regression is an example of parametric approach
# is assumes a linear form of our function f(X)
# parametric methods have several advantages:
        # easy to fit
        # only need to estimate a small amount of coefficents
        # coefficients have simple intrepretations
        # tests of statistical significance can be easily performed
# but parametric methods have a disadvantage: by default they make strong assumptions about the form of f(X)
# if the specified functional form is far from the truth then our predictions will be poor
# trying to fit a "straight line" linear model through curved data relationship will result in poor fits
# in contrast - non-parametric methods do not explicitly assume a form of the underlying function between Y and X
# they are an alternative an offer a more flexible approach for performing regression "problems"

# the simplest and best known is K Nearest Neighbors Regression (KNN)
# KNN is closely related to the KNN classification method in chapter 2
        # given a value for K and a prediction point x0
        # KNN regression first identifies the K training observations that are closest to x0 - the training set N
        # KNN then estimates f(x0) using the average of all training responses in the training set N
        # f(x0) = 1 / N (sum of yi in the training set)
# the higher K neighbors developed for each set will give us a better fit to our data
# the optimal value of K will depend on the bias-variance tradeoff
# a small value for K provides the most flexible fit but has HIGH VARIANCE
# the variance is that our set of closest neighbors is just one value
# larger values of K provide a smoother and less variable fit - the prediction is an average of many neighbors
# however the smoothness may overfit and mask some of the structure of the true F(X) 
# to identify the the best number of K we need to estimate test error rates!

# in what setting will a parametric approach such as linear regression outperform a non-parametric method?
# the answer is simple:
        # the parametric will outperform the non-parametric if the data is close to the true form of the parametric
# if we have a true linear form - linear regression will tend to be the best model!
# when K is large KNN will tend to get close to the linear model

# in real life - the true relationship is rarely exactly linear
# linear regression may perform better for small values of K - but is the form is slightly not linear - K will converge and perform better for large values of K
# if the true relationship is strictly non-linear - KNN will outperform linear regression at every value of K
# we can compare these multiple model by using TEST SET MSE

# should KNN be favored if we do not know the relationship?
# for large K we get close to linear performance in linear form data, way better in non-linear...right?
# EVEN IF THE RELATIONSHIP IS NON-LINEAR KNN MAY STILL PERFORM WORSE
# if we have mutliple predictors - KNN may perform worse than linear regression

# say we have an non-linear data with added random noise...predictors that are not associated with a repsonse
# when p = 1 or 2, KNN outperforms linear regression
# but for p > 3 the results start to favor linear regression...
# why?
# the decrease in performance as the number of predictors increases for KNN is a common problem
# THIS IS BECAUSE OUR SET OF NEIGHBORS MIGHT NOT BE A TRUE REPRESENTATIVE OF ALL DIMENSIONS OF THE PREDICTORS
# spreading 100 observations over 20 predictors makes it hard to establish representative sample sizes to predict based on a set of nearest neighbors
# this is noted as the curse of dimensionality 
# the K observations that are nearest to a given test observation x0 may be very far away from x0 in a p-dimensional space when p is large
# cannot predict when we have more predictors than observations!!
# we will get a very poor prediction of the true function and a poor KNN fit

# non-parametric will tend to outperform problems in which the dimension is small number of observations per predictor
# even in problems in which dimensionality is is small - we might perform linear regression to KNN based on interpretability
# if test MSE is only slight better than linear regression - we may prefer linear regression based on interpretation!!
# we will also get all the hypothesis tests with linear regression - we do not get this with KNN



## LAB: LINEAR REGRESSION IN R
library(MASS); library(ISLR)

## simple linear regression in R
# the MASS package contains the Boston dataset which records median house value for 506 neighborhoods
# we want to predict medv with all 13 predictors available
head(Boston,5)
# crim zn indus chas   nox    rm  age    dis rad tax ptratio  black lstat medv
# 1 0.00632 18  2.31    0 0.538 6.575 65.2 4.0900   1 296    15.3 396.90  4.98 24.0
# 2 0.02731  0  7.07    0 0.469 6.421 78.9 4.9671   2 242    17.8 396.90  9.14 21.6
# 3 0.02729  0  7.07    0 0.469 7.185 61.1 4.9671   2 242    17.8 392.83  4.03 34.7
# 4 0.03237  0  2.18    0 0.458 6.998 45.8 6.0622   3 222    18.7 394.63  2.94 33.4
# 5 0.06905  0  2.18    0 0.458 7.147 54.2 6.0622   3 222    18.7 396.90  5.33 36.2

# start with the lm function
# y is the repsonse, x is the predictor
# we formulate y ~ x
# specifiy the data to be used = the Boston dataset
lm.fit <- lm(medv ~ lstat, data = Boston)

# lm.fit gives us some basic information about our fit of the data
lm.fit
# Call:
#         lm(formula = medv ~ lstat, data = Boston)
# 
# Coefficients:
#         (Intercept)        lstat  
#               34.55        -0.95  

# for more detailed information use summary(lm.fit)
summary(lm.fit)
# Call:
#         lm(formula = medv ~ lstat, data = Boston)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -15.168  -3.990  -1.318   2.034  24.500 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 34.55384    0.56263   61.41   <2e-16 ***
#         lstat       -0.95005    0.03873  -24.53   <2e-16 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 6.216 on 504 degrees of freedom
# Multiple R-squared:  0.5441,	Adjusted R-squared:  0.5432 
# F-statistic: 601.6 on 1 and 504 DF,  p-value: < 2.2e-16



# look inside the lm object
# use names to see all available information we can use from the regression call
names(lm.fit)
# [1] "coefficients"  "residuals"     "effects"       "rank"          "fitted.values" "assign"       
# [7] "qr"            "df.residual"   "xlevels"       "call"          "terms"         "model"

# extracting the coefficients from our model
coef(lm.fit)
# (Intercept)       lstat 
# 34.5538409  -0.9500494 

# extracting more details from the coefficients of our model
coef(summary(lm.fit))
# Estimate Std. Error   t value      Pr(>|t|)
# (Intercept) 34.5538409 0.56262735  61.41515 3.743081e-236
# lstat       -0.9500494 0.03873342 -24.52790  5.081103e-88

# confidence interval for the coefficinets - where does the true value of our coefficient land?
confint(lm.fit)
#               2.5 %     97.5 %
# (Intercept) 33.448457 35.6592247
# lstat       -1.026148 -0.8739505



# predict on new values - using confidence interval
# the predict function can be used to produce confidence intervals and prediction intervals
# these will be on the prediction of medv given lstat
predict(lm.fit, data.frame(lstat = c(5, 10, 15)),
        interval = "confidence")
# fit      lwr      upr
# 1 29.80359 29.00741 30.59978
# 2 25.05335 24.47413 25.63256
# 3 20.30310 19.73159 20.87461


# predict on new values - using prediction interval
predict(lm.fit,data.frame(lstat = c(5, 10, 15)),
        interval = "prediction")
# fit       lwr      upr
# 1 29.80359 17.565675 42.04151
# 2 25.05335 12.827626 37.27907
# 3 20.30310  8.077742 32.52846



# the 95% confidence interval associated with a lstat value of 10 is (24.47, 25.63)
# the 95% prediction interval associated with a lstat value of 10 is (12.828, 37.28)
# as expected the confidence and prediction intervals are centered around the same point
# prediction intervals will always be wider than the confidence intervals

# lets plot the relationship to investigate
plot(Boston$medv, Boston$lstat)

# plot our fitted line through the data!!
abline(lm.fit, col = "red")

# there is some evidence of non-linearity in our relationship
# the abline can be used to to draw any line not just our fitted model
# we can draw and intercept line with abline(a,b) where a = some value, b = some value
# we can also add in different settings for plotting lines and points
# lwd gives line width, pch give point characters
plot(Boston$lstat, Boston$medv, pch = 21, col = "blue")
abline(lm.fit, lwd = 3, col = "red")
plot(1:20, 1:20, pch = 1:20)


# lets run some diagnostic plots of our model
# four diagnostic plots are automatically given in the plot command
# we apply the plot command to our lm function
# use the par function to put all four plots together into one window
par(mfrow = c(2,2))
plot(lm.fit)
par(mfrow= c(1,1))

# alternatively we can compute the resiudals from a linear regression fit using the residuals function
# the function rstudent will return the student residuals
# WE CAN PLOT THESE FUNCTIONS AGAINST THE FITTED VALUES!!!
# THIS HELPS US DETERMINE IF OUR MODEL MEETS THE ASSUMPTIONS OF LINEAR REGRESSION!!!
# based on these leverage charts...we see evidence that our data is not linear - we clearly see a pattern in residuals vs. fitted values
plot(predict(lm.fit), resid(lm.fit), col = "blue")
plot(predict(lm.fit), rstudent(lm.fit), col = "red")


# we can compute leverage statistics for any number of predictors using our hatvalues function
plot(hatvalues(lm.fit), col = "purple")

# which max calcualtes the max hatvalues of the dataset and returns the actual x value
# in this case the x with the highest hatvalue or leverage is 375
which.max(hatvalues(lm.fit))



## Multiple Linear Regression
# in order to fit a multiple linear regression model using least squares we use the lm() function
# y ~ x1 + x2 + x3 is the linear regression formula
# we use the same information functions to look "inside" the model
lm.fit2 <- lm(medv ~ lstat + age, data = Boston)
summary(lm.fit2)

# Call:
#         lm(formula = medv ~ lstat + age, data = Boston)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -15.981  -3.978  -1.283   1.968  23.158 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 33.22276    0.73085  45.458  < 2e-16 ***
#         lstat       -1.03207    0.04819 -21.416  < 2e-16 ***
#         age          0.03454    0.01223   2.826  0.00491 ** 
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 6.173 on 503 degrees of freedom
# Multiple R-squared:  0.5513,	Adjusted R-squared:  0.5495 
# F-statistic:   309 on 2 and 503 DF,  p-value: < 2.2e-16


# what if we want to throw all variables in a regression?
# we do not have to type them all just use .
lm.fit3 <- lm(medv ~. , data = Boston)
summary(lm.fit3) 
# Call:
#         lm(formula = medv ~ ., data = Boston)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -15.595  -2.730  -0.518   1.777  26.199 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  3.646e+01  5.103e+00   7.144 3.28e-12 ***
#         crim        -1.080e-01  3.286e-02  -3.287 0.001087 ** 
#         zn           4.642e-02  1.373e-02   3.382 0.000778 ***
#         indus        2.056e-02  6.150e-02   0.334 0.738288    
# chas         2.687e+00  8.616e-01   3.118 0.001925 ** 
#         nox         -1.777e+01  3.820e+00  -4.651 4.25e-06 ***
#         rm           3.810e+00  4.179e-01   9.116  < 2e-16 ***
#         age          6.922e-04  1.321e-02   0.052 0.958229    
# dis         -1.476e+00  1.995e-01  -7.398 6.01e-13 ***
#         rad          3.060e-01  6.635e-02   4.613 5.07e-06 ***
#         tax         -1.233e-02  3.760e-03  -3.280 0.001112 ** 
#         ptratio     -9.527e-01  1.308e-01  -7.283 1.31e-12 ***
#         black        9.312e-03  2.686e-03   3.467 0.000573 ***
#         lstat       -5.248e-01  5.072e-02 -10.347  < 2e-16 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 4.745 on 492 degrees of freedom
# Multiple R-squared:  0.7406,	Adjusted R-squared:  0.7338 
# F-statistic: 108.1 on 13 and 492 DF,  p-value: < 2.2e-16


# we can access the individual components of a summary object by name
# use names to see all available options

# R^2
summary(lm.fit3)$r.sq
# [1] 0.7406427

# RSE
summary(lm.fit3)$sigma
# [1] 4.745298

# we can use the vif() command function from the car package to calculate the variance inflation factor
# VIF gives us an idea of the collinearity of our predictors!!
# VIF close to 1 means no evidence of collinearity
library(car)
vif(lm.fit3)
# crim       zn    indus     chas      nox       rm      age      dis      rad      tax  ptratio    black 
# 1.792192 2.298758 3.991596 1.073995 4.393720 1.933744 3.100826 3.955945 7.484496 9.008554 1.799084 1.348521 
# lstat 
# 2.941491 


# removing variables from our regression
# maybe we no longer wish to include age in our model because of its high value
lm.fit4 <- lm(medv ~. -age, data = Boston)
summary(lm.fit4)
# Call:
#         lm(formula = medv ~ . - age, data = Boston)
# 
# Residuals:
#         Min       1Q   Median       3Q      Max 
# -15.6054  -2.7313  -0.5188   1.7601  26.2243 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  36.436927   5.080119   7.172 2.72e-12 ***
#         crim         -0.108006   0.032832  -3.290 0.001075 ** 
#         zn            0.046334   0.013613   3.404 0.000719 ***
#         indus         0.020562   0.061433   0.335 0.737989    
# chas          2.689026   0.859598   3.128 0.001863 ** 
#         nox         -17.713540   3.679308  -4.814 1.97e-06 ***
#         rm            3.814394   0.408480   9.338  < 2e-16 ***
#         dis          -1.478612   0.190611  -7.757 5.03e-14 ***
#         rad           0.305786   0.066089   4.627 4.75e-06 ***
#         tax          -0.012329   0.003755  -3.283 0.001099 ** 
#         ptratio      -0.952211   0.130294  -7.308 1.10e-12 ***
#         black         0.009321   0.002678   3.481 0.000544 ***
#         lstat        -0.523852   0.047625 -10.999  < 2e-16 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 4.74 on 493 degrees of freedom
# Multiple R-squared:  0.7406,	Adjusted R-squared:  0.7343 
# F-statistic: 117.3 on 12 and 493 DF,  p-value: < 2.2e-16

# we could also use the update() function if needed
lm.fit4 <- update(lm.fit4, ~.-age)
summary(lm.fit4)
# Call:
#         lm(formula = medv ~ crim + zn + indus + chas + nox + rm + dis + 
#                    rad + tax + ptratio + black + lstat, data = Boston)
# 
# Residuals:
#         Min       1Q   Median       3Q      Max 
# -15.6054  -2.7313  -0.5188   1.7601  26.2243 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  36.436927   5.080119   7.172 2.72e-12 ***
#         crim         -0.108006   0.032832  -3.290 0.001075 ** 
#         zn            0.046334   0.013613   3.404 0.000719 ***
#         indus         0.020562   0.061433   0.335 0.737989    
# chas          2.689026   0.859598   3.128 0.001863 ** 
#         nox         -17.713540   3.679308  -4.814 1.97e-06 ***
#         rm            3.814394   0.408480   9.338  < 2e-16 ***
#         dis          -1.478612   0.190611  -7.757 5.03e-14 ***
#         rad           0.305786   0.066089   4.627 4.75e-06 ***
#         tax          -0.012329   0.003755  -3.283 0.001099 ** 
#         ptratio      -0.952211   0.130294  -7.308 1.10e-12 ***
#         black         0.009321   0.002678   3.481 0.000544 ***
#         lstat        -0.523852   0.047625 -10.999  < 2e-16 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 4.74 on 493 degrees of freedom
# Multiple R-squared:  0.7406,	Adjusted R-squared:  0.7343 
# F-statistic: 117.3 on 12 and 493 DF,  p-value: < 2.2e-16



## Interaction Terms:
# it is easy to include interaction terms in a linear model using our trusty lm() function
# we use variable1:variable2to tell R to include an interaction term in our model
# the syntax variable1*variable2 includes variable1 + variable2 +variable1:variable2
lm.int <- lm(medv ~ lstat*age, data = Boston)
summary(lm.int)
# Call:
#         lm(formula = medv ~ lstat * age, data = Boston)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -15.806  -4.045  -1.333   2.085  27.552 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 36.0885359  1.4698355  24.553  < 2e-16 ***
#         lstat       -1.3921168  0.1674555  -8.313 8.78e-16 ***
#         age         -0.0007209  0.0198792  -0.036   0.9711    
# lstat:age    0.0041560  0.0018518   2.244   0.0252 *  
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 6.149 on 502 degrees of freedom
# Multiple R-squared:  0.5557,	Adjusted R-squared:  0.5531 
# F-statistic: 209.3 on 3 and 502 DF,  p-value: < 2.2e-16



## Non-Linear Transformations of the Predictors
# the lm function can also accommodate non-linear transformations of our predictors
# we use the I notation to modify x variables
# the near zero value of our pvalue tells us including this term most likely improves the model
lm.fit.trans <- lm(medv ~ lstat + I(lstat^2), data = Boston)
summary(lm.fit.trans)
# Call:
#         lm(formula = medv ~ lstat + I(lstat^2), data = Boston)
# 
# Residuals:
#         Min       1Q   Median       3Q      Max 
# -15.2834  -3.8313  -0.5295   2.3095  25.4148 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 42.862007   0.872084   49.15   <2e-16 ***
#         lstat       -2.332821   0.123803  -18.84   <2e-16 ***
#         I(lstat^2)   0.043547   0.003745   11.63   <2e-16 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 5.524 on 503 degrees of freedom
# Multiple R-squared:  0.6407,	Adjusted R-squared:  0.6393 
# F-statistic: 448.5 on 2 and 503 DF,  p-value: < 2.2e-16


# we can use anova to further investigate if the new transformed term is worth it in the model
# we quantitiy how much better including this term is versus the first model
# model one is our first model with just lstat
# model two is our model including the transformed lstat term
# the anova function performs a hypothesis test against these two models...does adding the term in the second model make it better?
# with high F statistic and low pvalue we are certain that including the transformed term improves our model
# a model containing lstat and lstat^2 is superior to the model containing just lstat
# THIS MATCHES OUR THINKING!! WE NOTICED EVIDENCE OF NON_LINEARITY IN THE SIMPLE MODEL BEFORE!
# THE QUADRATIC TERM HELPS US MODEL MORE OF THE CURVATURE OF OUR RELATIONSHIP BETWEEN MEDV AND LSTAT
lm.fit.nt <- lm(medv ~ lstat, data = Boston)
lm.fit.t <- lm(medv ~ lstat + I(lstat^2), data = Boston)
anova(lm.fit.nt, lm.fit.t)
# Analysis of Variance Table
# 
# Model 1: medv ~ lstat
# Model 2: medv ~ lstat + I(lstat^2)
# Res.Df   RSS Df Sum of Sq     F    Pr(>F)    
# 1    504 19472                                 
# 2    503 15347  1    4125.1 135.2 < 2.2e-16 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# residual plot to show that the pattern of residuals vs. fitted no longer exists
plot(lm.fit.t)[[1]]


# to create a cubic fit we need to include a transformed ^3 term into our model
# we can leverage the poly() function to create polynomial within our lm call
lm.fit5 <- lm(medv ~ poly(lstat,5), data = Boston)
summary(lm.fit5)
# Call:
#         lm(formula = medv ~ poly(lstat, 5), data = Boston)
# 
# Residuals:
#         Min       1Q   Median       3Q      Max 
# -13.5433  -3.1039  -0.7052   2.0844  27.1153 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)       22.5328     0.2318  97.197  < 2e-16 ***
#         poly(lstat, 5)1 -152.4595     5.2148 -29.236  < 2e-16 ***
#         poly(lstat, 5)2   64.2272     5.2148  12.316  < 2e-16 ***
#         poly(lstat, 5)3  -27.0511     5.2148  -5.187 3.10e-07 ***
#         poly(lstat, 5)4   25.4517     5.2148   4.881 1.42e-06 ***
#         poly(lstat, 5)5  -19.2524     5.2148  -3.692 0.000247 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 5.215 on 500 degrees of freedom
# Multiple R-squared:  0.6817,	Adjusted R-squared:  0.6785 
# F-statistic: 214.2 on 5 and 500 DF,  p-value: < 2.2e-16

# this suggests that including additional polynomial terms up to the fifth order leads to imporvmeents inour model!
# however, further investigation of the data reveals that no polynomial beyond the 5th order is significant

# we can also include other transformations in our lm call
# here is an example of a log transformation:
lm.fit.log <- lm(medv~log(rm), data = Boston)
summary(lm.fit.log)
# Call:
#         lm(formula = medv ~ log(rm), data = Boston)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -19.487  -2.875  -0.104   2.837  39.816 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  -76.488      5.028  -15.21   <2e-16 ***
#         log(rm)       54.055      2.739   19.73   <2e-16 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 6.915 on 504 degrees of freedom
# Multiple R-squared:  0.4358,	Adjusted R-squared:  0.4347 
# F-statistic: 389.3 on 1 and 504 DF,  p-value: < 2.2e-16


## Qualitative Predictors
# we will now examine the carseats data which is a part of the ISLR library
# we will attempt to predict sales in 400 locations based on a set of predictors
# this data includes qualitiative predictors that we can use to model sales in the lm function
# example: Shelvloc has three factor values: Bad, Medium, Good
data("Carseats")
names(Carseats)
# [1] "Sales"       "CompPrice"   "Income"      "Advertising" "Population"  "Price"       "ShelveLoc"  
# [8] "Age"         "Education"   "Urban"       "US" 

str(Carseats)
# 'data.frame':	400 obs. of  11 variables:
#         $ Sales      : num  9.5 11.22 10.06 7.4 4.15 ...
# $ CompPrice  : num  138 111 113 117 141 124 115 136 132 132 ...
# $ Income     : num  73 48 35 100 64 113 105 81 110 113 ...
# $ Advertising: num  11 16 10 4 3 13 0 15 0 0 ...
# $ Population : num  276 260 269 466 340 501 45 425 108 131 ...
# $ Price      : num  120 83 80 97 128 72 108 120 124 124 ...
# $ ShelveLoc  : Factor w/ 3 levels "Bad","Good","Medium": 1 2 3 3 1 1 3 2 3 3 ...
# $ Age        : num  42 65 59 55 38 78 71 67 76 76 ...
# $ Education  : num  17 10 12 14 13 16 15 10 10 17 ...
# $ Urban      : Factor w/ 2 levels "No","Yes": 2 2 2 2 2 1 2 2 1 1 ...
# $ US         : Factor w/ 2 levels "No","Yes": 2 2 2 2 1 2 1 2 1 2 ...


# for a qualitative value in a regression: R creates dummy variables automatically
# we can include factor variables with quantitative terms of all types
lm.fit.fact <- lm(Sales ~.+Income:Advertising +Price:Age,
                  data = Carseats)
summary(lm.fit.fact)
# Call:
#         lm(formula = Sales ~ . + Income:Advertising + Price:Age, data = Carseats)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -2.9208 -0.7503  0.0177  0.6754  3.3413 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
#         (Intercept)         6.5755654  1.0087470   6.519 2.22e-10 ***
#         CompPrice           0.0929371  0.0041183  22.567  < 2e-16 ***
#         Income              0.0108940  0.0026044   4.183 3.57e-05 ***
#         Advertising         0.0702462  0.0226091   3.107 0.002030 ** 
#         Population          0.0001592  0.0003679   0.433 0.665330    
#         Price              -0.1008064  0.0074399 -13.549  < 2e-16 ***
#         ShelveLocGood       4.8486762  0.1528378  31.724  < 2e-16 ***
#         ShelveLocMedium     1.9532620  0.1257682  15.531  < 2e-16 ***
#         Age                -0.0579466  0.0159506  -3.633 0.000318 ***
#         Education          -0.0208525  0.0196131  -1.063 0.288361    
#         UrbanYes            0.1401597  0.1124019   1.247 0.213171    
#         USYes              -0.1575571  0.1489234  -1.058 0.290729    
#         Income:Advertising  0.0007510  0.0002784   2.698 0.007290 ** 
#         Price:Age           0.0001068  0.0001333   0.801 0.423812    
# ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 1.011 on 386 degrees of freedom
# Multiple R-squared:  0.8761,	Adjusted R-squared:  0.8719 
# F-statistic:   210 on 13 and 386 DF,  p-value: < 2.2e-16


# the contrasts function returns the coding that R uses for the dummy variables
contrasts(Carseats$ShelveLoc)
# Good Medium
# Bad       0      0
# Good      1      0
# Medium    0      1


# R created a dummary variable that takes on a value of 1 if good, 0 otherwise = SHELVLOC GOOD
# it also created another variable that equals 1 if medium, 0 other wise = SHELVLOC MEDIUM
# a bad shelving is coded as 0 for both SHELVLOC GOOD AND SHELVLOC MEDIUM
# the SHELVLOCGOOD variable coefficient is positive indicates good shelving is good for sales
# SHELVLOCMEDIUM has a smaller positive coefficient indiicating that that a medium shelving leads to higher sales compared to bad but less than good


## Writing Functions
# R comes with lots of functions and many more are available in different packages
# what if we have an operation that is not available? WE CAN CREATE OUR OWN FUNCTIONS
# example: LOADLIBARIES() is a self created function that loads the libraries we need
LoadLibraries = function() {
        library(ISLR)
        library(MASS)
        print("your libraries have been loaded...bitch")
}

# now if we call our function LoadLibraries R will run our function
LoadLibraries()
# [1] "your libraries have been loaded...bitch"

# we can check the contents of ANY FUNCTION
LoadLibraries
# function() {
#         library(ISLR)
#         library(MASS)
#         print("your libraries have been loaded...bitch")
# }


## Practice Exercises: Chapter 3 Linear Regression

## Question 1: Describe the null hypothesis to which the pvalues are given in Table 3.4
# what conclusions do we draw from these values?
# phrase your answers in terms of SALES, TV, NEWSPAPER rather than just the coefficients

## Solution:
# the hypothesis test given testing the coefficient value is NOT zero
# if we see a significant pvalue we reject the null hypothesis that the coefficient is 0
# we accept the alternative hypothesis that the coefficient value is NOT EQUAL TO 0!!
# in this case we see TV and RADIO with significant pvalues meaning thier coefficient value is not 0 (holding other media constant)
# this means there is strong evidence that TV and RADIO have some relationship with the repsonse SALES
# NEWSPAPER has an insignificant pvalue - and we cannot conclude there is a relationship holding other media constant
# WE DID NOT HAVE ENOUGH EVIDENCE TO SAY THE RELATIONSHIP BETWEEN NEWSPAPER AND SALES IS NOT 0!!


## Question 2: KNN classification vs. KNN regression
# what are the differences between the KNN classification and the KNN regression?

## Solution:
# KNN works by creating a comparision neighbors "set" of obsverations to predict on a new point
# KNN will use a distance metric to determine the set of neighbors to include
# the analyst will determine the amount of observations to include in the set
# specifying 20 neighbors - the algorithm with find the 20 closest neighbors matching to the predictors in the new observation
# the algorithm will take in the new value - and calculate the distance between a set of other points - and use those points as the basis for prediction
# the main difference is basing KNN as regression or classification
        # classification: set of nearest neighbors will hold a majority vote to deicide predicted CLASSIFICATION
        # regression: set of nearest neighbors will be averaged together to deicide the prediction VALUE
# classification gives us what class the nearest neighbors think the new value is in
# regression gives us a predicted value based on the values of the nearest neighbors to the new value


## Question 3: suppose we have a dataset with five predictors with interaction terms
# X1 = GPA, X2 = IQ, X3 = GENDER (dummy), X4 = GPA*IQ, X5 = GPA*GENDER
# response is starting salary after graduation
# example model:  SALARY = GPA + IQ + GENDER + GPA*IQ + GPA*GENDER + error
# we fit a linear model and obtain the following coefficient values:
# final model = SALARY = 50 + 20(GPA) + .07(IQ) + 35(GENDER) + .01(GPA*IQ) -10(GPA*GENDER)

# which of the following are correct and why?
        # for a fixed value of IQ and GPA males earn more on average than females?
        # for a fixed value of IQ and GPA females earn more on average than males?
        # for a fixed value of IQ and GPA males earn more on average than females provided the GPA is high enough?
        # for a fixed value of IQ and GPA, females earn more on average than males provided the GPA is high enough?

# predict the salary of a female with IQ of 110 and GPA of 4.0

# TRUE or FALSE: since the coefficient for GPA*IQ interaction term is small - there is little evidence of interaction effect?

## Solution:
# we need to re-interpret the regression line for the dummy variable GENDER
# males == 0: Y = 50 + 20GPA + .07IQ + .01GPA*IQ
# females == 1: Y = 85 + 10GPA + .07IQ + .01GPA*IQ
# for a fixed value of IQ and GPA MALES EARN MORE ON AVERAGE THAN FEMALES PROVIDED THE GPA IS HIGH ENOUGH?
# this is because the terms 50+20*GPA for males is greater than 85+10*GPA for high enough GPAs
# if GPA is greater than equation to 3.5 males will earn more
# if less than 3.5 females will actaully earn more

# a prediction: SALARY = 50 + 20(4.0) + .07(110) + 35(1) + .01(4.0*110) -10(4.0*1) 
# 50 + 80 + 7.7 + 35 + 4.4 - 40 = 137.1 prediction salary for female with GPA 4.0 and IQ 110

# the coefficient value does not tell us that the interation term does not have an effect
# we would need to look into the pvalue of this term to determine if there is in fact a relationship between the interaction and SALARY
# we will look for the pvalue associated with the F statistic to determine if we should keep this relationship
# we could also fit several models one with and without the interaction term and use anova to determine the significance
# anova being significant with the interaction term will let us know that keeping it in the model is beneficial


## Question 4: we have a set of data (n = 100) containing a single predictor and one response
# we fit a linear model to the data and include a cubic predictor:
# example model: Y = B0 + B1X + B2X^2 + B3X^3 + e

# suppose that the true relationship between X and Y is linear:
# would we expect our plain linear model to have lower training RSS than the cubic regression? Why or why not?
# would we expect our plain linear model to have lower TEST RSS than the cubic regression? Why or why not?

# suppose we know that the true relationship is not linear...but not sure how far away from linear...
# would we expect training RSS to be lower for plain regression or cubic?
# would we expect TEST RSS to be lower for plain regression or cubic?


## Solution:

# if the true relationship is linear - we expect training RSS to be lower for the linear model
# but depending on our training set the cubic may model some of the noise and give a better training RSE - more flexible
# we expect the linear regression least squares to be closer to the true relationship and therefore lower RSE

# if the true relationship is linear - we expect testing RSS to be lower for the linear model
# cubic may benefit from extra variance explained in the training set but the test set should validate the true form
# with a true form of linear - the linear model will perform better on the testing RSS

# if the true relationship is not linear - the cubic regression training RSS will be lower than linear model
# the cubic model is more flexible and will decreasae the RSS in the training set (fits more naturally to curves instead of straight line)
# no matter what our underlying form is (AT LEAST NOT LINEAR) the more flexible model will drive down training RSS

# if the true relationship is not linear - the cubic regression testing RSS will be lower than linear model
# the cubic model is more flexible and will decreasae the RSS in the testing set (fits more naturally to curves instead of straight line)
# no matter what our underlying form is (AT LEAST NOT LINEAR) the more flexible model will drive down testing RSS


## Question 5: consider the fitted values that result from performing linear regression without an intercept
# the ith fitted value will take the form: yi = xiBhat
# where Bhat is sum(xi*yi) / sum(xi^2)
# we can write: yi = sum(ai * yi)
# what is ai?
# we interpret this result by saying that the fitted values from linear regression are linear combinations of the response variable

## Solution: ???



## Question 6: using 3.4 argue in the case of simple linear regression - the least squares line always passed through:
# (x mean, y mean)

## Solution: the least squares estimate must always pass through the mean of x and mean of y
# the mean is the exact point where the least squares estimate is minimized 
#- to fit a linear model we need to fit a line through this point!!
# if the goal is to fit a linear reducing the least squared error we must pass through the point where the RSE is minimized for X and Y!!



## Question 7: in the case of simple linear regression of Y on X:
# the R^2 statistic is equal to the square of the correlation between X and Y
# prove that this is the case (assuming x_ and y_ = 0)

## Solution:
# R^2 = 1 - RSS / TSS == 1 - sum(yi - yhat)^2 / sum(y^2)
# we can now substitute B1xi for yi because we known that yhati = B1hatxi
# completing this formula gives us:
# R^2 = sum(xi*yi)^2 / sum(xi^2) * sum(yi^2) = Cor(X,Y)^2












## Applied Practice Exercises: Chapter 3 Linear Regression

## Question 8: simple linear regression on the Auto data set
# use the lm function to perform model mpg ~ hp 
# use the summary function to see the results
        # is there a relationship between predictor and response?
        # how strong is the relationship between predictor and response?
        # is the relationship positive or negative?
        # what is the predicted mpg associated with a horsepower of 98? 
        # what are the the 95% confidence intervals and prediction intervals?

# plot the response and the predictor - plot the fitted line

# use the plot function to produce diagnostic plots of the least squares regression fit
# are there any problems?


## Solution:

# load data
data("Auto")
str(Auto)

# fit model
fit1 <- lm(mpg ~ horsepower, data = Auto)
summary(fit1)

# Call:
#         lm(formula = mpg ~ horsepower, data = Auto)
# 
# Residuals:
#         Min       1Q   Median       3Q      Max 
# -13.5710  -3.2592  -0.3435   2.7630  16.9240 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 39.935861   0.717499   55.66   <2e-16 ***
#         horsepower  -0.157845   0.006446  -24.49   <2e-16 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 4.906 on 390 degrees of freedom
# Multiple R-squared:  0.6059,	Adjusted R-squared:  0.6049 
# F-statistic: 599.7 on 1 and 390 DF,  p-value: < 2.2e-16

# there is a relationship between mpg and horsepower - the pvalue is significant - the relationship is NOT ZERO!!
# the R^2 statistic can give us an idea of how strong the relationship is between mpg and horsepower
        # with R^2 at 60% - 60% of the variation in mpg is explained by the variable horsepower
# the relationship between mpg and horsepower is negtive - increases in horsepower lead to decreases in MPG!!

# predict mpg for a car with 98 horsepower
# we predict based on our model that a car with 98 horsepower will get 23 mpg!!
pred1 <- predict(fit1, data.frame(horsepower = 98))
pred1
# 1 
# 23.41249

# what are the confidence and prediction intervals for this new prediction?
# the 95% confidence interval is (21.69, 25.13)
pred1 <-  predict(fit1, data.frame(horsepower = 98), interval = "confidence")
pred1
#       fit      lwr     upr
# 1 23.41249 21.69277 25.1322

# what are the confidence and prediction intervals for this new prediction?
# the 95% prediction interval is (15.33, 31.48)
# prediction intervals will always be wider than confidence intervals
# prediction intervals take into account IRREDUCIBLE ERROR!!
pred1 <-  predict(fit1, data.frame(horsepower = 98), interval = "prediction")
pred1
#       fit      lwr      upr
# 1 23.41249 15.33801 31.48697

# plot the data and the regression line
# ggplot
ggplot(data = mtcars) +
        geom_point(aes(x = mtcars$mpg, y = mtcars$hp), color = "blue", pch = 21) +
        geom_smooth(aes(x = mtcars$mpg, y = mtcars$hp),method = "lm", color = "red")

# base R
plot(Auto$horsepower, Auto$mpg, 
     main = "Scatterplot of mpg vs. horsepower", 
     xlab = "horsepower", ylab = "mpg", col = "blue")
abline(fit1, col = "red")


# diagnostic plots
# there is a clear pattern in residuals vs. fitted - residuals do not appear to be constant - no linearity in our data!
# normal Q-Q plot shows tails that indicate our residuals are mostly normal with a few tails
# leverage chart shows a few outliers (+/- 2) and a few high leverage points 117 and 94
plot(fit1)











## Question 9: linear regression on the Auto dataset
# produce a scatterplot matrix that includes all varaibles
# compute the matrix correlations between the variables using cor
# fit model mpg ~. - name
        # is there a relationship between predictors and response?
        # which predictors have a significant relationship to the repsonse?
        # what does the coefficient for the year variable suggest?
# use the plot function to see diagnostic plots
# fit a linear model with interactiom effect - are any interaction terms statistically significant
# fit a transformation linear regression - what do we find?

## Solution:

# load data
data("Auto")
str(Auto)

# scatterplot pairs plot
library(GGally)
ggpairs(data = Auto %>% select(.,-name))

# correlation matrix
cor(Auto %>% select(.,-name))
#                     mpg  cylinders displacement horsepower     weight acceleration       year     origin
# mpg           1.0000000 -0.7776175   -0.8051269 -0.7784268 -0.8322442    0.4233285  0.5805410  0.5652088
# cylinders    -0.7776175  1.0000000    0.9508233  0.8429834  0.8975273   -0.5046834 -0.3456474 -0.5689316
# displacement -0.8051269  0.9508233    1.0000000  0.8972570  0.9329944   -0.5438005 -0.3698552 -0.6145351
# horsepower   -0.7784268  0.8429834    0.8972570  1.0000000  0.8645377   -0.6891955 -0.4163615 -0.4551715
# weight       -0.8322442  0.8975273    0.9329944  0.8645377  1.0000000   -0.4168392 -0.3091199 -0.5850054
# acceleration  0.4233285 -0.5046834   -0.5438005 -0.6891955 -0.4168392    1.0000000  0.2903161  0.2127458
# year          0.5805410 -0.3456474   -0.3698552 -0.4163615 -0.3091199    0.2903161  1.0000000  0.1815277
# origin        0.5652088 -0.5689316   -0.6145351 -0.4551715 -0.5850054    0.2127458  0.1815277  1.0000000

# fit a base model
fit2 <- lm(mpg~. - name, data = Auto)
summary(fit2)
# Call:
#         lm(formula = mpg ~ . - name, data = Auto)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -9.5903 -2.1565 -0.1169  1.8690 13.0604 
# 
# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)    
#      (Intercept)  -17.218435   4.644294  -3.707  0.00024 ***
#      cylinders     -0.493376   0.323282  -1.526  0.12780    
#      displacement   0.019896   0.007515   2.647  0.00844 ** 
#      horsepower    -0.016951   0.013787  -1.230  0.21963    
#      weight        -0.006474   0.000652  -9.929  < 2e-16 ***
#      acceleration   0.080576   0.098845   0.815  0.41548    
#      year           0.750773   0.050973  14.729  < 2e-16 ***
#      origin         1.426141   0.278136   5.127 4.67e-07 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 3.328 on 384 degrees of freedom
# Multiple R-squared:  0.8215,	Adjusted R-squared:  0.8182 
# F-statistic: 252.4 on 7 and 384 DF,  p-value: < 2.2e-16

# F statistic pvalue indicates there is an overall relationship between repsonse and all predictors
# origin, year, displacement, and weight all give significant pvalues for their coeffiicents
# for an increase in year holding all other variables constant we have an increase in mpg about 1

# diagnostic plots
plot(fit2)

# residuals vs. fitted: evidence of a pattern in the residuals suggesting non-linearity in our data
# normal Q-Q has a strong tail at high values - residuals are not normally distributed
# residuals vs. leverage show outliers outside of +/- 2 and an extreme leverage point 14

# fit interaction term model
fit3 <- lm(mpg~. - name + horsepower*displacement + origin*year, data = Auto)
summary(fit3)
# Call:
#         lm(formula = mpg ~ . - name + horsepower * displacement + origin * 
#                    year, data = Auto)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -8.1108 -1.5223 -0.1222  1.2724 13.2009 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
#         (Intercept)              1.310e+01  7.987e+00   1.640   0.1017    
#         cylinders                6.071e-01  3.007e-01   2.019   0.0442 *  
#         displacement            -7.467e-02  1.086e-02  -6.875 2.53e-11 ***
#         horsepower              -1.905e-01  2.065e-02  -9.222  < 2e-16 ***
#         weight                  -3.170e-03  6.442e-04  -4.921 1.28e-06 ***
#         acceleration            -1.978e-01  9.042e-02  -2.188   0.0293 *  
#         year                     5.389e-01  9.987e-02   5.396 1.20e-07 ***
#         origin                  -8.581e+00  4.176e+00  -2.055   0.0406 *  
#         displacement:horsepower  5.084e-04  4.837e-05  10.512  < 2e-16 ***
#         year:origin              1.194e-01  5.371e-02   2.224   0.0268 *  
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.897 on 382 degrees of freedom
# Multiple R-squared:  0.8654,	Adjusted R-squared:  0.8622 
# F-statistic: 272.8 on 9 and 382 DF,  p-value: < 2.2e-16

# interactions of displacement and horsepower come up as "very" significant
# interactions of year:orgin is also signficiant

# fit model with transformations
fit4 <- lm(mpg~. - name - horsepower - weight + I(log(horsepower)) + I(sqrt(weight)), data = Auto)
summary(fit4)

# Call:
#         lm(formula = mpg ~ . - name - horsepower - weight + I(log(horsepower)) + 
#                    I(sqrt(weight)), data = Auto)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -9.2114 -1.8916 -0.1331  1.7318 12.6100 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
#         (Intercept)        33.029340   7.672170   4.305 2.12e-05 ***
#         cylinders          -0.440958   0.299056  -1.475  0.14117    
#         displacement        0.021703   0.006581   3.298  0.00107 ** 
#         acceleration       -0.179597   0.103306  -1.738  0.08293 .  
#         year                0.731219   0.047579  15.368  < 2e-16 ***
#         origin              1.317235   0.254328   5.179 3.60e-07 ***
#         I(log(horsepower)) -7.543770   1.553997  -4.854 1.76e-06 ***
#         I(sqrt(weight))    -0.585303   0.075578  -7.744 8.63e-14 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 3.1 on 384 degrees of freedom
# Multiple R-squared:  0.8451,	Adjusted R-squared:  0.8423 
# F-statistic: 299.3 on 7 and 384 DF,  p-value: < 2.2e-16


# log transformation of horsepower and sqrt transformation are significant










## Question 10: use the Carseats data
# fit a multiple linear regression to predict sales using price, urban, US
# provide an interpretation of each coefficient including qualitative variables
# which predictors do we reject the null hypothesis that Bj = 0?
# fit a new model with only significant predictors
# how well do our two models fit our data?
# using the second model give me confidence intervals for the coefficients
# is there evidence of outliers or high leverage points?

## Solution:

# load data
data("Carseats")
str(Carseats)

# fit our first model
fit1 <- lm(Sales ~ Price + Urban + US, data = Carseats)
summary(fit1)
# Call:
#         lm(formula = Sales ~ Price + Urban + US, data = Carseats)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -6.9206 -1.6220 -0.0564  1.5786  7.0581 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 13.043469   0.651012  20.036  < 2e-16 ***
#         Price       -0.054459   0.005242 -10.389  < 2e-16 ***
#         UrbanYes    -0.021916   0.271650  -0.081    0.936    
# USYes        1.200573   0.259042   4.635 4.86e-06 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.472 on 396 degrees of freedom
# Multiple R-squared:  0.2393,	Adjusted R-squared:  0.2335 
# F-statistic: 41.52 on 3 and 396 DF,  p-value: < 2.2e-16

# price has a -.05 effect on sales holding other variables constant
# Urban No has a +13 effect on sales holding other variables constant
# Urban Yes has a similiar +13 (intercept - UrbanYes coefficient) effect on sales holding other variables constant
# USYes has a 1.2 effect on sales holding other variables constant


# write out the model in equation form:
# SALES = 13.04  -.054*PRICE -.021*UrbanYes + 1.2*USYes 
# where UrbanYes = 1 and UrbanNo = 0
# where USYes = 1 and USNo = 0

# fit a smaller model with just significant coefficients
# Urban is not signifiicant lets leave it out - pvalue is not significant - cannot reject Bj = 0
fit2 <- lm(Sales ~ Price + US, data = Carseats)
summary(fit2)
# Call:
#         lm(formula = Sales ~ Price + US, data = Carseats)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -6.9269 -1.6286 -0.0574  1.5766  7.0515 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 13.03079    0.63098  20.652  < 2e-16 ***
#         Price       -0.05448    0.00523 -10.416  < 2e-16 ***
#         USYes        1.19964    0.25846   4.641 4.71e-06 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.469 on 397 degrees of freedom
# Multiple R-squared:  0.2393,	Adjusted R-squared:  0.2354 
# F-statistic: 62.43 on 2 and 397 DF,  p-value: < 2.2e-16

# how well do these two models fit?
# not well - our significant model performs miniscually over the first model
# only around 23% of the variation is explained in either model!

# what is the confidence interval for our coefficients in model 2?
# price = (-.06, -.04)
# US = (.69, 1.70)
confint(fit2)
#               2.5 %      97.5 %
# (Intercept) 11.79032020 14.27126531
# Price       -0.06475984 -0.04419543
# USYes        0.69151957  1.70776632

# is there evidence of leverage in our second model?
# there are several occurances of possible outliers - values that fall outside of +/-2 in the residuals vs. leverage plot
# value 368 has high leverage 
plot(fit2)







## Question 11: we will investigate the t-statistic for the null hypothesis in a regression without an intercept
# use the following code to answer the questions:
set.seed(1)
x = rnorm(100)
y = 2*x+rnorm(100)

# perform a simple linear regression y ~ x without an intercept
# report coefficients, standard error, t-statistic, p value
# at this level we will reject the null hypothesis that the coefficient estiamte is zero
fit1 <- lm(y ~ x + 0)
summary(fit1)$coef
# Estimate Std. Error  t value     Pr(>|t|)
# x 1.993876  0.1064767 18.72593 2.642197e-34

# perform a linear regressiom x ~ y without an intercept
# report coefficients, standard error, t-statistic, p value
fit2 <- lm(x ~ y +0)
summary(fit2)$coef
# Estimate Std. Error  t value     Pr(>|t|)
# y 0.3911145 0.02088625 18.72593 2.642197e-34



# what is the relationship between model 1 and model2?
# WE OBTAIN THE SAME VALUE FOR THE T-STAT AND P-VALUE
# BOTH RESULT REFLECT THE SAME LINE FROM A DIFFERENT VIEW OF PREDICTORS AND RESPONSE!!
# THE PVALUE AND TSTATISTIC WILL ALWAYS BE THE SAME FOR THE SAME RELATIONSHIP FLIPPED!!
# this is the same for regressions with an intercept!!!


# with and intercept y ~ x
fit3 <- lm(x~y)
summary(fit3)$coef
# Estimate Std. Error    t value     Pr(>|t|)
# (Intercept) 0.03880394 0.04266144  0.9095787 3.652764e-01
# y           0.38942451 0.02098690 18.5555993 7.723851e-34

# with an intercept x ~ y
fit4 <- lm(y~x)
summary(fit4)$coef
# Estimate Std. Error    t value     Pr(>|t|)
# (Intercept) -0.03769261 0.09698729 -0.3886346 6.983896e-01
# x            1.99893961 0.10772703 18.5555993 7.723851e-34

# the t-stat and p-value are the same in fit3 and fit4 both including an intercept!!!







## Question 12: simple linear regression without an intercept
# under what circumstance is the coefficient is the coefficient estimate for the regression of X~Y the same as Y ~ X
# the coefficient estimate Y ~ X is: Bhat = sum(xi*yi) / sum(x^2)
# the coeffiicent estimate of X~y is: Bhat = sum(xi*yi) / sum(y^2)
# this means that the two coefficient values will be the same when sum(x^2) == sum(y^2)

# generate an example with n = 100 in which the regression X~Y is different from Y~X
# here we need to generate a x and y with different values for sum(x^2) and sum(y^2)
# then we fit our regression to check the coefficient values are not the same!
x <- 1:100
sum(x^2)
# [1] 338350

y <- x+rnorm(100, sd = 0.1)
sum(y^2)
# [1] 338393.8

# fit1
fit1 <- lm(x~y)
summary(fit1)$coef
# Estimate   Std. Error      t value      Pr(>|t|)
# (Intercept) 0.001529991 0.0209405765 7.306347e-02  9.419045e-01
# y           0.999910949 0.0003599792 2.777691e+03 9.863043e-242

#fit2
fit2 <- lm(y~x)
summary(fit2)$coef
# Estimate   Std. Error       t value      Pr(>|t|)
# (Intercept) -0.0008886479 0.0209426864   -0.04243237  9.662404e-01
# x            1.0000763565 0.0003600388 2777.69054526 9.863043e-242


# generate an example with n = 100 in which the regressio ncoefificent is the same for X~Y and Y~X
# here we need to generate a x and y with the same values for sum(x^2) and sum(y^2)
# then we fit our regression model to check the coefficients is the same!!
x <- 1:100
sum(x^2)
# [1] 338350

y <- -1:-100
sum(y^2)
# [1] 338350

# fit x ~ y
fit3 <- lm(x~y)
summary(fit3)$coef
# Estimate   Std. Error       t value     Pr(>|t|)
# (Intercept) -5.684342e-14 5.598352e-15 -1.015360e+01 5.618494e-17
# y           -1.000000e+00 9.624477e-17 -1.039018e+16 0.000000e+00

# fit y ~ x
fit4 <- lm(y ~ x)
summary(fit4)$coef
# Estimate   Std. Error       t value     Pr(>|t|)
# (Intercept)  5.684342e-14 5.598352e-15  1.015360e+01 5.618494e-17
# x           -1.000000e+00 9.624477e-17 -1.039018e+16 0.000000e+00

# both coefficient estimates are the same!!







## Question 13: simulated data on a simple linear regression
# using the rnorm function create a vector x containing 100 obs from a normal distribution (0,1)
# using the rnorm function create a vector eps containing 100 observations from normal distribution (0,.25)
# using x and eps generate a vector y according to the model: Y = -1 .5(X) + e
set.seed(1)

# build x vector
x <- rnorm(100)

# build eps vector
eps <- rnorm(100, mean = 0, sd = sqrt(.25))

# build y vector
y <- -1 +.5*x + eps

# what is the length of y?
length(y)
# [1] 100

# what are the values of B0 and B1 in this model
# intercept = -1
# coefficient x = .5

# create a scatter plot between x and y
# we notice a strong positive linear relationship between x and y
ggplot() + geom_point(aes(x, y))

# fit a model y ~ x
# how do the coefficient differ from our "true" equation?
# estimates are very close to the true model but ever so slightly different!!
# this is due to the error term we added to the model
# we are honing in on the true model using ordinary least squares!!
fit1 <- lm(y ~x)
summary(fit1)
# Call:
#         lm(formula = y ~ x)
# 
# Residuals:
#         Min       1Q   Median       3Q      Max 
# -0.73194 -0.17888 -0.01419  0.16233  0.96349 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -1.01015    0.02942  -34.33   <2e-16 ***
#         x            0.50704    0.02978   17.03   <2e-16 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.2938 on 98 degrees of freedom
# Multiple R-squared:  0.7474,	Adjusted R-squared:  0.7448 
# F-statistic:   290 on 1 and 98 DF,  p-value: < 2.2e-16

# plot the results
plot(x,y, col = "dark blue")

# OLS regression line
abline(fit1, col = "orange")

# true regression line
abline(-1, .5, col = "red")

# define legend
legend("topleft", c("OLS", "True Function"), col = c("orange", "red"), lty = c(1,1))


# fit a polynomial model that predicts y ~ x and x^2
# does the quadratic term improve our fit?
# our R^2 actually gets worse !!!!!
# the true form is linear - adding quadratic will not improve fit!!!!
fit2 <- lm(y~x+I(x^2))
summary(fit2)
# Call:
#         lm(formula = y ~ x + I(x^2))
# 
# Residuals:
#         Min       1Q   Median       3Q      Max 
# -0.98252 -0.31270 -0.06441  0.29014  1.13500 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -0.97164    0.05883 -16.517  < 2e-16 ***
#         x            0.50858    0.05399   9.420  2.4e-15 ***
#         I(x^2)      -0.05946    0.04238  -1.403    0.164    
# ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.479 on 97 degrees of freedom
# Multiple R-squared:  0.4779,	Adjusted R-squared:  0.4672 
# F-statistic:  44.4 on 2 and 97 DF,  p-value: 2.038e-14



# repeat this process by adding more noise!!!
# noise will add more variance to our values of y and give us a more spread relationship
# this will drive down the R^2
# reducing noise will drive the predicited line closer and closer to the true line with less and less variance!!
# R^2 will improve as we add in less noise!!
set.seed(1)

# build x vector
x <- rnorm(100)

# build eps vector
eps <- rnorm(100, mean = 0, sd = sqrt(100))

# build y vector
y <- -1 +.5*x + eps

length(y)
ggplot() + geom_point(aes(x, y))

fit1 <- lm(y ~x)
summary(fit1)
# Call:
#         lm(formula = y ~ x)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -1.6254 -0.5315 -0.1208  0.4671  2.0318 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -1.03264    0.08399  -12.29  < 2e-16 ***
#         x            0.49908    0.09329    5.35 5.78e-07 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.8338 on 98 degrees of freedom
# Multiple R-squared:  0.226,	Adjusted R-squared:  0.2181 
# F-statistic: 28.62 on 1 and 98 DF,  p-value: 5.784e-07


# plot the results
plot(x,y, col = "dark blue")

# OLS regression line
abline(fit1, col = "orange")

# true regression line
abline(-1, .5, col = "red")

# define legend
legend("topleft", c("OLS", "True Function"), col = c("orange", "red"), lty = c(1,1))


fit2 <- lm(y~x+I(x^2))
summary(fit2)
# Call:
#         lm(formula = y ~ x + I(x^2))
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -1.7018 -0.5416 -0.1116  0.5025  1.9659 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  0.04912    0.10189   0.482    0.631    
# x            0.51486    0.09352   5.506 3.01e-07 ***
#         I(x^2)      -0.10299    0.07341  -1.403    0.164    
# ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.8297 on 97 degrees of freedom
# Multiple R-squared:  0.2414,	Adjusted R-squared:  0.2258 
# F-statistic: 15.43 on 2 and 97 DF,  p-value: 1.515e-06











## Question 14: collinearity

# set up the data with the following commands
# what is the model equation of this data?
# Y = 2 + 2(x1) + .3(x2) + error
# regression coefficients are 2 and .3
set.seed(1)
x1 <- runif(100)
x2 <- .5 * x1 + rnorm(100) / 10
y = 2 + 2*x1 + .3*x2 + rnorm(100)

# what is the correlation between x1 and x2?
# this shit is highly fucking correlated
cor(x1,x2)
# [1] 0.8351212
plot(x1, x2)

# fit a model y ~ x1 + x2: describe the results
fit1 <- lm(y~x1+x2)
summary(fit1)
# Call:
#         lm(formula = y ~ x1 + x2)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -2.8311 -0.7273 -0.0537  0.6338  2.3359 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)   2.1305     0.2319   9.188 7.61e-15 ***
#         x1            1.4396     0.7212   1.996   0.0487 *  
#         x2            1.0097     1.1337   0.891   0.3754    
# ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 1.056 on 97 degrees of freedom
# Multiple R-squared:  0.2088,	Adjusted R-squared:  0.1925 
# F-statistic:  12.8 on 2 and 97 DF,  p-value: 1.164e-05

# we are not getting close to our true function - correlated variables cause problems!
# we can reject B1 = 0 but not B2 = 0


# fit a model only using x1
fit2 <- lm(y~x1)
summary(fit2)
# Call:
#         lm(formula = y ~ x1)
# 
# Residuals:
#         Min       1Q   Median       3Q      Max 
# -2.89495 -0.66874 -0.07785  0.59221  2.45560 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)   2.1124     0.2307   9.155 8.27e-15 ***
#         x1            1.9759     0.3963   4.986 2.66e-06 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 1.055 on 98 degrees of freedom
# Multiple R-squared:  0.2024,	Adjusted R-squared:  0.1942 
# F-statistic: 24.86 on 1 and 98 DF,  p-value: 2.661e-06

# in this model we can reject B1 = 0!!!!


# fit a model only using x2
fit3 <- lm(y~x2)
summary(fit3)
# Call:
#         lm(formula = y ~ x2)
# 
# Residuals:
#         Min       1Q   Median       3Q      Max 
# -2.62687 -0.75156 -0.03598  0.72383  2.44890 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)   2.3899     0.1949   12.26  < 2e-16 ***
#         x2            2.8996     0.6330    4.58 1.37e-05 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 1.072 on 98 degrees of freedom
# Multiple R-squared:  0.1763,	Adjusted R-squared:  0.1679 
# F-statistic: 20.98 on 1 and 98 DF,  p-value: 1.366e-05

# now we can reject B1 = 0

# these answers differ because x1 and x2 are so highly correlated
# having both in will "mask" effects of each other
# having one or the other in will show as significant because they are very close to each other!!!


# re-fit on new data
x1 = c(x1, 0.1)
x2 = c(x2, .8)
y = c(y, 6)

# model 1 re-fit
fit1 <- lm(y~x1+x2)
summary(fit1)
# Call:
#         lm(formula = y ~ x1 + x2)
# 
# Residuals:
#         Min       1Q   Median       3Q      Max 
# -2.69309 -0.68184 -0.04583  0.75224  2.29389 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)   2.2665     0.2303   9.840 2.45e-16 ***
#         x1            0.1671     0.5246   0.318    0.751    
# x2            3.1371     0.7703   4.073 9.37e-05 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 1.079 on 99 degrees of freedom
# Multiple R-squared:  0.246,	Adjusted R-squared:  0.2308 
# F-statistic: 16.15 on 2 and 99 DF,  p-value: 8.501e-07
plot(fit1)

# model 2 re-fit
fit2 <- lm(y~x1)
summary(fit2)
# Call:
#         lm(formula = y ~ x1)
# 
# Residuals:
#         Min      1Q  Median      3Q     Max 
# -2.8848 -0.6542 -0.0769  0.6137  3.4510 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)   2.3921     0.2454   9.747 3.55e-16 ***
#         x1            1.5691     0.4255   3.687 0.000369 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 1.16 on 100 degrees of freedom
# Multiple R-squared:  0.1197,	Adjusted R-squared:  0.1109 
# F-statistic:  13.6 on 1 and 100 DF,  p-value: 0.0003686
plot(fit2)


# model 3 re-fit
fit3 <- lm(y~x2)
summary(fit3)
# Call:
#         lm(formula = y ~ x2)
# 
# Residuals:
#         Min       1Q   Median       3Q      Max 
# -2.66396 -0.67794 -0.06181  0.75541  2.32512 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)   2.3085     0.1879   12.28  < 2e-16 ***
#         x2            3.2981     0.5786    5.70 1.21e-07 ***
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 1.074 on 100 degrees of freedom
# Multiple R-squared:  0.2452,	Adjusted R-squared:  0.2377 
# F-statistic: 32.49 on 1 and 100 DF,  p-value: 1.214e-07
plot(fit3)

# what do we notice? we notice that the new observations actually pushed x2 toward significance in model 1
# in the model with x1 the last point is an outlier
# in the mdoel with x2 the last point is a high-leverage point
# in the model with both predictors the last point is a high leverage point











## Question 15: Boston data set: predict crime using all other variables in the dataset

# for each predictor fit a one variable model
# describe results
# which variables "stand out" as significant?

# load data
data("Boston")
str(Boston)

list <- names(Boston)
df <- NULL

# model loop
for (i in 1:length(list)) {
        
        column <- i
        
        model.df <- Boston %>% 
                dplyr::select(crim, column) %>% 
                lm(crim ~., data = .)
        
        results <- data.frame(coef(summary(model.df)), summary(model.df)$r.squared) %>% 
                rownames_to_column("name") %>% 
                filter(!str_detect(name, "Int"))
        
                
        
        df <- rbind(results, df)
}

(results2 <- df %>%  arrange(., -df$summary.model.df..r.squared) %>% 
        mutate_if(is.numeric, round, digits = 25) %>% 
        mutate(sig = ifelse(df$Pr...t..<.05, "*", "")))

# name    Estimate  Std..Error   t.value     Pr...t.. summary.model.df..r.squared
# 1      rad  0.61791093 0.034331820 17.998199 2.693844e-56                 0.391256687
# 2      tax  0.02974225 0.001847415 16.099388 2.357127e-47                 0.339614243
# 3    lstat  0.54880478 0.047760971 11.490654 2.654277e-27                 0.207590933
# 4      nox 31.24853120 2.999190381 10.418989 3.751739e-23                 0.177217182
# 5    indus  0.50977633 0.051024332  9.990848 1.450349e-21                 0.165310070
# 6     medv -0.36315992 0.038390175 -9.459710 1.173987e-19                 0.150780469
# 7    black -0.03627964 0.003873154 -9.366951 2.487274e-19                 0.148274239
# 8      dis -1.55090168 0.168330031 -9.213458 8.519949e-19                 0.144149375
# 9      age  0.10778623 0.012736436  8.462825 2.854869e-16                 0.124421452
# 10 ptratio  1.15198279 0.169373609  6.801430 2.942922e-11                 0.084068439
# 11      rm -2.68405122 0.532041083 -5.044819 6.346703e-07                 0.048069117
# 12      zn -0.07393498 0.016094596 -4.593776 5.506472e-06                 0.040187908
# 13    chas -1.89277655 1.506115484 -1.256727 2.094345e-01                 0.003123869


# it looks like rad and tax have the strongest relationships with crime
plot(Boston$crim, Boston$rad)
plot(Boston$crim, Boston$tax)
plot(Boston$crim, Boston$lstat)

pairs(Boston %>% dplyr::select(crim, rad, tax, lstat))

cor(Boston)
# crim          zn       indus         chas         nox          rm         age         dis
# crim     1.00000000 -0.20046922  0.40658341 -0.055891582  0.42097171 -0.21924670  0.35273425 -0.37967009
# zn      -0.20046922  1.00000000 -0.53382819 -0.042696719 -0.51660371  0.31199059 -0.56953734  0.66440822
# indus    0.40658341 -0.53382819  1.00000000  0.062938027  0.76365145 -0.39167585  0.64477851 -0.70802699
# chas    -0.05589158 -0.04269672  0.06293803  1.000000000  0.09120281  0.09125123  0.08651777 -0.09917578
# nox      0.42097171 -0.51660371  0.76365145  0.091202807  1.00000000 -0.30218819  0.73147010 -0.76923011
# rm      -0.21924670  0.31199059 -0.39167585  0.091251225 -0.30218819  1.00000000 -0.24026493  0.20524621
# age      0.35273425 -0.56953734  0.64477851  0.086517774  0.73147010 -0.24026493  1.00000000 -0.74788054
# dis     -0.37967009  0.66440822 -0.70802699 -0.099175780 -0.76923011  0.20524621 -0.74788054  1.00000000
# rad      0.62550515 -0.31194783  0.59512927 -0.007368241  0.61144056 -0.20984667  0.45602245 -0.49458793
# tax      0.58276431 -0.31456332  0.72076018 -0.035586518  0.66802320 -0.29204783  0.50645559 -0.53443158
# ptratio  0.28994558 -0.39167855  0.38324756 -0.121515174  0.18893268 -0.35550149  0.26151501 -0.23247054
# black   -0.38506394  0.17552032 -0.35697654  0.048788485 -0.38005064  0.12806864 -0.27353398  0.29151167
# lstat    0.45562148 -0.41299457  0.60379972 -0.053929298  0.59087892 -0.61380827  0.60233853 -0.49699583
# medv    -0.38830461  0.36044534 -0.48372516  0.175260177 -0.42732077  0.69535995 -0.37695457  0.24992873
# rad         tax    ptratio       black      lstat       medv
# crim     0.625505145  0.58276431  0.2899456 -0.38506394  0.4556215 -0.3883046
# zn      -0.311947826 -0.31456332 -0.3916785  0.17552032 -0.4129946  0.3604453
# indus    0.595129275  0.72076018  0.3832476 -0.35697654  0.6037997 -0.4837252
# chas    -0.007368241 -0.03558652 -0.1215152  0.04878848 -0.0539293  0.1752602
# nox      0.611440563  0.66802320  0.1889327 -0.38005064  0.5908789 -0.4273208
# rm      -0.209846668 -0.29204783 -0.3555015  0.12806864 -0.6138083  0.6953599
# age      0.456022452  0.50645559  0.2615150 -0.27353398  0.6023385 -0.3769546
# dis     -0.494587930 -0.53443158 -0.2324705  0.29151167 -0.4969958  0.2499287
# rad      1.000000000  0.91022819  0.4647412 -0.44441282  0.4886763 -0.3816262
# tax      0.910228189  1.00000000  0.4608530 -0.44180801  0.5439934 -0.4685359
# ptratio  0.464741179  0.46085304  1.0000000 -0.17738330  0.3740443 -0.5077867
# black   -0.444412816 -0.44180801 -0.1773833  1.00000000 -0.3660869  0.3334608
# lstat    0.488676335  0.54399341  0.3740443 -0.36608690  1.0000000 -0.7376627
# medv    -0.381626231 -0.46853593 -0.5077867  0.33346082 -0.7376627  1.0000000



# fit a model with all predictors
fit2 <- lm(crim~., data = Boston)
summary(fit2)
# Call:
#         lm(formula = crim ~ ., data = Boston)
# 
# Residuals:
#         Min     1Q Median     3Q    Max 
# -9.924 -2.120 -0.353  1.019 75.051 
# 
# Coefficients:
#         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  17.033228   7.234903   2.354 0.018949 *  
#         zn            0.044855   0.018734   2.394 0.017025 *  
#         indus        -0.063855   0.083407  -0.766 0.444294    
# chas         -0.749134   1.180147  -0.635 0.525867    
# nox         -10.313535   5.275536  -1.955 0.051152 .  
# rm            0.430131   0.612830   0.702 0.483089    
# age           0.001452   0.017925   0.081 0.935488    
# dis          -0.987176   0.281817  -3.503 0.000502 ***
#         rad           0.588209   0.088049   6.680 6.46e-11 ***
#         tax          -0.003780   0.005156  -0.733 0.463793    
# ptratio      -0.271081   0.186450  -1.454 0.146611    
# black        -0.007538   0.003673  -2.052 0.040702 *  
#         lstat         0.126211   0.075725   1.667 0.096208 .  
# medv         -0.198887   0.060516  -3.287 0.001087 ** 
#         ---
#         Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 6.439 on 492 degrees of freedom
# Multiple R-squared:  0.454,	Adjusted R-squared:  0.4396 
# F-statistic: 31.47 on 13 and 492 DF,  p-value: < 2.2e-16


# difference in coeffiicnets from model to model
mod2.coef <- data.frame(coef(summary(fit2))) %>% 
        rownames_to_column("name") %>% 
        filter(!str_detect(name, "Int"))

plot.df <- left_join(mod2.coef, results2, by = c("name")) 

ggplot(data = plot.df) +
        geom_point(aes(x = name, y = Estimate.x), color = "orange") +
        geom_point(aes(x = name, y = Estimate.y) ,color = "blue")


ggplot(data = plot.df) +
        geom_point(aes(x = Estimate.y, y = Estimate.x), color = "orange")


# polynomial model

list <- names(Boston %>% dplyr::select(., - crim))
df <- NULL

# model loop
for (i in 1:length(list)) {
        
        column <- list[i]
        
        model.df <- Boston %>% 
                dplyr::select(crim, column) %>% 
                lm(crim ~. + I(.[,2]^2) + I(.[,2]^3), data = .)
        
        results <- data.frame(coef(summary(model.df)), summary(model.df)$r.squared) %>% 
                rownames_to_column("name") %>% 
                filter(!str_detect(name, "Int"))
        
        
        
        df <- rbind(results, df)
}

(results2 <- df %>%  arrange(., -df$summary.model.df..r.squared) %>% 
                mutate_if(is.numeric, round, digits = 25) %>% 
                mutate(sig = ifelse(df$Pr...t..<.05, "*", "")))

# name      Estimate   Std..Error     t.value     Pr...t.. summary.model.df..r.squared sig
# 1         medv -5.094831e+00 4.338321e-01 -11.7437854 0.000000e+00                 0.420200257   *
# 2  I(.[, 2]^2)  1.554965e-01 1.719044e-02   9.0455201 3.260523e-18                 0.420200257   *
# 3  I(.[, 2]^3) -1.490103e-03 2.037895e-04  -7.3119688 1.046510e-12                 0.420200257   *
# 4          rad  5.127360e-01 1.043597e+00   0.4913162 6.234175e-01                 0.400036872    
# 5  I(.[, 2]^2) -7.517736e-02 1.485430e-01  -0.5060982 6.130099e-01                 0.400036872    
# 6  I(.[, 2]^3)  3.208996e-03 4.564010e-03   0.7031090 4.823138e-01                 0.400036872    
# 7          tax -1.533096e-01 9.567806e-02  -1.6023487 1.097075e-01                 0.368882080    
# 8  I(.[, 2]^2)  3.608266e-04 2.425463e-04   1.4876610 1.374682e-01                 0.368882080    
# 9  I(.[, 2]^3) -2.203715e-07 1.888705e-07  -1.1667862 2.438507e-01                 0.368882080    
# 10         nox -1.279371e+03 1.703975e+02  -7.5081586 2.758372e-13                 0.296977896   *
# 11 I(.[, 2]^2)  2.248544e+03 2.798993e+02   8.0334044 6.811300e-15                 0.296977896   *
# 12 I(.[, 2]^3) -1.245703e+03 1.492816e+02  -8.3446489 6.961110e-16                 0.296977896   *
# 13         dis -1.555435e+01 1.735967e+00  -8.9600490 6.374792e-18                 0.277824773    
# 14 I(.[, 2]^2)  2.452072e+00 3.464194e-01   7.0783330 4.941214e-12                 0.277824773    
# 15 I(.[, 2]^3) -1.185986e-01 2.040040e-02  -5.8135442 1.088832e-08                 0.277824773    
# 16       indus -1.965213e+00 4.819901e-01  -4.0772894 5.297064e-05                 0.259657858    
# 17 I(.[, 2]^2)  2.519373e-01 3.932212e-02   6.4070114 3.420187e-10                 0.259657858    
# 18 I(.[, 2]^3) -6.976009e-03 9.566596e-04  -7.2920492 1.196405e-12                 0.259657858    
# 19       lstat -4.490656e-01 4.648911e-01  -0.9659586 3.345300e-01                 0.217932432   *
# 20 I(.[, 2]^2)  5.577942e-02 3.011561e-02   1.8521764 6.458736e-02                 0.217932432   *
# 21 I(.[, 2]^3) -8.573703e-04 5.651667e-04  -1.5170220 1.298906e-01                 0.217932432   *
# 22         age  2.736531e-01 1.863796e-01   1.4682566 1.426608e-01                 0.174230994    
# 23 I(.[, 2]^2) -7.229596e-03 3.636995e-03  -1.9877935 4.737733e-02                 0.174230994   *
# 24 I(.[, 2]^3)  5.745307e-05 2.109355e-05   2.7237266 6.679915e-03                 0.174230994   *
# 25       black -8.355805e-02 5.632751e-02  -1.4834323 1.385871e-01                 0.149839829    
# 26 I(.[, 2]^2)  2.137404e-04 2.984193e-04   0.7162418 4.741751e-01                 0.149839829    
# 27 I(.[, 2]^3) -2.652453e-07 4.364264e-07  -0.6077665 5.436172e-01                 0.149839829    
# 28     ptratio -8.236054e+01 2.764394e+01  -2.9793340 3.028663e-03                 0.113781577   *
# 29 I(.[, 2]^2)  4.635347e+00 1.608321e+00   2.8821030 4.119552e-03                 0.113781577   *
# 30 I(.[, 2]^3) -8.476032e-02 3.089749e-02  -2.7432751 6.300514e-03                 0.113781577   *
# 31          rm -3.915014e+01 3.131149e+01  -1.2503440 2.117564e-01                 0.067786061    
# 32 I(.[, 2]^2)  4.550896e+00 5.009862e+00   0.9083875 3.641094e-01                 0.067786061   *
# 33 I(.[, 2]^3) -1.744770e-01 2.637470e-01  -0.6615314 5.085751e-01                 0.067786061   *
# 34          zn -3.321884e-01 1.098081e-01  -3.0251711 2.612296e-03                 0.058241974   *
# 35 I(.[, 2]^2)  6.482634e-03 3.860728e-03   1.6791221 9.375050e-02                 0.058241974   *
# 36 I(.[, 2]^3) -3.775793e-05 3.138615e-05  -1.2030123 2.295386e-01                 0.058241974    
# 37        chas -1.892777e+00 1.506115e+00  -1.2567274 2.094345e-01                 0.003123869  


# For “zn”, “rm”, “rad”, “tax” and “lstat” as predictor, 
# the p-values suggest that the cubic coefficient is not statistically significant;
# for “indus”, “nox”, “age”, “dis”, “ptratio” and “medv” as predictor, 
# the p-values suggest the adequacy of the cubic fit; 
# for “black” as predictor, the p-values suggest that the quandratic and cubic coefficients are not statistically significant, 
# so in this latter case no non-linear effect is visible.









# chapter 4: classification -----------------------------------------------






## Chapter 4: Classification
# linear regression assumes that the repsonse variable Y is quantitative
# classification assumes that the repsonse variable Y is qualitative!!
# quantitative variables are refered to as categorical
# approaches for predicting categorical variables are statistical learning techniques called classification
# we are classifying a new observation based on our model
# classification models can behave like regression models - assigning a probability to belong to a specific class!
# this chapter will focus on three of the most popular classification models:
        # logistic regression
        # linear discriminatn analysis
        # K nearest neighbors
# generalized additive models, trees, random forests, boosting and support vector machines will be discussed later


## Overview of classification
# classification problems occur often:
        # what of the three conditions does someone have?
        # is this transaction fraud? yes or no?
        # is this DNA sequence disease causing? yes or no?
# just as in the classification setting we have a set of training observations that we use to build a classifier
# we want our classification to perform well not only on the training data but also generalize to the test

## Why not linear regression?
# linear regression is not appropriate in the case of a qualitative response...why?
# example: predicting condition of a patient
# three possible categories a patient can be in: 1 = stroke, 2 = drug overdose, 3 = seizure
# using this dummy variable coding a least squares could be used to fit a linear model to predict Y based on X
# however - this coding puts an order to the result variables - drug overdose is between stroke and seizure
# this means the difference between stroke  and overdose is the same as the difference between overdose and seizure
# there is no reason that this needs to be the case - our coding scheme is arbitrary
# each coding scheme will imply different relationships between the outcome and predictors
# each type of coding will lead to different linear models with different interpretations
# this will give us different sets of predictions!!
# in general there is no natural way to convert qualitative repsonse variables with more than two levels into a quantitiative repsonse!!
# we can use linear regression on a binary case by dummy coding our variables!!
# for instance if we only had two classes: overdose and stroke: we could code them as 1 and 0 and fit a regression model
# here flipping the order will not change the interpreation of our linear model - the final predictions will be the same
# the predictions will be rough estimates of probabilities belonging to each class...
# the linear model results on a binary model will be the same as the results for a linear discriminant analysis
# this dummy variable approach for the Y cannot be easily extended to accomodate qualitative repsonses with more than two levels!!
# this calls for CLASSIFICATION!!!


## Logistic Regression:
# consider the Default dataset - the response Default falls into two categories Yes or No
# rather than modeling the response Y directly, logistic regression models the probability that Y belings to a category
# for the default dataset - logistic regression models the probability of default!!!
# example: the probability of a default given balance can be written as:
        # Pr(default = Yes | BALANCE)
# the values will range between 0 and 1
# for any given value of balance a prediction can be made for default
# one might predict deafult = Yes for any individual for whom p(BALANCE) > .5
# alternatively we wcould be conservative and choose a lower threshold p(BALANCE) > .1


## The Logistic Model
# how should we model the relationship between p(X) = Pr(Y = 1 | X) and X????
# here is the linear regression model: p(X) = B0 + B1X
# if we use this to predict default = Yes using BALANCE we get a bad model fit
# for balances close to zero we predict a negative probability of default
# for large balances we get probability values greater than 1!!
# these predictions are not sensible true probability must fall between 0 and 1
# any time a straight line fit is fit to a binary repsonse we will always predict negative and greater than 1 probabiliites
# the straight line fit will keep extraploting values as we get big or small Xs
# the straight line will not be bounded by 0 and 1 the line will cross through these points


# to avoid these problems  we must model p(X) using a function that gives outputs between 0 and 1 for ALL VALUES OF X
# in logistic regression - we use the logistic function to to this!!
# logistic function: p(X) = e^(B0 + B1(X)) / 1 + e^(B0 + B1(X))
# to fit this model we use a method called the maximum likelihood 
# a fit with a logsitc function will be bounded by 0 and 1 and model the binary repsonse better than the linear regression
# logistic regression will always give us the S shaped curve 
# regardless of the value of X we will always recieve a sensible prediction of the proabbility of belonging to a class
# the logistic model is better able to capture the range of probabiliites that is than the linear regresison model!

# we can manipulate the logistic function to give:
# p(X) / 1 - p(X)  = e^(B0 + B1(X))
# the quantitiy on the left hand side is called the odds and can take on any value between 0 and infinity
# values of the odds close to 0 and infinity indicate very low and high probabilies of default
# with odds of 1/4 = 1 in 5 people will default = p(X) = .2 implies .2 / 1 - .2 == .25
# with odds of 9, 9 out of every 10 people will default = p(X) = .9 implies .9 / 1 - .9 == 9
# by taking the log of both side of our odds equation we arrive at:
# log(p(X) / (1 - p(X)) = B0 + B1(X)
# the left hand side is now called the log-odds or logit function
# the logistic regression model has a logit that is linear in X
# in contrast to regression a change in X  by one unit increases the log odds of B1 - multiplying odds by e^B1
# HOWEVER because the relationship between p(X) and X is not a straight line - B1 DOES NOT correspond to the change in p(X) associated with a one unit increase in X
# the amount that p(X) changes will depend on the current value of X!!!!!!
# regradless of what x is - a positive coeffiicent will increase the probability and a negative coefficient will decrease the probability
# there is not a straight line relationship between p(X) and X!!
# the rate of change of P(X) depends on the value of X!!!

# logistic regression aims to model P(X) belonging to a certain class!!
# we do this by using the logistic function that is bounded between 1 and 0
# the logistic function is maximized by the maximum likehood method
# the logstic regression gets its name from the log odds - which is the basis for the logistic function
# predictions will be bounded between 1 and 0 and each estimate will be a probability of belonging to a certain class
# increasing a unit of X will multiply the odds by e^coefficient!!
# there is not a straight line relationship between P(X) and X
# P(X) will depend on the value of X; B1 does not correspond to the change in p (X) associated with a one unit increase in X
# the amount the P(X) changes based on a one unit increase will depend on the new value of X!!!!


## Estimating the Logistic Regression Coefficients
# the estimates for our classification model are unknown - we must estimate them from the training data
# in chapter 3 we dicsussed the least squares estimate of the unknown coefficients for a regression model
# although we can use the least squares criteria to fit our classification model - the MAXIMUM LIKLIHOOD ESTIMATE is PREFERRED
# the basic intuition behind the maximum liklihood estimate is as follows:
        # we seek estimates for B0 and B1 
        # such that the predicted probability p(xi) of default for each individual corresponds as closely as possible to the observed default status
        # we want a probability that is close as possible to the real class of the observation!!
# we try to find B0 and B1 such that plugging these estimates into the model for p(X) yields a number close to one for all who actually defaulted and close to 0 for those who did not
# this intuition can be formalized using the mathematical equation called a likelihood function:
        # l(B0, B1) = || for y = 1;  p(xi) || for y = 0;  (1 - p(xi))
# WE CHOOSE ESTIMATES OF B0 AND B1 THAT MAXIMIZE THIS LIKELIHOOD FUNCTION
# maxmimum likelihood is a very general approach that is used to fit many of the non-linear models that we examine throughout this book
# the least squares estimate is in fact a special case of maximum likelihood
# in general logistic regression and other models can be easily fit using a statistical software package such as R
# we do not need to concern ourselves with the actual fitting procedure of the maximum likelihood method

# example results:
# we fit a model: Default (Yes) ~ BALANCE
# we find that B1 = .0055
# this indicates that an increase in BALANCE is associated with an increase in the probability of DEFAULT
# a one unit increase in BALANCE is associated with an increase in the log odds of DEFAULT by .0055 units
# many aspects of our logistic regression fit will be similiar to our standard linear regression fit
        # we have standard errors associated with each coefficient
        # the z statistic plays the same role as our t statistic in linear regression: a large z statisitc indicates evidence that we can reject the null hypothesis that Ho can be equal to 0
        # the pvalue can let us know to reject or accpet the null hypothesis
# in our case the pvalue is very small so we can conclude that there is indeed an association between BALANCE and probability of DEFAULT
# the estimated intercept is typically not of interest - it's main purpose is to adjust the average fitted probabilities to the proportion of ones in the data

## Making Predictions
# once the coefficients have been estimated it is a simple matter to compute the probability of DEFAULT for any given credit card balance
# using the coefficient estimates from the model we predict that the default probability for an individual with a BALANCE of $1000 is:
        # p(X) = e^(B0 + B1(X)) / 1 + e^(B0 + B1(X))
        # e^(-10.65 + .0055 * 1000) / 1 + e^(-10.65 + .0055*1000) == .00576
# this is below 1% - with a balance of $1000 we expect DEFAULT in less than 1% of the cases
# in contrast the predicted probability of default for a BALANCE of $2000 is much higher = almost 58%!!!

# one can use qualitative predictors with the logistic regression model using the dummy variable approach
# we can combine quantitiative and qualitative predictors in our logistic regression!!!
# to do this we simply fit a model with a dummy variable for that specific predictor
# example: 1 if student = yes, 0 if student = no
# in this example the coefficient associated with the dummy variable is positive and the associated pvalue is significant
# this indicates that students tend to have higher DEFAULT probabilities than non-students:
        # Pr(default = Yes | Student = Y) == e^(-3.5 + .40 * 1) / 1 + e^(-3.5 + .49 * 1) = .0431
        # Pr(default = Yes | Student = N) == e^(-3.5 + .40 * 0) / 1 + e^(-3.5 + .49 * 0) = .0292

## Multiple Logistic Regression
# we now consider the problem of predicting a binrary response using multiple predictors
# we can generalize our log-odds formula for coefficent estimates to that of multiple coefficient estimates
        # log(p(X) / 1 - P(X)) == B0 + B1*X1 + ... + BpXp where X = (X1, ..., Xp) are p predictors
# we can re-write this equation as:
        # P(X) == e^(coefficient string) / 1 + e^(coeffiicent string)
# just as in our original logistic regression we will use the maximum liklihood estimate method to estimate our coefficients B0, B1...Bp

# example:
# we fit a model = DEFAULT ~ INCOME + STUDENT + BALANCE
# there is a suprising result here
# the pvalues associated with BALANCE and the dummy variable for STUDENT are very small indiciating that each of these is associated with the probability of DEFAULT
# however the coefficient for the dummy variable is negative - indicating that students are less likely to DEFULAT than non-students
# in our previous example - STUDENT had a positive association with DEFAULT
# how can student be associated with an increase in DEFAULT probability in the first model but a DECREASE in probability in the second model?
# the negative coefficient for STUDENT in the multiple logistic regression indicates that for a fixed value of BALANCE and INCOME a student is less likely to DEFAULT
# when we average the default rates of student vs. non-student across all values of BALANCE and INCOME when get a postiive coefficent
# why is this happening?
# THE VARIABLES STUDENT and BALANCE are highly correlated!!!!!!
# students tend to avoid higher levels of debt which is in turn associated with lower levels of default!!
# students are more likely to hold higher levels of debt which is associated with high levels of default
# even though a student with a given credit card balance will tend to have a lower probability of default than a non-student WITH THE SAME BALANCE, the fact is students on average have higher BALANCES OVERALL - students tend to have higher DEFAULT
# a student is riskier than a non-student if no information about the student's credit card balance is available
# HOWEVER THAT STUDENT IS LESS RISKY THAN A NON-STUDENT WITH THE SAME CREDIT CARD BALANCE!!!!

# this example illustrates the dangers of performing simple univariate regression models - WHEN OTHER PREDICTORS ARE RELEVANT!!
# AS WITH LINEAR REGRESSION THE RESULT OBTAINED USING ONE PREDICTOR CAN BE QUITE DIFFERENT FROM THOSE WITH MULTIPLE PREDICTORS!!
# ESPECIALLY WHEN THERE IS CORRELATION AMONG THE PREDICTORS!!!
# THIS IS KNOWN AS CONFOUNDING - correlation of predictors changes the results of the other predictors

# we can predict with our multiple logistic model too
# for example: a student with credit card balance of $1500 and income of $40000 has an estimated probability of default of:
        # P(X) = e^[coefficent string with 1500 and 40 and 1] / 1 + e^[coefficient string with 1500 and 40 and 1] === .058 percent
# a non-student with the same balance and income has an estimated probability of:
        # P(X) = e^[coefficent string with 1500 and 40 and 0] / 1 + e^[coefficient string with 1500 and 40 and 0] === .105 percent
# note we used 40 as the X variable for INCOME is in units of $1,000


## Logistic Regression for > 2 Repsonse Classes
# we sometimes wish to classify a response variable that has two or more classes
# for example we may have three categories we want to model probabilities for
# the repsonse has three categories = stroke, drug, seizure
# the two class logistic regression models discussed in the previous sections have muliple class extensions - but they are not used that often
# one of the reasons is that the method we discuss in the next section LINEAR DISCRIMINANT ANALYSIS is popular for multiple-class classification
# a multi-class logistic regression is possible and available in R - however consider LDA first


## Linear Discriminant Analysis
# logistic regression involves directly modeling Pr(Y = k | X = x) using the logistic function for the case of two repsonse classes
# we model the conditional distribution of the response Y given our predictors X
# we will now cover a less direct way of modeling these two class probabilities
# in this alternative approach we model the distribution of the predictors X separetly in each of the response classes "given Y"
# we then use Bayes theorm to flip these around into estiamtes for Pr(y = k | X = x)
# when these distributions are normal it turns out this new approach is similiar to logistic regression
# why do we need an alternative to logistic regression?
        # when the classes are well-separated the parameter estimates for the logistic model are unstable
        # if n is small and the distribution of the predictors X is approximately normal in each class - linear discriminat analysis is more stable
        # LDA is popular when we want to model 3 or more class repsonses

## Using bayes theorem for classification
# supppose we wish to classify an observation into one of K classes, where K >= 2
# the qualitative response variable Y can take on K possible distinct and unordered values
# we let pi.k represent the overall or prior probability that a randomly chosen observation is associated with the repsonse variable Y
# we let f(x) = Pr(X = x | Y = k) denote the density function of X for an observation that takes the kth class
# in other words f(x) is relatively large if there is a high probability that an observation in the kth class has X == x
# f(x) will be very small then it is unlikely that an observation in the kth class has X == x
# Bayes Theorem: 
        # Pr(Y = k | X = x) = (pi.k * f(x)) / sumof(pi.1 * f1(x))
# this suggests that instead of directly computing pk(X) we can simply plug in estimates of pi.k and f(X) to get our most probable class
# estimating pi.k is easy if we have a random sample of Ys from the population: we simply compute the fraction of the training observations that belong to the kth class
# estimating the f(X) is slightly more challenging
# we refer to pk(x) as the posterior probability that an observation X = x belongs to the kth class GIVEN THE PREDICTOR VALUE FOR THAT OBSERVATION
# we known that the Bayes classifier performs well
# if we are able to accurately estimate f(X) then we can develop a classifier that approximates the Bayes classifier
# Bayes classifier classifies an observation to the class for which pk(X) is the largest
# Linear Discriminant Analysis is how we will approximate the bayes classifier

## Linear Discriminant Analysis for p = 1
# assuming we only have one predictor
# we try to obtain an estimate for fk(x) that we plug into Bayes' theorem in order to estimate pk(x)
# we will then classify an observation to the class for which pk(x) is greatest
# to estimate fk(x) we will need to make assumptions about it's form
# suppose we assume that fk(x) is normal or Gaussian
# our density function in our one variable setting takes the form:
        # fk(x) = 1 / sqrt(2*pi*sigma) * exp(-1/2sigma^2 * (x - uk)^2)
# uk and sigma^2 k are the mean and variance paramenters for the kth class
# we also assume there is a shared variance term across all K classes = sigma^2
# plugging in our normal density function into bayes theorem givens us:
        # pk(x) = [pi.k * 1/sqrt(2*pi*sigma)] * exp(-(1/2simga^2) * (x - uk)^2) / [pi.l * 1/sqrt(2*pi*sigma)] * exp(-(1/2simga^2) * (x - ul)^2]
# note that pi.k denotes the prior probability that an observation belongs to the kth class not the constant pi
# the bayes classifier involves assigning an observation X = x to the class for which 4.12 is the largest
# taking the log of 4.12 and rearranging the terms:
        # theta.k(x) = x * (u.k / sigma^2) - (u.k^2 / 2*sigma^2) + log(pi.k)
# LDA assigns the classification of the new observation x for which 4.13 is the largest
# example: if K = 2, and pi.1 = pi.2, then the Bayes classifier assigns an observation to class 1 if 2x(u1 - u1) > u1^2 - u2^2 and class 2 otherwise
# in this case the Bayes decision boundary corresponds to the point where:
        # x = (u1^2 - u2^2) / 2(u1 - u2) ======= u1 + u2 / 2 (4.14)

# example: we display two normal density functions that are displayed f1(x) and f2(x) represent two distinct classes
# the mean and variance parameters for the two density functions are u1 = -1.35, u2 = 1.25 and sigma.1 = sigma.2 = 1
# these two densities overlap so given that X = x there is some uncertainty about the class to which a observation belongs
# if we assume that an observation is equally likely to come from each class pi.1 = pi.2 = .5 - bayes classifier will assign to class 1 if x > 0 and class 2 otherwise
# in this case: we can compute the Bayes classifier because we know all of the parameters invlolved
# in a real life situation we would not be able to calculate the bayes classifier

# in practice even if we are quite certain of our assumption that X is drawn from a Gaussian distribution within each class...
# we would still have to estimate parameters u1...uk, pi.k ... pi.k, and sigma^2
# the LINEAR DISCRIMINANT ANALYSIS (LDA) approximates the Bayes classifier by plugging estimates for pi.k, u.k and simga^2 into 4.13
# we use the following estimates:
        # u.k = 1 / n.k sumof(xi)
        # sigma^2 = 1 / (n - K) sumof(k=1 to K) sumof(yi = k: xi - uk)^2
# n is the total number of training observations and nk is the number of training observations in the kth class
# the estiamte for u.k is simply te average of all the training observations from the kth class
# simga^2 can be seen as a weighted average of the sample variances for each of the K classes
# sometimes we have knowledge of the class membership probabilities pi.1 ... pi.k which we can use directly
# in the abscense of any additional information LDA estimates pi.k using the proportion of the training observations that belong to the kth class
        # pi.k = n.k / n (4.16)
# the LDA classifier plugs the estimates given in 4.15 and 4.16 into 4.13 and assigns an Observation  X = x to the class for which:
        # theta(X) = x * (u.k / sigma^2) - (u.k^2 / 2*sigma^2) + log(pi.k) is the LARGEST
# the word linear in the classifiers name stems from the fact that the discriminant functinos d(x) are linear functions of x

# to implement LDA we begin by estimating pi.k, u.k and sigma^2 using 4.15 and 4.16
# we then computed the deciison boundary by assigning the obsveration that maximies our linear discriminant function
# it is possible if an observation lands on the decision boundary to assign the value to two classes!
# the decision boundary corresponds to the midpoint between the sample means for the two classes (u1 + u2) / 2 = 0

# how well does our LDA do in this example?
# we can generate a large number of test observations in order to compute the Bayes error rate and the LDA error rate
# remember we are only able to use the Bayes method because we know all the prior probabilities before hand (made up test case to show difference between Bayes and LDA)
# Bayes test error rate = 10.6% 
# LDA test error rate = 11.1%
# the LDA classifier rate is only .5% above the smallest possible error rate! (BAYES IS THE GOLD STANDARD)
# this indicates that LDA is performing pretty well and approximates well to the Bayes classifier
# the LDA classifier results from from:
        # assuming the observations from each class come from a normal distribution
        # we have a class specific mean vector
        # common variance sigma^2
        # plugging these estimates into the Bayes classifier


## Linear Discriminant Analysis for p > 1
# we are now going to extend the LDA classifier to the case of multiple predictors
# to do this we will need to assume that X = (X1, X2, ... Xp) is drawn from a multivariate Guassian / normal distribution
# we also assume we have class specific mean vectors and a common covariance matrix

# the multivariate normal distribution assumes that each predictor follows a one-dimensional normal distribution
# we also assume there is some correleation between pairs of predictors
# to indicate that a p-dimension random variable X has a multivariate normal distribution we write:
        # X ~ N(u, E)
# here E(x) = u; this is the mean of X a vector with p components
# here Cov(X) = E is a covariance matrix p * p  of X
# formally the multivariate normal distribution is defined as:
        # f(x) = (1 / (2pi)^p/2|E|^1/2) * exp(-1/2(x - u)^T E^-1(x - u))

# in the case of p > 1 predictors the LDA classifier assumes that the observations in the kth class are drawn from a multivariate normal distribution
# this means X ~ N (u.k, E) where u.k is a class specific mean vector and E is a covariance matrix that is common to all K classes

# plugging in the density function for the kth class; f(X = x) into 4.10 reveals that the Bayes classifier assigns an observation X = x to the class which:
        # d(x) = x^T E^-1 uk - 1/2(u^T E^-1 uk + log(pi.k)) is THE LARGEST
# this is the vector / matrix version of 4.13

# this representation will now draw multiple decision boundaries that seperate each of the pairs of classes
# note that there are three lines representing the Bayes decision boundaries because there are three paris of classes amoung the three classes
# that is one Bayes decision boundaries divide the predictor space into three regions
        # one bayes decision boundary separates class 1 from class 2
        # one bayes decision boundary separates class 2 from class 3
        # one bayes decision boundary separates class 2 from class 3
# these three bayes decision boundaries divide the predictor space into three regions
# the bayes classifier will classify an observation according to the region in which it is located

# once again we will need to estimate the unknown parameters u1...uk, pi1 ... pik and E
# these formulas are similiar to the one dimensional LDA case
# to assign a new observation X = x LDA plugs in these estimates into 4.19 and classifies based on where d(x) is the largest!!
# Note again: d(x) is a linear function of x: the LDA decision rule depends on x only through a linear combination of its elements
# this is why linear discriminant analysis is LINEAR!!!

# LDA will closely approximate the Bayes classifier in the multi-dimensional space
# in our example LDA classifier is .0746 and the Bayes classifier is .0770
# note we would not be able to compute the exact Bayes classifier in real-life - we need to approximate it using LDA

# we can perform LDA on the Default data in order to predict whether an individual will default on the basis of BALANCE and STUDENT
# the LDA model fit to the 10,000 training samples results in a TRAINING ERROR of 2.75%
# this sounds great - really low!!! but...

# training error is usually always lower than testing error - we really want low testing error
# we might expect our classifier to perform worse on a new set of observations
# remember - we specifically adjust the parameters of our LDA model to do well on the training data
# the higher ratio of parameters p to number of samples n the more we expect this overfitting to play a role
# in our case we have two predictors and 10,000 samples so we do not suspect overfitting!

# second since only 3.3% of the training sample defaultedk, a simple but useless classifier that always predicts NO DEFAULT WILL RESULT IN A LOW 3.3% DEFAULT RATE!!
# in other words, the trivial null classifier (all predictions are NO DEFAULT) will achieve an error rate that is only a bit higher than the LDA training set error rate

# in practice a binary classifier can make type types of errors:
# it can incorrectly assign an individual who does not default as a defualt
# it can incorrecetly assign an individual who does default as a non-default
# we investigate these two types of errors using the CONFUSION MATRIX
# our confusion matrix table shows:
        # LDA predicted that 104 people will default: 81 were correct (actually DEFAULTED) and 23 were incorrect(predicted DEFAULT but did NOT)
        # LDA predicted that 9,896 people will not default = 252 of these were incorrectly assigned!!
# 252 / 333 (total true default) = 75.7%!!
# while the overall error rate may be low the error rate amoung individuals who truely defaulted is VERY HIGH!!
# a company trying to identify high-risk individuals an error rate of 75.7% amoung individuals who default may well be unacceptable

# class specific performance is also important
# we use terms sensitivity and specificity to characterize the performance of a classifer
# sensitivity = is the percentage of true defaulters that are identified = only 24.3% in this case (81 / 333)
# specificity = the percentage of non-defualters that are correctly identified = 99.8% in this case (9,644 / 9,667)

# why does LDA do a poor job classifying the customers who deafult?
# why does LDA have such a low sensitivity??
# LDA is trying to approximate the Bayes classifier = Bayes classifier has the lowest TOTAL ERROR RATE (if normal form correct)
# Bayes classifier will yield the smallest total number of misclassified observations - IT DOES NOT MATTER WHERE THE MISCLASSIFICATIONS COME FROM!!!
# some misclassifications will results from incorrectly assigning a customer who does not default to the default class
# others will result from incorrectly assigning a customer defaults to the non-default class
# in our example: company may want to focus on not incorrectly classifiying a person who will default vs. the alternatve possibility
# it is possible to modify LDA in order to develop a classifier that better focused on sensitivity or specificity!!

# the bayes classifier works by assigning an observation to the class for which the posterior probability pk(X) is the greatest
# in the two-class case this amounts to assigning an observation to the default class if:
        # Pr(deafult = Yes | X = x) > .5
# thus the Bayes classifier and by extension the LDA uses a threshold of 50% for the posterior probability in order to assign an observation as DEFAULT
# if we are concerned about incorrectly predicting the default status for individuals who default then we consider lowering this threshold
# we might label any customer with a posterior probability of default above 20% to the default class
# instead of assigning an observation to the default class if the .5 threshold holds we could then update our decision rule too:
        # Pr(default = Yes | X = x) > .2
# the new error rates are available in Table 4.5
# the LDA now predicts that 430 indivduals will defualt
# of the 333 individuals who default LDA correctly predicts all but 138 - 41.1%
# this is a vast improvement over the error rate of 75.7% that resulted from using the threshold of 50%
# however this improvement comes at a cost = now 235 individuals who do not default are incorrectly classified
# as a result our OVERALL error rate has increased to 3.7%
# in our case we may consider this slight increase in the total error rate to be a small price to pay for more accurate indentification of individuals who DO DEFAULT!!

# using a threshold at .5 minimizes the OVERALL error rate
# this is to be expected because the Bayes classifier uses a threshold of .5 and is known to have the lowest overall error rate
# but when a threshold of .5 is used - the sensitivity error (individuals who do default) is quite high
# as the threshold is reduced - the error rate amoung non-default classification increases - there is always a trade-off!
# how can we decide which threshold value is best?
# such decisions are based on domain knowledge - detailed information regarding our specific problem and the goals we want to achieve

# the ROC curve is a popular graphic for displaying the two types of erros for all possible thresholds!!
# ROC = Reciever Operating Characteristics
# the overall performance of a classifier summarized over all possible thresholds is given by the area under the ROC!!! (AUROC)
# the best ROC fits the top left hand corner perfectly!!
# the larger the AUROC the better performance!!!
# for our example the AUROC is .95 which is very close to the maximun threshold!!
# a chance classifier will be a diagnoal line with AUC of .5
# ROC curves allow us to plot multiple classifiers and compare performance

# varying the classifier threshold changes its true positive and false positive rate (sensitivity and specificity)
# sensitivity is 1 - the specificity of our classifier
# summary table of metrics in classification:
        # False Positive Rate = FP / N = Type 1 Error, 1 - Specificity
        # True Positive Rate = TP / P = 1 - Type 2 Error, power, sensitivity, recall
        # Positive Predictive Value = TP / P(predicted) = Precision, 1 - false discovery proportion
        # Negative Predictive Value = TN / N(predicted)

## Quadratic Discriminant Analysis
# LDA assumes that the observations within each class are drawn from a multivariate normal distrbution
# this means there is a specific class mean vector and a covariance matrix common to all K
# Quadratic Discriminant Analysis (QDA) provides an alternative approach
# like LDA, QDA classifier results from assuming that the observations from each class are drawn from the multivariate normal distribution
# QDA will also plug in parameter estimates to approximate Bayes classifier in order to perform prediction
# unlike LDA, QDA assumes that each class has its own covariance matrix
# it assumes that an observation from the kth class is of the form X ~ N(u.k, E.k)
# this is a specific covariance matrix for each class
# under this assumption the bayes classifier assign an observation X = x to class for which the following formula is the largest!!
        # d(x) = 
        # -1/2(x - uk)^T Ek^-1(x - uk) - 1/2 * log(|Ek|) + log(pi.k)
# QDA classifier involves pluggin in estimates for Ek, uk, and pi.k into the above
# it then assigns an observation X = x to the class for which this formula is the largest
# the quantity x appears as a quadratic function instead of a linear function

# why does it matter whether or not we assume that the K classes share a common covariance matrix?
# why LDA vs. QDA??
# the answer lies in the BIAS-VARIANCE TRADE-OFF!!!
# when there are p predictors - estimating a covariance matrix requires estimating p(p+1)/2 parameters = LDA
# QDA estimates a separate covariance matrix for each class = Kp(p+1)/2 parameters!!
# with 50 predictors is some multiple of 1,275 parameters to estimate
# by assuming that the K classes have a common covariance matrix the LDA model becomes LINEAR!!!
# this means there are Kp linear coefficients to estimate
# LDA is a much less flexible classifier than QDA = meaning LDA has lower variance
# but there is a trade off - if the K classes do not share a common covariance matrix then LDA can suffer from high bias!!
# LDA tends to be a better bet than QDA if there are few training observations - we need to reduce the variance!
# QDA is reccomended if the training set is very large so that the variance of each classifier is not a major concern
# QDA is also better if the common covariance matrix across all classes is completely asinine



## Comparison of Classification Methods
# in this chapter we have seen three different classification statistical learning methods
# logistic regression, LDA, QDA
# in chapter 2 we discussed the KNN classification method (KNN can be used for classification and regression)
# which scenario works best for which method?

# logistic regression and LDA are closely related
# consider a two class setting with 1 predictor
# let p1(x) and p2(x) = 1 - p1(x) be the probabilities that the observation X = x belongs to class 1 and class 2 respectively
# in the LDA framework we can compute the log odds (like logistic regression)
        # log(p1(x) / 1 - p1(x)) = log(p1(x) / p2(x)) = c0 + c1(x)
# where c0 and c1 are functions of u1, u2 and signma^2
# we know that logistic regression:
        # log(p1 / 1 - p1) = B0 + B1x
# both are linear functions of x
# both logistic regression and LDA produce linear decision boundaries!!!
# the only difference is that in logistic regression B0 and B1 are estimated through maximum liklihood function
# in LDA c0 and c1 are computed using estimated mean and variance from a normal distribution (approximating the bayes classifier)
# this same connect holds for multidimensional data with predictors > 1

# since logistic regression and LDA only differ in fitting procedures we might expect they have similiar results
# this is often the case but not always
# LDA assumes the observations are drawn from the normal distribution with a common variance matrix
# LDA can outperform logistic regression if we know that the observations are drawn from a normal distribution
# conversely logistic regression can outperform LDA if guassian and shared varaince matrix assumption are not met with the form of the data

# KNN takes a completely different approach from the classifiers LDA and logistic regression
# in order to make a prediction for X = x, the K training observations that are closest to x are identified
# the X is assigned to the class to which the plurality of these observations belong
# KNN is a completely non-parameteric approach: no assumptions are made about the SHAPE OF THE DECISION BOUNDARY OR FORM OF UNDERLYINH DATA
# we can expect KNN to outperform LDA and logistic regression when the decision boundary is NOT LINEAR
# however KNN does not tell us which predictors are important in predicting the x's class = we do not get a coefficients table

# QDA serves as a compromise between the non-parametric KNN method and the linear LDA and logistic regression approaches
# since QDA assumes a quadratic decision boundary it can accurately model a wider number of training observations than linear models
# though not as flexible as KNN -  QDA can perform better in the presence of a limited number of traiing observations - there is no assumption about the form of the decision boundary

# example of all in action:
# data = six scenarios:
# 3 of the scanarios the decision boundary is linear the other three is non-linear
# for each scenario we produce 100 random training data sets
# we then fit each of our methods to the data and then predict to get the test error
# in each scenario there are two predictors
# KNN was chosen with K = 1 and with a cross-validation K


# Scenario 1:
# there were 20 training observations in each of the two classes
# the observations within each class were uncorrelated random normal variables with a different mean in each class
# our figure shows LDA performed well in this setting as we have linear data = linear decision boundary
# KNN performed poorly because it generated variance that was not accounted for in a reduction of bias
# QDA performed worse that LDA in TEST ERROR since it fit a more flexible model than necessary
# logistic regression also uses a linear decision boundary and results were only slightly worse than LDA

# Scenario 2:
# details are the same as scenario 1 except:
# within each class the two predictors had a correlation of -.5
# there is little change to the performace of our models in this case

# Scenario 3:
# we generated X1 and X2 from the t distribution with 50 observations per class
# the t distribution has a similiar shape to the normal but has a tendancy to yeild more extreme results
# in this setting our decision boundary is still linear!
# this fits the logistic regression framework!!
# this set-up violated the assumptions of LDA = observations were not drawn from a normal distribution
# now logistic regression outperforms LDA in TEST ERROR
# both LDA and logistic regression were better than the other methods
# QDA results quickly deteriorated as a consequence of non-normality


# Scenario 4:
# data generated from a normal distribution with correlation of .5 between predictors in the first class and -.5 correlation between predictors in the second class
# this setup matches the QDA assumption = quadratic decision boundaries!
# because of this QDA outperformed all other methods!!

# Scenario 5:
# within each class the observations were generated from a normal distribution with uncorrelated predictors
# however the responses were sampled from the logistic function using X1^2 and X2^2 and X1 * X2 as predictors
# becuase of this we have another quadratic decision boundary
# QDA performs the best followed closely by KNN cross validated choice of K
# linear model perform poorly as expected with this underlying form of the data

# Scenario 6: 
# details are the same as the previous scenario
# except the responses were sampled from a more complicated non-linear function
# as a result even the flexible QDA could not fit the complex function = quadratic decision boundaries could not model the data
# QDA did slightly better than the linear methods (LDA, logsistic)
# the much more flexilbe KNN_cv method gave the best results
# but KNN with K = 1 gave the worse results!!
# this highlights that even with a non-linear relationship a non-parametric method such as KNN can still give porr results if we do not tune the method properly

# these examples show that no one method will outperform the others in every situation
# when the true decision boundaries are linear  then LDA and logistic will do the best
# when boundaries are slightly non-linear QDA may give better results
# for more complicated decision boundaries a non-parametric approach like KNN can be superior
# but the level of smoothness for a non-parametric approach must be choosen carefully
# next chapter will show how to choose the correct level of smoothness and help choose the best method overall

# finally recall from chapter 3 that in a regression setting we can accommodate a non-linear relationship between response and predictors by transformations of the predictors
# WE CAN DO THIS WITH CLASSIFICATION AS WELL!
# we can fit a logistic regression by including X^2, X^3 and X^4 as predictors!!
# this gives us a more flexible logistic regression statistical learning method!!
# however - this may or may not improve performance! the VARIANCE- BIAS tradeoff
# will an increase in variance due to added flexibility be offset by a sufficiently large reduction in bias??
# WE COULD ALSO DO THE SAME FOR LDA!!!!
# if we added all possible quadratic terms to the LDA model the form would be the same as QDA!!!
# but we would have different parameter estimates (LDA = common variance matrix, QDA = unique variance matrix per p)



## Chapter 4: Lab
# Logistic Regression, LDA, QDA, KNN

## Stock Market data
# view the summary
library(ISLR)
names(Smarket)

# [1] "Year"      "Lag1"      "Lag2"      "Lag3"      "Lag4"      "Lag5"      "Volume"    "Today"    
# [9] "Direction"

dim(Smarket)
# [1] 1250    9

summary(Smarket)
# Year           Lag1                Lag2                Lag3                Lag4          
# Min.   :2001   Min.   :-4.922000   Min.   :-4.922000   Min.   :-4.922000   Min.   :-4.922000  
# 1st Qu.:2002   1st Qu.:-0.639500   1st Qu.:-0.639500   1st Qu.:-0.640000   1st Qu.:-0.640000  
# Median :2003   Median : 0.039000   Median : 0.039000   Median : 0.038500   Median : 0.038500  
# Mean   :2003   Mean   : 0.003834   Mean   : 0.003919   Mean   : 0.001716   Mean   : 0.001636  
# 3rd Qu.:2004   3rd Qu.: 0.596750   3rd Qu.: 0.596750   3rd Qu.: 0.596750   3rd Qu.: 0.596750  
# Max.   :2005   Max.   : 5.733000   Max.   : 5.733000   Max.   : 5.733000   Max.   : 5.733000  
# Lag5              Volume           Today           Direction 
# Min.   :-4.92200   Min.   :0.3561   Min.   :-4.922000   Down:602  
# 1st Qu.:-0.64000   1st Qu.:1.2574   1st Qu.:-0.639500   Up  :648  
# Median : 0.03850   Median :1.4229   Median : 0.038500             
# Mean   : 0.00561   Mean   :1.4783   Mean   : 0.003138             
# 3rd Qu.: 0.59700   3rd Qu.:1.6417   3rd Qu.: 0.596750             
# Max.   : 5.73300   Max.   :3.1525   Max.   : 5.733000 


# examine correlation
# cor() function provides the correlation matrix of all the predictors
cor(Smarket[,-9])
# Year         Lag1         Lag2         Lag3         Lag4         Lag5      Volume        Today
# Year   1.00000000  0.029699649  0.030596422  0.033194581  0.035688718  0.029787995  0.53900647  0.030095229
# Lag1   0.02969965  1.000000000 -0.026294328 -0.010803402 -0.002985911 -0.005674606  0.04090991 -0.026155045
# Lag2   0.03059642 -0.026294328  1.000000000 -0.025896670 -0.010853533 -0.003557949 -0.04338321 -0.010250033
# Lag3   0.03319458 -0.010803402 -0.025896670  1.000000000 -0.024051036 -0.018808338 -0.04182369 -0.002447647
# Lag4   0.03568872 -0.002985911 -0.010853533 -0.024051036  1.000000000 -0.027083641 -0.04841425 -0.006899527
# Lag5   0.02978799 -0.005674606 -0.003557949 -0.018808338 -0.027083641  1.000000000 -0.02200231 -0.034860083
# Volume 0.53900647  0.040909908 -0.043383215 -0.041823686 -0.048414246 -0.022002315  1.00000000  0.014591823
# Today  0.03009523 -0.026155045 -0.010250033 -0.002447647 -0.006899527 -0.034860083  0.01459182  1.000000000

# correlations between our lag variables and today's returns are almost 0
# the only substantial correlation is between Year and Volume predictors
# plotting volume shows that volume of trading has increased over time
plot(Smarket$Volume)

## Logistic Regression
# fit a logistic regression in order to predict DIRECTION of the stock market
# we will use Lag1:Lag5 and VOLUME as our predictors
# the glm() function fits all generalized linear models including logistic regression
# we must specifiy family = "binomial" in our call to glm()

glm.fit <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
               data = Smarket, family = "binomial")
summary(glm.fit)
# Call:
#         glm(formula = Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + 
#                     Volume, family = "binomial", data = Smarket)
# 
# Deviance Residuals: 
#         Min      1Q  Median      3Q     Max  
# -1.446  -1.203   1.065   1.145   1.326  
# 
# Coefficients:
#         Estimate Std. Error z value Pr(>|z|)
# (Intercept) -0.126000   0.240736  -0.523    0.601
# Lag1        -0.073074   0.050167  -1.457    0.145
# Lag2        -0.042301   0.050086  -0.845    0.398
# Lag3         0.011085   0.049939   0.222    0.824
# Lag4         0.009359   0.049974   0.187    0.851
# Lag5         0.010313   0.049511   0.208    0.835
# Volume       0.135441   0.158360   0.855    0.392
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1731.2  on 1249  degrees of freedom
# Residual deviance: 1727.6  on 1243  degrees of freedom
# AIC: 1741.6
# 
# Number of Fisher Scoring iterations: 3

# the smallest pvalue is associated with Lag1
# the negative correlation coefficient suggests that a positive return yesterday would lead to a negative return today
# however the Lag1 pvalue is not significant at the alpha = .05 level

# we will use the coef() function in order to access just the coefficients of the model
coef(glm.fit)
# (Intercept)         Lag1         Lag2         Lag3         Lag4         Lag5       Volume 
# -0.126000257 -0.073073746 -0.042301344  0.011085108  0.009358938  0.010313068  0.135440659 

# summary table of coefficients
summary(glm.fit)$coef
#               Estimate Std. Error    z value  Pr(>|z|)
# (Intercept) -0.126000257 0.24073574 -0.5233966 0.6006983
# Lag1        -0.073073746 0.05016739 -1.4565986 0.1452272
# Lag2        -0.042301344 0.05008605 -0.8445733 0.3983491
# Lag3         0.011085108 0.04993854  0.2219750 0.8243333
# Lag4         0.009358938 0.04997413  0.1872757 0.8514445
# Lag5         0.010313068 0.04951146  0.2082966 0.8349974
# Volume       0.135440659 0.15835970  0.8552723 0.3924004


# we can use the predict function to give the probability that the market will go up (DIRECTION) based on our model and its predictors
# we will need to specifiy type = "response"
# if no data set is supplied to the predict function - R will give predictions for the fitting dataset
# the values given are probabilities of the stock market going up!
# we can check this using the contrasts function to determine the correct interpretation of the dummy variables
glm.probs <- predict(glm.fit, type = "response")
glm.probs[1:20]
# 1         2         3         4         5         6         7         8         9        10        11 
# 0.5070841 0.4814679 0.4811388 0.5152224 0.5107812 0.5069565 0.4926509 0.5092292 0.5176135 0.4888378 0.4965211 
# 12        13        14        15        16        17        18        19        20 
# 0.5197834 0.5183031 0.4963852 0.4864892 0.5153660 0.5053976 0.5319322 0.5167163 0.4983272

# this means our predicted probabilities are referenced in terms of up
# an observation with prediction of .51 means there is a .51 probability of the stock market going up that day
contrasts(Smarket$Direction)
# Up
# Down  0
# Up    1

# we can convert our model results to make this easier for a human to read
glm.pred <- rep("Down", 1250)
glm.pred[glm.probs > .5] = "Up"

# the first command creates a vector of 1,250 Down elements
# the second line transforms to Up all of the elements for which the predicted probability is greater than .5

# we can use the table() function to produce a confusion matrix in order to determine how many observations were correctly or incorrectly classified
table(glm.pred, Smarket$Direction)
# glm.pred Down  Up
# Down  145 141
# Up    457 507

# find the total correctly predicted
cm <- table(glm.pred, Smarket$Direction)

# accuracy
(cm[1,1] + cm[2,2]) / nrow(Smarket)
# [1] 0.5216

# another version of accuracy
# take the average of predictions that are equal to the correct DIRECTION
mean(glm.pred == Smarket$Direction)
# [1] 0.5216

# the diagnoal elements of the matrix indicate correct predictions
# the off-diagnoal represent the incorrect predictions
# our model correctly predicted that the market would go up on 507 days and that it would go down 145 days
# this equates to 652 correct predictions out of 1250 observations

# we use the mean function to compute the fraction of days for which the prediction was correct
# in this case our logistic regression correctly predicted the movement of the market 52.2% of the time

# is this better than random guessing?
# is this this the case? 52% > 50%??
# THIS IS MISLEADING = TRAINING ACCURACY IS NOT EQUAL TO TESTING ACCURACY
# we trained and tested  our model on the same set of data!!
# our TRAINING ERROR rate is 1 - accuracy
# AS WE KNOWN TRAINING ERROR IS OFTEN OVER OPTIMISTIC - IT TENDS TO UNDERESTIMATE THE TESTING ERROR!!
# we need to use TEST and TRAINING datasets to accuracy assess our model!!

# training error
1 - mean(glm.pred == Smarket$Direction)
# [1] 0.4784
# training error
mean(glm.pred != Smarket$Direction)
# [1] 0.4784


# test and training datasets
train = Smarket %>% 
        as.data.frame() %>% 
        filter(Year < 2005)

dim(train)
# [1] 998   9

test = Smarket[!Smarket$Year %in% train$Year,]
dim(test)
# [1] 252   9

# we filter our test and training to data before 2005 to build our model on = we will then test it on post 2005 data
# cleaner code
train = (Smarket$Year < 2005) # vector with Year less than 2005

# filter smarket to only the observations less than 2005
Smarket.2005 =Smarket[!train,]
dim(Smarket.2005)
# [1] 252   9

# create worded direction variable on the holdout data set
Direction.2005 <- Smarket$Direction[!train]


# we can now fit logistic regression model using only the training data
# we will then test our fitted model on the training set against the test set!!
# we need to re-take our model through all the previous steps!

glm.fit2 <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
                data = Smarket, family = "binomial", subset = train)
summary(glm.fit2)
# Call:
#         glm(formula = Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + 
#                     Volume, family = "binomial", data = Smarket, subset = train)
# 
# Deviance Residuals: 
#         Min      1Q  Median      3Q     Max  
# -1.302  -1.190   1.079   1.160   1.350  
# 
# Coefficients:
#         Estimate Std. Error z value Pr(>|z|)
# (Intercept)  0.191213   0.333690   0.573    0.567
# Lag1        -0.054178   0.051785  -1.046    0.295
# Lag2        -0.045805   0.051797  -0.884    0.377
# Lag3         0.007200   0.051644   0.139    0.889
# Lag4         0.006441   0.051706   0.125    0.901
# Lag5        -0.004223   0.051138  -0.083    0.934
# Volume      -0.116257   0.239618  -0.485    0.628
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1383.3  on 997  degrees of freedom
# Residual deviance: 1381.1  on 991  degrees of freedom
# AIC: 1395.1
# 
# Number of Fisher Scoring iterations: 3

coef(glm.fit2)
# (Intercept)         Lag1         Lag2         Lag3         Lag4         Lag5       Volume 
# 0.191212621 -0.054178292 -0.045805334  0.007200118  0.006440875 -0.004222672 -0.116256960 

summary(glm.fit2)$coef
# Estimate Std. Error     z value  Pr(>|z|)
# (Intercept)  0.191212621 0.33368991  0.57302488 0.5666278
# Lag1        -0.054178292 0.05178534 -1.04620896 0.2954646
# Lag2        -0.045805334 0.05179687 -0.88432632 0.3765201
# Lag3         0.007200118 0.05164416  0.13941784 0.8891200
# Lag4         0.006440875 0.05170561  0.12456821 0.9008654
# Lag5        -0.004222672 0.05113838 -0.08257344 0.9341907
# Volume      -0.116256960 0.23961830 -0.48517564 0.6275518

# predict new model onto the TEST SET
glm.probs2 <- predict(glm.fit2, type = "response", newdata = Smarket.2005)

# human readable format
glm.pred2 <- rep("Down", 252)
glm.pred2[glm.probs2 > .5] = "Up"

# confusion matrix
table(glm.pred2, Direction.2005)
# Direction.2005
# glm.pred2 Down Up
# Down   77 97
# Up     34 44

# accuracy
mean(glm.pred2 == Direction.2005)
# [1] 0.4801587

# misclassification
mean(glm.pred2 != Direction.2005)
# [1] 0.5198413

# these results fucking suck!
# the test error rate is 52% which is worse than random guessing!
# this is not suprising - how could we predict todays price using yesterday's price?

# recall that our initial model had no significant pvalues for any coefficient
# the smallest being Lag1 which even then was not that small
# maybe by removing the variables that appear not to be helpful we can get a better model?
# predictors that have no relationship with repsonse tend to deteriorate the TEST ERROR!!
# more predictors cause an increase in variance without a corresponding decrease in bias

# let's refit the model will the "most important" variables

# fit the model
glm.fit3 <- glm(Direction ~ Lag1 + Lag2, 
                data = Smarket, family = binomial, 
                subset = train)

# predict with new model
glm.probs3 <- predict(glm.fit3, type = "response", newdata = Smarket.2005)

# create human readable format
glm.pred3 <- rep("Down", 252)
glm.pred3[glm.probs3 > .5] = "Up"

# confusion matrix
table(glm.pred3, Direction.2005)
# Direction.2005
# glm.pred3 Down  Up
# down   35  35
# Up     76 106

# test error metrics
mean(glm.pred3 == Direction.2005)
# [1] 0.5595238

mean(glm.pred3 != Direction.2005)
# [1] 0.4404762


# the results now seem better: 56% of the daily movements have been recorded correctly
# but wait - predicting the market will increase every day  will also be correct around 56% of the time!!
# in terms of overall error rate our new model is better than the first but may not hold up to scrutiny
# our logistic regression in this case is no better than a naive approach
# our confusion matrix shows that on days when logistic regression predicts an increase in the market it has a 58% accuracy
# this suggests possible trading strategy of buying on days when the model predicts an increasing market and avoid days when a decrease is predicted
# we would need to dive deeper to decide if this could happen just by chance

# suppose that we want to predict the return associated with particular values of Lag1 and Lag2
# in particular we want to predict Direction on a day when Lag1 and Lag2 equal 1.2 and 1.1 respectively and also on a day when they are 1.5 and -.8
# we can do this using the predict functino

predict(glm.fit3, type = "response",
        newdata = data.frame(
                Lag1 = c(1.2, 1.5), 
                Lag2 = c(1.1, -.8)
                ))
# 1         2 
# 0.4791462 0.4960939



## Linear Discriminant Analysis
# now we will perform LDA on the same Smarket data
# we fit an lda model using the lda() function which is apart of the MASS library
library(MASS)

# fit model
lda.fit <- lda(Direction ~ Lag1 + Lag2, data = Smarket, 
               subset = train)
lda.fit

# Call:
#         lda(Direction ~ Lag1 + Lag2, data = Smarket, subset = train)
# 
# Prior probabilities of groups:
#         Down       Up 
#       0.491984 0.508016 
# 
# Group means:
#         Lag1        Lag2
# Down  0.04279022  0.03389409
# Up   -0.03954635 -0.03132544
# 
# Coefficients of linear discriminants:
#         LD1
# Lag1 -0.6420190
# Lag2 -0.5135293

# view data
plot(lda.fit)

# what do these results mean?
# the LDA output indicates that pi1 = .492 and pi2 = .508
# this means 49% of the observations show the market going down
# it also provides the group means - the average of each predictor within each class (UP or DOWN)
# THESE ARE USED AS THE LDA ESTIMATES OF Uk to approximate to the BAYES CLASSIFIER

# we need to use the coefficients of linear discriminants output to provide the combinations used for the LDA decision rule
# these are the multipliers of the elements of X = x
# if -.642*Lag1 - -.514*Lag2 is large then the LDA classifier will predict a market increase
# if this function is small the LDA classifier will predict a market decrease

# the plot function provides a plot of the linear discriminants
# this is obtained by calculating our multiplier function for every training set observation

# the predict function returns a list of three elements
# the first elements, class = contains the LDA predictions about the movement of the stock market
# the second element, posterior = is a matrix whose kth column contains the posterior probability that the corresponding observation belongs to the kth class
# the last element, x = contains the linear discriminants (the multipliers)

lda.pred <- predict(lda.fit, Smarket.2005)
names(lda.pred)
# [1] "class"     "posterior" "x"

# as we observed in chapter the LDA and logistic regression predictions are almost identical
lda.class <- lda.pred$class

# view results
table(lda.class, Direction.2005)
# Direction.2005
# lda.class Down  Up
# Down   35  35
# Up     76 106

# we apply the 50% threshold to the posterior probabilities allows us to recreate the posterior probabilities built in the LDA model
sum(lda.pred$posterior[,1] >= .5)
# [1] 70

sum(lda.pred$posterior[,1] < .5)
# [1] 182

# notice that the posterior probability output by the model corresponds to a the probability the market WILL DECREASE!!
lda.pred$posterior[1:20,1]
# 999      1000      1001      1002      1003      1004      1005      1006      1007      1008      1009 
# 0.4901792 0.4792185 0.4668185 0.4740011 0.4927877 0.4938562 0.4951016 0.4872861 0.4907013 0.4844026 0.4906963 
# 1010      1011      1012      1013      1014      1015      1016      1017      1018 
# 0.5119988 0.4895152 0.4706761 0.4744593 0.4799583 0.4935775 0.5030894 0.4978806 0.4886331

lda.class[1:20]
# [1] Up   Up   Up   Up   Up   Up   Up   Up   Up   Up   Up   Down Up   Up   Up   Up   Up   Down Up   Up  
# Levels: Down Up

# if we wanted to use the posterior probability threshold other than 50% in order to make predictions we could easily do so
# suppose we want to predict a market decrease only if we are very certain that the market will indeed decrease that day
# the posterior probability could be set at a threshold of .9
sum(lda.pred$posterior[,1] > .9)
# [1] 0

# oh shit - no days in 2005 meet this strict threshold!!
# the greatest posterior probability of decrease in all of 2005 is only 52%!!


## Quadratic Discriminant Analysis
# we will now fit a quadratic discriminant analysis to our Smarket data
# QDA is implemented in R using the qda() function - which is also a part of the MASS library
qda.fit <- qda(Direction ~ Lag1 + Lag2, data = Smarket,
               subset = train)
qda.fit
# Call:
#         qda(Direction ~ Lag1 + Lag2, data = Smarket, subset = train)
# 
# Prior probabilities of groups:
#         Down       Up 
# 0.491984 0.508016 
# 
# Group means:
#         Lag1        Lag2
# Down  0.04279022  0.03389409
# Up   -0.03954635 -0.03132544

# the output contains the group means
# we do not see the linear discriminant multipliers because qda is a quadratic function
# the predict function works exactly the same way as lda
qda.class <- predict(qda.fit, Smarket.2005)$class

table(qda.class, Direction.2005)
# Direction.2005
# qda.class Down  Up
# Down   30  20
# Up     81 121

# performance metric
mean(qda.class == Direction.2005)
# [1] 0.5992063

# qda predictions are accurate almost 60% of the time
# this level of accuracy is impressive for stock market analysis
# this suggests that using the quadratic form more closely represents the true form of the stock market data
# the relationship between direction and our predictors are more accuracy modeled in this case
# QDA performed better than LDA and logistic regression!
# we should recommend evaluation this model's performance on a larger test set before betting that this approach will consistently beat the market!!

## K Nearest Neighbors
# we will now perform a KNN fit using knn() which is a part of the class library
# this function works slightly different than our tradiitonal lm, glm model fit calls
# knn will perform predictions in one single line
# the knn function requires four inputs:
        # a matrix containing the predictors associated with the training data
        # a matrix containing the predictors asscociated with the prediction dataset (observations we want to predict)
        # a vector containing the class labels for the training observations
        # a value for K the number of nearest neighbors to be used by the classifier

# we will use cbind() function to bind the Lag1 and Lag2 variables into two matricies - one for training and one for test

# knn package
library(class)

# define training and test set matricies
train.X <- cbind(Smarket$Lag1, Smarket$Lag2)[train,]

test.X <- cbind(Smarket$Lag1, Smarket$Lag2)[!train,]

# set up names vector
train.Direction  <- Smarket$Direction[train]


# we now have all the peices to use knn to predict the stock market direction
# we need to set the seed before we apply knn() - we need this for reproducibility
set.seed(1)

# train the knn model and predict all in one step
# start with K = 1 as initial test case
knn.pred <- knn(train.X, test.X, train.Direction, k = 1)

# view knn results
table(knn.pred, Direction.2005)
# Direction.2005
# knn.pred Down Up
# Down   43 58
# Up     68 83

# performance metric
mean(knn.pred == Direction.2005)
# [1] 0.5

# the results using K = 1 are not very good since only 50% of the observations are correctly predicted
# it may be that K = 1 results in an overly flexible fit to our data
# let's re-fit our model with K = 3

# new model fit with k = 3
knn.pred = knn(train.X, test.X, train.Direction, k = 3)

# view the knn results
table(knn.pred, Direction.2005)
# Direction.2005
# knn.pred Down Up
# Down   48 54
# Up     63 87

# performance metric
mean(knn.pred == Direction.2005)
# [1] 0.5357143

# our results with using more "nearest neighbors" have improved our model slightly
# it turns out that increasing K actually deteriorates our results 
# from this it appears that QDA provides the best results of the methods we have examined so far


## Application to the Caravan Insurance Data
# in our final test case we will apply the KNN approach to the Caravan data set
# we will aim to predict Purchase - whether a person will buy a Caravan policy
# in this dataset only 6% of people actually purchases a policy
data("Caravan")
dim(Caravan)
# [1] 5822   86

summary(Caravan$Purchase)
# No  Yes 
# 5474  348

# view the % of yes
mean(Caravan$Purchase == "Yes")
# [1] 0.05977327

# because the KNN classifier predicts the class of a given test observation by indentifying the observations nearest to it - THE SCALE MATTERS!
# THE SCALE OF OUR VARIABLES MATTER!!
# any variables that are on a large scale will have a much larger effect on the distance between the observations and deeply effect our KNN classifier
# distance is key in our KNN classification framework
# consider a dataset with two variables salary and age (measured in dollars and years)
# KNN will think that a difference of $1,000 in salary is enormous compared to a difference of 50 years in age (real life is almost the opposite)
# the importance of scale reveals another important issue: if we measures salary in yen or age in minutes - our KNN results will be wildly different
# SCALE MATTERS TO A KNN CLASSIFIER

# a good way to handle this problem is to standardize the data so that all variables are given a mean of zero and sd of 1
# this puts all of our variables on a comparable scale
# we can use the scale() function to do this
# however we can only scale() quantitative values

# scale
standardized.X <- scale(Caravan[,-86])

# check results - scaled data will have mean of 0 and standard deviation of 1
var(Caravan[,1])
# [1] 165.0378

var(Caravan[,2])
# [1] 0.1647078

var(standardized.X[,1])
# [1] 1

var(standardized.X[,2])
# [1] 1

# now every column of standardized.X has a standard deviation of 1 and a mean of 0
# we now need to split our standardized data into test and training datasets

# test will be the first 1000 observations
test = 1:1000

# training will be everything but test
train.X <- standardized.X[-test,]

# test will be the first 1000 observations
test.X <- standardized.X[test,]

# we will also need our split repsonse variables to use in the knn function
train.Y <- Caravan$Purchase[-test]
test.Y <- Caravan$Purchase[test]

# set seed for reproducibility
set.seed(2)

# fit knn model on training data set
# remember we need a training x matrix, and testing X matrix, and a training Y vector with classes to traing on
# also need to give the number of nearest neighbors we would like to use in the set for class prediction
knn.pred <- knn(train.X, test.X, train.Y, k = 1)

# view results
mean(test.Y != knn.pred)
# [1] 0.116

mean(test.Y != "No")
# [1] 0.059


# our KNN error rate on our 1000 test observations is just under 12%
# at first glance this may seem great!
# but...we could get a 6% error rate by always predicting NO regardless of our predictors!!

# consider a real life application:
# there is a cost involved to sell insurance to a person
# the success rate of only 6% may be far too low to cover the costs involved
# we want to target insurance to customers who are likely to buy it - our overall rate is not of interest!!
# we want to investigate the fraction of people that are correctly predicted to buy insurance is of interest

# it turns out that KNN with K = 1 does far better than random guessing among the customers that are predicted to buy insurance
# amoung 77 such customers 9 or 11.7% actually do purchase
# this is double the rate we would get with random guessing

table(knn.pred, test.Y)
# test.Y
# knn.pred  No Yes
# No  872  49
# Yes  69  10

tab <- table(knn.pred, test.Y)

tab[2,2] / (tab[2,1] + tab[2,2])
# [1] 0.1265823

# using K = 3 our success rate increases to 19%
# using K = 5 our success rate increases to 26.7%
# it appears that KNN is finding some real patterns in a difficult data set!!

# nearest neighbor set K = 3
knn.pred <- knn(train.X, test.X, train.Y, k = 3)
tab <- table(knn.pred, test.Y)
# test.Y
# knn.pred  No Yes
# No  920  54
# Yes  21   5
 tab[2,2] / (tab[2,1] + tab[2,2])
 # [1] 0.2

 # nearest neighbor set K = 5
 knn.pred <- knn(train.X, test.X, train.Y, k = 5)
 tab <- table(knn.pred, test.Y)  
 # test.Y
 # knn.pred  No Yes
 # No  930  55
 # Yes  11   4
 
 tab[2,2] / (tab[2,1] + tab[2,2])
 # [1] 0.2666667

# for comparison let's also fit a logistic regression model to the same data
# if we use .5 as the threshold for a classifier we have a big fucking problem...
# only 7 of the test observations are predicted to purchase insurance - and our predictions are wrong about all of these!
# WE DO NOT HAVE TO USE A THRESHOLD OF .5!!!
# if we instead predict a purchase any time the predicted probability of purchase exceeds .25 we get much better results
# we predict 33 people will purchase insurance and we are correct about 33% of these people
# this is over five times better than random guessing!!

 # fit logistic regression
glm.fit <- glm(Purchase ~ ., data = Caravan, family = binomial, subset = -test)

# predict with logistic regression
glm.probs <- predict(glm.fit, Caravan[test,], type = "response")

# get list of class names to easily read logistic regression output
glm.pred <- rep("No", 1000)

# flag can prediction with a probability of greater than .5 with Yes
glm.pred[glm.probs > .5] = "Yes"

# view results of the .5 threshold
table(glm.pred, test.Y)
# test.Y
# glm.pred  No Yes
# No  934  59
# Yes   7   0

# re-run results with the lower threshold for YES purchase insurance
glm.pred[glm.probs > .25] = "Yes"

# view results
t <- table(glm.pred, test.Y)
# test.Y
# glm.pred  No Yes
# No  919  48
# Yes  22  11

# metric
t[2,2] / (t[2,1] + t[2,2])
# [1] 0.3333333





## Exercises

## Conceptual:


## Question 1:
# show that the logistic function representation and the logit representation for the logistic regression model are the same

# solution:

# logistic function = p(X) = e^(b0 + b1X) / 1 + e^(b0 + b1X)
# logit function = p(X) / 1 - p(X) = e^(B0 + B1X)

# we can manipulate the first formula to equal the second - the logistic function and logit function for logistic regression are the same


## Question 2:
# classifying an observation to the class for which 4.12 is the largest is equivalanet to classifiying an observation for which 4.13 is the largest
# prove this is the case for both classes belonging to a normal distribution
# the Bayes classifier assigns an observation to the class for which the discriminant function is maximized

# solution:

# Assuming that fk(x)is normal, the probability that an observation x is in class k is given by:

        # pk(x) = πk12π√σexp(−12σ2(x−μk)2) / ∑πl12π√σexp(−12σ2(x−μl)2)

# while the discriminant function is given by :

        # δk(x) = xμk / σ2 − μ^2k /2σ2+ log(πk)

# we find that maximizing pk is the same as maximizing dk


## Question 3:
# this relates to the QDA model in which observations within each class are drawn from a normal distribution with:
# class specific mean and class specific covariance matrix
# we consider a simple model with p = 1
# prove that the bayes classifier is not linear in this case - argue that is in fact quadratic

# solution
# if an observation belongs to the Kth class then X comes from a one-dimensional normal distribution X ~ N(uk, sigma^2k)
# after taking the log of both sides we find that in this case d(x) is a quadratic function of x


## Question 4: 
# when the number of features is large we tend to see performance degrade in KNN classifier
# we also see performance downgrades in other "local" classifiers that perform prediction using only observations "close" to the test observation
# this is known as the curse of dimensionality = non-parametric methods perform poorly when p is large
# we will investigate this curse in the next few questions:

# a. 
# suppose that we have a set of test observations each with measurements on p = 1 feature X
# we assume that X is uniformly distributed on [0,1]
# we have a response value with each observation
# suppose we wish to predict a test observations repsonse using only observations that are within 10% of the range of X closest to the test observation
# for example: to predict X = .6 we will only use observations in the range of [.55, .65]
# on average - what fraction of the available observations will we use to make the prediction?

# solution:
# on average we will only use 10% of the available observations to predict a new X
# this will obviously change if we predict at the extremes of X: [X < .05, X > .95] we do not have a 10% sample

# b.
# now suppose we have a set of obs each with measaurement on p = 2 features, X1 and X2
# both X1 and X2 are uniformly distributed around [0,1] * [0,1]
# we want to predict X response using 10% of available X1 range and 10% of the available X2 range
# example: predict X1 = .6 and X2 = .35 we will use the ranges:
        # [.55, .65] for X1
        # [.3, .4] for X2
# what fraction of available observations will we use to make the prediction?

# solution:
# on average we will only use 10% of the range from X1 and 10% of the range from X2
# this equates to almost 1% of the space of observations available
# .1 ^ p = percent of observations used in the prediction


# c. 
# now suppose we have a set of observations with p = 100 features
# again they are uniformly distributed on each feature and each feature ranges from [0,1]
# we will predict using 10% of each features range that is closest to the test observation
# what fraction of available observations will we use to make a prediction?

# solution:
# the percent for p = 100 is .1 ^ 100 = 10^ -98 % == almost zero percent of the observations we will use on average!
# the more predictors that harder it is to find a 10% space of observations to develop a classifier for!!


# d. 
# using our answers in (a) - (c) argue that a drawback of KNN when is p is large is there there are:
# very few training observations near the testing observation

# solution:
# we have seen that as the number of p increases, it is harder to find a comparison set of observations to match the testing observation
# as p increases, observations that are "near" decrease exponentially!!
# with enough features we won't find any training observations to classifiy the result into

# e. 
# now suppose we wish to make a prediction for a test observation by creating a p-dimensional hypercube
# this hypercube  is centered around the test observation that contains on average 10% of the training observations
# for p = 1,2 and 100 what is length of each side of the hypercube?

# solution:
# our cube will grow expentially based on the number of predictors
# p = 1, l = .1
# p = 2, l = sqrt(.1) = .01
# p = 100, l = .1 ^ 1/100 = .999
# general, p = N, l = .1 ^ 1/N


## Question 5:
# we now examine the differences between LDA and QDA

# a.
# if the Bayes decision boundary is linear, do we expect LDA or QDA to perform better on the training set? test set?

# solution:
# LDA will perform better on the test set if the true form of the data is linear
# if the Bayes decision boundary is linear - LDA will be approxmiate this with its own linear approximation
# QDA could possiblly perform better on the training set as it is a more flexible model...
# this could be to the determiment of the test error - and it will model more noise and not pick up on the true linear decision boundary

# b. 
# if the bayes decision boundary is non-linear, do we expect LDA or QDA to perform better on training and test?

# solution:
# non-linear decision boundaries will call for a more flexible model over LDA
# QDA could accruately model the non-linear nature of the deicsion boundary resulting in less training and test error than LDA

# c. 
# as the sample size n increases, do we expect the test prediction accuracy of QDA relative to LDA improve? why>

# solution:
# we do expect the test set error to decline as n increases
# QDA is a more flexible model - as n increases the flexible model will fit better than the inflexible model
# as n increases we can reduce the variance of the flexible model by using a large sample - hence testing error should decrease
# high sample size will favor the flexible model - high sample size helps us guard against overfitting

# d. 
# true or false: even if the bayes decision boundary for a given problem is linear...
# we would probably achieve a superior test error rate using QDA vs. LDA
# QDA is flexible enough to model a linear boundary?

# solution:
# LDA will perform better than QDA if the true decision boundary is linear for a small amount of samples
# QDA would require a large amount of samples to fit to the linear boundary
# QDA being more flexible will tend to overfit the linear boundary giving us a higher test error rate


## Question 6:
# suppose we collect data for a group of students:
# X1 = hours studied, X2 = undergrad GPA, Y = recieve an A
# we fit a logistic regression and produce these results: B0 = -6, B1 = .05, B2 = 1

# a. 
# estimate the probability that a student with 40 hours and 3.5 GPA gets an A in the class?

# solution:
# push the new observation inputs into our fitted logistic regression model
# the logistic regression fit is not as simple as plugging in the coefficients into the linear formula
# we will use:
        # p(X) = exp(B0 + B1X1 + B2X2) / (1 + exp(B0 + B1X1 + B2X2))
        # where X1 = hours studied, and X2 = undergrad GPA

# plug in results to the logistic regression prediction formula
hours = 40
gpa = 3.5
b1 = .05
b0 = -6
b2 = 1

(p_x <- exp(b0 + b1*hours + b2*gpa) / (1 + exp(b0 + b1*hours + b2*gpa)))
# [1] 0.3775407

# the probability that a student with 40 study hours and a 3.5 undergrad gpa gets an A in this class is 37.75%

# b. 
# how many hours would a student need to study in part A to have a 50% chance of getting an A in the class?

# solution:
# we need to flip our logistic regression formula around to solve this equation
# we have all our inputs we just to to use math to solve for hours
        # p(X) = exp(B0 + B1X1 + B2X2) / (1 + exp(B0 + B1X1 + B2X2))
        # where X1 = hours studied, and X2 = undergrad GPA
# here we will have X1 as unknown and p(X) = .5

# define variables to plug into prediction equation
px = .5
gpa = 3.5
b1 = .05
b0 = -6
b2 = 1

# manipulating the logistic regression formula for hours we get
        # log(1) = -2.5 + .05X1)
        # solve for X1
        # X1 = 2.5 / .05 = 50 hours
(x1 <- 2.5 / .05)
# [1] 50

# the student would need to study 50 hours to achieve a 50% chance of getting an A based on gpa and our logistic regression



## Question 7:
# suppose we wish to predict whether a stock will issue a dividend this year (yes or no)
# we examine a large number of companies and discover that the mean value of X companies that issued was X = 10
# the mean for those who didn't issue was X = 0
# the variance of X for these TWO sets of companies was sigma^2 = 36
# 80% of the companies issued
# assuming that X follows a normal distribution...
# predict the probability that a company will issue a dividend  this year given its percentage profit was X = 4

# solution:
# we need to set up the bayes classification therom to solve this problem
# we first need to write it down - then extrapolate it for each yes and no case
        # bayes theorem:
        # p(x) = equation 4.12 
# this simplfies to:
        # pi(yes)exp(-1/2sigma^2 (x - u(yes)^2))
        # divided by:
        # pi(yes)exp(-1/2sigma^2 (x - u(yes)^2)) + pi(no)exp(-1/2sigma^2 (x - u(no)^2))
# based on this formula we can plug in our known quantities to get a probability of issuing a dividend

# define our input terms
pi_yes <- .8
pi_no <- .2
sigma <- 6
mu_yes <- 10
mu_no <- 0
x <- 4

# build the prediction formula - numerator with yes issue divedend
(top_bayes <- pi_yes * exp((-1/(2*(sigma^2))) * ((x - mu_yes)^2)))
# [1] 0.4852245

# build the prediction formula - denominator with yes and no issue divedend
(bottom_bayes <- (pi_yes * exp((-1/(2*(sigma^2))) * ((x - mu_yes)^2))) + (pi_no * exp((-1/(2*(sigma^2))) * ((x - mu_no)^2))))
# [1] 0.645372

# bayes prediction
(bayes_db <- top_bayes / bottom_bayes)
# [1] 0.7518525

# the probability that a company with X = 4 percentage profit will issue a dividend is 75.18%


## Question 8:
# suppose we take a data set and split it into equal test and training datasets
# we then try two different classification procedures
# first we try logistic regression and get a error rate of 20% on training and 30% on testing
# we also using KNN (n = 1) and get an average error rate of 18% over both test and training
# based on these results - which method should we use for classification?

# solution:
# this is a tricky question
# recall that the KNN with K = 1 will find the "single" datapoint closest to our new observation
# in the training set - each prediction will find "itself" in the training data - and perfectly classify the result
# this means that the training error rate for our KNN classifier will be 0
# the average of the training and test error rate for KNN is 18%
# this means that the test error rate for the KNN classifier is 36%!!!
# logistic regression outperforms KNN with a test error rate of 30%!!!
# we should use the logistic regression as our classifier!!


## Question 9:
# this question has to do with the odds

# a.
# on average what fraction of people with an odds of .37 of credit card default will in fact default?

# solution:
# we need to understand the definition of odds and how it relates back to probability
# the formula for odds is:
        # p(X) / 1 - p(X)
# we set this equal to .37 and then solve for p(X) to get the fraction of people who will actually default
# manipulting the odds formula we get:
        # p(X) / 1 - p(X) = .37
        # 1.37p(X) = .37
        # p(X) = .37 / 1.37

# define odds
odds <- .37

# calculate probability equation
(p_X <- odds / 1.37)
# [1] 0.270073

# 27% of people with an odds of .37 to default will actually default on thier credit card

# b.
# suppose an indivdiual has a 16% chance of defaulting on a credit card payment
# what are the odds that she will default??

# solution:
# take the previous solution of odds > probability and reverse it
# we now have the probability p(X) and we want to extrapolate into odds
        # odds = p(X) / 1 - p(X)
        # odds = .16 / .84

# define variables to plug into odds formula
p <- .16
q <- 1-p

# odds formula
(odds_ratio <- p / q)
# [1] 0.1904762





## Applied Exercises


## Question 10:
# use the weekly dataset to answer the following questions

# load the data
library(ISLR)
data("Weekly")

# a. 
# produce some numerical and graphical summaries of the weekly dataset

# view the strucutre
str(Weekly)
# 'data.frame':	1089 obs. of  9 variables:
# $ Year     : num  1990 1990 1990 1990 1990 1990 1990 1990 1990 1990 ...
# $ Lag1     : num  0.816 -0.27 -2.576 3.514 0.712 ...
# $ Lag2     : num  1.572 0.816 -0.27 -2.576 3.514 ...
# $ Lag3     : num  -3.936 1.572 0.816 -0.27 -2.576 ...
# $ Lag4     : num  -0.229 -3.936 1.572 0.816 -0.27 ...
# $ Lag5     : num  -3.484 -0.229 -3.936 1.572 0.816 ...
# $ Volume   : num  0.155 0.149 0.16 0.162 0.154 ...
# $ Today    : num  -0.27 -2.576 3.514 0.712 1.178 ...
# $ Direction: Factor w/ 2 levels "Down","Up": 1 1 2 2 2 1 2 2 2 1 ...

# quick summary
summary(Weekly)
# Year           Lag1               Lag2               Lag3               Lag4         
# Min.   :1990   Min.   :-18.1950   Min.   :-18.1950   Min.   :-18.1950   Min.   :-18.1950  
# 1st Qu.:1995   1st Qu.: -1.1540   1st Qu.: -1.1540   1st Qu.: -1.1580   1st Qu.: -1.1580  
# Median :2000   Median :  0.2410   Median :  0.2410   Median :  0.2410   Median :  0.2380  
# Mean   :2000   Mean   :  0.1506   Mean   :  0.1511   Mean   :  0.1472   Mean   :  0.1458  
# 3rd Qu.:2005   3rd Qu.:  1.4050   3rd Qu.:  1.4090   3rd Qu.:  1.4090   3rd Qu.:  1.4090  
# Max.   :2010   Max.   : 12.0260   Max.   : 12.0260   Max.   : 12.0260   Max.   : 12.0260  
# Lag5              Volume            Today          Direction 
# Min.   :-18.1950   Min.   :0.08747   Min.   :-18.1950   Down:484  
# 1st Qu.: -1.1660   1st Qu.:0.33202   1st Qu.: -1.1540   Up  :605  
# Median :  0.2340   Median :1.00268   Median :  0.2410             
# Mean   :  0.1399   Mean   :1.57462   Mean   :  0.1499             
# 3rd Qu.:  1.4050   3rd Qu.:2.05373   3rd Qu.:  1.4050             
# Max.   : 12.0260   Max.   :9.32821   Max.   : 12.0260 

# pairs plot
pairs(Weekly, col = Weekly$Direction)

# correlation matrix
(Weekly_df <- Weekly %>% 
        keep(is.numeric) %>% 
        cor(.) %>% 
        as.data.frame(.))
#               Year         Lag1        Lag2        Lag3         Lag4         Lag5      Volume
# Year    1.00000000 -0.032289274 -0.03339001 -0.03000649 -0.031127923 -0.030519101  0.84194162
# Lag1   -0.03228927  1.000000000 -0.07485305  0.05863568 -0.071273876 -0.008183096 -0.06495131
# Lag2   -0.03339001 -0.074853051  1.00000000 -0.07572091  0.058381535 -0.072499482 -0.08551314
# Lag3   -0.03000649  0.058635682 -0.07572091  1.00000000 -0.075395865  0.060657175 -0.06928771
# Lag4   -0.03112792 -0.071273876  0.05838153 -0.07539587  1.000000000 -0.075675027 -0.06107462
# Lag5   -0.03051910 -0.008183096 -0.07249948  0.06065717 -0.075675027  1.000000000 -0.05851741
# Volume  0.84194162 -0.064951313 -0.08551314 -0.06928771 -0.061074617 -0.058517414  1.00000000
# Today  -0.03245989 -0.075031842  0.05916672 -0.07124364 -0.007825873  0.011012698 -0.03307778
# Today
# Year   -0.032459894
# Lag1   -0.075031842
# Lag2    0.059166717
# Lag3   -0.071243639
# Lag4   -0.007825873
# Lag5    0.011012698
# Volume -0.033077783
# Today   1.000000000

# we notice a strong correlation between Year and Volume


# b. 
# use the full dataset to perform a logistic regression with Direction as the repsonse
# use the five lag variables and Volume as predictors
# are there any significant features? what are they?

# fit our logistic regression model
log_fit <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume - 1,
               family = "binomial", data = Weekly)

summary(log_fit)

# Call:
# glm(formula = Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + 
#                     Volume - 1, family = "binomial", data = Weekly)
# 
# Deviance Residuals: 
#         Min      1Q  Median      3Q     Max  
# -1.726  -1.191   1.033   1.148   1.553  
# 
# Coefficients:
#         Estimate Std. Error z value Pr(>|z|)  
# Lag1   -0.032730   0.026177  -1.250   0.2112  
# Lag2    0.068196   0.026685   2.556   0.0106 *
# Lag3   -0.008099   0.026447  -0.306   0.7594  
# Lag4   -0.019420   0.026231  -0.740   0.4591  
# Lag5   -0.006856   0.026230  -0.261   0.7938  
# Volume  0.056925   0.026792   2.125   0.0336 *
#         ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1509.7  on 1089  degrees of freedom
# Residual deviance: 1496.1  on 1083  degrees of freedom
# AIC: 1508.1
# 
# Number of Fisher Scoring iterations: 4

# it seems that Lag2 and Volume are statisticall significant



# c.
# compute the confusion matrixand overall fraction of correct guesses for our model
# explain what the confusion matrix is giving us and the types of mistakes our classifier is giving us...

library(caret)

log_pred <- predict(log_fit, type = "response") %>% 
        as.tibble() %>% 
        mutate(class = if_else(value > .5, "Up", "Down"))

confusionMatrix(log_pred$class, reference = Weekly$Direction)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction Down  Up
#       Down   54  48
#       Up    430 557
# 
# Accuracy : 0.5611         
# 95% CI : (0.531, 0.5908)
# No Information Rate : 0.5556         
# P-Value [Acc > NIR] : 0.369          
# 
# Kappa : 0.035          
# Mcnemar's Test P-Value : <2e-16         
# 
# Sensitivity : 0.11157        
# Specificity : 0.92066        
# Pos Pred Value : 0.52941        
# Neg Pred Value : 0.56434        
# Prevalence : 0.44444        
# Detection Rate : 0.04959        
# Detection Prevalence : 0.09366        
# Balanced Accuracy : 0.51612        
# 
# 'Positive' Class : Down 

# we see here from the confusion matrix output the overall accuracy is .56
# the specificity is .92 - we are able to predict the market going up most of the time
# the sensitivity is .11 - we do not do a good job of predicting when the market goes down


# d.
# now fit a logistic regression using a training period from 1990 to 2008
# use Lag2 as the only feature
# compute the confusion matrix on the test data

# define test and training datasets
train <- Weekly %>% 
        filter(Year < 2009)

test <- Weekly %>% 
        filter(Year >= 2009)


# fit our logistic regression model
log_fit <- glm(Direction ~ Lag2,
               family = "binomial", data = train)

summary(log_fit)
# Call:
# glm(formula = Direction ~ Lag2, family = "binomial", data = train)
# 
# Deviance Residuals: 
#         Min      1Q  Median      3Q     Max  
# -1.536  -1.264   1.021   1.091   1.368  
# 
# Coefficients:
#         Estimate Std. Error z value Pr(>|z|)   
# (Intercept)  0.20326    0.06428   3.162  0.00157 **
#         Lag2         0.05810    0.02870   2.024  0.04298 * 
#         ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1354.7  on 984  degrees of freedom
# Residual deviance: 1350.5  on 983  degrees of freedom
# AIC: 1354.5
# 
# Number of Fisher Scoring iterations: 4


# predict on test
log_pred <- predict(log_fit, type = "response", newdata = test) %>% 
        as.tibble() %>% 
        mutate(class = if_else(value > .5, "Up", "Down"))

# develop confusion matrix
confusionMatrix(log_pred$class, reference = test$Direction)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction Down Up
# Down    9  5
# Up     34 56
# 
# Accuracy : 0.625          
# 95% CI : (0.5247, 0.718)
# No Information Rate : 0.5865         
# P-Value [Acc > NIR] : 0.2439         
# 
# Kappa : 0.1414         
# Mcnemar's Test P-Value : 7.34e-06       
# 
# Sensitivity : 0.20930        
# Specificity : 0.91803        
# Pos Pred Value : 0.64286        
# Neg Pred Value : 0.62222        
# Prevalence : 0.41346        
# Detection Rate : 0.08654        
# Detection Prevalence : 0.13462        
# Balanced Accuracy : 0.56367        
# 
# 'Positive' Class : Down  

# accuracy is .625
# predicting specificity is .91 - slight degradation in predicting the market will go up
# predicting sensistivity is .21 - big improvement in predicting the market will go down


# e. 
# repeat steps above using LDA

library(MASS)

# lda fit
lda_fit <- lda(Direction ~ Lag2, data = train)

# lda summary stats

# group means
lda_fit$means
# Lag2
# Down -0.03568254
# Up    0.26036581

# the singluar values which give ratio of between and within group standard deviations on linear discriminant variables
# these squares are the F-statistics
lda_fit$svd
# [1] 2.039443

# see the prior probabilities the lda call estimated to approximate the bayes decision boundary
lda_fit$prior
# Down        Up 
# 0.4477157 0.5522843


# predict using lda
lda_predict <- predict(lda_fit, newdata = test, type = "response") %>% 
        as.data.frame() %>% 
        dplyr::select(class)

# confusion matrix
confusionMatrix(lda_predict$class, test$Direction)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction Down Up
# Down    9  5
# Up     34 56
# 
# Accuracy : 0.625          
# 95% CI : (0.5247, 0.718)
# No Information Rate : 0.5865         
# P-Value [Acc > NIR] : 0.2439         
# 
# Kappa : 0.1414         
# Mcnemar's Test P-Value : 7.34e-06       
# 
# Sensitivity : 0.20930        
# Specificity : 0.91803        
# Pos Pred Value : 0.64286        
# Neg Pred Value : 0.62222        
# Prevalence : 0.41346        
# Detection Rate : 0.08654        
# Detection Prevalence : 0.13462        
# Balanced Accuracy : 0.56367        
# 
# 'Positive' Class : Down  


# we notice the exact same results as the logistic regression...?



# f.
# repeat the above steps using QDA

# qda fit
qda_fit <- qda(Direction ~ Lag2, data = train)
summary(qda_fit)

# qda summary stats

# group means
qda_fit$means
# Lag2
# Down -0.03568254
# Up    0.26036581


# see the prior probabilities the qda call estimated to approximate the bayes decision boundary
qda_fit$prior
# Down        Up 
# 0.4477157 0.5522843


# predict using qda
qda_predict <- predict(qda_fit, newdata = test, type = "response") %>% 
        as.data.frame() %>% 
        dplyr::select(class)

# confusion matrix
confusionMatrix(qda_predict$class, test$Direction)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction Down Up
# Down    0  0
# Up     43 61
# 
# Accuracy : 0.5865          
# 95% CI : (0.4858, 0.6823)
# No Information Rate : 0.5865          
# P-Value [Acc > NIR] : 0.5419          
# 
# Kappa : 0               
# Mcnemar's Test P-Value : 1.504e-10       
# 
# Sensitivity : 0.0000          
# Specificity : 1.0000          
# Pos Pred Value :    NaN          
# Neg Pred Value : 0.5865          
# Prevalence : 0.4135          
# Detection Rate : 0.0000          
# Detection Prevalence : 0.0000          
# Balanced Accuracy : 0.5000          
# 
# 'Positive' Class : Down 

# the qda model picks the market going up every time!!
# we get a .58 accuracy just by picking up every time!


# g.
# complete the same steps using a knn with K = 1 classifier

library(class)

# knn fit
# the input into the knn call is a little different than our typical lm type calls
# knn(train, test, cl, k = 1, l = 0, prob = FALSE, use.all = TRUE)

knn_fit <- knn(
        train = as.matrix(train$Lag2), # need to supply matrix of training variables
        test = as.matrix(test$Lag2), # need to supply matrix of testing variables - there is no model call
        cl = train$Direction, # need to provide a vector of the "right answer" to train our model
        k = 1 # specify the number of nearest neighbors to use in the decision process
        )

# knn does the fitting and prediction in one step
confusionMatrix(knn_fit, test$Direction)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction Down Up
# Down   21 30
# Up     22 31
# 
# Accuracy : 0.5             
# 95% CI : (0.4003, 0.5997)
# No Information Rate : 0.5865          
# P-Value [Acc > NIR] : 0.9700          
# 
# Kappa : -0.0033         
# Mcnemar's Test P-Value : 0.3317          
# 
# Sensitivity : 0.4884          
# Specificity : 0.5082          
# Pos Pred Value : 0.4118          
# Neg Pred Value : 0.5849          
# Prevalence : 0.4135          
# Detection Rate : 0.2019          
# Detection Prevalence : 0.4904          
# Balanced Accuracy : 0.4983          
# 
# 'Positive' Class : Down 


# h.
# which of these methods appear to be the best for this data?
# lda or logistic regression appear to be the best with test accuracy of .625
# for reference the qda model picked UP everytime in the test data - and got accuracy of .58
# our logistic and lda models improved slightly over this


# i.
# experiement with other models and different transformations

# knn experiment
# model with k = 5, 10, 15, 20, 25, 100

ks <- c(5,10,15,20,25,100, 150, 250)

cm_list <- list(k5 = NULL, k10 = NULL, k15 = NULL, k20 = NULL, k25 = NULL, k100 = NULL, k150 = NULL,
                k250 = NULL)

for (i in seq_along(ks)) {
        
        set.seed(100)
        
        knn_fit <- knn(
                train = as.matrix(train$Lag2), # need to supply matrix of training variables
                test = as.matrix(test$Lag2), # need to supply matrix of testing variables - there is no model call
                cl = train$Direction, # need to provide a vector of the "right answer" to train our model
                k = ks[[i]] # specify the number of nearest neighbors to use in the decision process
        )
        
        cm_list[i] <- confusionMatrix(knn_fit, test$Direction)$overall %>% 
                as.tibble() %>% 
                filter(row_number() == 1)  
        
}

(cm_df <- as.data.frame(cm_list))

#              k5       k10       k15       k20       k25      k100  k150      k250
# 1     0.5480769 0.5673077 0.5865385 0.5961538 0.5384615 0.5769231 0.625 0.5865385

# we see that k20 has the highest test accuracy



# qda with transformed variables
# qda fit
qda_fit <- qda(Direction ~ Lag2 + I(Lag2^2), data = train)
summary(qda_fit)

# qda summary stats

# group means
qda_fit$means
# Lag2
# Down -0.03568254
# Up    0.26036581


# see the prior probabilities the qda call estimated to approximate the bayes decision boundary
qda_fit$prior
# Down        Up 
# 0.4477157 0.5522843


# predict using qda
qda_predict <- predict(qda_fit, newdata = test, type = "response") %>% 
        as.data.frame() %>% 
        dplyr::select(class)

# confusion matrix
confusionMatrix(qda_predict$class, test$Direction)$overall %>% as.data.frame()
# .
# Accuracy       6.250000e-01
# Kappa          1.281169e-01
# AccuracyLower  5.246597e-01
# AccuracyUpper  7.180252e-01
# AccuracyNull   5.865385e-01
# AccuracyPValue 2.439500e-01
# McnemarPValue  2.989608e-07

# using a squared term gives us the accuracy of our logistic and lda models!!


# our best models are the original logistic regression, lda and the transformed qda
# these all meet at 62% accuracy
# note predicting up on the test data gives us 58% accuracy


## Question 11:
# in this problem you will develop a model to predict high or low gas mileage
# this will be based on the auto dataset
data("Auto")

# a. 
# create a classification (binary) variable that contains 1 if mpg is a value above the dataset median
# 0 otherwise

(auto_df <- Auto %>% 
        as.tibble() %>% 
        mutate(mpg01 = if_else(mpg > median(mpg), 1, 0)) %>% 
        dplyr::select(., -mpg))
# A tibble: 392 x 9
#     cylinders displacement horsepower weight acceleration  year origin name                      mpg01
# <      dbl>        <dbl>      <dbl>  <dbl>        <dbl> <dbl>  <dbl> <fct>                     <dbl>
# 1      8.00          307        130   3504        12.0   70.0   1.00 chevrolet chevelle malibu     0
# 2      8.00          350        165   3693        11.5   70.0   1.00 buick skylark 320             0
# 3      8.00          318        150   3436        11.0   70.0   1.00 plymouth satellite            0
# 4      8.00          304        150   3433        12.0   70.0   1.00 amc rebel sst                 0
# 5      8.00          302        140   3449        10.5   70.0   1.00 ford torino                   0
# 6      8.00          429        198   4341        10.0   70.0   1.00 ford galaxie 500              0
# 7      8.00          454        220   4354         9.00  70.0   1.00 chevrolet impala              0
# 8      8.00          440        215   4312         8.50  70.0   1.00 plymouth fury iii             0
# 9      8.00          455        225   4425        10.0   70.0   1.00 pontiac catalina              0
# 10      8.00          390        190   3850         8.50  70.0   1.00 amc ambassador dpl           0
# # ... with 382 more rows


# b. 
# explore the data graphically in order to investigate the association between mpg01 and other features

# correlation matrix
cor(keep(auto_df, is.numeric))
#               cylinders displacement horsepower     weight acceleration       year     origin      mpg01
# cylinders     1.0000000    0.9508233  0.8429834  0.8975273   -0.5046834 -0.3456474 -0.5689316 -0.7591939
# displacement  0.9508233    1.0000000  0.8972570  0.9329944   -0.5438005 -0.3698552 -0.6145351 -0.7534766
# horsepower    0.8429834    0.8972570  1.0000000  0.8645377   -0.6891955 -0.4163615 -0.4551715 -0.6670526
# weight        0.8975273    0.9329944  0.8645377  1.0000000   -0.4168392 -0.3091199 -0.5850054 -0.7577566
# acceleration -0.5046834   -0.5438005 -0.6891955 -0.4168392    1.0000000  0.2903161  0.2127458  0.3468215
# year         -0.3456474   -0.3698552 -0.4163615 -0.3091199    0.2903161  1.0000000  0.1815277  0.4299042
# origin       -0.5689316   -0.6145351 -0.4551715 -0.5850054    0.2127458  0.1815277  1.0000000  0.5136984
# mpg01        -0.7591939   -0.7534766 -0.6670526 -0.7577566    0.3468215  0.4299042  0.5136984  1.0000000

library(GGally)

# pairs plot
ggpairs(auto_df %>% dplyr::select(., -name) %>% mutate(mpg01 = factor(mpg01)),aes(color = mpg01))

# we see that cars with high horsepower are bad with mpg!
# we see that the more weight a car has the worse it is at mpg!
# 

# final data set for modeling
auto_mod <- auto_df %>% dplyr::select(., -name) %>% mutate(mpg01 = factor(mpg01)) %>% 
        as.tibble()


# c. 
# create a test and training data set
library(caret)

# define splits
intrain <- createDataPartition(auto_mod$mpg01, p = .6, list = F)

# create test and training
train <- auto_mod[intrain,]
test <- auto_mod[-intrain,] 


# d.
# perform lda on the training data in order to predict mpg01 using the variables most associated with mpg01 from our EDA

library(MASS)

# lda fit
lda_fit <- lda(mpg01 ~ horsepower + weight + displacement, data = train)

# lda summary stats

# group means
lda_fit$means
# horsepower   weight displacement
# 0  128.90678 3587.339     271.0424
# 1   78.84746 2346.585     115.5254

# the singluar values which give ratio of between and within group standard deviations on linear discriminant variables
# these squares are the F-statistics
lda_fit$svd
# [1] 18.36453

# see the prior probabilities the lda call estimated to approximate the bayes decision boundary
lda_fit$prior
# 0   1 
# 0.5 0.5 


# predict using lda
lda_predict <- predict(lda_fit, newdata = test, type = "response") %>% 
        as.data.frame() %>% 
        dplyr::select(class)

# confusion matrix
cm <- confusionMatrix(lda_predict$class, test$mpg01)

cm$table
#         Reference
# Prediction  0  1
#          0 67  4
#          1 11 74

cm$overall
# Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull AccuracyPValue  McnemarPValue 
# 9.038462e-01   8.076923e-01   8.463703e-01   9.451789e-01   5.000000e-01   3.680652e-27   1.213353e-01



# e. 
# perform QDA on the training data and test on the test dataset

# qda fit
qda_fit <- qda(mpg01 ~ horsepower + weight + displacement, data = train)

# predict using qda
qda_predict <- predict(qda_fit, newdata = test, type = "response") %>% 
        as.data.frame() %>% 
        dplyr::select(class)

# confusion matrix
cm <- confusionMatrix(qda_predict$class, test$mpg01)

cm$table
#                 Reference
# Prediction      0  1
#              0 69  7
#              1  9 71

cm$overall
# Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull AccuracyPValue  McnemarPValue 
# 8.974359e-01   7.948718e-01   8.387888e-01   9.402300e-01   5.000000e-01   3.271911e-26   8.025873e-01


# f. 
# perform a logistic regression 

# logistic regression fit
log_fit <- glm(mpg01 ~ horsepower + weight + displacement, data = train,
               family = "binomial")

# logistic regression predict
(log_pred <- predict(log_fit, newdata = test, type = "response") %>% 
        as.tibble() %>% 
        mutate(class = if_else(value > .5, "1", "0")))

# develop confusion matrix
confusionMatrix(log_pred$class, test$mpg01)$table
# Reference
# Prediction  0  1
# 0 70  8
# 1  8 70
confusionMatrix(log_pred$class, test$mpg01)$overall
# Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull AccuracyPValue  McnemarPValue 
# 8.974359e-01   7.948718e-01   8.387888e-01   9.402300e-01   5.000000e-01   3.271911e-26   1.000000e+00 


# g. 
# perform a KNN on the training data in order to predict mpg01

ks <- c(5,10,15,20,25)

cm_list <- list(k5 = NULL, k10 = NULL, k15 = NULL, k20 = NULL, k25 = NULL)

for (i in seq_along(ks)) {
        
        
        knn_fit <- knn(
                train = as.matrix(cbind(train$horsepower, train$weight, train$displacement)),
                test = as.matrix(cbind(test$horsepower, test$weight, test$displacement)), 
                cl = train$mpg01,
                k = ks[[i]]
        )
        
        cm_list[i] <- confusionMatrix(knn_fit, test$mpg01)$overall %>% 
                as.tibble() %>% 
                filter(row_number() == 1)  
        
}

(cm_df <- as.data.frame(cm_list))

#       k5       k10       k15       k20       k25
# 1 0.8910256 0.8974359 0.8910256 0.8782051 0.8910256

# K at 10 seems to give the best overall prediction - test accuracy is 89.74%



## Question 12:
# this problem invloves writing functions

# a. 
# write a function, Power() that prints out the result of raising 2 to the 3rd power
# compute 2^3 and print the results

# create function
power2 <- function() {
        calc <- 2^3
        print(calc)
}

# run function
power2()
# [1] 8


# b. 
# create a new function that allows you to pass in any two numbers x and a and prints out the calc x^a

# create function
powerCalc <- function(x, a) {
        calc <- x ^ a
        print(calc)
}

# run function
powerCalc(10,4)
# [1] 10000

# c. 
# compute 10^3, 8^17, and 131^3 with this function

# 10^3
powerCalc(10,3)
# [1] 1000

# 8^17
powerCalc(8,17)
# [1] 2.2518e+15

# 131^3
powerCalc(131, 3)
# [1] 2248091


# d.
# create a new function that stores the power calc as an R object


# create function
powerStore <- function(x, a) {
        calc <- x ^ a
        # print(calc)
        return(calc)
}


# e. 
# create a plot of f(x) = x^2
# x axis = 1:10
# y axis = x^2

x = 1:10

map(1:10, ~powerStore(.x, 2)) %>% 
        as.data.frame() %>% 
        gather(X1:X100, key = "x", value = "x2") %>% 
        ggplot(., aes(x = 1:10, y = .$x2)) +
        geom_point() +
        xlab("scale 1 - 10") +
        ylab("x2") +
        theme_few()

# f. 
# create a function that allows you to create a plot of x against x^a for a fixed a and range of x

# create function
plotPower <- function(x, a) {
        
        map(x, ~powerStore(.x, a)) %>% 
                as.data.frame() %>% 
                gather(., key = "name", value = "x2") %>% 
                ggplot(., aes(x = x, y = .$x2)) +
                geom_point() +
                xlab("range") +
                ylab("x^a") +
                theme_few()
}

# test function
plotPower(1:3, 3)
plotPower(1:8, 2)
plotPower(1:10, 10)
plotPower(1:10, 3)



## Question 13:
# using the Boston dataset 
# fit classification models in order to predict whether a given suburb has a crime rate above or below the median
# explore logistic regression, lda and KNN

library(MASS)
data("Boston")

# summary stats of boston dataset
summary(Boston)
# crim                zn             indus            chas              nox               rm       
# Min.   : 0.00632   Min.   :  0.00   Min.   : 0.46   Min.   :0.00000   Min.   :0.3850   Min.   :3.561  
# 1st Qu.: 0.08204   1st Qu.:  0.00   1st Qu.: 5.19   1st Qu.:0.00000   1st Qu.:0.4490   1st Qu.:5.886  
# Median : 0.25651   Median :  0.00   Median : 9.69   Median :0.00000   Median :0.5380   Median :6.208  
# Mean   : 3.61352   Mean   : 11.36   Mean   :11.14   Mean   :0.06917   Mean   :0.5547   Mean   :6.285  
# 3rd Qu.: 3.67708   3rd Qu.: 12.50   3rd Qu.:18.10   3rd Qu.:0.00000   3rd Qu.:0.6240   3rd Qu.:6.623  
# Max.   :88.97620   Max.   :100.00   Max.   :27.74   Max.   :1.00000   Max.   :0.8710   Max.   :8.780  
# age              dis              rad              tax           ptratio          black       
# Min.   :  2.90   Min.   : 1.130   Min.   : 1.000   Min.   :187.0   Min.   :12.60   Min.   :  0.32  
# 1st Qu.: 45.02   1st Qu.: 2.100   1st Qu.: 4.000   1st Qu.:279.0   1st Qu.:17.40   1st Qu.:375.38  
# Median : 77.50   Median : 3.207   Median : 5.000   Median :330.0   Median :19.05   Median :391.44  
# Mean   : 68.57   Mean   : 3.795   Mean   : 9.549   Mean   :408.2   Mean   :18.46   Mean   :356.67  
# 3rd Qu.: 94.08   3rd Qu.: 5.188   3rd Qu.:24.000   3rd Qu.:666.0   3rd Qu.:20.20   3rd Qu.:396.23  
# Max.   :100.00   Max.   :12.127   Max.   :24.000   Max.   :711.0   Max.   :22.00   Max.   :396.90  
# lstat            medv      
# Min.   : 1.73   Min.   : 5.00  
# 1st Qu.: 6.95   1st Qu.:17.02  
# Median :11.36   Median :21.20  
# Mean   :12.65   Mean   :22.53  
# 3rd Qu.:16.95   3rd Qu.:25.00  
# Max.   :37.97   Max.   :50.00  

# create the classifier repsonse variable to model on
boston_df <- Boston %>% 
        mutate(crim = if_else(crim > median(crim), 1, 0)) %>% 
        as_tibble()

unique(boston_df$crim)


# explore the boston dataset to find best features to select
cor(keep(boston_df, is.numeric))

#               crim          zn       indus         chas         nox          rm         age         dis
# crim     1.00000000 -0.43615103  0.60326017  0.070096774  0.72323480 -0.15637178  0.61393992 -0.61634164
# zn      -0.43615103  1.00000000 -0.53382819 -0.042696719 -0.51660371  0.31199059 -0.56953734  0.66440822
# indus    0.60326017 -0.53382819  1.00000000  0.062938027  0.76365145 -0.39167585  0.64477851 -0.70802699
# chas     0.07009677 -0.04269672  0.06293803  1.000000000  0.09120281  0.09125123  0.08651777 -0.09917578
# nox      0.72323480 -0.51660371  0.76365145  0.091202807  1.00000000 -0.30218819  0.73147010 -0.76923011
# rm      -0.15637178  0.31199059 -0.39167585  0.091251225 -0.30218819  1.00000000 -0.24026493  0.20524621
# age      0.61393992 -0.56953734  0.64477851  0.086517774  0.73147010 -0.24026493  1.00000000 -0.74788054
# dis     -0.61634164  0.66440822 -0.70802699 -0.099175780 -0.76923011  0.20524621 -0.74788054  1.00000000
# rad      0.61978625 -0.31194783  0.59512927 -0.007368241  0.61144056 -0.20984667  0.45602245 -0.49458793
# tax      0.60874128 -0.31456332  0.72076018 -0.035586518  0.66802320 -0.29204783  0.50645559 -0.53443158
# ptratio  0.25356836 -0.39167855  0.38324756 -0.121515174  0.18893268 -0.35550149  0.26151501 -0.23247054
# black   -0.35121093  0.17552032 -0.35697654  0.048788485 -0.38005064  0.12806864 -0.27353398  0.29151167
# lstat    0.45326273 -0.41299457  0.60379972 -0.053929298  0.59087892 -0.61380827  0.60233853 -0.49699583
# medv    -0.26301673  0.36044534 -0.48372516  0.175260177 -0.42732077  0.69535995 -0.37695457  0.24992873
# rad         tax    ptratio       black      lstat       medv
# crim     0.619786249  0.60874128  0.2535684 -0.35121093  0.4532627 -0.2630167
# zn      -0.311947826 -0.31456332 -0.3916785  0.17552032 -0.4129946  0.3604453
# indus    0.595129275  0.72076018  0.3832476 -0.35697654  0.6037997 -0.4837252
# chas    -0.007368241 -0.03558652 -0.1215152  0.04878848 -0.0539293  0.1752602
# nox      0.611440563  0.66802320  0.1889327 -0.38005064  0.5908789 -0.4273208
# rm      -0.209846668 -0.29204783 -0.3555015  0.12806864 -0.6138083  0.6953599
# age      0.456022452  0.50645559  0.2615150 -0.27353398  0.6023385 -0.3769546
# dis     -0.494587930 -0.53443158 -0.2324705  0.29151167 -0.4969958  0.2499287
# rad      1.000000000  0.91022819  0.4647412 -0.44441282  0.4886763 -0.3816262
# tax      0.910228189  1.00000000  0.4608530 -0.44180801  0.5439934 -0.4685359
# ptratio  0.464741179  0.46085304  1.0000000 -0.17738330  0.3740443 -0.5077867
# black   -0.444412816 -0.44180801 -0.1773833  1.00000000 -0.3660869  0.3334608
# lstat    0.488676335  0.54399341  0.3740443 -0.36608690  1.0000000 -0.7376627
# medv    -0.381626231 -0.46853593 -0.5077867  0.33346082 -0.7376627  1.0000000

# pairs plot
ggpairs(boston_df, aes(color = crim))

# let's model with rad, dis, age, nox, indus

# create test and training datasets
inTrain <- createDataPartition(boston_df$crim, p = .65, list = F)

# create splits
train <- boston_df[inTrain,]
test <- boston_df[-inTrain,]


# logistic regression fit
log_fit <- glm(crim ~ lstat + tax + dis, data = train, family = "binomial")
summary(log_fit)

# Call:
# glm(formula = crim ~ lstat + tax + dis, family = "binomial", 
#             data = train)
# 
# Deviance Residuals: 
#         Min        1Q    Median        3Q       Max  
# -3.08513  -0.50248   0.03739   0.29076   2.89157  
# 
# Coefficients:
#         Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -1.319367   0.850909  -1.551    0.121    
# lstat        0.038419   0.030038   1.279    0.201    
# tax          0.008609   0.001626   5.293 1.20e-07 ***
#         dis         -0.653754   0.124248  -5.262 1.43e-07 ***
#         ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 457.48  on 329  degrees of freedom
# Residual deviance: 222.73  on 326  degrees of freedom
# AIC: 230.73
# 
# Number of Fisher Scoring iterations: 6

# logistic regression predict
log_pred <- predict(log_fit, newdata = test, type = "response") %>% 
        as.tibble() %>% 
        mutate(class = if_else(value > .5, 1, 0))

# confusion matrix and model performance on test set
round(confusionMatrix(log_pred$class, test$crim)$overall,3)
# Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull AccuracyPValue  McnemarPValue 
# 0.744          0.489          0.673          0.807          0.500          0.000          0.074


# lda fit
lda_fit <- lda(crim ~ lstat + tax + dis, data = train, family = "binomial")

# lda predict
lda_predict <- predict(lda_fit, newdata = test, type = "repsonse")

# confusion matrix and test set accuracy
round(confusionMatrix(lda_predict$class, test$crim)$overall,3)
# Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull AccuracyPValue  McnemarPValue 
# 0.739          0.477          0.667          0.802          0.500          0.000          0.027


# knn experiment
# model with k = 5, 10, 15, 20, 25, 100

ks <- c(5,10,15,20,25)

cm_list <- list(k5 = NULL, k10 = NULL, k15 = NULL, k20 = NULL, k25 = NULL)

for (i in seq_along(ks)) {
        
        
        knn_fit <- knn(
                train = as.matrix(cbind(train$lstat, train$tax, train$dis)),
                test = as.matrix(cbind(test$lstat, test$tax, test$dis)), 
                cl = train$crim,
                k = ks[[i]]
        )
        
        cm_list[i] <- confusionMatrix(knn_fit, test$crim)$overall %>% 
                as.tibble() %>% 
                filter(row_number() == 1)  
        
}

(cm_df <- as.data.frame(cm_list))

#         k5       k10       k15       k20       k25
# 1 0.9034091 0.8806818 0.8181818 0.8181818 0.8068182




# chapter 5: resampling methods -------------------------------------------

## Resampling methods
# resampling methods are an indispensible tool in modern statistics
# repeatedly draw samples from a training set and refit a model of interest on each sample to gain additional information
# we can then compare all our "re-fitted" models to see how they differ
# this gives us more range than just using one model on our training data

# resampling approaches can be computationally expensive - we fit many models using different subsets of the training data
# this chapter will focus on two common resampling methods - cross validation and bootstrap
# cross validation is used to estimate test error associated with a given model
# cross validation can also be used to guage the level of flexibility in a model
# this process is known as a model assessment and model selection
# the bootstrap is used to provide a measure of accuracy of a parameter estimate given a model

## Cross validation
# we already discussed the distinction between the training error and the test error rate
# test error is the average error we get from applying our model build on a training dataset to the test dataset
# we use the results of test error to decide if our model is good
# we cannot in theory calculate the true test error - we need to estimate it by splitting our training dataset
# calculating the traiing error is easy but can differ wildly from the test error estimates

# we can a large amount of techniques we can use to estimate the testing error set using splits of our training data
# some make an adjustment to our training error
# some use cross validation or "holding out" some data to approximate our true test data
# we learn on the training data then apply the model to our test dataset - the dataset subset we held our from the learning process
# the key concepts of cross validation hold true for quantitiative and qualitative repsonses

## Validation Set Approach
# we want to estimate test error on a fitted model
# the validation set approach is a very simple strategy
# we randomly divide the available set of observations into two parts the training set and the hold out set
# we fit the model to the training set and that model is used to predict the observations in the hold out set
# we then guage the performance of the model based on a metric on the developed from predictions on the hold out set
# we typically use MSE in the case of a quantitiative repsonse - this provides and estimate of the test error rate

# we will go through an example with the Auto cars dataset
# we think that this relationship is not truely linear - a quadratic fit gave us better accuracy
# would a cubic or higher order polynomical fit give even better results? how do we figure this out?
# we could fit the model and check the significance of the polynomical terms (pvalue)
# we could also answer this questions using the validation set approach
# will randomly split the auto cars dataset into training and validation sets
# the validation error rates from predicting the model trained on the training dataset will be our estimate of test error rate
# we will use MSE as a measure of validation - the performance of our model
# we find that the MSE for a quadratic is lower than the simple linear regression with no polynomial term
# however we find that the MSE is for a cubic term is slightly larger than our quadratic regression
# this implies that including the cubic term does not lead to a better prediction of using just the quadratic term!!

# in order to create the validation set - we randomly split the data into two halves
# if we repeat the process of randomly splitting our data - we get slightly different subsamples and different MSE
# every time we use a new random split we will get slightly different samples the model is training and validated on
# we can examine all of these curves to hone in on our hypothesis
# after creating 10 random splits of our training and validation set we found:
# quadratic performs better than simple linear in all 10 random partitions
# quadratic may be better than cubic in all 10 random training - test splits
# this happens even though the MSE is slightly different for each sample!!
# we can determine that the linear model is not a great fit for our data

# the validation set appraoch is very simple and is easy to implement
# it does have to potential drawbacks:
# the estimate of our test error can be highly variable depending on the random sample we get to train and test our model
# in the validation approach - only a subset of observations are used in the training set
        # this can limit the potential sample size of the data we need to model
        # we could be training on fewer observations depending on the data
# this suggests that the validation set error rate may tend to overestimate the test error rate for the model fit on the entire dataset

# the next sections discuss cross validation - a improvement on the validation set approach that improves these two problem areas


## Leave on Out Cross Validation
# leave on out cross validation (LOOCV) is closely related to the valdiation set appraoch
# it attempst to address the validation sets drawbacks

# like the validation approach - we split the set of observations into two parts
# instead of creating two subsets of similiar size we use a single observation for our validation set!
# the remaining observations will be used as the training set
# we fit our model on the n - 1 dataset = all observations!
# we get our MSE test error estimate by fitting our model onto the one observation that we held out
# this estiamte is unbiased because we use more data to fit the model - but it is highly variable becuase we are only testing on one observation!
# TO DO THIS WE REPEAT THIS PROCESS N NUMBER OF TIMES - each using a different held out observation
# the LOOCV estimate for the test MSE is the AVERAGE OF ALL THIS INDIVIDUAL iterations of train on n-1 obs and predict on 1 observation
#  CV = 1/n sum(MSE)(i)

# LOOCV has a couple of major advantages over the validation set approach
# IT HAS FAR LESS BIAS!
# we repeatedly fit the statistical learning method using training sets that contain n - 1 observations
# we are training almost as many models as there are observations in the dataset!
# LOOCV tends not to overestimate the test error rate as much as validation approach
# in contrast to the validation approach will yields different results when apploed repeatedly due to randomness
# performing LOOCV multiple times will always yeild the same results - there is no randomness in the training / validation splits

# example: we used LOOCV on the Auto data set in order to obtain a test set MSE error estimate for a linear regression
# LOOCV has the potential to be expensive to implement since the model has to be fit n times
# this can be time conusming if N is large - and we are fitting a complex model

# with least squares linear or polynomial regression we have a cost reducing shortcut that takes the same time as a model fit!
# the following formula holds:
        # CV(n) = 1 / n sum((yi - yhat) / 1 - hi) ^ 2
# yi is the ith fitted value from the original least squares fit and hi is the leverage
# this is like the ordinary MSE except the ith residual is divided by 1 - leverage
# the leverage lies between 1/n and 1 and reflects the amount that an observation influences the fit of a model
# hence the residuals for high-leverage points are inflated in this formula by exactly the right amount for this equality to hold

# LOOCV is a very general method and can be used with any type of predictive modeling
# we could use it with logistic regressin or linear discriminant analysis
# be wary that using the leverage to short cut the MSE fit is only used in linear regression
# in classification and other cases - we will need to refit our model n times

## K Fold Cross Validation
# an alternative to LOOCV is k-fold cross validation
# this approach involves randomly dividing the set of observations into k groups or folds of almost equal size
# the first fold is treated as a validationset and the method is fit on the remaining k - 1 folds
# the MSE is then computed on the observations on the held-out fold
# this procedure is then repeated k times, each time, a different group of observations is treated as the validation set
# this process results in k estimates of the test error - MSE1, MSE2 ... MSEn
# the CV estimate is computed by averaging all these MSE estimates
        # CVk = 1 / k sum(MSE(i))
# it is not hard to see that LOOCV is a special case of k-fold CV in which k is set equal to n!!!
# k fold validation is usually used with k = 5 or k = 10
# why should we use k = 5 or 10 rather than LOOCV of k = n?
# the obvious advantage is computational cost
# LOOCV requires fitting our model n times
# this has the potential to being computationally expensive (expect of linear model which we have the leverage shortcut)

# cross validation is a very general appraoch that can be applied to almost as learning method
# some learning methods are very intensive in their model fit and LOOCV may take alot of time, especially if n is large
# in contrast perofrming 10-fold CV means only fitting the model k times which may be much easier
# k fold CV has other advantages involving the bias-variance tradeoff!!!
# k fold CV will give us some variability in the MSE test error estimate but it will be more stable than pure validation CV

# remember when we examine MSE we do not know the true MSE - we are estimating the true MSE with CV
# using simulation we can create a dataset and calculate the "true" MSE and compare it to our estimate to see how well we are doing
# using this method we can see that CV does do a good job of modelling the true MSE!!

# when we perform CV our goal might be to determine how well a learning method can be expected to perform on independent data
# at other times we are interested in the only the location of the MINIMUM POINT IN THE ESTIMATES MSE CURVE
# this is because we might be performing CV on a number of models, or on a single model with differently tuned flexibility parameters
# this helps us dial-in our model to the best fit with the lowest test error estimate
# for this case - the location of the minimum point in the estimated test MSE is not
# altought CV might underestimate the true MSE (from our simluation example) - CV does accruately model the level of flexibility needed to minimize MSE!!!


## Bias-Variance Trade-Off for K-Fold Validation
# k-fold CV has a computational advantage to LOOCV
# but k fold CV also gives more accurate estimates of the test error rate than LOOCV
# this has to do with the BIAS VARIANCE TRADEOFF

# we know that the validation cv appraoch can lead to onverestimates of the test error rate - we only train on roughly half the dataset
# using this we can see that LOOCV will give approximately unbiased estimates of the test error - we use all but one observation to fit our model
# k fold CV is between these two cv methods - we essentially scale pure validation cv by a level of K
# k fold will give us (k -1)*n / k observations to train on - fewer than LOOCV but more than VCV
# from the prospective of BIAS - LOOCV is preferred to k fold CV
# however we known that bias is not the only source for concern in an estimating procuedure we MUST CONSIDER VARIANCE!!
# it turns out that LOOCV has higher variance than k - fold CV
# why is this the case?
# when we perform LOOCV we are in effect averaging the output of n fitted models, each trained on almost a identical set of observations!
# therefore - all of our outputs are highly correlated with each other!
# in contrast, when we perform k-fold CV with k < n we are averaging outputs of k fitted models that are less correlated with each other
# the correlation is reduced because the overlap between the k training sets is each model is smaller!!!
# since the mean of many highly correlated quantities has higher variance than the mean of many quantities that are not highly correlated...
# the test error estiamte resulting from LOOCV tends to have higher variance than does the test error estiamte resulting from k fold CV

# to summarize: there is a bias-variance tradeoff associated with the choice of k in k fold CV
# given these considerations - one performs k fold CV using k = 5, or k = 10
# these k values have been shown to yeild test error rate estimates that minimize bias and variance!!!


## Cross Validation in Classification Problems
# cross validation can also be used with classification problems
# we cannot use MSE like we used in regression problems with CV
# instead of MSE we will use the number of misclassified observations
# for example, in the classification setting, the LOOCV error rate takes the form:
        # CVn = 1 / n sum(Err(i))
# where Err = I(yi != yhat)
# all CV methods error rates for classification problems are defined in the same way
# we can apply the same methods to classification to see if CV actually works...
# we can simulate data to give us a true  misclassiification error rate and see if CV gets close to it
# it is proven that misclassification error can be a good judge as to finding the "true" misclassification error rate
# remember that in practice - WE DO NOT KNOW THE TRUE MISCLASSIFICAITON ERROR RATE!!
# we can use CV to determine the best model to fit our data 
# as we know the training error will decrease as we ramp up the flexibility
# we known that the training error will match the characterisitic U-shape as complexiity increases
# CV can give us a good approximation of true error rate and can be the decider on which model and how flexible a model should be




## Bootstrap
# sampling with replacement!!
# the boostrap is a widely applicable and extrememly powerful statistical tool
# we can use it to quantify the uncertainty associated with a given estimator or statistical learning method
# as a simple exmaple: the bootsrap can be used to estimate the standard errors of the coefficients from a linear regression fit
# the power of the bootstrap lies in the fact that it can be easily applied to a wide range of statistical learning methods
# including some for which a measure of variability is otherwise difficult to obtain and is not given from a statistical software

# our example for this chapter will be a toy example trying to find the best way to allocate money under a simple model
# suppose we wish to invest a fixed sum of money in two financial assets that yield returns of X and Y 
# we will invest a fraction of our money in X and will invest the remaining in Y
# since there is variability associated with the returns on these two assets, we wish to minimize the variance of our investment
# we want to minimize Var($X + (1 - $)Y)
# one can show that the value that minimizes the risk is given by:
        # $ = sigma^2Y - sigma^2XY / sigma^2X + sigma^2Y - 2sigmaXY
# where sigma^2X = Var(X), sigma^2Y = Var(Y) and sigma(X,Y)  = Cov(X,Y)
# in reality these quantities are unknown
# we can estimate these quantities using a dataset that contains past measurements for X and Y
# we can then estimate the value of $ that minimizes the variance of our investment using this equation:
        # $hat = sigma^2Y - sigma^2XY / sigma^2X + sigma^2Y - 2sigmaXY
# we can simulate 100 different observatoins of the returns for X and Y many times to give us estimates for future X and Y
# then we want to understand the accuratacy of $ - the value that minimizes our risk
# to do this we continue to simluate our X and Y pairs 1,000 times - this gives us 1,000 estimates of our risk minimizer
# we have $1, $2, ... $1000
# using the simulation parameters we know as "true" we can figure out the truen value of $ is .6
# we can compare this to the mean of our 1000 estimates of $ and get very close to the true mean!!
# we can also use this approach to esimate the sd of $ - in our example we get .083
# this means - we have a very good idea of the accuracy of our estimates $: SE($) = .083
# this means - for a random sample from the population we would expect $ to differ from the true $ by almost .08 on average

# in practicem the procedure for estimating SE($) cannot be applied...
# for real data we cannot generate new samples from the original population
# this is where the bootstrap comes in
# the bootstrap approach allows us to emulate the process of obtaining new sample sets
# THIS ALLOWS US TO ESTIMATE THE VARIABILITY OF $ WITHOUT GENERATING ADDIITONAL SAMPLES!!
# rather than repeatedly obtaining independent datasets from the population...
# we instead obtain distinct data sets by repeated sampling observations FROM THE ORIGINAL DATASET!!!

# bootstrap example on a small dataset n = 3
# Z contains only n = 3 observations
# we randomly select n observations from the data set in order to produce a bootstrap dataset
# THIS SAMPLING IS DONE WITH REPLACEMENT!!
# THIS MEANS THAT THE SAME OBSERVATION CAN OCCUR MORE THAN ONCE IN THE BOOTSTRAP SET BUILDING!!
# WE REPLACE THE SAMPLE BACK IN THE ORIGINAL DATASET AFTER CREATING Z*1 AND THE OBSERVATION IS AVAILABLE FOR GENERATING Z*2 and all other boostrap datasets
# in this example Z*1 contains the third observation twice, the first observation once and no second observation
# we can then use Z*1 to produice a new bootstrap estimate for $ which we will call $*1
# this procedure is repeated B times for some large value of B in order to produce B different bootstrap datasets
# our bootstrap datasets are now: Z*1, Z*2, ..., Z*B
# for each of these datasets we now have B different $ estimates!!
# these are outlined as $*1, $*1, ..., $*B
# we can compute the standard error of these bootstrap estimates to get an approximation of $ from our original dataset
# boostraping can be used to effectively estimate the variability associated with $!!!!


## Lab: Cross Validation and the Bootstrap
# we will explore the resampling techniques covered in this chapter
library(ISLR)
data("Auto")
set.seed(1)

## Validation CV
# let's explore the validation cross validation method
# we will use this to estimate the test error rates that result from fitting various linear models
# we will use the Auto dataset
# whenever dealing with resampling methods - it is important to set our seed - this will ensure we can reproduce our results
# resampling is randomly sampling our dataset - setting the seed allows us to lock in the same sample as before
# we will get different results in resampling - we are taking random splits of the data at every resample!!

# using sample() to split our data into training and validation

# define our split of the data
train = sample(392, 196)

# fit a model on the training data using subset within the lm() call
lm.fit <- lm(mpg ~ horsepower, data = Auto, subset = train)
lm.fit$coefficients
# (Intercept)  horsepower 
# 40.3403772  -0.1617013 

# use our fitted model built on the training data to predict onto the test data
lm.pred <- predict(lm.fit, Auto[-train,])
head(lm.pred, 20)
# 1         2         3         6         7         8        11        12        13        14 
# 19.319212 13.659667 16.085186  8.323525  4.766097  5.574604 12.851161 14.468174 16.085186  3.957591 
# 16        17        19        20        22        23        25        27        30        31 
# 24.978756 24.655354 26.110665 32.902119 25.787263 24.978756 25.787263  8.000123 26.110665 25.787263

# now that we have predictions we can calculate the estimate of MSE!!
mean((Auto[-train,]$mpg - lm.pred)^2)
# [1] 26.14142

# therefore our estimated test MSE for the linear regression fit is 26.14

# we can continue to use this process of other polynomical linear regressions
lm.fit2 <- lm(mpg~poly(horsepower, 2), data = Auto, subset = train)
lm.fit2$coefficients
# (Intercept) poly(horsepower, 2)1 poly(horsepower, 2)2 
# 23.59853           -122.13747             40.19920 

# lets see our TESTING ERROR ESTIMATE!!
mean((Auto$mpg - predict(lm.fit2, Auto))[-train]^2)
# [1] 19.82259

# fit the third order polynomial and see the TESTING ERROR ESTIMATE
lm.fit3 <- lm(mpg~poly(horsepower, 3), data = Auto, subset = train)
lm.fit3$coefficients
# (Intercept) poly(horsepower, 3)1 poly(horsepower, 3)2 poly(horsepower, 3)3 
# 23.59585           -122.64299             40.03607             -2.81603 

# TESTING ERROR ESTIMATE
mean((Auto$mpg - predict(lm.fit3, Auto))[-train]^2)
# [1] 19.78252


# our testing error estimtaes are 19.8 and 19.7
# if we choose a different training set instread, then we will obtain somewhat different errors on the validation set
set.seed(2)

# define training set
# the training set size will not change - THE RANDOM SAMPLE TO BUILD THE TRAINING SAMPLE WILL!!
train = sample(392,196)

# model fit
lm.fit <- lm(mpg ~ horsepower, data = Auto, subset = train)
lm.fit$coefficients
# (Intercept)  horsepower 
# 40.2638277  -0.1565434 

# MSE estimate
mean((Auto$mpg - predict(lm.fit, Auto))[-train]^2)
# [1] 23.29559


# we can continue to use this process of other polynomical linear regressions
lm.fit2 <- lm(mpg~poly(horsepower, 2), data = Auto, subset = train)
lm.fit2$coefficients
# (Intercept) poly(horsepower, 2)1 poly(horsepower, 2)2 
# 23.66258           -122.62358             43.38035 

# lets see our TESTING ERROR ESTIMATE!!
mean((Auto$mpg - predict(lm.fit2, Auto))[-train]^2)
# [1] 18.90124

# fit the third order polynomial and see the TESTING ERROR ESTIMATE
lm.fit3 <- lm(mpg~poly(horsepower, 3), data = Auto, subset = train)
lm.fit3$coefficients
# (Intercept) poly(horsepower, 3)1 poly(horsepower, 3)2 poly(horsepower, 3)3 
# 23.670262          -123.239704            42.937391             4.328204

# TESTING ERROR ESTIMATE
mean((Auto$mpg - predict(lm.fit3, Auto))[-train]^2)
# [1] 19.2574

# using this split of observations into a training set and a validation set...
# we find that the validation set error rates for the models with linear, quadratic, and cubic terms are...
# 23.3, 18.9, and 19.26

# these results are consistent with our previous findings: 
# a model that predicts mpg using a quadratic function of horsepower performs better than a model that involves only a linear functinon of hp
# there is little evidence that we should use a model with a cubic function


## LOOCV
# the LOOCV estimate can be automatically computed for any generalized linear model using glm() and cv.glm() functions
# glm without a call to family will perform a linear regression!
# lm() and glm() with no family call will give the exact same results!!
# we want to use glm() in order to take advantage of cv.glm()
# the cv. option is available in the boot library
library(boot)

# fit glm.model
glm.fit <- glm(mpg ~ horsepower, data = Auto)
glm.fit$coefficients
# (Intercept)  horsepower 
# 39.9358610  -0.1578447 

# employ cv estimate of the test error
# provide cv.glm the dataset and the fitted model
cv <- cv.glm(Auto, glm.fit)

# what is the test error estimate?
cv$delta
# [1] 24.23151 24.23114

# what does cv.glm give us?
# the cv.glm() function produces a list with several components
# the two numbers inside delta contain the cross validation results
# in this case the numbers are identical
# the cross validation error estimate on the testing dataset is ~24.3

# we can repeat this process for increasingly complex polynomial fits
# to automate this process we will use a for loop
# we will fit poly regressions from order i = 1:5
# compute the cv error for each fit and then compare

# define vector to store our for loop outputs into
cv.error <- rep(0,5)

# define for loop
for (i in 1:5) {
        glm.fit <- glm(mpg ~ poly(horsepower, i), data = Auto) # fit the model with polynomial order = i
        cv.error[i] = cv.glm(Auto, glm.fit)$delta[1] # put the test estimate result into the cv.error list
        
}

# view results
cv.error
# [1] 24.23151 19.24821 19.33498 19.42443 19.03321

# we see a sharp drop in the estimated test MSE between the linear and quadratic fits
# but then no clear improvement from using higher order polynomials


## k Fold CV
# we can also use the cv.glm() function to perform k fold cv
# we will use k = 10 in our example
# we will use the Auto dataset
# make sure to set the random seed for reproducible results
set.seed(17)

# define list to store values
cv.error.10 <- rep(0,10)

# define for loop of fitting models
for (i in 1:10) {
        glm.fit = glm(mpg ~ poly(horsepower), data = Auto) # fit the model at different degrees of polynomial order
        cv.error.10[i] <- cv.glm(Auto, glm.fit, K = 10)$delta[1] # store the kfcv result for each fit 
}

# view results
cv.error.10
# [1] 24.20520 24.24309 24.30761 24.20218 24.22832 24.21199 24.14792 24.31201 24.09372 24.50921

# notice that our computational time is much shorter than LOOCV
# we still see little evidence that using cubic or high-order polynomials terms leads to lower test error
# it seems the quadratic fit is the best fit for this data problem

# we saw in the previous example that under LOOCV the cross valdation results as essentially identical
# look at the delta estimates
# when we perform kfcv the two numbers actually differ slightly
# in kfcv the first delta is the standard k fold estiamte
# the second delta is the bias corrected version
# even with the difference both are very close to each other


## Bootstrap
# now let's illustrate an example of the bootstrap
# to do this we will use a simple example from the auto dataset

## Estimating accuracy of a statistic of interest
# one of the advantages of the bootstrap is that it can be applied in almost any situation
# no complicated mathmatical calculations are required
# performing a bootstrap in R requires two simple steps:
# first we create a function that calculates a statistic of interest
# second we use the boot function from the boot library to perform the bootstrap repeatedly sampling observations from the data WITH REPLACEMENT!
library(ISLR)
data("Portfolio")

# first we create the function alpha.fn()
# the function takes in (X, Y) data as well as a vector indicating which observations should be used to estimate alpha
# the function then outputs the estimate for alpha based on the selected observations
# alpha is the value that minimizes our portfolio risk
alpha.fn <- function(data, index) {
        X <- data$X[index]
        Y <-  data$Y[index]
        return((var(Y) - cov(X,Y)) / (var(X) + var(Y) - 2*cov(X,Y)))
}

# this function returns or outputs an estimate for alpha based on applying equation 5.7 to the observations given in index
# for example - the following command tells R to estimate alpha using 100 observations
# samples with 1,2,...1000
alpha.fn(Portfolio, 1:100)
# [1] 0.5758321

# the next command uses the sample() function to randomly select 100 observations from the range 1:100 with replacement
# this is equivalent to constructing a new bootstrap data set and recomputing alpha based on the new dataset
# samples with a random mix of 100 observations from the set 1:100 - we can repeat numbers!!
set.seed(1)
alpha.fn(Portfolio, sample(100, 100, replace = T))
# [1] 0.5963833

# we can implement a bootstrap analysis by performing this command many times
# recording all of the corresponding estimates for alpha and then computing the resulting standard deviation
# however the boot function automates this apporach
# below we will produce R = 1000 bootstrap estimates for alpha
# creates 1000 iterations of the 100 observation random sample!!
library(boot)

# implement bootstrap with R = 1000
boot(Portfolio, alpha.fn, R = 1000)

# ORDINARY NONPARAMETRIC BOOTSTRAP
# 
# Call:
# boot(data = Portfolio, statistic = alpha.fn, R = 1000)
# 
# 
# Bootstrap Statistics :
# original        bias    std. error
# t1* 0.5758321 -7.315422e-05  0.08861826

# the final output shows that using the original data, alpha = .5758 and the bootstrap estaimte for SE(alpha) = .0886!!!


## Estimating the Accuracy of a linear regression model
# the bootstrap approach can be used to assess the variability of the coefficients estimates and predictions from a statistical learning method
# we will use the bootstrap to assess the variability of the estimates for B0 and B1 of a linear regression model
# we will compare these estimates against the formulas for SE(B0) and SE(B1)

# we first create a simple function boot.fn() that fits a linear model on our data and returns B0 and B1 estimates
# we then apply this function to the full set of 392 observations in order to compute the esitmates of B0 and B1 on the entire data set

# coefficient estimate function in one line!
boot.fn <- function(data, index) return(coef(lm(mpg ~ horsepower, data = data, subset = index)))

# apply the function to the entire dataset
# gives us estimates for B0 and B1
boot.fn(Auto, 1:max(nrow((Auto))))
# (Intercept)  horsepower 
# 39.9358610  -0.1578447

# the boot.fn() function can also be used in order to create bootstrap estimates
# we can randomly sample among the observations to minmic bootstrap resampling

# example
set.seed(1)

# run the bootstrap function with a random sample of the actual data
boot.fn(Auto, sample(392, 392, replace = T))
# (Intercept)  horsepower 
# 38.7387134  -0.1481952

# another iteration - different results!!
# we are estimating the Intercept and Coefficient by randomly sampling the data
# we expect slightly different results for each random sample
# this is how bootstrapping works - we iterate this process many times and average all the estimates together!
boot.fn(Auto, sample(392, 392, replace = T))
# (Intercept)  horsepower 
# 40.0383086  -0.1596104 

# next we use the boot() function to compute the standard errors of 1000 bootstrap estimates for the slope and coefficients

# boot function for coefficent and slope estimates R = 1000
boot(Auto, boot.fn, 1000)
# ORDINARY NONPARAMETRIC BOOTSTRAP
# 
# Call:
# boot(data = Auto, statistic = boot.fn, R = 1000)
# 
# 
# Bootstrap Statistics :
# original      bias    std. error
# t1* 39.9358610  0.02972191 0.860007896
# t2* -0.1578447 -0.00030823 0.007404467

# this indicates that the bootstrap estimate for SE(BO) is .86
# the bootstrap estimate for SE(B1) is .0074
# we now will compare these estimates to the standard formulas for standard errors of regression coefficients
# we can get these from the summary function
summary(lm(mpg~horsepower, data = Auto))$coef
#                 Estimate  Std. Error   t value      Pr(>|t|)
# (Intercept) 39.9358610 0.717498656  55.65984 1.220362e-187
# horsepower  -0.1578447 0.006445501 -24.48914  7.031989e-81

# the standard errors for B0 is .717 and the standard error estimate for B1 is .006
# these are different from our bootstrap estimates
# what does this mean?
# it means the bootstrap is actually performing well!
# recall that the standard errors given in equation 3.8 (page 66) rely on certain assumptions
# these equations depend on the unknown parameter simga^2 the variance
# we estimate simga^2 using the residual sum of squared error!!
# the formula for standard errors does not need the linear model to be correct - our estimate of variance does!!
# we see in figure 3.8 on page 91 that there is a non-linear relationship in the data!
# the residuals from a linear fit will be inflated and so will our variance!!
# secondly our formulas assume that the x are fixed and all the variability comes from the variation in the errors ei
# THE BOOTSTRAP APPROACH DOES NOT RELY ON ANY OF THESE ASSUMPTIONS!!
# the bootstrap is likely giving a more accurate estimate of the standard errors of B0 and B1 than the summary function!

# below we compute the bootstrap standard error estimates and the standard linear regression estimates that result from fitting a quadratic model to the same data
# since this model provides a good fit to the data - we should see a closer match between SE formulas and SE bootstrap estimates

# quadratic coefficient function
boot.fn = function(data, index) coefficients(lm(mpg~horsepower + I(horsepower ^2), data = data, 
                                                subset = index))

# bootstrap the quadratic fit
boot(Auto, boot.fn, 10000)

# ORDINARY NONPARAMETRIC BOOTSTRAP
# 
# 
# Call:
# boot(data = Auto, statistic = boot.fn, R = 10000)
# 
# 
# Bootstrap Statistics :
#         original        bias    std. error
# t1* 56.900099702  4.987521e-02 2.102357413
# t2* -0.466189630 -8.649327e-04 0.033462035
# t3*  0.001230536  3.583313e-06 0.000120856


# compare with the summary formulas
# the results are much closer!
# a better fitting model will give us better estimates of standard error of the coefficients!
# the bootstrap can narrow in on the the true standard error estimates becuase it does not rely on assummptions only the given data!!
summary(lm(mpg~horsepower + I(horsepower ^2), data = Auto))$coef
# Estimate   Std. Error   t value      Pr(>|t|)
# (Intercept)     56.900099702 1.8004268063  31.60367 1.740911e-109
# horsepower      -0.466189630 0.0311246171 -14.97816  2.289429e-40
# I(horsepower^2)  0.001230536 0.0001220759  10.08009  2.196340e-21





## Chapter 5: Conceptual Exercises


## Question 1:
# using the basic statistical properites of the variance - derive equation 5.6
# in other words prove that alpha given by 5.6 does indeed maximize Var(alphaX + (1-alpha)Y)

## Solution:
# Using the following rules:
#         Var(X+Y)=Var(X)+Var(Y)+2Cov(X,Y)Var(cX)=c2Var(X)Cov(cX,Y)=Cov(X,cY)=cCov(X,Y)
#         Var(cX)=c2Var(X)
#         Cov(cX,Y)=Cov(X,cY)=cCov(X,Y)

# Minimizing two-asset financial portfolio:
#         Var(αX+(1−α)Y)
#         =Var(αX)+Var((1−α)Y)+2Cov(αX,(1−α)Y)
#         =α2Var(X)+(1−α)2Var(Y)+2α(1−α)Cov(X,Y)
#         =σ2Xα2+σ2Y(1−α)2+2σXY(−α2+α)        

# Take the first derivative to find critical points:
#         0=ddαf(α)
#         0=2σ2Xα+2σ2Y(1−α)(−1)+2σXY(−2α+1)
#         0=σ2Xα+σ2Y(α−1)+σXY(−2α+1)
#         0=(σ2X+σ2Y−2σXY)α−σ2Y+σXY
#         α=σ2Y−σXY / σ2X+σ2Y−2σXY





## Question 2:
# we will now derive the probability that a given observation is part of a bootstrap sample
# suppose we obtain a bootstrap sample from a set of n observations

# a. 
# what is the probability that the first bootstrap observation is not the jth observation from the original sample?
# we take a sample from n - what is the probability it is not j?
# first - what is the probability a sample in n will be in j?
        # we can infer this is 1 / the total set of n 
        # if n = 30, the chance that j is in n is 1/30
# now - we should flip this probability around to understand the probability a sample of will NOT have j
        # this is simply = 1 - (1 / N)
        # if n = 30, the chance that j is NOT in n is 1 - (1/30) = 29/30 = ~96%

# b. 
# what is the probability that the second bootstrap is not the jth observation from the original sample
# WE KNOW BOOTSTRAPPING IS SAMPLING WITH REPLACEMENT
# this means that the second bootstrap observation will have the same probability of not being the jth observation!
# 1 - (1/n)

# c. 
# argue that the probability that the jth observation is not in the bootstrap sample  is (1 - 1/n)^n
# to solve this question we need to chain the single bootstraps probabilites together
# we are trying to see if j with be in the entire bootstrap sample AT ALL
# this probabilitiy is the generalized chained independent proabilites (1 - (1/n)) multiplied to the total number of samples n
# this generalizes to (1 - (1/n))^n!!

#d. 
# when n = 5, what is the probability that the jth observation IS in the bootstrap sample? (the entire bootstrap sample)
# to solve this we use our formula from the question above
# solution: (1 - (1 / 5)) ^ 5
n = 5
(boot_n <- (1 - 1/n)^n)
(p_in <- 1 - boot_n)
# [1] 0.67232

# e. 
# when n = 100, what is the probability that the jth observation IS in the bootstrap sample? (the entire bootstrap sample)
# to solve this we use our formula from the question above
# solution: (1 - (1 / 100)) ^ 100
n = 100
(boot_n <- (1 - 1/n)^n)
(p_in <- 1 - boot_n)
# [1] 0.6339677

# f. 
# when n = 1000, what is the probability that the jth observation IS in the bootstrap sample? (the entire bootstrap sample)
# to solve this we use our formula from the question above
# solution: (1 - (1 / 1000)) ^ 1000
n = 1000
(boot_n <- (1 - 1/n)^n)
(p_in <- 1 - boot_n)
# [1] 0.6323046

# g. 
# create a plot the displays...
# for each integer value of n from 1 to 100000 the probability that the jth observation is in the bootstrap sample
# need to make a data frame with 1 - 100000 with the calculated bootstrap IN probabilites to graph
df <- tibble(
        value = rep(1:1000), bootstrap = 1 - (1 - 1/value)^value
)

plot(df$value, df$bootstrap)
# It's clear from the graph that the function \( 1-(1-\frac{1}{n})^n \) reaches an asymptote (of around 63% given the previous answers).

# h.
# we now investigate numerically the probability that a bootstrap sample of size n = 100 contains the jth observation
# here we use j = 4
# we repeatedly create bootstrap samples and record whether or not we find j contained in the sample
store = rep(NA, 100000)

for (i in 1:100000) {
        store[i] = sum(sample(1:100, rep = T) == 4) > 0
}

mean(store)
# [1] 0.63283

# As expected, with a large number of observations, 
# the chance that a bootstrap sample will contain a particular data point from the original sample is about 2/3
# - as demonstrated in the asymptotic function graph in part g.



## Question 3:
# we now review k-fold cross validation

# a. 
# explain how k-fold cross validation is implemented
# k fold cv is implemented by taking a set of n observations and randomly splitting them into k non-overlapping groups
# each group gets a chance to be the validation set with the remaining k - 1 groups set to the training set
# we get our test error estimate by averaging the MSE over each of our k paritions

# b.
# what are the advantages and disadvantages of k fold cv?
# how is it better or worse than validation set approach?
# how is it better than LOOCV?
# KFCV is better than VSCV becuase it helps reduce variance that a simple test and train split may have (VSCV)
# in VSCV the testing error estimate can vary dramatically based on the splits we used and observations we included in test and training
# the validation set may also tend to overestimate the test error on the entire dataset
# LOOCV is a special case of KFCV - we create k = n number of paritions - training on all but one observation and predicting on just one observation
# this alone is very computationally expensive for large datasets
# also because we are predicting on just one observation every time - LOOCV tends to have higher variance than KFCV
# KFCV tends to be the great middle ground between LOOCV and VSCV
# LOOCV will have higher variance, VSCV will have higher bias - K FOLD CV attempts to reduce both in estiamting testing error!



## Question 4:
# suppose we use some statistical learning method to make prediction for response Y for a particular value of the predictor X
# carefully describe how we may estimate the standard deviation of our prediction

## Solution:
# to solve this question we need to understand bootstrapping
# we can bootstrap samples from our dataset and compute the stand deviation for each bootstrap sample
# bootstrapping is sampling with replacement
# we can then average all bootstrap samples together to get an idea of the predictions standard deviation





## Chapter 5: Applied Exercises


## Question 5:
# in chapter 4 we used logistic regression to predict the probability of default using income and balance
# we will now estimate the test error of this logistic regression model using the validation set approach
# do not forget to set seed before performing the analysis

# a. 
# fit a logistic regression model that uses income and balance to predict default

# load data
library(ISLR)
data("Default")

# fit our logistic regression model
# do not forget to specify the family = binomial
log_fit  <- glm(default ~ balance + income, data = Default, family = 'binomial')
summary(log_fit)
# Call:
# glm(formula = default ~ balance + income, family = "binomial", 
#             data = Default)
# 
# Deviance Residuals: 
#         Min       1Q   Median       3Q      Max  
# -2.4725  -0.1444  -0.0574  -0.0211   3.7245  
# 
# Coefficients:
#         Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -1.154e+01  4.348e-01 -26.545  < 2e-16 ***
# balance      5.647e-03  2.274e-04  24.836  < 2e-16 ***
# income       2.081e-05  4.985e-06   4.174 2.99e-05 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 2920.6  on 9999  degrees of freedom
# Residual deviance: 1579.0  on 9997  degrees of freedom
# AIC: 1585
# 
# Number of Fisher Scoring iterations: 8


# b. 
# use the validation set approach  to estimate the test error of this model
# we must split our data into test and training
# fit a multiple logistic regression model on the training observations
# obtain the predictions on the validation set
# set the probability to > .5 Yes, else No
# compute the validation set error using a confusion matrix

set.seed(2)

# create training and test splits
inTrain <- createDataPartition(Default$default, p = .6, list = F)

train <- Default %>% 
        filter(row_number() %in% inTrain) %>% 
        as_tibble()

test <- Default %>% 
        filter(!row_number() %in% inTrain) %>% 
        as_tibble()

# fit our model
log_fit  <- glm(default ~ balance + income, data = train, family = 'binomial')

# get predictions
(log_pred <- predict(log_fit, newdata = test, type = 'response') %>% 
        as_tibble() %>% 
        mutate(predict = if_else(value > .5, 'Yes', 'No')))

# get confusionmatrix
confusionMatrix(test$default, log_pred$predict)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  3850   16
# Yes   91   42
# 
# Accuracy : 0.9732         
# 95% CI : (0.9678, 0.978)
# No Information Rate : 0.9855         
# P-Value [Acc > NIR] : 1              
# 
# Kappa : 0.4282         
# Mcnemar's Test P-Value : 8.438e-13      
# 
# Sensitivity : 0.9769         
# Specificity : 0.7241         
# Pos Pred Value : 0.9959         
# Neg Pred Value : 0.3158         
# Prevalence : 0.9855         
# Detection Rate : 0.9627         
# Detection Prevalence : 0.9667         
# Balanced Accuracy : 0.8505         
# 
# 'Positive' Class : No 

# our test set error estimate is 97%!


# c.
# repeat this process three times using three different splits of the dat

# create training and test splits
inTrain <- createDataPartition(Default$default, p = .75, list = F)

train <- Default %>% 
        filter(row_number() %in% inTrain) %>% 
        as_tibble()

test <- Default %>% 
        filter(!row_number() %in% inTrain) %>% 
        as_tibble()

# fit our model
log_fit  <- glm(default ~ balance + income, data = train, family = 'binomial')

# get predictions
(log_pred <- predict(log_fit, newdata = test, type = 'response') %>% 
                as_tibble() %>% 
                mutate(predict = if_else(value > .5, 'Yes', 'No')))

# get confusionmatrix
confusionMatrix(test$default, log_pred$predict)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  2410    6
# Yes   53   30
# 
# Accuracy : 0.9764         
# 95% CI : (0.9697, 0.982)
# No Information Rate : 0.9856         
# P-Value [Acc > NIR] : 0.9999         
# 
# Kappa : 0.494          
# Mcnemar's Test P-Value : 2.115e-09      
# 
# Sensitivity : 0.9785         
# Specificity : 0.8333         
# Pos Pred Value : 0.9975         
# Neg Pred Value : 0.3614         
# Prevalence : 0.9856         
# Detection Rate : 0.9644         
# Detection Prevalence : 0.9668         
# Balanced Accuracy : 0.9059         
# 
# 'Positive' Class : No 

# create training and test splits
inTrain <- createDataPartition(Default$default, p = .5, list = F)

train <- Default %>% 
        filter(row_number() %in% inTrain) %>% 
        as_tibble()

test <- Default %>% 
        filter(!row_number() %in% inTrain) %>% 
        as_tibble()

# fit our model
log_fit  <- glm(default ~ balance + income, data = train, family = 'binomial')

# get predictions
(log_pred <- predict(log_fit, newdata = test, type = 'response') %>% 
                as_tibble() %>% 
                mutate(predict = if_else(value > .5, 'Yes', 'No')))

# get confusionmatrix
confusionMatrix(test$default, log_pred$predict)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  4818   15
# Yes  110   56
# 
# Accuracy : 0.975           
# 95% CI : (0.9703, 0.9791)
# No Information Rate : 0.9858          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.4619          
# Mcnemar's Test P-Value : <2e-16          
# 
# Sensitivity : 0.9777          
# Specificity : 0.7887          
# Pos Pred Value : 0.9969          
# Neg Pred Value : 0.3373          
# Prevalence : 0.9858          
# Detection Rate : 0.9638          
# Detection Prevalence : 0.9668          
# Balanced Accuracy : 0.8832          
# 
# 'Positive' Class : No 

# create training and test splits
inTrain <- createDataPartition(Default$default, p = .99, list = F)

train <- Default %>% 
        filter(row_number() %in% inTrain) %>% 
        as_tibble()

test <- Default %>% 
        filter(!row_number() %in% inTrain) %>% 
        as_tibble()

# fit our model
log_fit  <- glm(default ~ balance + income, data = train, family = 'binomial')

# get predictions
(log_pred <- predict(log_fit, newdata = test, type = 'response') %>% 
                as_tibble() %>% 
                mutate(predict = if_else(value > .5, 'Yes', 'No')))

# get confusionmatrix
confusionMatrix(test$default, log_pred$predict)
# Confusion Matrix and Statistics
# 
#           Reference
# Prediction No Yes
#       No  96   0
#       Yes  2   1
# 
# Accuracy : 0.9798          
# 95% CI : (0.9289, 0.9975)
# No Information Rate : 0.9899          
# P-Value [Acc > NIR] : 0.9206          
# 
# Kappa : 0.4923          
# Mcnemar's Test P-Value : 0.4795          
# 
# Sensitivity : 0.9796          
# Specificity : 1.0000          
# Pos Pred Value : 1.0000          
# Neg Pred Value : 0.3333          
# Prevalence : 0.9899          
# Detection Rate : 0.9697          
# Detection Prevalence : 0.9697          
# Balanced Accuracy : 0.9898          
# 
# 'Positive' Class : No 


# d.
# now consider a logistic regression model that predicts the probability of default using income balance and student
# estimate the test error for this model using the validation set approach
# does including our student variable help our model?
# create training and test splits
inTrain <- createDataPartition(Default$default, p = .73, list = F)

train <- Default %>% 
        filter(row_number() %in% inTrain) %>% 
        as_tibble()

test <- Default %>% 
        filter(!row_number() %in% inTrain) %>% 
        as_tibble()

# fit our model
log_fit  <- glm(default ~ balance + income + student, data = train, family = 'binomial')

# get predictions
(log_pred <- predict(log_fit, newdata = test, type = 'response') %>% 
                as_tibble() %>% 
                mutate(predict = if_else(value > .5, 'Yes', 'No')))

(gather_pred <- test %>% 
        mutate(pred = predict(log_fit, newdata = test, type = 'response'),
               pred_class = if_else(pred > .5, 'Yes', 'No'),
               missed = default == pred_class)
        )

1 - mean(gather_pred$missed)
# [1] 0.02371249

# get confusionmatrix
confusionMatrix(test$default, log_pred$predict)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  2597   13
# Yes   51   38
# 
# Accuracy : 0.9763          
# 95% CI : (0.9698, 0.9817)
# No Information Rate : 0.9811          
# P-Value [Acc > NIR] : 0.9683          
# 
# Kappa : 0.5316          
# Mcnemar's Test P-Value : 3.746e-06       
# 
# Sensitivity : 0.9807          
# Specificity : 0.7451          
# Pos Pred Value : 0.9950          
# Neg Pred Value : 0.4270          
# Prevalence : 0.9811          
# Detection Rate : 0.9622          
# Detection Prevalence : 0.9670          
# Balanced Accuracy : 0.8629          
# 
# 'Positive' Class : No    





## Question 6
# we will compute the estimates for the standard errors of our logistic regression using bootstrap and the normal formulas

# a. 
# using the summary function and glm function - determine the estimated standard errors for the coefficients of our model

# fit our model
log_fit <- glm(default ~ income + balance, data = Default, family = 'binomial')

# fit the standard error estimates
summary(log_fit)$coef

#               Estimate   Std. Error    z value      Pr(>|z|)
# (Intercept) -1.154047e+01 4.347564e-01 -26.544680 2.958355e-155
# income       2.080898e-05 4.985167e-06   4.174178  2.990638e-05
# balance      5.647103e-03 2.273731e-04  24.836280 3.638120e-136

# b. 
# write a function that takes as input the defualt data set as well as an index of observations...
# and then outputs the coefficient estimates for income and balance in the multiple logistic model
boot.fn <- function(data, index) return(coef(glm(default ~ income + balance,
                                                 data = data, family = 'binomial', subset = index)))

# c. 
# use the boot() function together with your boot.fn() functino to estimate the standard errors of our model
library(boot)

# iterate over our boot.fn 50 times
boot(Default, boot.fn, R = 50)

# ORDINARY NONPARAMETRIC BOOTSTRAP
# 
# 
# Call:
# boot(data = Default, statistic = boot.fn, R = 50)
# 
# 
# Bootstrap Statistics :
#         original        bias     std. error
# t1* -1.154047e+01 -1.075424e-01 4.383801e-01
# t2*  2.080898e-05  1.088770e-06 5.810581e-06
# t3*  5.647103e-03  3.994582e-05 2.114837e-04

# d. 
# comment on the differences we see in the standard "formula" estimates and the bootstrap estimates
# we see slight differences in std error by formula and by bootstrap
# it is possible that our formula estimates are biased due to confounding variables


## Question 7:
# we saw that the cv.glm() function can be used in order to compute the LOOCV test error estimate
# alternatively we could compute this using glm() and predict() and a common for loop
# let's test this out
data("Weekly")

# a. 
# fit a logistic regression model that predictions Direction using Lag1 and Lag2
log_fit <- glm(Direction ~ Lag1 + Lag2, data = Weekly, family = 'binomial')

# b. 
# fit a logistic regression model that predicts Direction using the Lags using all but the first observation

# build out test and traing splits
train <- Weekly %>% 
        filter(row_number() != 1)

test <- Weekly %>% 
        filter(row_number() == 1)

# fit our model on the training dataset
log_fit <- glm(Direction ~ Lag1 + Lag2, data = train, family = 'binomial')


# c. 
# use the model in b to predict the direction of the first observation
# was our observation correctly classified?
# predict back on to the single held out value!
(log_pred <- test %>% 
        as_tibble() %>% 
        mutate(pred = predict(newdata = test, log_fit, type = 'response'),
               pred_class = if_else(pred > .5, 'Up', 'Down'),
               accuracy = pred_class == Direction)
)

# we predicted Up but the true direction was actually down!!


# d. 
# write a loop from i = 1:n where n is the number of observations in the dataset that performs that following steps:
# fit a logistic regression using all but the ith observation
# compute the posterior probability of the market moving up for the ith observation
# use the posterior probability for the ith observation in order to predict UP or DOWN
# determine whether or not an error was made in predicting the direction for the ith observation

misses <- NULL
flips <- NULL

for (i in 1:1089) {
        
        train <- Weekly %>% 
                as_tibble() %>% 
                filter(row_number() != i)
        
        test <- Weekly %>% 
                as_tibble() %>% 
                filter(row_number() == i)
        
        # fit our model on the training dataset
        log_fit <- glm(Direction ~ Lag1 + Lag2, data = train, family = 'binomial')
        
        log_pred <- test %>% 
                        as_tibble() %>% 
                        mutate(pred = predict(newdata = test, log_fit, type = 'response'),
                               pred_class = if_else(pred > .5, 'Up', 'Down'),
                               accuracy = (pred_class == Direction) * 1)
        
        misses <- rbind(misses, log_pred$accuracy) 
        
        post_prob <- sum(train$Direction == 'Up') / nrow(train)
        
        
        if (rbinom(n = 1, 1, p = post_prob) == 1) {
                flip = 1
        } else {
                flip = 0
        }
        
        
        flips <- rbind(flips, flip)
   
}

# d. 
# determine the test error estimates for the logistic model and the prior probability model
sum(misses)
sum(flips)
mean(misses)
mean(flips)



## Question 8:
# we will now perform cross validation on a simulated data set

# a. 
# generate a simulated data set as follows:
# in this dataset what is n and what is p?
# write out the model to generate this in equation form
set.seed(1)
x <- rnorm(100)
y <- x - 2*x^2 + rnorm(100)

df <- data.frame(x,y)

# solution:
# n is 100
# p is the number of predictions = we have two predictors x, y
# the equation is 
y = x - 2*x^2 + rnorm(100)

# b. 
# create a scatterplot of X against Y - what do we find?
ggplot(data = df , aes(x, y)) + 
        geom_point()

# c. 
# set a seed and then compute the LOOCV erros that result from fitting the following four models
# list of higher order polynomial functions
library(boot)
set.seed(1)


# normal glm fit
glm.fit <- glm(y ~ x, data = df)

# cv 
cv.glm(df, glm.fit)$delta
# [1] 21.16342 21.14793


# poly 2 fit glm
glm.fit2 <- glm(y~poly(x,2), data = df)

# cv 
cv.glm(df, glm.fit2)$delta
# [1] 1.005410 1.004873


# poly 3 fit glm
glm.fit3 <- glm(y~poly(x,3), data = df)

# cv 
cv.glm(df, glm.fit3)$delta
# [1] 1.111552 1.109926


# poly 4 fit glm
glm.fit4 <- glm(y~poly(x,4), data = df)

# cv 
cv.glm(df, glm.fit4)$delta
# [1] 1.057551 1.056014



# d. 
# repeat the steps above with another random seed set
set.seed(159753456)

# normal glm fit
glm.fit <- glm(y ~ x, data = df)

# cv 
cv.glm(df, glm.fit)$delta
# [1] 21.16342 21.14793


# poly 2 fit glm
glm.fit2 <- glm(y~poly(x,2), data = df)

# cv 
cv.glm(df, glm.fit2)$delta
# [1] 1.005410 1.004873


# poly 3 fit glm
glm.fit3 <- glm(y~poly(x,3), data = df)

# cv 
cv.glm(df, glm.fit3)$delta
# [1] 1.111552 1.109926


# poly 4 fit glm
glm.fit4 <- glm(y~poly(x,4), data = df)

# cv 
cv.glm(df, glm.fit4)$delta
# [1] 1.057551 1.056014

# THE RESULTS ARE EXACTLY THE SAME!!
# LOOCV WILL DO THE SAME VALIDATION STEPS FOR ANY DATASET!!
# WE WILL EVENTUALLY PARTIION THE DATA N TIMES!!

# e. 
# the lowest is the quadratic polynomial model has the lowest LOOCV test error rate
# this was expected because it matches the true form of Y

# f. 
# comment on the coefficient statistical significance  that result from fitting each of the models in c

summary(glm.fit)$coef
# Estimate Std. Error   t value     Pr(>|t|)
# (Intercept) -2.7850616  0.4313668 -6.456366 4.129669e-09
# x            0.2627009  0.3704604  0.709120 4.799336e-01
summary(glm.fit2)$coef
# Estimate Std. Error    t value     Pr(>|t|)
# (Intercept)  -2.795342   0.097079 -28.794509 2.546417e-49
# poly(x, 2)1   3.057180   0.970790   3.149167 2.176685e-03
# poly(x, 2)2 -41.594249   0.970790 -42.845772 7.891902e-65
summary(glm.fit3)$coef
# Estimate Std. Error    t value     Pr(>|t|)
# (Intercept)  -2.795342 0.09672345 -28.900358 3.690289e-49
# poly(x, 3)1   3.057180 0.96723446   3.160743 2.105768e-03
# poly(x, 3)2 -41.594249 0.96723446 -43.003274 1.578678e-64
# poly(x, 3)3  -1.266471 0.96723446  -1.309373 1.935331e-01
summary(glm.fit4)$coef
# Estimate Std. Error    t value     Pr(>|t|)
# (Intercept)  -2.795342 0.09590439 -29.147177 3.572455e-49
# poly(x, 4)1   3.057180 0.95904390   3.187737 1.941708e-03
# poly(x, 4)2 -41.594249 0.95904390 -43.370537 2.062358e-64
# poly(x, 4)3  -1.266471 0.95904390  -1.320556 1.898221e-01
# poly(x, 4)4  -1.560252 0.95904390  -1.626883 1.070746e-01

# p-values show statistical significance of linear and quadratic terms, which agrees with the CV results.



## Question 9
# we will now consider the Boston housing data set from the MASS library
library(MASS)
data("Boston")

# a. 
# based on this dataset provide an estimate for the population mean of medv
(u_medv <- mean(Boston$medv))
# [1] 22.53281

# b. 
# based on this dataset provide an estimate for the standard error of u
# remember = standard error is the standard deviation divided by the squared root of n
nobs_bos <- nrow(Boston)
sd_medv <- sd(Boston$medv)
(se_medv <- sd_medv / sqrt(nobs_bos))
# [1] 0.4088611


# c. 
# now estimate the standard error of u using the bootstrap
# how does this compare to our "formula" standard error?
library(boot)
set.seed(2)

# se function
bootMean <- function(data, index) {
        mean(data[index])
        
}

# iterate over our boot.fn 50 times
boot(Boston$medv, bootMean, R = 1000)

# ORDINARY NONPARAMETRIC BOOTSTRAP
# 
# 
# Call:
# boot(data = Boston$medv, statistic = bootMean, R = 1000)
# 
# 
# Bootstrap Statistics :
# original      bias    std. error
# t1* 22.53281 -0.01969308   0.4192166


# d. 
# based on your bootstrap estimate provide a 95% confidence interval for the mean of medv
# compare it to the results obstained using t.test(Boston$medv)

# one sample t test to estimate 95% CI
(t.test.CI <- t.test(Boston$medv))
# One Sample t-test
# 
# data:  Boston$medv
# t = 55.111, df = 505, p-value < 2.2e-16
# alternative hypothesis: true mean is not equal to 0
# 95 percent confidence interval:
#         21.72953 23.33608
# sample estimates:
#         mean of x 
# 22.53281 

# CI using our bootstrap computed SE
boot.ci(boot_u, conf = .95, type = 'all')
# BOOTSTRAP CONFIDENCE INTERVAL CALCULATIONS
# Based on 1000 bootstrap replicates
# 
# CALL : 
#         boot.ci(boot.out = boot_u, conf = 0.95, type = "all")
# 
# Intervals : 
#         Level      Normal              Basic         
# 95%   (21.71, 23.30 )   (21.68, 23.31 )  
# 
# Level     Percentile            BCa          
# 95%   (21.75, 23.39 )   (21.69, 23.33 )  
# Calculations and Intervals on Original Scale


# e. 
# based on this data set provide an estimate for the median of medv
(u_medv <- median(Boston$medv))
# [1] 21.2


# f. 
# now estimate the standard error of the median using the bootstrap
library(boot)
set.seed(2)

# se function
bootMedian <- function(data, index) {
        median(data[index])
        
}

# iterate over our boot.fn 1000 times
boot(Boston$medv, bootMedian, R = 1000)
# ORDINARY NONPARAMETRIC BOOTSTRAP
# 
# 
# Call:
# boot(data = Boston$medv, statistic = bootMedian, R = 1000)
# 
# 
# Bootstrap Statistics :
#       original  bias    std. error
# t1*     21.2 -0.0414   0.3793151


# g. 
# based on this dataset provide an estimate for the 10th percentile of medv in Boston
(q.10 <- quantile(Boston$medv, .1))
# 10% 
# 12.75 


# f. 
# use the bootstrap to estimate the standard error of q.10
library(boot)
set.seed(2)

# se function
boot_q10 <- function(data, index) {
        quantile(data[index], .1)
        
}

# iterate over our boot.fn 1000 times
boot(Boston$medv, boot_q10, R = 1000)
# ORDINARY NONPARAMETRIC BOOTSTRAP
# 
# 
# Call:
# boot(data = Boston$medv, statistic = boot_q10, R = 1000)
# 
# 
# Bootstrap Statistics :
#         original  bias    std. error
# t1*    12.75 -0.0412   0.5196177




# chapter 6: linear model selection and regularization --------------------

# in a regression setting we have our standard linear model
# Y = B0 + B1*X1 + ... + BN * XN + e
# this is used to describe the relationship between a response Y and a set of variables (X1:XN)
# we typically fit this model using least squares

# in the next chapters we consider approaches for extending the linear model framework
# Chapter 7: accomodate non-linear but additive models
# Chapter 8: more general non-linear models

# the linear model has distinct advantages in terms of inference
# in this chapter we will discuss ways in which the simple linear model can be improved
# we replace our least squares fitting with some alternative fitting procedures

# why might we want to use another fitting procedure instead of least squares?
# alternative fitting procedures can yield better prediction accuracy and model intpretability

## prediction accuarcy:
# provided that the true form of the relationship between Y and set of X is approximately linear...
# the least squares estimates will have low bias
# if n > p (observations greater than number of predictors) least squares fit will have low variance
# this will mean least squares will perform well on the test dataset
# if n is not much larger than p - we can have a lot of variability in the least squares fit
# this results in overfitting and poor predictions on future observations
# if p > n = there is no longer a valid least squares estimate
# the variance is infinite so there is no way we can use the least squares estimate at all
# by constraining or shrikning the estimated coefficnets we can often reduce the variance at the cost of a small increase in bias
# this can lead to substantial improvements in the accuracy with which we can predict

## model interpretability
# if is often the case that some or many of the variables used in a multiple regression setting are NOT associated with the response
# including such irrelevant variables leads to unnecessary complexity in the resulting model
# by removing these variables (setting the coefficients to zero) we obtain a model that is easily interpretted
# this chapter will explore feature selection or variable selection automatically
# this is excluding irrelevant variables from a regression model


# there are many alternatives both classic and modern to using the least squares fit
# in this chapter we discuss three important methods:

## Best Subset Selection:
# involves identifying a subset of the p predictors that we believe to be related to the response
# we then fit a model on the reduced set of features

## Shrinkage:
# this approach involves fitting a model with all p predictors...
# however the estimated coefficients are shrunken towards zero relative to thier least squares estiamtes
# this skrinkage also known as REGULARIZATION has the effect of reducing variance
# depending on what type of skrinkage is performaed some of the coefficents may be estimated at zero
# some skrinkage methods can perform variable selection automatically

## Dimension Reduction
# the approach involves projecting the p predictors into a M dimensional subspace where M < p
# this is achieved by computing M different linear combinations or projections of the variables
# then these M projections are used a predictors to fit a linear regression model by least squares

# these methods can apply to classificaiton and regression problems!
# the rest of this chapter will go through each method and discuss the advantages and disadvantages of each


## Subset Selection:
# we consider some methods for selecting subsets of predictors
# these include best subset and stepwise model selection procedures

# Best Subset Selection
# to perform a best subset selection we fit separate least squares regression for each possible combination of the p predictors
# that is we fit all p models with one predictor, all p(p-1)/2 models with two predictors etc...
# we then take a look at all the resulting models with the goal of finding our best model with the best subset of p

# the problem of selecting the best model from the 2^p possibilities considered by best subset selection is not trivial
# this is usually broken up into two stages


## Best Subset Algorithm
# let M0 denote the null model, which contains no predictors - this predicts the sample mean for each observation
# for k = 1, 2...p
        # fit all (p over k) models that contain exactly k predictors
        # pick the best amoung these (p over k) models and call it Mk
        # here best is having small RSS or large R^2 (portion of variance explained)
# select a single best model from amoung M0...Mp using cross validated prediction error, Cp (AIC), BIC or adjusted R^2


# in step 2 we identify the best model for each subset size in order to reduce the problem from one of 2^p possible models...
# to a problem of of p + 1 possible models
# now in order to select a single best model we must simply choose amoung the p + 1 options
# this must be done carefully...
# the RSS of these p + 1 models decreases as we add more features...we need a way to adjust our metric to account for this
# using the plain RSS or R^2 will give us the model with the most predictors every time!
# like always we want to find a model with low test error!
# training error tends to be smalled than test error  but low training error does not guarentee low test error
# therefore in step 3 we use cross validated test error, Cp, BIC or adjusted R^@ in order to select the best model

# best subset selection can be used for least squares regression and can also be used for other types of models
# in the case of logistic regression we order our models by deviance instead of RSS
# deviance is a measure that is similar to RSS for a broader range of class models
# deviance is negative two times the maximized log-likehood
# the smaller the deviance the bettter the fit!!

# while best subset is a simple approach - it is computationally expensive
# the number of possible models that must be fit rapidly grows as p increases!!
# in general there are 2^p models that invloves subsets of p predictors
# if p = 10 then there are approximately 1,000 possible models to be considered
# if p = 20 then there are over one million possibilites!!
# best subset becomes computationally infeasible for predictors around p = 40

# there are some special shortcuts computationally such as branch and bound techniques to help eliminate model choices...
# but these still have problems with large p and are only available with linear regression
# next we discuss computationally efficient alternatives to best subset selection





## Stepwise Selection
# for computational reasons the best subset selection cannot be applied with very large set of predictors
# best subset selection may also suffer from statistical problems when p is large
# the larger the search space the higher chance of finding models that look good on training data but might be bad on test data or future data
# an enormous search space can lead to overfitting and high variance of coefficient estimates
# for these reasons stepwise methods are used
# stepwise methods explore a for more restricted set of models compared to best subset selection


## forward stepwise selection
# forwrad stepwise selection is a computationally efficient alternative to best subset selection
# best subset will consider all 2^p possible models containing subsets of the p predictors
# forward step will consider a much smaller smaller set of models

# forward step begins with a model containins no predictors and then adds predictors one-by-one until all p are included in the model
# at each step the variables that gives the greatest ADDITIONAL improvement to the fit is added to the final model

## Forward Stepwise Selection Algorithm
# let M0 denote the null model which contains no predictors
# for k = 0, ..., p -1
        # consider all p - k models that augment the predictors in Mk with one additional p
        # choose the best among these p - k models and call it MK+1 - best is highest R^2 or lowest RSS
# select a single best model from among M0 ... Mp using cv prediction error (Cp, BIC, adjusted R^2)

# unlike best subset selection which involved fitting 2^p models...
# forward stepwise selection involves fitting one null model along with p - k models in the kth iteration
# this amounts to a total of 1 + sum(k = 0, p = 1) (p - k) models
# this is a substantial difference
# when p = 20 best subset requires fitting 1 million models
# when p = 20 forward stepwise selection requires fitting only around 200 models

# in step 2 of our fstep algorithm we must identify the best model from among the p-k that augment Mk with one additional predictor
# we can do this by simply choosing the model with the lowest RSS or highest R^2
# however in step 3 we must identify the best model among a set of models with different number of variables

# fstep computational advantage over best subset selection is clear
# fstep is not gaurenteed to to find the best possible model becuase it does not search the 2^p space
# for instance the best possible one-variable model contains X1 and the best two variable model contains X2 and X3
# fstep selection will fail to select the best possible two variable model:
# M1 in fstep will contains X1 and M2 in fstep must also contain X1!! 

# fstep selection can be applied even in the high-dimensional setting where n < p
# fstep cannot work is p > = nß


## Backwards Step Selection
# like fstep selection backwards step selection provides an efficient alternative to best subset selection
# bstep selection begins with the full least squares model containing all p predictors
# it then iteratively removes the least useful predictor one at a time

## Backward Step Selection Algorithm
# let M0 denote the FULL model which contains all p predictors
# for k = p, p - 1, ...1:
        # consider all k models that contain all but one of the predictors in Mk for a total of k -1 predictors
        # choose the best models among the k models and call it Mk-1  hbest is having smallest RSS or highest R^2 (training)
# select a single best model from among M0, ..., Mp using cross validated prediction error, Cp, BIC, or adjusted R^2

# like fstep the bstep selections searches through only 1 + p(p+1) / 2 models
# and so can be applied in settings where p is too large to apply best subset selection
# also like fstep, bstep is not guarenteed to yeild the best model conatining a subset of p predictors (does not serch the entire 2^p space)
# bstep also requires the number of samples n is lalrger than the number of variables p (so that the FULL model can be fit)
# fstep can be used even when n < p and so it is the only viable subset method when p is larger than the number of data observations


## Hybrid Approaches 
# the best subset, fstep selection and bstep selection generally give similiar but not identical models
# there are also hybrid versions of forward and backwards step selection
# variables are added to the model sequentially...
# however after adding a new variable the method may also remove any variables that no longer provide an improvement in model fit
# such an approach attempts to more cloesly mimic best subset selection while retaining the computational advantages of forward and backwards stepwise selection


## Choosing the Optimal Model
# best subset selection, forward selection, and backward selection result in the creation of a set of models..
# each containing a subset of p predictors
# in order to implement these methods we need a way to determine which model is best

# remember the model containing all the predictors will by default have the smallest RSS and largest R^2 related to TRAINING ERROR
# we want to find the best model as a result of TEST ERROR!!
# WE KNOW THAT TRAINING ERROR CAN BE A BAD ESTIMATE OF TEST ERROR!!

# therefore our training RSS and R^2 are not suitable for selecting our best model among a collection of models with various p

# in order to select the best model with respect to test error WE NEED TO ESTIMATE THE TEST ERROR
# there are two common approaches to do this:

# we can indirectly estimate test error by making an adjustmnet to the training error to account for bias due to overfitting
# we can directly estimate the test error using cross validation
# we will consider both of these approaches below:

## Cp, AIC, BIC, and Adjusted R^2
# we show in chapter 2 that the training set MSE is generally an underestimate of test MSE (MSE = RSS / n)
# this is becuase when we fit a model to the training data using least squares...
# we specifically estimate the regression coefficients such that the training RSS is small as possible (but not the TEST RSS!)
# in particular - training RSS will decrease as more variables are added to the model but test error may not!
# this means that training RSS and R^2 CANNOT BE USED to select the best model from a set of models with various predictors

# however there are a number of techniqnues for adjusting the training error for the model size!!
# these approaches CAN BE USED to select amoung a set of models with different number of variables!!
# there are four common approaches: Cp, AIC, BIC, adjusted R^2

## Cp estimate of test error
# for a fitted least squares model containing d predictors the Cp estimate of test MSE is computed using the equaiton:
# Cp = 1/n(RSS + 2dsigma^2)
# here: 
# d is the number of predictors
# simga^2 is an esitmate of the variance of the error e assosicated with each response measurement
# sigma^2 is usually estimated using the full model containing all predictors
# essentially the Cp statistic adds a penalty of 2*d*simga^2 to the training RSS in order to adjust for the fact that training error underestimates the test error

# the penalty increases as the number of d increases
# we are adding some to the ERROR (RSS) to make this adjustment!
# this is intended to adjust for the corresponding decrease in RSS as we add more predictors to the model!

# one can show that if sigma^2 is an unbiased estimate of sigma^2 then Cp is an unbiased estimate of TEST MSE!!
# as a result: 
# the Cp statistic tends to take on a small value for models with LOW TEST ERROR
# THIS MEANS WE CHOOSE THE BEST MODEL AS THE MODEL WITH THE LOWEST Cp!!
# in our example: 
# Cp selects a 6 predictor model containing income, limit, rating, cars, age and student



## AIC
# the AIC criteria is defined for large class of models fit by maximum liklihood
# in the case of a model with Gaussian errors, maximum likelihood and least squares are the same thing...
# therefore: AIC is given by:
# AIC = 1 / n*simga^2 (RSS + 2*p*sigma^2)
# this means that in linear models Cp and AIC are proportional to each other!

## BIC
# BIC is derived from a Bayesian point of view but ends of looking similiar to Cp and AIC as well
# for the least squares model with d predictors the BIC is given by:
# BIC = 1/n*sigma^2(RSS + log(n)D*sigma^2)
# like Cp, BIC will tend to take on a small value for a model with low test error
# we generally select the modelthat has the lowest BIC value

# notice that BIC uses a log term - the BIC generally places a heavier penalty on models with many variables..
# this results in a smaller selection of models than Cp
# in our example we see that BIC chooses a model that contains only four predictors:
# income, limit, cards and student
# we see the curves of the RSS are flat between four and six predictors models - so maybe there is not much performance gain in giving the 6 variable model as best


## Adjusted R^2
# the adjusted R^2 is another popular approach for selecting amoung a set of models that contain differnt numbers of variables
# the normal R^2 is:
# 1 - RSS / TSS
# TSS is the total sum of squares given by sumof(yi - y hat)^2
# since RSS always decreases as more variables are added to the model the R^2 always increases as more variables are added
# for a least squares model with d variables, the adjusted R^2 statistic is calculated as:
# Adjusted R^2 = 1 - ((RSS / (n - d - 1)) / (TSS / (n - 1)))

# unlike our previous criteria - adjusted R^2 with a large value is indicates a model with low test error
# maximizing the adjusted R^2 is equivalent to minimizing RSS / (n - d - 1)
# while RSS always decreases as the number of variables in the model increases, RSS / (n - d- 1) may INCREASE OR DECREASE...
# DEPENDING ON THE NUMBER OF d PREDICTORS INCLUDED IN THE MODEL!!!

# the intuition behind the adjuted R^2 is that once all of the correct variables have been included in the model
# adding some adiditonal noise variables will lead to only a very small decrease in RSS
# since adding noise variables leads to an increase in d, such variables will lead to an increase in RSS(n - d- 1)...
# and consequently a decrease in the adjusted R^2
# therefor in theory: the model with the largest R^2 will have only correct variables and no noise variables
# unlike the regular R^2, the adjusted R^2 pays a price for inclusion of unneccessary variables in the model!


## all criteria for selecting the best model given many models with various predictors have theoretical justifications
# these rely on asymptotic arguements (where sample size is very large)
# despite it's popularity the adjusted R^2 is not as well motivated in statistical theory such as AIC, BIC, and Cp
# all of these measures are simple to use and compute
# these quantities can also be generalized to other models, not just linear form models as presented here


## Validation and Cross Validation
# as an alternative to the approaches just discussed...
# we can directly estimate the test error using the validation set and cross validation methods
# we can compute the validation set error or the cv set error ...
# and then select the model for which the resulting estimated test error is smallest
# this procedure has an advantage relative to AIC, BIC, Cp and adjusted R^2 in that is provides a direct estimate of the test error
# it also makes fewer assumptions about the true underlying model
# it can also be used on a wider range of model selection tasks where the degrees of freedom are hard to pinpoint ( i.e. number of predictors)

# in the past performing cross validation was computationally prohibitive for many problems with large p or large n
# so AIC, BIC, Cp and adjusted R^2 were more attractive approaches for choosing among a set of models
# with today's fast computer the computaitons to perform cv are hardly and issue
# thus cross validation is a very attractive approach for selecting among a number of models under consideration

# an example shows the "best" model for validation cv, k fold cv and BIC
# both cv methods result in a six variable model
# however all three methods suggest the 3, 4 5 and 6 variables models are relatively close in terms of test errors

# while the 3 variable model has lower test error than the two variable model...
# the estimated test errors for 3 - 11 variable models are very close
# if we repeated both cv methods with new random samples we may see some slight changes to their model reccomendation
# in this setting we can select a model using the one-standard-error rule
# we first calculate the standard error of the estimated MSE for each model size...
# then select the smallest model for which the estimated test error is within one standard error of the lowest point on the curve
# the rationale here is that if a set of models appear to be more or les equally good then we might as well choose the simplest model
# the simplest model is the model with the smallest amount of predictors
# applying the one-standard-error rule to the validation set or cv approach leads to the selection of the three variable model



## Skrinkage Methods:
# the subset selection methods described in the previous section invlove using least squares to fit a linear model that contains a subset of predictors
# as an alternative we can fit a model containing all p predictors using a technique that constrains or regularizes the coefficient etsimates...
# or equivalently that skrinks the coefficient estimates towards zero!
# skrinkage the coefficient estimates can significantly reduce variance
# the two best known techniques for shrinking the regression coefficients towards zero are ridge regression and lasso regression

## Ridge Regression
# recall that the least squares fitting procedure estimates B0, B1,..., Bp using values that minimize:
# RSS = sumof(yi - B0 - sumof(Bj*xj))^2 = this is just actual - prediction squared!!

# Ridge Regression is very similiar to least squares except that the coefficients are estimated by minimizing a slightly differnt quantity
# in particular the ridge regression coefficient estimates B^R are the values that minimize:
# sumof(yi - B0 - sumof(Bj*xj))^2 + alpha*sumof(Bj^2) = RSS + alpha*sumof(Bj^2)
# here alpha is >= 0 and is the TUNING PARAMETER to be determined separately

# this equation trades off two different cirteria
# as with least squares ridge regression seeks coeffiient estimates that fit the data well by making the RSS small
# however the second term alpha*sumof(Bj^2) is called the SHRINKAGE PENALTY, is small when the set of coefficients are close to zero
# this EFFECTIVELY SHRINKS THE ESTIMATES OF Bj TOWARDS ZERO!!
# the tuning parameter serves to control the relative impact of these two terms on the regression coefficient estimates

# when our tuning parameter = 0 the penalty term has no effect and ridge regression will be the same as least square regression
# as alpha our tuning parameter goes to infinity the impact of the skrinkage penalty grows
# the ridge regression coefficient estimates will approach zero!
# unlike least squares, ridge regression will produce a different set of coefficient estimates for each value of alpha (tuning parameter)
# selecting a good value for alpha is critical can we can do this with cross validation!!

# note that our shrinkage penalty is applied to the set of Bj coefficients but not the intercept B0
# we want to shrink the estimated association of each variable with the repsonse
# however WE DO NOT WANT TO SHRINK THE ESTIMATE OF THE INTERCEPT - MEANT AS THE MEAN REPSONSE WHEN ALL Bj ARE 0!!

## Application of Ridge Regression to the Credit Data
# as alpha increases all coefficient esitmates in the set will shrink towards zero (see the chart)
# with alpha at 0 our ridge regression will match the standard linear regression
# when alpha is extrememly large all of the ridge regression coefficients are basically zero - representing the null model

# the next panel shows the same ridge coefficients but instead of alpha...
# we now display ||Balpha^R||2 / ||B||2 where B denotes the vector of least square coefficient estimates
# the ||B||2 notation denotes the l2 norm of a vector and is defined as ||B||2 = sqrt(sumof(Bj^2))
# this measures the distance of B from zero!!
# as alpha increases the l2 norm will ALWAYS DECREASE!!
# when alpha is zero = the l2 norms are the same as the standard linear regression coefficent estimates
# when alpha is large = the l2 norm is equal to 0!!

# therefore we can think of the x axis in the left panel chart as the amount that the ridge regression coefficient estimates have been shrunken towards zero
# a small value indicates that they have been shrunken very close to zero!

# the standard least squares coefficient estimates are scale equivariant:
# multiplyig Xj by a constant c simply leads to a scaling of the least square coefficinets by a factor of 1/c
# this means regardless of how the jth predictor is scaled the COEFFICIENT ESTIMATE will be the same
# in contrast the ridge regression coefficient estimates can change substantially when multiplying a given predictor by a constant
# in other words Xj*Bj^Ralpha will depend not only on the value of alpha but the scaling of the jth predictor
# IN FACT IT MAY EVEN DEPEND ON THE SCALING OF OTHER PREDICTORS!!

# therefor it is best to apply ridge regression after STANDARDIZING THE PREDICTORS USING THE FORMULA:
# xij = xij / sqrt(1/n * sumof(xij - xmeanj)^2)
# this puts all the predictors on the same scale with a standard deviation of 1
# as a result the final fit will not depend on the scale on which the predictors are measured - they are all on the same scale!!


## Why does Ridge Regression Improve Over Least Squares?
# ridge regression advantage over least squares is rooted in the bias-variance trade off
# as our tuning paramenter lambda increases the flexibility of the ridge regression fit decreases
# the leads to decreased variance but increased bias
# with lambda at 0 we have the normal least squares coefficients the variance is high but there is no bias
# as lambda increases the shrinkage of the ridge coefficient estimates leads to a substantial reduction in variance of predictions...
# at the expense of a slight expense increase in bias
# recall that the test mean squared error is a function of the variance plus squared bias
# as lambda goes to 10: the variance decreases rapidly with very little increase in bias (shrinking the coefficients reduces our fit to the data)
# consequenctly the MSE drops considerably as our tuning parameter increases from 0 to 10
# beyond this point lambda = 10: the decrease in variance due to increase lambda slows...
# and shrinking of the coefficients causes a gross underestimation of the coefficients and bias skyrockets!
# there is a certain "sweet spot" with our tuning parameter that minimizes our error by reducing variance but not adding too much bias!!
# our chart shows the minimal MSE is around lambda = 30
# interestingly - because of the high variance the MSE associated with the least squares fit (lambda = 0) is almost as high as that of the null model for which all coeffiient estiamtes are 0
# remember if all coefficents are zero its the same as our tuning parameter at infiinity!!
# ridge with a high lambda shrinks all coeffiient estimates to zero!

# the second chart shows the l2 norm of the RIDGE regression as a function of the NORMAL REGRESSION
# we see that as we get more flexible - our test MSE actually increases
# note the reasons:
# NORMAL REGRESSION - we are overfitting and increasing VARIANCE!!
# RIDGE REGRESSION - we are shrinking our coefficients too much and producing a bad fit - we are increasing BIAS!!

# in general in situations where the relationship between the response and the predictors is close to linear...
# the least squares estiates will have low bias but high variance!
# this means that a small change in the training data can cause a large change in the least squares coefficient estimates!

# in particular when the number of variables p is almost as large as the number of observations n...
# the least square estimates will be extremely variable!
# if p > n then the least square estimates do not even have a unique solution!
# ridge regression can still perform very well by trading off a small increase in bias for a large decrease in variance!
# ridge regression works best in situations where the least square estimates have high variance

# ridge regression also has substantial computational advantages over best subset selection which requires searching through 2^p models
# as we discussed, even for moderate values of p a search through the space of all models can be computationally infeasible!
# in contrast for any fixed value of lambda ridge regression only fits a single model and does this quickly
# in fact one can show that the computations required to solve 6.5 for ALL VALUES OF LAMBDA are almost identical to those for fitting a model using least squares!!



## The Lasso
# ridge regression does have one obvious disadvantage:
# unlike best subset, forward selection, and backward selection which will try to select all possible models with subsets of possible variables
# ridge regression will include all p predictors in its one fit model that reduces variance for slight increases in bias
# ridge regression will shrink all coefficents close to zero but not exactly zero!
# this may not be a problem for prediction but might be a problem for interpretability!
# is p is large - coefficients will be shrunken close to zero but still kept in the model!

# in our credit example: it appears that the most important variables are income, limit, rating, and student
# we might wish to fit a model including just these important variables!
# however the ridge regression will always generate a model involving all ten predictors
# increasing the value of lambda (tuning parameter) will tend to reduce the magnitudes of the coeffiecnets but will not result in exclusion of any of the variables

## the lasso
# the lasso is a recent alternative to ridge regression that overcomes the disadvantage of having to keep all the variables in the fitted model
# the lasso coefficients B^L minimize the quantitiy:
# RSS + lambda * sumof(abs(Bj))
# comparing the two functions from ridge vs. lasso together we see they have similiar formulations...
# but the only difference is that the B^2 term in ridge has been replaced with abs(Bj) term in lasso
# the lasso uses an l1 penalty!
# the ridge uses an l2 penalty!
# the l1 norm of a coefficient vector B is given by: ||B||1 = sumof(abs(Bj))

# as with ridge, lasso shrinks the coefficients estimates towards zero
# however in the case of the lasso the l1 penalty has the effect of forcing some of the coefficients to be exactly 0 when the tuning parameter is large
# hence much like the best subset selection the lasso performs variable selection for us!
# as a result models generated from the lasso are generally much easier to interpret than those produced by ridge regression
# we say that the lasso yields sparse models
# sparse models involve only a subset of the original variables
# as in ridge regression selecting a good value of lambda our tuning parameter is critical

# as an exmaple consider the coefficient plots produced from applying a lasso on the credit data set
# when lambda  is 0
# then the lasso still gives exactly the least squares fit
# when lambda our tuning parameter becomes sufficiently large the lasso gives the null model in which the coefficient estimates are all 0!!

# in between these two extremes is where ridge and lasso differ
# as we grow tuning parameter lasso can produce a model involving any number of variables
# ridge regression will always include all of the variables in the model - but some coefficient estimates may be very small


## Another Formulation for Ridge and Lasso Regression
# one can show that the lasso and ridge regression coefficient estimates solve the problems
# minimize a formula subject to sumof(abs(Bj)) <= s = LASSO
# minimize a formula subjet to sumof(abs(Bj^2)) <= s = RIDGE
# in other words for every value of lambda there is some s that our equations will give the same lasso coefficient estimates
# this means there is also a minimizing value of s that will give the same ridge coefficients esitmates

# when p = 2
# then our lasso formula indicates that the lasso coefficient estimates have the smallest RSS out of all points that lie within abs(B1) + abs(B2) <= s
# the ridge regression formula estimates have the smallest RSS out of all points that lie within the circle defined by B1^2 + B2^2 <= s

# we can think of this relationship as follows:
# when we perform the lasso we are trying to find the set of coefficients estimates that lead to the smallest RSS, subject to the constraint that there is...
# BUDGET S for how large sumof(abs(Bj)) can be!
# when s is extremely large this budget is not very restrictive and so coefficient estimates can be large (and mirror the normal regression coefficient estimates)
# if our BUDGET S is small then our penalty must be small in order to avoid voilating the budget - and dropping the variable
# similiarly the ridge regression formula indicates that when we perform ridge regression...
# we seek a set of coefficient estimates such that the RSS is as small as possible, subject to the requirement that the ridge penalty not exceed the BUDGET S

# the formulations reveal a close connection between the lasso, ridge and best subset selection
# we consider the problem of:
# minimizing our B subject to sumof(I(Bj != 0)) <= s
# here I(Bj != 0) is an indicator variable: it takes on the value of 1 if Bj != 0 and equals zero otherwise
# then this formula amounts to finding a set of coefficients estimates such that RSS is as small as possible, subject to the constraint that no more than s coefficients can be non-zero
# the problem here is equivalent to BEST SUBSET SELECTION!

# we know that best subset selection is computationally expensive when p is large (fit a model for every iteration of predictors)
# therefore we can interpret ridge regression and the lasso as computationally feasible alternatives to best subset selection
# these methods replace the intractable form of BUDGET S with much more manageable constraints that are easier to solve
# of course, lasso (l1) is closer to best subset selection because lasso can perform feature selection for us!


## The variable Selection Property of the Lasso
# why is it that the lasso, unlike ridge, results in coefficient estimates that are exactly zero?
# we can use our formulations to shed light on this issue
# with a large tuning parameter we know that both lasso and ridge will produce the normal least squares regression estimates
# remember that the contrainst functions of the lasso and ridge differ:
# lasso: DIAMOND CONSTRAINT FUNCTION WHEN PLOTTED ON A GRAPH
# ridge: CIRCLE CONSTRAINT FUNCTION WHEN PLOTTED ON A GRAPH
# the interestion of our RSS ellipsis will be able to meet our LASSO DIAMOND at a specific point - this indicates reducing the variable to zero
# in contrasts the interestion of our RSS ellipsis will not be able to meet our RIDGE CIRCLE at a specific point - this indicates a variable cannot be reduced to zero
# this LASSO DIAMOND and RIDGE CIRCLE idea is in the case of p = 2! this shapes will scale as we increase dimensionality!
# however the key idea of our expanding RSS ellipsis being 'stopped' by a constraint geometric shape still holds!
# further - depending on that shape - coefficient estimates can be driven to zero (lasso 'diamond' with points in the contraint function)...
# or coefficents can never reach zero due to the ridge 'circle' with smooth edges in the constraint function!
# this means lasso leads to feature selection when p >2 due to the sharp corners of it constraint function geosphere!




## Comparing Lasso and Ridge Regression
# it is clear that the lasso has a major advantage over ridge regression - lasso produces simpler and more interpretable models that involve only a subset of predictors
# which method results in better prediction accuracy???
# let's compare the variance, squared bias and test MSE of the lasso compared to the ridge regression
# clearly both methods have similiar behavior - if lambda increases - we increase bias but reduce variance
# our example shows lasso and ridge have almost identical biases
# however - the variance of the ridge regression is slightly lower than the variance of the lasso
# consequently the minimum MSE of ridge regression is slightly smaller than that of the lasso - IN THIS EXMAPLE

# but wait...
# this example was generated in a way such that all 45 predictors were related to the repsonse - none of the true coefficients equaled zero!!
# the lasso implicitly assumes that a number of the coefficients truly equal zero
# consequently it is not surprising that ridge regression outperforms the lasso in terms of prediction error in this setting

# let's re-do the experiment with non-associated predictors to our repsonse
# in this new case only 2 out of the 45 predictors are truely associated with the repsonse variable
# now we see drastic changes
# lasso outperforms ridge in terms of bias, variance and MSE

# these two examples show that neither ridge nor lasso will universally dominate the other
# in general one might expect the lasso to perform better in a setting where a relatively small number of predictors have substantial coefficients, with the remaining near zero
# ridge regression will perform better when the response is a function of many predictors all with coefficents of roughly equal size
# however - the number of predictors associated with the response is never truely known BEFOREHAND!!
# we can use cross validation to determine which approach is better on a particular data set

# as with ridge regression when the least squares estimates have excessively high variance, the lasso solution can yield a reduction in variance as the expense of a small increase in bias
# this can generate more accurate predictions
# unlike ridge regression (l2), the lasso performs variable selection and results in models that are easier to interpret
# there are very efficiient algorithms for fitting both ridge and lasso models:
# in both cases the entire coefficient paths can be computed with about the same amount of work as a single least squares fit!

## Quick Check:
# remember the bias variance trade-off
# variance: error given by fitting a model too closely to a specific training set
# if we fit our model as a flexible model to the training set too closely - our model will suffer in MSE when applied to real live data
# variance is linked to the change in MSE if we had even small changes in training data points
# bias: the idea of error from not fitting the true 'form' of the model
# trying to fit a linear regression through a true 'S' shaped form will result in high bias - no matter how we angle our line we will under or overpredict at certain points along the true function's cruves!
# the ultimate goal is to reduce both variance and bias!
# in general - the more flexible - the higher variance but lower bias (we for sure model close to the true form but are suceptible to fluctations based on training data)
# simple models will have lower variance - not change much with small changes in training data - but may have high bias - not pick up on the true form of the data!!


## A Simple Special Case for Ridge Regression and Lasso Regression
# let's consider a case with n = p to build intuition about the behavior or ridge and lasso regression
# we will also have an X diagonal matrix with 1s in the diagnol and 0s otherwise
# we will perform a regression without an intercept 
# with these assumptions: the usual least squares problems simplifies to finding B1...Bp that minimize
        # sumof((y - Bj)^2
# in this case the least squares solution is given by: Bj = yi
# in this setting the ridge regression amounts to finding B1...Bp such that:
        # sumof(yj - Bj)^2 + lambda * sumof(Bj^2) is minimized
# in this setting the lasso regession amounts to finding the coefficients such that:
        # sumof(yj - Bj)^2 + lambda * sumof(abs(Bj) is minimized
# one can show in this setting the ridge regression estimates take the form: Bj(ridge) = yj / (1 + lambda)
# one can show that the lasso regression estimates take the form: 
        # Bj(lasso) = yj - lambda/2 if yj > lambda / 2
        # Bj(lasso) = yj + lambda/2 if yj < -lambda / 2
        # 0 if abs(yj) <= lambda / 2

# we can see that ridge regression and the lasso perform two very different types of skrinkage
# in ridge regression - each least squares estimate is shrunken by the same proportion
# the lasso regression - shrinks each least squares coefficient towards 0 by a constant amount lambda / 2
# the least squares coefficients that are less than abs(lambda / 2) are shrunken entirely to zero
# the shrinkage performed by lasso regression in the simple case is known as soft thresholding
# the fact that some lasso coefficients are shrunken entirely to zero explains why the lasso performs better in feature selection

# in the case of a more general data matrix X the case for both regularization methods is a little more complex
# the main idea should hold approximnately:
# ridge regression more or less shrinks every dimension of the data by the same proportion
# lasso regression more or less shrinks all coefficients toward zero by a similiar amount - sufficiently small coefficients are shrunken all the way to zero

## Bayesian Interpretation of Ridge Regression and Lasso Regression
# we now show that one can view ridge regression and the lasso through a bayesian lens
# a bayesian viewpoint for regression assumes that the coefficient vector B has some prior distribution
# we say p(B) where B = (B0, B1,...Bp)^T
# the liklihood of the data can be written as f(Y | X, B) where X = (X1, ... Xp)
# multiplying the prior distribution by the likelihood gives us the posterior distribution which takes the form:
        # f(Y | X, B)*p(B)
# where the proportionality above follows from Bayes theorem and the equality above follows from the assumption that X is fixed
# we assume the usual linear model: Y = B0 + B1*X1 + ... + Bp*Xp + Ei
# and we suupose that the errors are independent and drawn from a normal distribution
# furthermore assume that p(B) = productof(g(Bj)) for some density function g
# it turns out that ridge regression and the lasso follow naturally from two special cases of g
# if g is normal distribution with mean 0 and standard deviation a function of lambda, then it follows that the posterior mode for B...
# that is the most likely value for B...
# is given by the ridge regression solution ( the ridge regression solution is also the posterior mean)
# if g is a double exponential (Laplace) distribution with mean 0 and scale parameter a function of lambda...
# then it follows that the posterior mode for B is the lasso solution!
# the lasso solution is not the posterior mean, and in fact, the posterior mean does not yeild a sparse coefficient vector)

# the gaussian and double-exponential priors are displayed in figure 6.11
# therefore from a bayesian viewpoint the ridge regression and the lasso follow directly from assuming the usual linear model with normal errors, together with a simple prior distribution for B
# notice that the lasso prior is steeply peaked at zero, while the guassian is flatter and fatter at zero
# hence the lasso expects a priori that many of the coefficients are exactly zero, while ridge assumes the coefficents are randomly distributed about zero


## Selecting the regularization tuning parameter
# just as the subset selection approaches considers in the beginnning of this chapter require a method to determine the "best" model
# implementing ridge and lasso regression requires a method for selecting a value for the tuning parameter lambda
# we need a way to pick the best constraints parameter for s
# cross validation provides a simple way to tackle this problem
# we choose a grid of lambda values and compute the cross validation error for each value of lambda
# we then select the tuning parameter value for which the cross validation error is smallest
# finally the model is re-fit using all of the available observations and the selected value of the tuning parameter

# in our example we see the best tuning parameter for the ridge regression on the  Credit data set is very small
# this indicates that the optimal fit only involves a small amount of shrinkage relative to the least squares solution (remember lambda at 0 is the least squares)
# in addition the dip is not very pronounced so there is a rather wide range of values that would give very similiar error
# in this case we may simply use the least squares solution

# in our example we see the best tuning parameter for the lasso regression on the Credit data set is also very small
# not only has the lasso correctly given much large coefficients estimates to the two signla predictors...
# but also the minimum cross validation error corresponds to a set of coefficients estimates for which only the signal variables are non-zero
# hence cross validation together with the lasso has correctly identified the two signal variables in the model even though this is a challenging setting
# we have p = 45, and only n = 50 observations!!
# in contrast the least squares solution assigns a large coefficient estimate to only one of the two signal variables!!


## Dimension Reduction Models
# the methods that we have discussed so far in this chapter have controlled variance in two different ways
# either by using a subset of the original variables, or by shrinking thier coefficients toward zero
# all of these methods are defined using the original predictors X1, X2, ... Xp
# we now will explore a class of approaches that transforms the predictors and then fit a least suqares model using the transformed variables
# we will refer to these techniques as dimension reduction models

# let Z1, Z2 ... Zm represent M < p linear combinations of our original p predictors
# this is:
# Zm = sumof(omegajm * Xj)
# for some constants omega1m, omega2m, ... omegapm, m = 1,..., M
# we can then fit the linear regression model:
# yi = theta0 + sumof(omegam * zim + ei), i = 1, ..., n: using least squares
# note that in this formula the regression coefficients are given by theta0, theta1, ... thetaM
# if the constants omega1m, omega2m ... omegapm are chosen wisely, then such dimension reduction approaches can often outperform least squares regression
# in other words, fitting our formula using least squares can lead to better results than fitting by least squares

# the term dimesion reduction comes from the fact that this approach reduces the problem of estimating the p + 1 coefficients B0, B1, ...Bp ...
# to the simpler problems  of esimating the M + 1 coefficients theta0, theta1, ... thetaM where M < p 
# in other words the dimension of the problem has been reduced from p + 1 to M + 1

# notice from our formula:
# sumof(thetam* zim) = sumof(Bj * x*ij) where Bj = sumof(thetam * omegajm)
# this emans that this formula can be thought of a special case of the original linear regression model
# dimension reduction serves to constrain the estimated Bj coefficients, since now they must take the form or out theta * omega formula
# this constraint on the form of the coefficients has the potential to bias the coefficieint estimates
# however in situations where p is large relative to n, selecting a value of M << p can significantly reduce the variance of the fitted coefficients
# if M = p, and all the Zm are linearly independent, then our formula does not constrain any of the coefficients
# in this case no dimension reducion occurs and so fitting is equivalent to the normal least squares model on the original p predictors

# all dimension reduction methods work in two steps
# first the transformed predictors Z1...Zm are obtained
# second the model is fit using these M predictors
# however the choice of our subsets Z1...Zm or the selections of omegajm can be achieved in different ways
# this chapter will cover two approaches for this task: principal components and partial least squares


## Principal Component Regression
# principal component analysis (PCA) is a popular approach for deriving a low dimensional set of features from a large set of variables
# PCA is discussed in greater detail as a tool of UNSUPERVISED LEARNING in chapter 10
# here we describe its use as a dimension reduction technique for regression

## Overview of Principal Component Analysis
# PCA is a technique for reducing the dimensin of a n X p data matrix X
# the first principal component direction of the data is that along which the observations VARY THE MOST
# consider a example which shows population in tens of thousands of people and ad spending  for a particular company in thousands of dollars
# we want to find the line that has the most variability! that is residuals from the true data points to the line!!
# if we projected the 100 observations onto this 1st principal component line we would have the most variance!!
# projecting these data points on to any other line would give us LESS VARAINCE!
# in a way were are sort of fitting a regression model line to give us the most variance on our oriignal observations!
# this means that the subset predictors used to build this line will account for the most possible variance!!
# projecting a poiint onto a line simply involves finding the loaction on the line which is closest to the point

# principal component mathmatically:
# Z1 = .839 * (pop - pophat) + .544 * (ad - adhat)
# here theta11 = .839 and omega21 = .544 are the principal component loadings which define the direction
# in this formula pophat indicates the mean of all pop values in the dataset and adhat indicates the mean of all ad spending
# the idea is that out of every possible linear combination of pop and ad such that omega11^2 + omega21^2 = 1...
# this particular linear combination yeilds the highest variance!!
# this is the linear combination for which Var(omega11 * (pop - pophat) + omega21 * (ad - adhat)) is maximized
# it is necessary to consider only linear combinations of the form omega11^2 + omega21^2 = 1 since otherwise we could increase these parameters arbitratily and blow up the variance
# in this formula the two loadings are both positive and have similiar size and so Z1 is almost an average of the two variables

# since n = 100, pop and ad are vectors of length 100, and so is Z1
# for instance:
# zi1 = .839 * (popi - popbar) + .544 * (ad - adbar)
# the values of z11 ... zn1 are known as the principal component scores


# there is also another interpretation for PCA:
# the first principal component vector defines the line that is as close as possible to the data
# for instance the the first principal component line minimizes the sum of the squared perpendicular distances betwen each point and the line
# we choose the first principal component so that the projected observations are as close as possible to the original observations

# we can think of the values of the principal component Z1 as a single number summaries of the joint pop and ad for each location
# if our formula is less than zero then this indicates a city with below-average population size and below average ad spending
# a positive PCA1 score suggests the opposite
# how well can a single number represent both pop and ad?
# we see in a few plots that there is a strong relationship  between the first principal component and the two features
# in other words the first principal component appears to capture most of the information contained in the pop and ad predictors

# so far we can concentrated on the first principal component
# in general, one can construct up to p distinct principal components
# the second principal component Z2 is a linear combination of the variables that is UNCORRELATED WITH Z1 and has the LARGE VARIANCE SUBJECT TO THE UNCORRELATED RESTRAINT!!
# in our exmample it turns out that the zero correlation condition of Z1 and Z2 is equivalent to the condiitno that the direction must be perpendicular or orthogonal to the first principal component direction!
# the second principal component is given by the formula:
# Z2 = .544 * (pop - popbar) - .839*(ad - adbar)
# since the advertising data has two predictors the first two principal components contain all of the information that is in pop and ad
# however by construction the first component will contain the most information
# we will see much large variability of zi1 versus zi2
# the fact that the second principal component scores are much closer to zero indicates that this component captures less information that the first principal component

# when we plot them against each other we should see little relationship between the two components! PCA2 needs to be uncorrelated with PCA1
# in our example there is little relationship between PCA2 and our predictors - so we may only need to use PCA1 for our regression!
# we only need to use PCA1 to accruately represent the pop and ad budgets

# with two-dimensional data such as our advertising exmaple we can construct at most two components
# however if we had other predictors then additional components could be constructed
# they would successively maximize variance subject to the constraint of being UNCORRELATED with the PRECEEDING PRINCIPAL COMPONENT


## Principal Components Regression Approach
# the PCA regression (PCR) approach involves constructing the first M principal components Z1...Zm and then using these components as the predictors in a linear regression model using least squares
# the key idea is that often a small number of principal components suffice to explain most of the variability in the data as well as the relationship with the response
# in other words, we assume that the directions in which X1 ... Xp show the most variation are the directions that are assoicatied with response Y
# while this assumption is not guaranteed to be true, it often turns out to be a reasonable enough approximation to give good results

# if the assumption underlying PCR holds, then fitting a least squares model to Z1...Zm will lead to better results than fitting a least squares model on the original set of predictors!!
# since most of all of the information in the data that relates to the response is contained in Z1...Zm:
# and by estimating only M << p coefficenties we tend to mitigate overfitting

# in our advertising example the first principal component explains most of the variance in both pop and ad
# so a principal component regression that uses this single variable to predict some response of interest such as sales will likely perform quite well
# as more principal components are used in the regression model, the bias decreases, but the variance increases
# this results in our typical U -shape for the mean squared error
# when we select M to equal the number of predictors - the PCR amounts simply to a least squares fit using all of the original predictors
# using PCR with an appropriate choice of M can result in a improvement over least squares


# in our example we fit PCR on the simulated datasets
# both datasets where generated using n = 50 observations and p = 45 predictors
# the response in the first dataset used all predictors
# the response in the second dataset was generated using just two of the predictors
# as more PCs are used in the regression model the bias decreases but the variance increases
# this results in the typical U shape for the meas squared error
# the M = p = 45 the PCR simply amounts to a least squares fit using all of the original predictors
# the figure indicates that using a good choice of M can lead to improvement over the least squares model

# when examining the ridge regression and lasso applied to the same dataset we see PCR does not perform as well
# the worse performance is a consequence of how we set up our data
# the data were generated in a way such that many PCs are needed to accurately model the data
# PCR tend to do well in cases when the first few PCs are sufficient to capture most of the variation in the predictors AND the relationship with the response
# in an example dataset developed using the first 5 principal components to develop the response...PCR performs well
# now the bias drops to zero rapidly was M the number of PCs increases
# the mean squared error displays a clear minimum at M = 5
# all three methods, ridge, lasso, PCR offer improvements over ordinary least squares
# PCA and ridge regression outperform our lasso model

# we note that even though PCR provides a simple way to perform regression using M < p predictors, it is NOT A FEATURE SELECTION METHOD
# this is because each of the M principal components used in the regression IS A LINEAR COMBINATION of all p ORIGINAL FEATURES!!!
# for instance Z1 was a combination of both pop and ad
# therefore while PCR often performs well in many practical settings, ...
# it does not result in the development of a model that relies upon a small set of the original features
# in this sense PCR is more closely related to ridge regression than lasso regression
# one can even think of ridge regression as a continous version of ridge regression

# in PCR the number of principal components M is typically chosen using cross validation
# in our example the lowest cross validation error occrus when M = 10
# this corresponds to almost no dimension reduction at all since PCR with M = 11 will simply fit the oridinary least squares

# when performing PCR we generally reccomend standardizing each predictor!!
# this standardization ensures that all variables are on the same scale
# in the absence of standardization the high-variance variables will tend to play a larger role in the PCs obtained...
# AND THE SCALE ON WHICH THE VARIABLES ARE MEASURED WILL ULTIMATELY HAVE AN EFFECT ON THE FINAL PCR MODEL!
# if the variables are all measured in the same units then we may forego the standardization step
# ALWAYS INVESTIGATE SCALING PREDICTORS BEOFRE APPLYING PCA




## Partial Least Squares
# the PCR approach we just described invloves identifying linear combinations, or directions, that best represent the predictors, X1,...,Xp
# these directions are indentified in an unsupervised way, since the response Y is not used to help determine the PC directions
# the repsonse does not SUPERVISE the identification of the principal components
# because of this - PCR suffers from a drawback: there is no guarantee that the dirctions that best explain the predictors will also be the best directions for PREDICTING THE RESPONSE!!

# we will now review Partial Least Squares (PLS)
# PLS is a supervised alternative to PCR
# PLS is a dimension reduction method, which first identifies the new set of features, Z1...Zm that are linear combinations of the original features
# PLS then fits a linear model via least squares using these M new features
# unlike PCR, PLS identifies these new features in a supervised way
# PLS will make use of the repsonse Y in order to identify new features that not only approximate the features well but are also related to the RESPONSE
# PLS approach attempts to find directions that help explain both the repsonse and the set of predictors!

# here is how the first PLS direction is computed
# after standardizing the p predictors, PLS computes the first direction Z1 by setting each thetaji (loadings) equal to the coffienct from a simple linear regression
# in computing Z1 = sumof(thetaj1*Xj) PLS places the highest weight on the variables that are most strongly related to the response

# PLS on our sales example
# we have sales in 100 regions with Population Size and Ad Spending as predictors
# PLS has chosen a direction that has less change in ad dimension per unit change in the population dimension relative to PCA
# this suggests that population is more highly correlated with the response than is AD
# the PLS direction does not fit the predictors as closely as PCA, but it does a better job at explaining the repsonse

# here is how the second PLS direction is computed
# we first adjust each of the variables in Z1 by regressing each varaible on Z1 and taking residuals
# These residuals can be interpretted as the remaining information that has not been explained by the first PLS direction
# we then compute Z2 using this orthogonal data in exactly the same fashion as Z1 was computed based on the original data
# this iterative approach can be repeated M times to indentify multiple PLS components Z1...Zm
# at the end of this procedure we use least squares to fit a linear model to predict Y using Z1...Zm in exactly the same fashion as for PCR

# as with PCR, the number of M of partial least squares directions is used in PLS is a tuning parameter that is typically chosen by cross validation
# we generally standardize the predictors and response before performing PLS
# in practice it often performs no better than ridge regression or PCR
# while the supervised dimension reduction of PLS can reduce bias, it has the potential to increase variance 
# the overall benefit of PLS vs. PCR is almost the same


## Considerations in High Dimensions
# most traditional statistical techniques for regression and classification are intended for low-dimensional setting
# low dimensional setting is where n the number of observations is much greater than p the number of predictors
# most traditional problems are based on having lots of observations but only a couple of predictors
# this means that n >> p and the problem is low dimensional
# dimension refers to the size of p relative to the size of n

# times have changed
# it is now common place that our number of predictors can be extremely large
# while our number of n can be restricted by cost, availability or other factors

# data sets containing more p features than observations are often referred to as high dimensional
# classical approaches to least squares are not useful at this setting
# many issues arise in high dimensional data sets and apply to the same n > p cases
# these include the always relevant bias - variance tradeoff and the danger of overfitting
# these become particularly important when the number of features is very large relative to the number of observations

# we defined high-dimensional setting as the case where the number of features p is larger than the number of observations n
# these considerations also apply if p is only slightly smaller than n too
# always keep these ideas in mind when performing supervised learning

## What goes wrong in high dimensions?
# example will focus on linear regression
# the same problems here will also plague logistic regression, linear discriminant analysis and other approaches
# when the number of features p is as large as or large than the number of observations n...
# least squares cannot and should not be performed
# the reason is simple:
# regardless of whether or not there is a true relationship between features and the response...
# least squares will yeild a set of coefficients estimates that result in a perfect fit to the data - and residuals will be zero

# example: p = 1 feature and case where we have 20 observations, p = 1 feature and case were there are 2 observations
# when there are 20 observations, n > p the least squares regression line does not perfectly fit the data
# instead the regression line seeks to approximate the 20 observations as well as possible
# when there are 2 observations, the regressino line will fit the data exactly
# this is problematic because this perfect fit will almost certainly be overfitting our data
# in other words - although possible to fit perfectly to the training data - we would see poor results on the test data or new data
# this will make the model useless
#  the problem is simple: when p > n or p ~~ n a simple least squares regression line is TOO FLEXIBLE and OVERFITS OUR DATA!!!

# this can be extrapolated to the case when p is very large
# data were simulated with n = 20 and regression was performed on models with 1:20 features included
# each of the 1:20 features were completely unrelated to the response
# as the features increase R^2 increases to 1 and the training MSE goes down completely to zero - EVEN THOUGH THE FEATURES ARE COMPLETELY uNRELEATED TO THE RESPONSE
# MSE on an independent test set becomes extremely large as the number of features included in the model increases - including the additional predictions increases VARIANCE
# looking at the test MSE it is clear the best model only uses a few predictors
# however, carelessly examining R^2 or the training set MSE might erroneously conclude that the model with the greatest number of variables is best
# this indicates the importance of applying extra care when analyzing data sets with a large number of variables - always evaluate on the TEST SET INDEPENDENT FROM TRAINING

# we saw a number of approaches for adjusting the training set RSS or R^2 in order to account for the number of variables used to fit a least squares model
# unfortunately, Cp, AIC, BIC approaches are also NOT APPROPRIATE in the high dimensional setting ( p > n)
# this is because our estimates of sigma^2 are problematic - fitting in high dimensions can reduce sigma^2 to 0
# similiarly, problems arise in the application of adjusted R^2 in the high-dimensional setting - we can easily obtain an adjusted R^2 of 1
# clearly alternative approaches that are better-suited to the high-dimensional settings are requried

## Regression in High Dimensions
# it turns out many of the methods in this chapter for fitting less flexible least squares models, such as forward stepwise selection, ridge rregression and lasso regression, PCR are useful in high dimensional settings
# all these approaches avoid overfitting by using a less fleixble fitting approach than least squares
# here is an example:
# there are p = 20, 50 or 2,000 features of which only 20 are truely associated with the repsonse
# the lasso was performed on n = 100 training observations and the meas squared error was evaluated on an independent test set
# as the number of features increases, the test set error increases
# when p = 20, the lowest validation set error was achieved when lambda in lasso was small
# when p was larger than then the lowest validation set error was achieved using a larger value of lambda - we needed more shirnkage of the coeffiicents

# we view the lasso fit and its degrees of freedom = the number of non-zero coefficents of the fit was we tune the regularization parameter
# this highlights three key points:
# regularization plays a key role in high0dimensional problems
# appropriate tuning parameter selection is crucial for good predictive performance (tune with cross validation)
# the test error tends to increase as the dimensionality of the problem (the number of features or predictors) increases, unless the additional features are truly associated with the response (we decrease bias but add significant variance)

# the third point above is a key principle in the analysis of high-dimensional data which is known as the curse of dimensionality
# one might think that as the number of features used to fit a model increases the quality of the fitted model increases as well ---- THIS IS NOT THE CASE
# in general - adding additional signal features that are truly associated with the response will improve the fitted model
# adding noise features that are not truely associated with the response will lead to a deterioration in the fitted model performance
# noise features increase the dimensionality of the problem - exacerbating the risk of overfitting ( noise may be assigned non-zero coefficients due to chance)
# this will come without any potential upside in terms of improved test set error!!

# technology that allows for collection of 10000s of features is a double edged sword:
# they can lead to predictive models if these features are in fact relevant to the response but will lead to worse performance if not relevant at all
# even if they are relevant the variance incurred in fitting thier coefficents may outweigh the reductino in bias that they bring!

## Interpreting Results in High Dimensions
# when we perform the lasso, ridge, or other regression procedures in the high dimension setting we must be cautious in reporting the results we obtain
# multicolinearity is the concept that the variables in a regression setting might be correlated with each other
# in the high-dimensional setting the multicolinearity problem is extreme
# any variable in the model can be written as a linear combination of all the other variables in the model
# this means that we can never know exactly which variables if any are truly predictive of the outcome
# and we can never identify the best coefficients for use in the regression
# at most we can hope to assign large regression coefficents to variables that are correlated with the variables that are truly predictive of the outcome

# suppose that we are trying to predict blood pressure on the basis of .5 million SNPs
# forward stepwise selection indicates about 17 of those SNPs lead to a good predictive model on the training data
# it would be incorrect to conclude that these 17 SNPs predict blood pressure better than the other SNPs not included
# there are likely to be as many sets of 17 SNPs that would predict blood pressure just as well as the seleted model
# if we were to obtain an independent data set and perform stepwise selection on the dataset we would likely obtain a model containing different and even non-overlapping SNPs
# this does not detract from the value of the model obtained - for instance the model might turn out to be very effictive in predicting blood pressure on an independent set of patients
# we must be careful not to overstate the results obtained and to make it clear that what we have identified is simply one of many possible models for predicting blood pressure
# these all need to be further validated on the independent test datasets

# it is also important to be particularly careful in reporting errors and measures of model fit in the high dimensional setting
# we have seen that when p > n it is easy to obtain a useless model that has zero resiudals
# therefor one sohuld never use sum of squared errors, p values, R^2 statistics or other traditional measures of model fit on the training data as evidence of good fit in the high dimensional setting
# for instance in the high dimensional setting it can be easy to obtain a model with R^2 = 1 when p > n
# reporting this fact might mislead others into thinking that a statistically valid and useful model has been obtained - when it fact this provides no evidence of such a model
# it is important instead report results on an independent test set or cross validation errors
# for instance the MSE or R^2 on an independent test set is a valid measure of model fit but the MSE on the training set certainly is not!


## Lab 1: Subset Selection Methods
# apply best subset selction to the hitters dataset
# we wish to predict a baseball player's salary on the basis of various batting statistics
# note that the salary vallue is missing for some baseball players 
library(ISLR)
data('Hitters')
names(Hitters)
mean(is.na(Hitters$Salary))
# [1] 0.1832298

# let's remove all these players
Hitters = Hitters %>% na.omit()
mean(is.na(Hitters$Salary))
# [1] 0

# let's use best subset selection
# the regsubsets() function apart of the leaps library performs best subset selection
# it identifies the best model that contains a given number of predictors where best is quantified using RSS
library(leaps)

# fit model
regfit_full = regsubsets(Salary ~ ., data = Hitters)
summary(regfit_full)
# Subset selection object
# Call: regsubsets.formula(Salary ~ ., data = Hitters)
# 19 Variables  (and intercept)
# Forced in Forced out
# AtBat          FALSE      FALSE
# Hits           FALSE      FALSE
# HmRun          FALSE      FALSE
# Runs           FALSE      FALSE
# RBI            FALSE      FALSE
# Walks          FALSE      FALSE
# Years          FALSE      FALSE
# CAtBat         FALSE      FALSE
# CHits          FALSE      FALSE
# CHmRun         FALSE      FALSE
# CRuns          FALSE      FALSE
# CRBI           FALSE      FALSE
# CWalks         FALSE      FALSE
# LeagueN        FALSE      FALSE
# DivisionW      FALSE      FALSE
# PutOuts        FALSE      FALSE
# Assists        FALSE      FALSE
# Errors         FALSE      FALSE
# NewLeagueN     FALSE      FALSE
# 1 subsets of each size up to 8
# Selection Algorithm: exhaustive
# AtBat Hits HmRun Runs RBI Walks Years CAtBat CHits CHmRun CRuns CRBI CWalks LeagueN DivisionW PutOuts Assists Errors
# 1  ( 1 ) " "   " "  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "    " "     " "       " "     " "     " "   
# 2  ( 1 ) " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "    " "     " "       " "     " "     " "   
# 3  ( 1 ) " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "    " "     " "       "*"     " "     " "   
# 4  ( 1 ) " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "    " "     "*"       "*"     " "     " "   
# 5  ( 1 ) "*"   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "    " "     "*"       "*"     " "     " "   
# 6  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    " "   "*"  " "    " "     "*"       "*"     " "     " "   
# 7  ( 1 ) " "   "*"  " "   " "  " " "*"   " "   "*"    "*"   "*"    " "   " "  " "    " "     "*"       "*"     " "     " "   
# 8  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   " "    " "   "*"    "*"   " "  "*"    " "     "*"       "*"     " "     " "   
# NewLeagueN
# 1  ( 1 ) " "       
# 2  ( 1 ) " "       
# 3  ( 1 ) " "       
# 4  ( 1 ) " "       
# 5  ( 1 ) " "       
# 6  ( 1 ) " "       
# 7  ( 1 ) " "       
# 8  ( 1 ) " "

# we can use the nvmax variable to tell the model the max variables to fit within a model
refit_full = regsubsets(Salary ~ . , data = Hitters, nvmax = 19)
summary(refit_full)
# Subset selection object
# Call: regsubsets.formula(Salary ~ ., data = Hitters, nvmax = 19)
# 19 Variables  (and intercept)
# Forced in Forced out
# AtBat          FALSE      FALSE
# Hits           FALSE      FALSE
# HmRun          FALSE      FALSE
# Runs           FALSE      FALSE
# RBI            FALSE      FALSE
# Walks          FALSE      FALSE
# Years          FALSE      FALSE
# CAtBat         FALSE      FALSE
# CHits          FALSE      FALSE
# CHmRun         FALSE      FALSE
# CRuns          FALSE      FALSE
# CRBI           FALSE      FALSE
# CWalks         FALSE      FALSE
# LeagueN        FALSE      FALSE
# DivisionW      FALSE      FALSE
# PutOuts        FALSE      FALSE
# Assists        FALSE      FALSE
# Errors         FALSE      FALSE
# NewLeagueN     FALSE      FALSE
# 1 subsets of each size up to 19
# Selection Algorithm: exhaustive
# AtBat Hits HmRun Runs RBI Walks Years CAtBat CHits CHmRun CRuns CRBI CWalks LeagueN DivisionW PutOuts Assists Errors
# 1  ( 1 )  " "   " "  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "    " "     " "       " "     " "     " "   
# 2  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "    " "     " "       " "     " "     " "   
# 3  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "    " "     " "       "*"     " "     " "   
# 4  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "    " "     "*"       "*"     " "     " "   
# 5  ( 1 )  "*"   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "    " "     "*"       "*"     " "     " "   
# 6  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    " "   "*"  " "    " "     "*"       "*"     " "     " "   
# 7  ( 1 )  " "   "*"  " "   " "  " " "*"   " "   "*"    "*"   "*"    " "   " "  " "    " "     "*"       "*"     " "     " "   
# 8  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   "*"    "*"   " "  "*"    " "     "*"       "*"     " "     " "   
# 9  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"    " "     "*"       "*"     " "     " "   
# 10  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"    " "     "*"       "*"     "*"     " "   
# 11  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"    "*"     "*"       "*"     "*"     " "   
# 12  ( 1 ) "*"   "*"  " "   "*"  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"    "*"     "*"       "*"     "*"     " "   
# 13  ( 1 ) "*"   "*"  " "   "*"  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"    "*"     "*"       "*"     "*"     "*"   
# 14  ( 1 ) "*"   "*"  "*"   "*"  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"    "*"     "*"       "*"     "*"     "*"   
# 15  ( 1 ) "*"   "*"  "*"   "*"  " " "*"   " "   "*"    "*"   " "    "*"   "*"  "*"    "*"     "*"       "*"     "*"     "*"   
# 16  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   " "   "*"    "*"   " "    "*"   "*"  "*"    "*"     "*"       "*"     "*"     "*"   
# 17  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   " "   "*"    "*"   " "    "*"   "*"  "*"    "*"     "*"       "*"     "*"     "*"   
# 18  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   "*"   "*"    "*"   " "    "*"   "*"  "*"    "*"     "*"       "*"     "*"     "*"   
# 19  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   "*"   "*"    "*"   "*"    "*"   "*"  "*"    "*"     "*"       "*"     "*"     "*"   
# NewLeagueN
# 1  ( 1 )  " "       
# 2  ( 1 )  " "       
# 3  ( 1 )  " "       
# 4  ( 1 )  " "       
# 5  ( 1 )  " "       
# 6  ( 1 )  " "       
# 7  ( 1 )  " "       
# 8  ( 1 )  " "       
# 9  ( 1 )  " "       
# 10  ( 1 ) " "       
# 11  ( 1 ) " "       
# 12  ( 1 ) " "       
# 13  ( 1 ) " "       
# 14  ( 1 ) " "       
# 15  ( 1 ) " "       
# 16  ( 1 ) " "       
# 17  ( 1 ) "*"       
# 18  ( 1 ) "*"       
# 19  ( 1 ) "*" 


# summary can also given us different performance metrics
names(summary(refit_full))

# let's look at rsq
summary(refit_full)$rsq
# [1] 0.3214501 0.4252237 0.4514294 0.4754067 0.4908036 0.5087146 0.5141227 0.5285569 0.5346124 0.5404950 0.5426153 0.5436302
# [13] 0.5444570 0.5452164 0.5454692 0.5457656 0.5459518 0.5460945 0.5461159

par(mfrow = c(2,2))
# let's plot the outputs
plot(summary(refit_full)$rss, xlab = 'Number of Variables', ylab = 'RSS')

plot(summary(refit_full)$adjr2, xlab = 'Number of Variables', ylab = 'Adjusted RSq')

# find the maxmimum point of our metrics
which.max(summary(refit_full)$adjr2)

# plot these as points
points(11, summary(refit_full)$adjr2[11], col = 'red', cex = 2, pch = 20)


# we can plot other metrics using which.min
plot(summary(refit_full)$cp, xlab = 'Number of Variables', ylab = 'Cp')

# find the min to plot
which.min(summary(refit_full)$cp)

# extract the points to plot
points(10, summary(refit_full)$cp[10], col = 'red', cex = 2, pch = 20)

par(mfrow = c(1,1))
# our function regsubsets has a built in plot function!!
plot(refit_full, scale = 'r2')
plot(refit_full, scale = 'adjr2')
plot(refit_full, scale = 'Cp')
plot(refit_full, scale = 'bic')

# access the best subset model
coef(refit_full, 6)
# (Intercept)        AtBat         Hits        Walks         CRBI    DivisionW 
# 91.5117981   -1.8685892    7.6043976    3.6976468    0.6430169 -122.9515338 
# PutOuts 
# 0.2643076 


## Forward and Backward Stepwise Selection
# we can also use regsubsets function to perform forward and backward stepwise selection
# we use the method 'forward' and 'backward'

# fit forward stepwise model
regfit_fwd <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19,
                         method = 'forward')
summary(regfit_fwd)
# Subset selection object
# Call: regsubsets.formula(Salary ~ ., data = Hitters, nvmax = 19, method = "forward")
# 19 Variables  (and intercept)
# Forced in Forced out
# AtBat          FALSE      FALSE
# Hits           FALSE      FALSE
# HmRun          FALSE      FALSE
# Runs           FALSE      FALSE
# RBI            FALSE      FALSE
# Walks          FALSE      FALSE
# Years          FALSE      FALSE
# CAtBat         FALSE      FALSE
# CHits          FALSE      FALSE
# CHmRun         FALSE      FALSE
# CRuns          FALSE      FALSE
# CRBI           FALSE      FALSE
# CWalks         FALSE      FALSE
# LeagueN        FALSE      FALSE
# DivisionW      FALSE      FALSE
# PutOuts        FALSE      FALSE
# Assists        FALSE      FALSE
# Errors         FALSE      FALSE
# NewLeagueN     FALSE      FALSE
# 1 subsets of each size up to 19
# Selection Algorithm: forward
# AtBat Hits HmRun Runs RBI Walks Years CAtBat CHits CHmRun CRuns CRBI CWalks
# 1  ( 1 )  " "   " "  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "   
# 2  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "   
# 3  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "   
# 4  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "   
# 5  ( 1 )  "*"   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "   "*"  " "   
# 6  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    " "   "*"  " "   
# 7  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    " "   "*"  "*"   
# 8  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    "*"   "*"  "*"   
# 9  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"   
# 10  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"   
# 11  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"   
# 12  ( 1 ) "*"   "*"  " "   "*"  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"   
# 13  ( 1 ) "*"   "*"  " "   "*"  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"   
# 14  ( 1 ) "*"   "*"  "*"   "*"  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"   
# 15  ( 1 ) "*"   "*"  "*"   "*"  " " "*"   " "   "*"    "*"   " "    "*"   "*"  "*"   
# 16  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   " "   "*"    "*"   " "    "*"   "*"  "*"   
# 17  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   " "   "*"    "*"   " "    "*"   "*"  "*"   
# 18  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   "*"   "*"    "*"   " "    "*"   "*"  "*"   
# 19  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   "*"   "*"    "*"   "*"    "*"   "*"  "*"   
# LeagueN DivisionW PutOuts Assists Errors NewLeagueN
# 1  ( 1 )  " "     " "       " "     " "     " "    " "       
# 2  ( 1 )  " "     " "       " "     " "     " "    " "       
# 3  ( 1 )  " "     " "       "*"     " "     " "    " "       
# 4  ( 1 )  " "     "*"       "*"     " "     " "    " "       
# 5  ( 1 )  " "     "*"       "*"     " "     " "    " "       
# 6  ( 1 )  " "     "*"       "*"     " "     " "    " "       
# 7  ( 1 )  " "     "*"       "*"     " "     " "    " "       
# 8  ( 1 )  " "     "*"       "*"     " "     " "    " "       
# 9  ( 1 )  " "     "*"       "*"     " "     " "    " "       
# 10  ( 1 ) " "     "*"       "*"     "*"     " "    " "       
# 11  ( 1 ) "*"     "*"       "*"     "*"     " "    " "       
# 12  ( 1 ) "*"     "*"       "*"     "*"     " "    " "       
# 13  ( 1 ) "*"     "*"       "*"     "*"     "*"    " "       
# 14  ( 1 ) "*"     "*"       "*"     "*"     "*"    " "       
# 15  ( 1 ) "*"     "*"       "*"     "*"     "*"    " "       
# 16  ( 1 ) "*"     "*"       "*"     "*"     "*"    " "       
# 17  ( 1 ) "*"     "*"       "*"     "*"     "*"    "*"       
# 18  ( 1 ) "*"     "*"       "*"     "*"     "*"    "*"       
# 19  ( 1 ) "*"     "*"       "*"     "*"     "*"    "*" 

# backwards selection
regfit_bwd <- regsubsets(Salary ~ . , data = Hitters, nvmax = 19,
                         method = 'backward')
summary(regfit_bwd)
# Subset selection object
# Call: regsubsets.formula(Salary ~ ., data = Hitters, nvmax = 19, method = "backward")
# 19 Variables  (and intercept)
# Forced in Forced out
# AtBat          FALSE      FALSE
# Hits           FALSE      FALSE
# HmRun          FALSE      FALSE
# Runs           FALSE      FALSE
# RBI            FALSE      FALSE
# Walks          FALSE      FALSE
# Years          FALSE      FALSE
# CAtBat         FALSE      FALSE
# CHits          FALSE      FALSE
# CHmRun         FALSE      FALSE
# CRuns          FALSE      FALSE
# CRBI           FALSE      FALSE
# CWalks         FALSE      FALSE
# LeagueN        FALSE      FALSE
# DivisionW      FALSE      FALSE
# PutOuts        FALSE      FALSE
# Assists        FALSE      FALSE
# Errors         FALSE      FALSE
# NewLeagueN     FALSE      FALSE
# 1 subsets of each size up to 19
# Selection Algorithm: backward
# AtBat Hits HmRun Runs RBI Walks Years CAtBat CHits CHmRun CRuns CRBI CWalks
# 1  ( 1 )  " "   " "  " "   " "  " " " "   " "   " "    " "   " "    "*"   " "  " "   
# 2  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    "*"   " "  " "   
# 3  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    "*"   " "  " "   
# 4  ( 1 )  "*"   "*"  " "   " "  " " " "   " "   " "    " "   " "    "*"   " "  " "   
# 5  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    "*"   " "  " "   
# 6  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    "*"   " "  " "   
# 7  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    "*"   " "  "*"   
# 8  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    "*"   "*"  "*"   
# 9  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"   
# 10  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"   
# 11  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"   
# 12  ( 1 ) "*"   "*"  " "   "*"  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"   
# 13  ( 1 ) "*"   "*"  " "   "*"  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"   
# 14  ( 1 ) "*"   "*"  "*"   "*"  " " "*"   " "   "*"    " "   " "    "*"   "*"  "*"   
# 15  ( 1 ) "*"   "*"  "*"   "*"  " " "*"   " "   "*"    "*"   " "    "*"   "*"  "*"   
# 16  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   " "   "*"    "*"   " "    "*"   "*"  "*"   
# 17  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   " "   "*"    "*"   " "    "*"   "*"  "*"   
# 18  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   "*"   "*"    "*"   " "    "*"   "*"  "*"   
# 19  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   "*"   "*"    "*"   "*"    "*"   "*"  "*"   
# LeagueN DivisionW PutOuts Assists Errors NewLeagueN
# 1  ( 1 )  " "     " "       " "     " "     " "    " "       
# 2  ( 1 )  " "     " "       " "     " "     " "    " "       
# 3  ( 1 )  " "     " "       "*"     " "     " "    " "       
# 4  ( 1 )  " "     " "       "*"     " "     " "    " "       
# 5  ( 1 )  " "     " "       "*"     " "     " "    " "       
# 6  ( 1 )  " "     "*"       "*"     " "     " "    " "       
# 7  ( 1 )  " "     "*"       "*"     " "     " "    " "       
# 8  ( 1 )  " "     "*"       "*"     " "     " "    " "       
# 9  ( 1 )  " "     "*"       "*"     " "     " "    " "       
# 10  ( 1 ) " "     "*"       "*"     "*"     " "    " "       
# 11  ( 1 ) "*"     "*"       "*"     "*"     " "    " "       
# 12  ( 1 ) "*"     "*"       "*"     "*"     " "    " "       
# 13  ( 1 ) "*"     "*"       "*"     "*"     "*"    " "       
# 14  ( 1 ) "*"     "*"       "*"     "*"     "*"    " "       
# 15  ( 1 ) "*"     "*"       "*"     "*"     "*"    " "       
# 16  ( 1 ) "*"     "*"       "*"     "*"     "*"    " "       
# 17  ( 1 ) "*"     "*"       "*"     "*"     "*"    "*"       
# 18  ( 1 ) "*"     "*"       "*"     "*"     "*"    "*"       
# 19  ( 1 ) "*"     "*"       "*"     "*"     "*"    "*" 


# we see that using forward selection the best one variable model contains obly CRBI
# the best two variable model additionally includes hits
# for this data the best one-variable through six variable models are each identical for best subset and forward selection
# however, the best seven variable models identified by forward stepwise selection, backwards, and best subset selection are different

## Choosing Among Models Using the Validation Set Approach and CV
# it is possible to chose among a set of models of different sizes using Cp, BIC, and adjusted R^2
# we will now consider how to do this using validation and crossvalidation approaches

# in order for these approaches to yeild accurate estimates of the test error we must only use THE TRAINING SET to perform all aspects of model fitting
# this includes variables selection
# the determination of which model of a given size is best must be made using ONLY THE TRAINING OBSERVATIONS
# this point is important!
# if the full data set is used to perform the best subset selection step the validation set errors and cross validation erros that we obtain will not be accurate estimates of test error
# in order to use the validation set approach we begin by splitting the observations into a training set and a test set
# we do this by creating a random vector train and use that to parition our data set

# get the partition
set.seed(1)
train = sample(c(T, F), nrow(Hitters), rep = T)
test =(!train) 

train_hitters <- Hitters[train,]
test_hitters <- Hitters[-train,]

# now we can apply of best selection to the training data
regfit_best <- regsubsets(Salary~., data = train_hitters, nvmax = 19)
# Subset selection object
# Call: regsubsets.formula(Salary ~ ., data = train_hitters, nvmax = 19)
# 19 Variables  (and intercept)
# Forced in Forced out
# AtBat          FALSE      FALSE
# Hits           FALSE      FALSE
# HmRun          FALSE      FALSE
# Runs           FALSE      FALSE
# RBI            FALSE      FALSE
# Walks          FALSE      FALSE
# Years          FALSE      FALSE
# CAtBat         FALSE      FALSE
# CHits          FALSE      FALSE
# CHmRun         FALSE      FALSE
# CRuns          FALSE      FALSE
# CRBI           FALSE      FALSE
# CWalks         FALSE      FALSE
# LeagueN        FALSE      FALSE
# DivisionW      FALSE      FALSE
# PutOuts        FALSE      FALSE
# Assists        FALSE      FALSE
# Errors         FALSE      FALSE
# NewLeagueN     FALSE      FALSE
# 1 subsets of each size up to 19
# Selection Algorithm: exhaustive

# we can now compute the validation erro for the best model of each size!
# we first make a model matrix from the test data
test_mat <- model.matrix(Salary~., data = test_hitters)

# the model matrix function is used for building an X matrix of all predictors
# now we loop through each model:
# for each i we extract the coefficients from the model...
# then multiply them into the appropriate columns of the TEST model matrix to form the predictions
# then we compute the test MSE

# set up blank list to put MSE values into outside of the loop
val.errors = rep(NA,19)

# MSE for loop
for (i in 1:19) {
        
        coeff = coef(regfit_best, id = i) # select the coefficients from the ith best fit model
        pred = test_mat[,names(coeff)]%*%coeff
        val.errors[i] = mean((test_hitters$Salary - pred)^2)
}

# view the mse for each model
val.errors
# [1] 140115.6 118979.9 141962.3 123722.7 113679.4 122861.2 111861.6 116507.2 116429.0
# [10] 116942.9 112350.2 112061.5 112669.8 112787.2 112910.5 113019.4 112776.1 112758.4
# [19] 112680.0

# find the model with the lowest MSE
# remember to use which.min to look over a list of values - this will give us the index!
which.min(val.errors)
# [1] 7

# use the best model and find the coefficients
coef(regfit_best,7)
# (Intercept)        Walks       CAtBat        CHits         CRBI       CWalks 
# 71.3715653    5.0905823   -0.4014289    1.4814354    1.3608444   -0.9906385 
# DivisionW      PutOuts 
# -157.3340571    0.3510259 

# this was hard because there is no predict function in regsubsets
# let's create our own
predictRegsubsets <- function(object, newdata, id, ...) {
        form = as.formula(object$call[[2]])
        mat = model.matrix(form, newdata)
        coeff = coef(object, id = id)
        xvars = names(coeff)
        mat[,xvars]%*%coeff # matrix multiplcation of coefficents by the test data columns
}


# finally we perform best subset selection on the full dataset and select the best 7 variable model!
# it is important that we make use of the full dataset and select the best 10 variable model
# using more data will give us more accurate estimates - we know how many variables in the model we need from our validation

# fit model on full dataset
regfit_best_7 <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19)

# get the coeffiicents from the 7 variable model
coef(regfit_best_7,7)
# (Intercept)         Hits        Walks       CAtBat        CHits       CHmRun 
# 79.4509472    1.2833513    3.2274264   -0.3752350    1.4957073    1.4420538 
# DivisionW      PutOuts 
# -129.9866432    0.2366813 


# we notice that the 7 variable model on the entire dataset has a different set of variables!


# we will now choose amoung the various sized models using cross validation
# this approach is involved
# we must perform best subset selection within each of the k training sets!
# first create a vector that allocates each observation to one of k = 10 folds...
# then we create a matrix in which we will stor the results
k = 10
set.seed(1)

# define which observations go into which folds
folds <- sample(1:k, nrow(Hitters), replace = T)

# build a blank matrix to store the MSE metrics for each fold in
# buids an 10X19 matrix: we will store the MSE fro the jth fold and the ith best model
cv.errors <- matrix(NA, k, 19, dimnames = list(NULL, paste(1:19)))




























































































































































































































































