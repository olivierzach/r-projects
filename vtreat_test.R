
# vtreat missing values examples ------------------------------------------



# workflow and data -------------------------------------------------------


library('ggplot2')
library('tidyverse')
library('vtreat')
data('msleep')


# view data ---------------------------------------------------------------



msleep %>% as_tibble
msleep %>% View
str(msleep, width = 70, strict.width = 'cut')



# profile missing values --------------------------------------------------


# what percent is our values missing?
mean(is.na(msleep$brainwt))


# all columns: vore, conversation, sleep_rem, sleep_cycle, brainwt are missing
# missing at random or missing systematic?
sapply(msleep, function(x) {mean(is.na(x))})



# missing numeric values --------------------------------------------------



# missing at random: ok to replace with standins...
# inferred values, distribution of values, expected value, mean value
# correct in the aggregate easy and cost effective to implement
# model approach: impute missing values based on other variables available
# Kabacoff (2015) for imputation methods...can we random forest?


# missing systematically:
# it is possible that missing values are systematically different that other values
# some animals do not have REM sleep...missing values would be based on a "system"
# this means we cannot use traditional mean imputation to fill in values
# practical solution is to fill in the missing values with a nominal value...
# mean of missing data only or 0
# we should also add a variable that tracks which data has been altered

# code example
msleep$sleep_rem_isBAD <- is.na(msleep$sleep_rem)
msleep$sleep_rem_clean <- ifelse(
  msleep$sleep_rem_isBAD, # if ue give mean of sleep_rem, else, original value
  mean(msleep$sleep_rem, na.rm = T),
  msleep$sleep_rem
)

# check out the data
msleep %>% as_tibble
msleep$sleep_rem_clean
msleep$sleep_rem_isBAD


# motivation: linear models
# suppose predict y ~ x where x has missing values
# if we fill in the missing values with mean(x)...
# we are saying that x has no net effect on y when x is missing...
# if we also have an indicator variable "isBAD"...
# then this indicator will estimate the expected value of y for missing rows of x
# more flexible models = might be able to model non-linear effects...
# i.e interactions between missingness and other variables

# when it is not known whether MAR or MS...vtreat is conservative and assumes MS
# missingness is often an indictor of data provenance in a business setting
# the missing indicator column can be a highly informative variable



# missing categorical variables -------------------------------------------

# one can treat missing categorical values as another level in the factor
# if NA occurs during variable treatment design we get stats...
# on the relationship between missing and outcome
# if NA does not occur during treatment design it is treated as a novel level


# novel categorical levels and indicators
# R can accepts modeling function with categorical variables directly
# while this is great...
# sometimes R does not handle categorical levels that were not present in training

# "train" data
df <- data_frame(
  x = c('a', 'a', 'b', 'b', 'c', 'c'),
  y = 1:6
)

df

# model fit 
model <- lm(y ~ x, data = df)

# "test" data with a new factor level
newdata <- data_frame(
  x = c('a', 'b', 'c', 'd')
)
newdata

tryCatch(
  predict(model, newdata = newdata),
  error = function (x) print(strwrap(x))
)


# to avoid this we would like to detect novel levels in new data and encode them...
# in a way our models can understand
# we can do this by representing categorical variables as indicators

# implementation in vtreat
# creates new derived variables + indicators from original data
treatplan <- designTreatmentsN(df, 'x', 'y') 

# indicator variables have the designiation "lev" in treatment plan
varnames <- treatplan$scoreFrame$varName[treatplan$scoreFrame$cod == 'lev'] 

# prepare applies a treatment plan to a new dataframe
newdata_treat <- prepare(
  treatplan,
  newdata,
  pruneSig = NULL, 
  varRestriction = varnames
)

# this puts data from 1 columns to a combines indicator column
# vtreat encodes these facotrs as indicators even in novel level presence
newdata_treat %>% View
newdata %>% View



treat_train <- prepare(
  treatplan, 
  df, 
  pruneSig = NULL,
  varRestriction = varnames
)

treat_test <- prepare(
  treatplan,
  newdata,
  pruneSig = NULL,
  varRestriction = varnames
)


vtreat_model <- lm(y ~ ., data = treat_train)
vtreat_predict <- predict(vtreat_model, newdata = treat_test)

# this results in data that can be safely input into a model that was trained on original data...
# where our level d was not present!


# how are novel levels represented?
# suppose our training data has x with values {a,b,c,d,e}
# we also know the observed frequency tables {fa, fb, fc, fd, fe}
# for example d and e are "rare" levels 
# we will represent a value of x as the tuple (sa, sb, sc, sd, se)
# values usually take on 0 or 1: ex. value c can be (0,0,1,0,0)

# after we fit a model on the training data we apply it to new data...
# in this case we see x take on the previously unseen value w
# how do we encode w?
# ...three possible solutions

# Novel levels are represented as "no level"
# w <- (0,0,0,0,0)
# this is the most straight forward representation
# ...if x takes on a previously unseen value, it has no effect on the outcome

# novel levels are weighted proportional to known levels
# w <- (fa, fb, fc, fd, fe)
# we assume that the novel level is really one of the known levels in our data
# we assume it is proportional to the prevalence of each level in the training data
# linear models tend to predict the weighted average of the outcomes that would be predicted for each known level

# novel levels are treated as uncertainty among rare levels
# w <- (0,0,0,.5, .5)
# d and e levels are rare
# we pool the rare levels into a single category level "rare" before modeling
# and then re-encode novels as rare during model deployment
# inuition: previously unobserved values are simply rare, and they similiarily effect the output



# high cardinality categorical variables ----------------------------------

# problem variable: categorical variable with many possible values / levels
# why are these problems?
# computational costs: varaible with k levels is transformed into k-1 numerical varaibles
# are the number of levels grows - they might not show up in our training data
# we eventually run into the novel level problem!
# to fix this we should convert high-cardinality categorical variables into numeric variables

# Look up codes
# can try to map domain specific numeric data to each of our categorical levels
# this might not be available or appropriate in all situations

# Impact or Effects coding
# idea is to convert the problematic variable into a small number of numeric variables
# this is known as effects coding or impact coding
# in vtreat - we replace high-cardinality varaibles with one-variable model vs. outcome of interest!!


# this example creates a dataframe with zip code and a random value
# the zip code z03 takes up 80% of the data
# there are 25 possible values that zip can take on
set.seed(235)
Nz <- 25
zip <- paste0('z', format(1:Nz, justify = 'right'))
zip <- gsub(' ', '0', zip, fixed = T)
zipval <- 1:Nz; names(zipval) <- zip
n <- 3; m <- Nz - n
p <- c(numeric(n) + (0.8/n), numeric(m) + 0.2/m)
N <- 1000
zipvar <- sample(zip, N, replace=TRUE, prob=p)
signal <- zipval[zipvar] + rnorm(N)
d <- data.frame(zip=zipvar,y=signal + rnorm(N))


# here is an implementation of the treatment plan
treatplan <- designTreatmentsN(
  d, 
  varlist = "zip",
  outcome = "y",
  verbose = F
)

# check the mean of the outcome
treatplan$meanY


# the lev variables are indicator variables that were created for the more prevalent levels
# all levels are impact coded into the variable zip_catN
# impact coded variable encodes the difference between the expected outcome on zip code and the overall expected outcome
# the expected "impact" of a particular zip code on the outcome y
# Impact(zip) = E[y | zip] - E[y]
scoreFrame <- treatplan$scoreFrame
scoreFrame[, c('varName', 'sig', 'extraModelDegrees', 'origName', 'code')]


# representing levels of a categorical variable as both impact and indicator values is redundant but can be useful
# indicator variables can model interactions between specific levels and other variables...
# impact coding cannot
# we can leave this choice up to the downstream model to decide

# this function will prepare the zip variable into the indicator and impact-coded variables
vars <- scoreFrame$varName[!(scoreFrame$code %in% c('catP', 'catD'))]
dtreated <- prepare(
  treatplan, 
  d, 
  pruneSig = NULL,
  varRestriction = vars
)

# impact coding also works similarly when the outcome of interest is categorical
# binary two class classification
# in the case of categorical outcome y with target class target...
# the impact code represents levels of a categorical variable x
# Impact(xi) = logit(P[y == target|xi|]) - logit(P[y == target])

# novel level impact codes
# all possible zip codes were present in the training data
# as we noted previously this might not always be true
# especially if we run into high cardinality categorical values
# if these levels are encountered in the future - they are treated as having zero impact

# small training set to have new levels pop out of new possible zip codes
N <- 100
zipvar <- sample(zip, N, replace=TRUE, prob=p)
signal <- zipval[zipvar] + rnorm(N)
d <- data.frame(zip=zipvar, y=signal+rnorm(N))
length(unique(d$zip))

# here are the levels we do not have in the training set
omitted <- setdiff(zip, unique(d$zip))
print(omitted)

# create treatment plan
treatplan <- designTreatmentsN(
  d, 
  varlist = 'zip',
  outcome = 'y',
  verbose = T
)

# "test" data that includes the missing factor values
dnew <- data.frame(zip = zip)

# prepare the data using the treatment plan
dtreated <- prepare(
  treatplan, 
  dnew,
  pruneSig = NULL,
  varRestriction = vars
)

# results: zip codes which were missing in the training data are encoded as...
# no impact on the outcome (missing levels treated as zeros)
dtreated[dnew$zip %in% omitted, "zip_catN"]



# nested model bias -------------------------------------------------------

# we must be careful when impact coding variables
# the data used to do the impct coding should not be the same as the data used to fit the model
# this is because impact coding are complex high cardinality level variables that are encoded to low degree of freedom variables
# impact coded high cardinality categorical variables may not be handled correctly by downstream modeling

# example:
# we run the risk of overfitting
# our binary classification only depends on the "good" variables not on the "bad" high level variables

set.seed(2262)
nLev <- 500
n <- 3000

d <- data.frame(
  xBad1=sample(paste('level', 1:nLev, sep=''), n, replace=T),
  xBad2=sample(paste('level', 1:nLev, sep=''), n, replace=T),
  xGood1 = sample(paste('level', 1:nLev, sep = ''), n, replace = T),
  xGood2 = sample(paste('level', 1:nLev, sep = ''), n, replace = T)
)

# sets up the outcome variable to be related only to the "good" variables
d$y <- (
  0.2*rnorm(nrow(d)) +
  0.5*ifelse(as.numeric(d$xGood1)>nLev/2, 1, -1) + 
  0.3*ifelse(as.numeric(d$xGood2)>nLev/2, 1, -1)
  ) > 0

d$rgroup <- sample(
  c("cal", "train", "test"),
  nrow(d), 
  replace=TRUE,
  prob=c(0.6, 0.2, 0.2)
  )

# check sample 
mean(d$rgroup == 'cal')



## THE WRONG WAY
# naive data partitioning
# first we will parition the data into a training set and test set
# then create a treatment plan
# this plan will include impact codings for the categorical variables xBadi

# subset into train and test
dTrain <- d[d$rgroup!='test', , drop=FALSE]
dTest <- d[d$rgroup=='test', , drop=FALSE]

# setup treatment plan
treatments <- vtreat::designTreatmentsC(
  dTrain,
  varlist = c('xBad1', 'xBad2', 'xGood1', 'xGood2'),
  outcomename='y',
  outcometarget=TRUE,
  verbose=FALSE
  )

# prepare data using the treatment plan
dTrainTreated <- vtreat::prepare(treatments, dTrain, pruneSig=NULL)


# fit model to our training set
m1 <- glm(
  y ~ xBad1_catB + xBad2_catB + xGood1_catB + xGood2_catB,
  family = binomial(link = 'logit'),
  data = dTrainTreated
)

# note the low residual deviance and the signifcant "bad" variables
summary(m1)


# see performance on train data
dTrain$predM1 <- predict(m1, newdata = dTrainTreated, type = 'response')

dTrain$preds <- ifelse(dTrain$predM1 > .5,  TRUE, FALSE)
dTrain$acc_train <- dTrain$y == dTrain$preds
mean(dTrain$acc_train)


# see performance drop on test data - classic case of overfitting
dTestTreated <- vtreat::prepare(treatments, dTest, pruneSig=NULL)
dTest$predM1 <- predict(m1, newdata=dTestTreated, type='response')

dTest$preds <- ifelse(dTest$predM1 > .5,  TRUE, FALSE)
dTest$acc_test <- dTest$y == dTest$preds
mean(dTest$acc_test)



## WHAT IS THE RIGHT WAY? a calibration set!!
# consider a trained stats model as a two-arguement function f(A,B)
# A = training data, B = application data
# designTreatmentsC(A) %>% prepare(B) produces a treated data frame
# when we use the same data in both places to build our training frame...
# TrainTreated = f(TrainData, TrainData) we are not doing a good job simulating new data
# future application of f() = f(TrainData, FutureData)

# we can improve the quality of the simulation using TrainTreated = f(CalibrationData, TrainData)
# calibration data and traindata and disjoint datasets
# we expect this to be a good estimation of future data
# this is the idea behind cross validation

# re-do the same problem above but partition the data into training, calibration, and holdout
# the impact coding is fit to the calibration set
# the overall model is fit to the training set

dCal <- d[d$rgroup == 'cal', , drop = F]
dTrain <- d[d$rgroup == 'train', ,drop = F]
dTest <- d[d$rgroup == 'test', , drop = F]

# build the impact coding on the calibration set
treatments <- designTreatmentsC(
  dCal,
  varlist = c('xBad1', 'xBad2', 'xGood1', 'xGood2'),
  outcomename =  'y',
  outcometarget = T,
  verbose = T
)

# prepare the training set using the treatment built on the calibration set
dTrainTreated <- prepare(
  treatments, 
  dTrain, 
  pruneSig = NULL
)

# remove the output or response from our newvars
newvars <- setdiff(colnames(dTrainTreated), 'y')

# fit a model on the prepared training data - built up from a treatment on calibration set
m2 <- glm(
  y ~ .,
  data = dTrainTreated,
  family = 'binomial'
)

# note that this model successfully recognizes the bad variables are not significant!
print(summary(m2))

# see performance on train data
dTrainTreated$predM2 <- predict(m2, newdata = dTrainTreated, type = 'response')

# 86% accuracy on training
dTrainTreated$preds <- ifelse(dTrainTreated$predM2 > .5,  TRUE, FALSE)
dTrainTreated$acc_train <- dTrainTreated$y == dTrainTreated$preds
mean(dTrainTreated$acc_train)

# what do the results look like on test?
# first need to prepare the test data
dTestTreated <- prepare(
  treatments,
  dTest,
  pruneSig = NULL
)


# predict the treated test set using our model built on calibration and train
dTestTreated$predM2 <- predict(m2, newdata = dTestTreated, type = 'response')

# what is the accuracy on test? we expect smaller variance than previous
# performance on holdout set is a lot closer to performance on train!
dTestTreated$preds <- ifelse(dTestTreated$predM2 > .5, TRUE, FALSE)
dTestTreated$acc_test <- dTestTreated$y == dTestTreated$preds
mean(dTestTreated$acc_test)

# overfitting in this example has been solved in two ways:
# training performance is closer to test performance
# test performance is better than that with the model fit using the naive data partition


## ANOTHER CORRECT IMPLEMENTATION
# cross validation 
# we want to build a cross validated function of our f()
# we split TrainData into a list of 3 disjoint row intervals:
# [Train1, Train2, Train3]
# our goal is to compute:
# TrainTreated = f(Train2 + Train3, Train1) + f(Train1 + Train3, Train2) + f(Train1 + Train2, Train3)
# + here denotes rbind()
# no row in the right hand side is ever worked on by a model built using that row!
# this mimics future data !! i.e. f(TrainData, FutureData)
# cross validatin is used to simluate future data
# check the form buildSubModels(A) %>% combineModels(B) == ensemble methods




# wide data: variable significance and pruning ----------------------------

# wide datasets with many variables relative to number of observations are:
# costly computationally
# lead to overfitting that generalized poorly to new data
# in the extreme cases wide data can fool modeling procedures into finding models that perform on training even through there is no signal
# it is wise to prune back or remove irrelevant variables before modeling

# standard approaches:
# stepwise regression, regularized regression, variable importance estimates
# stepwise regression suffers from multiple experiment bias - bias caused by repeated evaluation of interim models on the same data set


# vtreat offers estimates of variable significance and the option of pruning variables based on the signifances during the data preparation step
# variable significances are based on the significance of a corresponding single variable model
# for problems with a numberic outcome, variables significances are based on the F statistic of a single variable linear regression
# for classification problems, the significance is based on the x^2 statistic of a single variable logistic regression
# note: a categorical variable with k levels is equivalent to k-1 indicator variables
# these additional degrees of freedom must be accounted for when estimating the significance of F or x^2
# the variable pruning of these significances is only based on hueristics
# idea: a useful variable has a signal that could be detected by a small linear model even if the original relation is complex or non-linear
# this technique can miss variables that are significant in a large joint model

# Choosing the significance threshold:
# we can iterpret the significance of a variable as the probability that a non-signaling variable would have an F or x^2 statistic as large as the variable observed (stochastic)
# if we have 100 non-signaling variables with a significance of less than p = .05
# then we expect to erroneously accept about 100 * .05 = 5 non-significant varibles
# the significance threshold for varible pruning is the false positive rate we are willing to accept
# robust modeling algorithms should be able to tolerate a few insignificant variables
# note: p should be set to p = 1 / nvar where nvar is the number of considered variables


# example:
# regression problem with two numeric inputs (one signal one noise) and two high-cardinality categoric inputs (100 levels each)

set.seed(22451)
N <- 500
# numeric value inputs sampled from normal distribution; one signal one noise
sigN <- rnorm(N)
noiseN <- rnorm(N)

# set up high cardinal variables
Nlevels <- 100
zip <- paste0('z', format(1:Nlevels, justify = 'right'))
zip <- gsub(' ', '0', zip, fixed = T)
zipval <- runif(Nlevels)
names(zipval) = zip

# set up the categorical values one signal one noise
sigC <- sample(zip, size = N, replace =T)
noiseC <- sample(zip, size = N, replace = T)

# set up the repsonse to be  function of the signal variables but not the noise variables; but include some noise in true function
y <- sigN + zipval[sigC] + rnorm(N)

# create model data frame to test the experiments
df <- data.frame(
  sN = sigN, 
  nN = noiseN,
  sC = sigC,
  nC = noiseC, 
  y = y
)

# set up treatment plan
treatplan <- designTreatmentsN(
  df, 
  varlist = setdiff(colnames(df), 'y'),
  outcomename = 'y',
  verbose = T
)

sframe <- treatplan$scoreFrame
vars <- sframe$varName[!(sframe$code %in% c('catP', 'catD'))]
sframe[sframe$varName %in% vars, c('varName', 'sig', 'extraModelDegrees')]

# for each derived variable - we get a significance estimate and any extra degrees of freedom in the 'one variable model'
# this helps the analyst reproduce the corresponding significance calculation
# categorical: # of degrees of freedom - 1

sframe$varType <- ifelse(grepl('s', sframe$varName), 'signal', 'noise')
sframe$varType

# plotting the signifcance estimates
ggplot(data = sframe, aes(y = sig, x = reorder(varName, sig), fill = varType)) +
  geom_bar(stat = 'identity') +
  coord_flip() +
  geom_hline(yintercept = 1 / nrow(sframe), linetype = 'dotted')

# take a look at the chart
# the dotted line is the proposed pruning significance level
# WE WANT THE VALUES TO THE LEFT OF THE THRESHOLD LINE
# these are variables with significance based on the threshold level!
# we would select sN, impact coding sC_catN, and sC level 8
# notice tht all noise variables have been rejected
# also notice that several signal variables have also been rejected - this could be that our impact coding variable still picks up the levels information!

# lets prepare the out dataframe with this treatment plan
prune <- 1 / nrow(sframe)
dfTreat <- prepare(treatplan, df, pruneSig = prune)
head(dfTreat)







