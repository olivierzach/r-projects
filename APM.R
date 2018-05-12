### Applied Predictive Modeling


# chapter 1: Introduction ---------------------------------------------------------------------

## Introduction
# we all want to predict future events
# we usually make decisions based on information
# sometimes we combine intuition with actual hard data
# we are predicting future events based on experience and information

# we cannot process all the data that is out there
# data has grown exponentially in the past n years and keeps growing
# we use learning procedures to sift through all this data and extract patterns

# the process of developing tools through a number of fields
# the ultimate goal is to make an accurate prediction about the future!
# for this book we will pool these terms into the common phrase predictive modeling

# predictive modeling is defined as 'the process by which a model is created or chosen to try to best predict the probability of an outcome'
# we will use this definition:
# the process of developing a mathmatical tool or model that generates and accurate prediction

# predictive models now permeate our existence!
# but they can have fall backs...
# they regularly provide innacurate predictions and wrong answers...

# predictive models regularly fail because they do not account for complex variables
# these realities of not "knowning all the variables" shouldn't deter us from seeking to improve our process and build better models

# there are many common reasons by models fail:
# inadequate pre=processing of the data
# inadequate model validation
# unjustified extraploation or generalization (predicting a model onto data it hasn't seen before)
# overfitting the model to the data that we do have available!

# ths book tries to help modelers produce reliable, trust worthy models by providing a step-by-step guide
# we also want to provide intuitive knowledge of a wide range of common models
# this book will provide:
# foundational principles for building predictive models
# intuitive explanations of many commonly used predictive modeling methods for both classification and regression problems
# principles and steps for validating a predictive model
# computer code to perform the necessary foundational work to biuld and validate predictive models

# we first want to set the stage by examining the fundamental challenge of modeling: the trade-off between prediction and interpretation

## Prediction vs. Interpretation
# for most examples we have a lot of historical data we can use to build our prediction models
# the foremost objective of these examples is not to understand exactly WHY something happens...
# but rather build an accurate prediction about the future!!
# this type of modeling is to optimize prediction accuracy

# the tension between prediction and interpretation is present everywhere
# the critical idea is the prediction not exactly the how we got to the prediction

# while the primary interest of predictive modeling is to generate accurate predictions
# a secondary interest may be to interpret the model and understand why it works
# the reality is that as we push towards higher accuracy - models become more complex and thier interpreability becomes more difficult
# there is always the trade-off we make when prediction accuracy is the primary goal


## Key Ingredients of Predictive Models
# we now have lots of data
# and the computation to get started with building models easily
# becuase anyone can get started with this...maybe the creditability of modeling has eroded...

# if a predictive signal exists in a set of data many different models should pick up on it
# this is regarless of the care taken in building a the model...
# naive model application can therefore be effective to some extent
# 'even a blind squirrell can find a nut'

# but the best most predictive models are fundamentally influenced by a modeler with expert knowledge and context of the problem
# this expert knowledge should first be applied in obtaining relevant data for the desired research objectives
# irrelevant information can be drive down the true prediction of a model
# domain knowledge can help separate potentially meaningful information from irrelevant information
# this eliminates determental noise and helps tease out the true signal

# still - undesired confounding signal may also exist in the dat and may not be able to be identified without domain knowledge
# in the end predictive modeling is not a substitute for intuition but rather a complement!
# simply put neither data-driven models nor the expert knowledge is the answer - we need to use a combination of the two
# secondly - 
# traditional experts make better decisions when they are provided with the results of statistical prediction
# those who cling to the authority of traditional experts tend to embrace the idea of combining the two forms of 'knowledgeable' by giving the experts 'staistical support'
# humans usually make better predictions when they are provided with the results of statistical prediction

# we can use pure stats modeling or we can use a combination but we have to fit to the problem at hand
# the foundation of an effective predictive model is laid with intuition and deep knowledge of the problem context
# the modeling process begins with relevant data
# the third ingrediant is a versatile computational toolbox which includes processing, visualiztion and modeling techniques


## Terminology
# predictive modeling is one of the many ames that refers to the process of uncovering relationships within data for predicting an outcome
# in this field there are many scientific domains involved and hence many different terms for common entities:

# sample, data point, observation, instance refer to a single independent unit of data
# the term sample can also refer to a subset of data points

# the training set consists of the data used to develop models
# the test or validation sets are used solely for evaluating the performance of a final set of candidate models

# the predictors, independent variables, attributes or descriptors are the data used as input for the prediction equation

# outcome, dependent variable, target, class or response refer to the outcome event or quantity that is being predicted

# continous data have natural numeric scales
# categorical data or nominal, attribute or discrete data take on specific values that have no scale

# model building, model training and parameter esitmation all refer to the process of using data to determine values of the model equations

## Example Data Sets and Data Scenarios
# this section will cover some common examples and the ways to solve them with predictive modeling

# these cases highlight the common characteristics of datasets and modelling problems
# the repsonse may be continous or categorical
# categorical responses there may be two or more categorical variables
# for continous response data the distribution of the repsonse may be symmetric or skewed
# for categorical response data the distribution may be balanced or unbalanced
# understanding the distribution of the response is critically necessary for one of the first steps in the modeling process:
# splitting our data into training and testing sets
# understanding the response distribution will guide the modeler towards better ways of partitioning the data
# not understanding repsonse characteristics can lead to computational difficulties for certain kinds of models and models with less than optimal predictive ability

# these examples also highlight the characterisitcs of predictors that are universal to datasets too
# predictors may be continous, count, categorical
# they may have missing values and could be on different scales of measurement
# additionally predictors within a data set may have high correlation or association...
# thus indicating that the predictor set contains numerically redundant information
# predictors can also be sparse - or have missing information
# like the response - predictors can follow a symmetric or skewed distribution (continous predictors)
# also predictors can be balanced or unbalanced in categorical variables
# predictors may or may not have an underlying relationship with the response!!

# different kinds of models handle these types of predictor characteristics in different ways
# for example - partial least squares naturally manages correlated predictors but is numerically more stable if the predictors are on a similiar scale
# recursive partitioning in unaffected by similiar scales but has less stable strucutre when predictors are correlated
# multiple linear regression cannot handle missing values but recursive partitioning can be used with missing information
# any of these techniques can fail if we do not pre-process our data to fit into our model

# the final characterisitc of data is very fundamental: the relationship between the sample N and the number of predictors P
# if the samples is greater than predictors all models can handle this problem...but may take some computational effort

# but if we have a dataset with significantly fewer samples than actual predictors certain models fail outright
# some models like recursive partitioning and k nearest neighbors can be used directly under this condition
# we will identify a method's ability to handle n < P datasets

# in summary we must have a detailed understanding of the predictors and the repsonse for any data set prior to attempting to build a model
# lack of understanding can lead to computational difficulites and less than optimal model performance
# most data sets require some degree of pre-processing in order to expand the universe of possible predictive models and optimize each model's predictive performance


# chapter 2 tour of the predictive model process ----------------------------------------------

# this chapter covers the broad concepts of building a predictive model
# we will look into data 'spending', building candidate models and selecting the optimal model

## Case Study: Predicting Fuel Economy
# we have a dataset of cars, thier attributes, and thier fuel economy
# we wish to build a predictive model that predicts fuel economy based on the cars attributes

# in practice we would build a model on as many vehicle attributes as possible to find the most predictive model
# in this case study we will focus on a simple one predictor model
# we will predict mpg based on displacement for the 2010 and 2011 model year cars

# the first step in any model is to understand the data
# to do this we should visualize the data if we can
# in the one variable model we can easily plot mpg by displacement
# if we plotted this data we would see that as engine displacement increases, mpg decreases
# there is a strong negative relationship between mpg and displacement
# this relationship holds true regardless of the model year
# we notice the relationship is somewhat linear but does have some curvature 

# if we had more than one predictor we would need to further understand the characterisits of all predictors associated with the response
# these characteristis may imply further pre-processing steps important to building a good model

# after first understanding the data the next step is to build and evaluate a model on the data
# a standard approach is to take a random sample of the data for model building and use the rest for understanding model performance
# in this case we want to predict mpg for a new carline:
# we will separate our data into 2010 for trianing and post 2010 for testing
# the 2010 data will be used as the training set
# the post 2010 data will be used as the model test or validation set

# now that we defined the data used for model building and evaluation we need to pick a metric to judge model performance
# for regression we predict a numeric value so residuals are important sources of information
# residuals are the actual - predicted value
# when predicting numeric values the RMSE (root mean squared error) is commonly used to evaluate models
# RMSE is interpretted as how far, on average, the resiudals are from 0 (0 being the predicted value equals the observed value!)

# at this point the modeler will try various techniques to mathmatically define the relationship between predictor and outcome
# to do this the training set is used to estimate the various values needed by the model equations
# the test set will be used only when a few strong candidate models have been finalized

# suppose we create a linear regression that predicted mpg based on a slope and intercept
# using the training data we determine the intercept to be 50 and the slope to be -4.5 displacementß using least squares
# if we plot this line throughs the original data we can see our model does well but perofrmance degrades at the extremes of cylinders

# when working with a training set, one must be careful not to simply evaluate model performance using the same data to build the model
# if we simply re-predict on the training set - we may produce overly optimistic estimates of coefficients and overestimate how well the model works
# an alternative approach for quantifying how well the model works is to use resampling
# we use different subversions of the training data set to fit our model
# we can use a resampling method of k fold cross validation to estimate test RSME
# using this technique we fit the coefficient estimate is -4.6*displacements

# looking at our model line over the original data - we see the relationship is not quite linear
# we may be able to get a better fit by introducing some non-linearity into our model
# there are many ways to do this
# the basic idea is we augment our original straight line model with additional complexity
# we can ddd a squared term for engine displacement - this will be a new predictor in our model
# this changes the model equation to:
# mpg = intercept + B1*displacement + B2*displacement^2
# this model is refered as a quadratic model since it includes a squared term
# unquestionably, this added term improves our model fit
# the RSME is now estimated to be 4.2 mpg using cross validation
# one issue with quadratic models is that they can perform poorly on the extremes of one predictor
# predicting new vehicles with large displacement values may produce significantly inaccurate results

# there are many other techniques for creating sophisticated relationships between predictors and outcomes
# one appraoch is the multivariate adaptive regression splines (MARS) model
# when used with a single predictor...
# MARS can fit seperate linear regression lines for different ranges of engine displacement
# the intercept, slopes, and size of the regression space are estimated using this model
# unlike the standard least squares regression model this technique has a tuning parameter...
# the tuning parameter cannot be directly estimated from the data
# there is no analytical equation to determine how many regression segments "spaces" we should use in this model
# while MARS has internal algorithms for making this determination...
# the users can try different values and use resampling to determine the "best" tuning parameter value
# once we find our tuning parameter a final MARS model will be fit using all the training set and used for prediction

# for a simple predictor MARS can allow up to five model terms
# using cross validation we evaluate four candidate values for our tuning parameter to create a resampling profile
# the lowest RMSE value indicates that there is some insensitivity to this tuning parameter
# the RSME associated with the optimal model was 4.2 mpg
# after fitting the final MARS model with four terms, the training model is produced
# based on these models, we then predict onto the test set
# the test RMSE values for the quadratic model was 4.72 MPG and 4.69 MPG
# based on this - either model would be appropriate for the prediction of new carlines


## Modeling Themes
# there are several key aspects of the model building process that are fundamental to the process

## Data Splitting
# how we allocate data to certain predictive model tasks is an important aspect of modeling
# which data goes to training? which goes to test?
# for our example we wanted to predict the fuel economy of new vehicles!
# this drives our choice for test and training
# this means we are testing how well a model extrapolates to a different population
# if we were interested in interpolation - predicting on the same populatino of vehicles we would split as a random sample of all vehicles in the dataset
# how test and training sets are determined should reflect how the model will be applied!!
# how much data should be in each set?
# this generally depends on the situation
# if the pool of data is small, data splitting decisions can be critical!
# a small test would have limited utility as a judge on performance
# in this case a sole reliance on resampling techniques might be more effective (i.e. no test set)
# large datasets reduce the critically of these decisions


## Predictor Data
# this example has revolved around one of many possible predictors: engine displacement
# the original data contain many other factors we could throw into our model
# an earnest attempt to predict mpg would include as many predictors as possible
# using more predictors we could drive down our RSME further
# example: none of our models did well when displacement was small...if we add in predictors that could account for this we may find improvement
# this aspect is called feature selection: the process of determining the minimum set of relevant predictors needed by the model


## Estimating Performance
# before using the test set, two techniques were employed to determine the effectiveness of the model
# we use qualtitative assessments of statistics using resampling (k fold cv) to understand how each model would perform on new data
# the other tool was to create simple visualizations of a model such as plotting the observed and predicted values to discover where the model goes right and wrong
# this type of qualitative information is critical for improving moels and is lost when only guaging models on stats!!!


## Evaluating Serveral Models
# for these data three different models were evaluated
# "No Free Lunch" - one model will not unanamously beat out all other models in all situations!!!
# because of this we should aim to test out as many different modelling techniques as possible - then determine which models to focus on
# a simple plot can help determine where to get started - a non-linear form of underlying data will rule out linear models...



## Model Selection
# at some point in the process we need to choose a model
# our example demonstrated two types of model selection
# first we choose some models over other different types of models
# the linear regression model did not fit well and was dropped - we choose between different models
# for our MARS model the tuning parameter was chosen using cross validation
# this type of model selection decided on the tupe of MARS model to use - we choose within the same model
# in either case we relied on cross validation and the test set  to produce an assessment of performance


## Summary
# at face value: the model process seems simple
# pick a model, plug in data, and generate a prediction
# while this approach will generate a model - it most likely will not produce a good predictive model
# to get a good predictive model we must:
# understand the data
# undertand the objective of the model
# we then pre-process and split our data
# only after these steps are complete do we finally proceed to building, evaluating and selecting a model!!!





# chapter 3 data pre-processing ---------------------------------------------------------------

## Data Pre-Processing
# data pre-processing techniques refer to the addition, deletion or transformation of the training data set
# data preparation can make or break a model's predictive ability
# different models have different sensitivities to the type of predictors in the model
# how the predictors enter the model is also important

# transformations of the data to reduce the impact of data skewness or outliers can lead to significant model improvements
# feature extraction is an empirical technique for creating surrogate variables that are combinations of multiple predictors
# simpler strageties such as removing predictors based on thier lack of information content can also be effective

# the need for data pre-processing is determined by the type of model being used
# tree based models are notably insensitive to characteristics of predictor data
# linear regression is not insensitive to the characteriitics of predictor data

# this chapter outlines approaches to unsupervised data processing 
# the outcome variable is not considered by the pre-processing techniques in unsupervised data processing
# in other chapters supervised methods where the outcome is utilized to pre-process data are also discussed
# for example parital least squares (PLS) model are essentially supervised versions of principal component analysis (PCA)

# how the predictors are encoded or scaled is called feature engineering
# fetaure engineering can have a significant impact on a model
# using combinations or predictors may be more effective than using two independent predictors
# encoding is informed by the modeler's understanding of the problem and thus is not determined from a mathmatical technique

# there are usually several different methods for encoding predictor data
# for example the date can be represented as a predictor in many different ways
        # days since a reference date
        # isolating the month year and day of the week and separate predictors
        # the numeric day of the year
        # whether the date was within the school year
# the 'correct' feature engineering depends on several factors
# some encodings may be optimal for some models but not for others
# some models contain built in feature selection
# the model will try to automatically include predictors that help maximize accuracy
# the model can pick and choose which representation of the predictors is best

# the relationship between the predictor and the outcome is a second factor
# if there is a seasonal component to the data then numeric day of year would be best
# encomding by month would be best if we see increased performance by month

# as with many stats questions - 'which feature engineeriring or selection is best' - is the answer 'it depends'
# this answer depends on the model being used and the true relationship with the outcome


## Data Transformation for Individual Predictors
# transformations of predictor variables may be needed for reveral reasons
# some modeling techniques may have strict requirements - such as a common scale for all predictors
# in other cases a good model may be hard to build due to outliers

## Centering and Scaling
# the most straight forward and common data transformation is to center and scale the predictor varaibles
# to center a predictor variable - the average value is subtracted from all the values
# as a result of centering the predictor will now have a mean of 0
# to scale the data, each value of the predictor is divided by the standard deviation
# scaling the data makes the predictor have a standard deviation of 1
# these manipulations are generally used to improve the numeric stability of some calculations
# some models such as PLS benifit from predictors being on a common scale
# the only real downside to these transformations is a loss of intepretability of the individual values since the data or no longer in the original units

## Transformations to Resolve Skewness
# another common reason for transformations is to remove distribution skewness
# an un-skewed distribution is roughly symmetric
# the probability of falling on either side of the mean is roughly equal
# a right-skewed ditribution has a large number of point on the left side of the distribution

# a gneral rule of thumb to consider is that skewed data whose ratio of the highest value to the lowest value is greater than 20 have significant skewness
# the skewness statistic can be used as a diagnostic
# if the predictor distribution is roughly symmetric the skewness values will be close to zero
# as the distribution becomes more left skewed the value becomes negative
# here is the formula for the skewness statistic:
        # skewness = sumof((xi - xbar)^3) / (n - 1)v^(3/2)
# where v = 
        # v = sumof((xi - xbar)^2) / (n - 1)
# x is the predictor variable, n is the number of values, and xbar is the sample mean of the predictor

# replacing the data with the log, square root, or inverse may help remove the skew
# after a log transformation we may see the distribution change from a skewed distribution to a rouguhly symmetric distribution

# there are statistical methods that can be used to empirically identitfy the appropriate transformation
# Box and Cox propose a family of transformations that are indexed by a parameter lambda
        # x* = x^lambda - 1 / lambda if lambda != 0
        # log(x) if lambda = 0
# in addition to the log transformation this family  can identify square (lambda = 2)
# the square root transformation (lambda = .5) and inverse transformations (lambda = -1)
# we can use the training data set to estimate lambda and therefore apply the "correct" transformation to that data
# this procedure would be applied indpeendently  to each predictor data than contain values greater than zero

# in our example 69 predictors were not transformed due to zero or negative values
# the remaining 44 predictors had parameter value estimates between -2 and 2
# for each of these values the correct recommendation of transformation was applied
# transformation value of .1 = log transformation
# transformation value of -1.1 so the inverse transformation was applied

## DAta Transformations for Multiple Predictors
# these transformation discussed below act on groups of predictors typically the entire set under consideration
# of primary importance are methods to resolve outliers and reduce the dimension of the data

## Transformation to Resolve Outliers
# outliers are samples that are expectionally far from the mainstream of the data
# under certain assumptions there are formal statistical definiitons of an outlier
# even with a through understanding of the data outliers can be hard to define
# we can often identify an unusal value by looking at the data with a graph
# we we think we found an outlier - we need to figure out if that outlier is possible for the set or if an error of entry happened
# we should not hastily remove or change values especially if the sample size is small
# with small sample sizes apparents outliers might be a result of a skewed distribution where there are not yet enough data to 'see' the skewness
# also the outlying data may be a part of a special part of the population under study that we are just starting to sample
# depending on how that data were collected a 'cluster' of valid points that reside outside the mainstream of the data might belong to a different population that the other samples

# there are several predictive models that are resistant to outliers
# tree based classification models create splits of the training data  and the predictioni equation is set to logical statements
# this makes it so the outliers do not have influence over the model
# also support vector machines for classification generally disregar a portion of the training set samples  when creating a prediction equation

# one data transformation we can use to reduce the effect of outliers is the spatial sign transformation
# this trasnformation is applied to the entire set of predictors
# each sample is divided by its squared norm
        # spatial sign = x*ij = (xij / (sqrt(sumof(xij^2))))
# centering and scaling is not needed before making this transformation
# removing predictor variables after applying the spatial sign transformation may be problematic - spatial sign is calculated on the group of predictors!



## Data Reduction and Feature Extraction
# data reduction techniques are another class of predictor transformations
# these methods reduce the data by generating a smaller set of predictors that seek to capture a majority of the information of the original variables
# in this way, fewer variables can be used that provide reasonable fidelity to the original data
# for most data reduction techniques the new predictors are functions of the original predictors
# therefore - all the original predictors are still needed to create these surrogate variables
# the class of methods is often called signal extraction or feature extraction techniques

# PCA is a commonly used data reduction technique
# this method seeks to find linear combinations of the predictors known as principal components
# these principal components aim to capture the most possible variance
# the first PC is defined as the linear combination of the predictors that captures the most variability of all possible linear combinations
# the the subsequent PCs are derived such that these linear combinations caputre the most remaining variability and they are uncorrelated with the previous PCs!
# we can write the PC mathmatically
# PCj = (aj1 * Predictor1) + (aj2 * Predictor2) + ... + (ajP * PredictorP)
# p is the number of predictors
# the coefficents aj1, aj2, ... ajP are called component weights and help us understand which predictors are most important to each PC

# we will use an example to illustrate PCA
# we have a set of two correlated predictors: pixel intensity and intesity values and a categorical response
# given the high correlation between these two (.93) we could infer that these values model redundant information about the true response
# we could use either predictor or a linear combination of the predictors in place of the original predictors
# in this example two PCAs can be derived
# this represents a rotation of the data about the axis of greatest variation
# the first PC summarizes 97% of the original variability while the second summarizes 3%
# it is reasonable to use only the first PC for modeling since it accounts for the majority of information in the data

# the primary advantage of PCA is that it creates components that are uncorrelated
# some predictive models prefer predictors to be uncorrelated in order to find solutions to improve the model's numeric stability
# PCA pre-processing creates new predictors with desirable characteristics for these kinds of models

# PCA must be used with understanding and care
# modelers must understand that PCA seeks predictor-set variation without regard to any further understanding of the predictors (measurement scales / distribution)
# or to knowledge of the modeling objectives (repsonse variable)
# without proper guidance, PCA can generate components that summarizes characteristics of the data that are irrelevant to the underlying structure of the data and the ultimate modeling objective

# because PCA seeks linear combinations of predictors that maximize variability - it will be naturally drawn to summarizing predictors that have more variation
# if the original predictors are on measurement scales that differ in magnitude, then the first few components will focus on summarizingthe higher magnitude predictors
# while the latter components will summarize lower variance predictors
# high variance = income, low variance = height
# this means that PC weights will be larger for the higher variability predictors on the first few components
# it also means that PCA will be focusing on identifying the data structure based on measurement scales rather than based on the important relationships of the current data

# for most dataset predictors are on different scales
# in addition predictors may have skewed distributions
# to help PCA avoid summarizing distributional differences and predictor scale information it is best to first transform skewed predictors
# we should transform skewed predictors then center and scale the predictors prior to performing PCA
# centering and scaling enables PCA to find the underlying relationships in the data without being influenced by the measurement scales

# the second caveat of PCA is that is does not consider the modeling objective or response variable when summarizing variability
# PCA is blind to the response - it is an UNSUPERVISED TECHNIQUE
# if the predictive relationship between the predictors and response is not connected to the predictors' variability...
# then the derived PCA will not provide a suitable relationship with the response
# in this case a SUPERVISED TECHNIQUE, like Partial Least Squares will derive components while simultaneously considering the corresponding response

# once we have decided on the appropriate transformations of the predictor variables we can apply PCA
# for datasets with many predictors we must decide how many components to retain
# a hueristic approach for determining the number of components to retain is to create a skree plot -
# this shows the number of PCs and the scaled variability
# we should pick the number of PCs based right when the scaled variability starts to tail off
# the component number prior to the tapering off of scaled variance is the maximal component that we should choose
# we can also select the optimal PCs being used through cross validation

# visually examining the PCs is a critical step for assessing data quality and gaining intuition for the problem
# to do this we can plot the first few PCs against each other 
# this can show separation between classes
# this can sent the expectations of the modeler
# if there is little clustering of the classes, the PC plot should show lots of overlap between classes
# plotting the components need to have matching scales - PCs scale tend to become smaller has they account for less variation


# example:
# PCA was applied to the entire set of segmentation data predictors
# there were some predictors with significant skewness
# since skewed predictors can have an impact on PCA, there were 44 variables that were transformed using Box Cox procedure
# after these transformations all predictors were centered and scaled prior to conducting PCA

# the first three components accounted for 14 %, 12.6% and 9.4% of the total variance
# after four components there is a sharp decline in the percentage of variation being explained
# although these components only describe about 42% of the information in the data set
# from a plot of the first three PCs...
# there appears to be separation between the classes when plotting the first and second components
# however, the distribution of the well-segmented cells is roughly contained within the distribution of the poorly identified cells
# one conclusion is that the cell types are not easily separated
# this does not mean that other models, especially models that can take in highly non-linear relationships, cannot find this separation
# we also see no blatant outliers

# another exploratory use of PCA is characterizing which predictors are associated with each component
# recall that each component is a linear combination of the predictors and the coefficient for each predictor is called the loading or component weight
# loadings close to zero indicate that the predictor variable did not contribute much to that PC
# we can examine each components and understand which variables contributed the most to that particular PC
# in our example...
# cell body characteristics  have the largest effect on PC1 (check the loadings) and therefore the predictor values
# a majority of the loadings for the third channel are close to zero and have no effect on PC!
# even though the cell body measurements account for more variation in the data...
# this does not mean that cell body measurement  will be associated with predicting the outcome!!

## Dealing with Missing Values
# in many cases some predictors have no values for a given sample
# these missing data could be structurally misses - i.e. the number of children a man has given birth to
# in other cases the value cannot or was not determined or recorded at the time of the model building
# it is important to understand why the values are missing

# first it is important to understand if the pattern of missing data is related to the outcome
# this is called informative missingness since missing data has a pattern that is instructional on its own
# informative missingness can induce significant bias in the model
# example: modeling drug effects
# if drug is so bad it causes patients to be sick or die - we would not have subseqenut visits and informatino if they dropped out the study (they don't show we can't know why)
# in this case there is clearly a relationship between the probability of missing values and the treatment
# customer ratings can often have informative missingness - people are more compelled to rate products when they have strong opinions
# in this case the data are more likely to be polarized by having a few values in the middle of the rating scale
# you either love it or hate it if you rate it phenom

# missing data should not be confused with censored data where the exact value is missing but something is known about its value
# if a customer has not yet returned a movie we do not know the actual time span only that is is as least as long as the current duration

# are censored data treated differently that missing data?
# when building traditional models focused on interpretation or inference - the censoring is usually taken into account in a formal manner by making assumptions about the censoring mechanism
# for predictove models, it more common to treat these data as simple missing data or use the censored value as the observated value
# for example:
# when a sample has a value below the limit of detection the actual limit can be used in place of the real value
# for this situation it is also common to use random number between  zero and the limit of the detection

# in our experience missing values are more often related to predictor variables than the sample
# because of this - the amount of missing data may be concentrated in a subset of predictors rather than occuring randomly across all the predictors
# in some cases the percentage of missing data is substantial enough to remove this predictor from subsequent modeling activiites
# There are cases where the missing values might be concentrated in specific samples
# for large datasets, removal of samples based on missing values is not really a problem - assuming that the missingness is not informative
# in smaller datasets there is a steep price in removing samples


# if we do not remove the missing data there are two general approaches
# a few predictive models especially tree based techniques can specifically account for missing data
# alternatively, missing data can be imputed
# in this case we can use information in the training set predictors to estimates the values of the other predictors
# this amounts to a predictive model within a predictive model (we are predicting what the missing values would be)

# imputation has been extensively studied in stats
# in model building - we are concerned with the accuracy of the model rather than the inference
# we want our missing data to be given the best value possible

# imputation is just another layer of modeling we can try to estimate the values of the predictors based on other predictor varaiables
# the most relevant scheme for accomplishing this is to use the training set to build an imputation model for each predictor in the data set
# prior to model training or the prediction of new samples, missing values are filled with the imputation model
# note that this extra layer adds uncertainty
# if we are using resampling to select tuning parameter values or to estimate performance the imputation should be incorporated within the resampling
# this will increase the computational time for building models but will also provide honest estimates of model performance

# the number of predictors affected by missing values is small an exploratory analysis of the relationships between predictors is a good idea
# for example, visualizations and methods such as PCA can be used to determine if there are strong relationships between the predictors
# if a variable with missing values is highly correlated with another predictor that has few missing values a focused model can often be effective for imputation

# one popular technique for imputation is a K nearest neighbor model
# a new sample is imputed by finding the samples in the training set closest to it and averages these nearby points to fill in the missing value
# one advantage of this approach is that the imputed data are confimed to be within the range of the training set values
# one disadvantage is that the entire training set is required every time a missing value needs to be imputed
# also the number of neighbors is a tuning parameter as is the method of calculating "closeness" between a set of points

# alternatively a simpler approach can be used to imputate
# we can scan for other predictors in the model that are highly correlated with the predictor that as missing data
# we can create a simple linear regression model using the highly correlated variable to predict the missing values!!


## Removing Predictors
# there are potential advantages to removing predictors prior to modeling
# first fewer predictors means decreased computational time and complexity
# second if two predictors are highly correlated this implies that they are redundant - the carry the same information into the model
# removing on of the highly correlated predictors should not compromise the accuracy of the model too much and we may get a more parsimonious model
# third, something models can be crippled by predictors with degenerate distributions
# in these cases there can be a significant improvement in model performance and or stability without the problematic variables

# consider a predictor variable that has a single unique value - this is a zero variance predictor
# for some models an non-informative variable may have little effect on the calculations
# a tree based model is impervious to this type of predictor since if would never be used in a split
# a model such as linear regression would find these data problematic and likely to cause an error in the computations
# in either case these data have no information and can easily be discarded
# similiarly  some predictors might only have a handful of unique values that occur with very low frequencies
# these are the near-zero variance predictors may have a single value for the vast majority of the samples


# how can the user diagnose this problematic data?
# the number of unique points in the data must be small relative to the number of samples
# a small percentage of unique values is in itself not a cause for conern as many dummy variables generated form categorical predictors would fit this description
# the problem occurs when the frequency of these unique values is severly disproportionate
# the ratio of the most common frequency to the second most common reflects and imbalance in the frequency

# a rule of thumb for detecting near zero variables is:
# the fraction of unique values over the sample size is low (around 10%)
# the ratio of the frequency of the most prevalent value to the frequency of the second most prevelanet value is large ( around 20)
# in both of these criteria are true and the model in question is susceptible to this type of predictor it may be advantageous to remove the variable from the model

## Between Preidctor Correlations
# Collinearity is the technical term for the situation where a pair of predictors variables have substantial correlation with each other
# it is also possible to have relationships between multiple predictors at once - multicollinearity
# for example:
# a dataset may have a number of predictors that reflect the size of some aspect
# measurements like perimeter, width, length may all try to explain the size of a data
# to examine colinearity we investigate a correlation matrix
# each pairwise correlation is computed using our training data and can be colored to reflect the magnitutide of that correlation
# in our example dark red is strong negtive correlations and dark blue is strong positive correlation
# scanning down the diagnol of a correlation matrix we can see blocks of strong postitive correlations indicating "blocks" of collinearity
# we see that the "size" variables are grouped together and all exhibit positive correlation between each other

# when the dataset consists of too many predictors to examine visually techniques such as PCA can be used to characterize the magnitude of the colliearity problem
# for example if our first principal component accounts for a large percentage of variance this implies that there is at least one group of predictors that represent the same information
# for example say the first 3-4 components in a PCA have relative contributions to the total variance
# this would indicate that there are at least 3-4 significant relationships between predictors
# the PCA loadings (coefficient weights on each variable within a PC) can be used to understand which predictors are associated with each component

# in general there are good reasons to avoid data that are highly correlated with each other
# first redundant predictors frequenctly add more complexity to the model than information they provide to the model
# there are also mathematical disadvantages to having correlated predictor data
# using highly correlated predictors in techniques like linear regression can result in highly unstable models, errors, and degraded predictive performance

# classical regression analysis has several tools to diagnose multicollinearity in linear regression
# since colliear predictors can impact the variance of parameter estimates in this model we can use Variance Inflation Factor (VIF)
# the VIF can locate variables where that are suffering from collieararity
# a heuristic approach to dealing with this issue is to remove the minimum number of predictors to ensure that all pairwise correlations are below a certain threshold
# here is the algorithm:
        # calculate the correlation matrix
        # determine the two predictors associated with the largest absolute pairwise correlation (call them predictors A and B)
        # determine the average correlation between A and all other variables
        # determine the average correlation between B and all other variables
        # if A has a larger average correlation, remove it, otherwise remove predictor B
        # repeat steps 2-4 for all other variables in the model - this will result in no absolute correlations are above the threshold level
# the idea is to first remove predictors that have the most correlated relationships

# suppose we wanted to use a model that is particularly sensistive to between predictor correlations we might apply a threshold of .75
# this means that we want to eliminate the minimum number of predictors to achieve all pairwise correlations less than .75
# in our example we actually remove 43 predictors

# as previously mentioned dimension reduction models are another techinque for mitigating the effect of strong correlations between predictors
# these techniques make the connection between predictors and repsonse more complex
# since signal extraction methods are usually unsupervised there is no guarantee that the resulting surrogate predictors have any relationship with the outcome!



## Adding Predictors
# when a predictor is categorical, such as gender or race, it is common to decompose the predictor into a set of more specifc variables
# in an example we may bins certain predictor groups into groups of observations
# to use data in these models the categories are re-encoded into smaller bits of information called "dummy variables"
# each category gets its own dummy variable that is zero or one as an indicator of the group
# n-1 dummy variables are ususally created - the final variable can be inferred
# the decision to include all of the dummy variables can depend on the choice of model
# models that include include an intercept term such as simple linear regression would have numerical issues if each dummy variable was included in the model
# for each sample these variables all add up to one and this would provide the same information as the intercept
# if the model is insensitive to this type of issue using the complete set of dummy varibales would help improve interpretation of the model

# many of the models described in this text automatically generate highly complex nonlinear relationships between the predictors and the outcome
# more simplistic models do not unluess the user manually specifies which predictors should be nonlienar and in what way
# logistic regression is a classification model that generates simple linear classification boundaries
# we can mold this linear decision boundary to model more complex curves by using polynomial terms
# we can also augment the prediction data with addition of complex combinations of our data
# for classification models we can calculate the "class centriods" which are the centers of the predictor data for each class
# then for each predictor the distance to each class centroid can be calculated and these distances can be added to the model


## Binning Predictors
# while there are recommended techniquess for pre=processing data there are also methods to avoid
# one common approach to simplify a dataset is to take numeric predictor and pre-categorize it to a bin into two or more groups prior to analysis
# these includes examples like: temperature less than 36 degress - yes / no
# the perceived advantages to this appraoch are:
        # ability to make simple statements to interpret the model
        # the modeler does not know the exact relationship between the predictors and outcome
        # a higher response rate for survey questions where the choices are binned
# there are many issues with manual binning of continoues data
# the first is that there can be significant loss of performance in the model
# manually binning our predictors limits the potential of models to pick up on the true signal of the data
# we also lose precision in the predictions when they are binned
# for example: if there are only two binned predictors only four combinatoins exisit in the dataset so only simple predictions can be made
# categorizing predictors can lead to a high rate of false positives (noise predictors determined to be informative)

# unfortunately the predictive models that are most powerful are usually the least interpretable
# the bottom line is that the perceived improvement in interpretability gained by manual categorization is usually offset by a significant loss in performance
# since this book is concerned with predictive models (interpretation is not the primary goal) loss of performance should be avoided
# if a medical diagnositc is used for such important determinations patients desire the most accurate prediction possible
# as long as complex models are properly validated it may be improper to use a model that is built for interpretation rather than predictive performance

# note that the arguement here is related to the manual categorization of predictors prior to model building
# there are several models such as classification / regression trees and regression splines that estimate cut points in the process of model building
# the difference between these methodologies and manual binning is that the models use all the predictors to derive bins based on a single objective (like maximizing accuracy)
# they evaluate many variables siumultaneously and are usually based on statistically sound methodologies


## Computing
# this section uses data from the AppliedPredictiveModeling package and functions from caret, corrplot, e1071 and lattice packages
# examine the chapters directory of the APM package to see the specific code to build the models shown
# many chapters in this book contain sections at the end of the chapter that detail computation in R
# the computing sections generally explain how to generally do the computations while the code is the best source of the calculations
# there are a few useful R functions than can be used to find existing functions or classes of interest
# the function apropos will search any loaded R packages for a given term
# for example: to find the functions for creating a confusion matrix within the currently loaded packages

# load the APM packages for the examples in this chapter
install.packages('AppliedPredictiveModeling', dependencies = T)
library(AppliedPredictiveModeling); library(caret); library(e1071);library(corrplot)


## Example 1: General Computing

# search laoded packages for a term
apropos('confusion')
# [1] "confusionMatrix"       "confusionMatrix.train"

# search the R website for a term
RSiteSearch('confusion', restrict = 'functions')

# get data for segmentation example
data("segmentationOriginal")

# filter to training set for chapter examples
segData <- segmentationOriginal %>% filter(Case == 'Train') %>% 
        as_tibble()

# the class and cell field will be saved into separte vectors then removed from the main object
cellID <- segData$Cell
class <- segData$Class
case <- segData$Case

# now remove these from the original data
segData <- segData %>% 
        dplyr::select(., -Cell, -Class, -Case)

# the original data contained serveral "status" columns which were binary versions of the predictors
# to remove these we find the column names containing status and remove them
segData <- segData %>% dplyr::select(., -contains('Status'))



## Exmaple 2: Transformations
# as discuseed some features in our data show significant skewness
# we can use the skewness functino in e1071 pacage to calculate the sample skewness statistics for each predictor

# skewness for one predictor
skewness(segData$AngleCh1)
# [1] -0.02426252

# applying the skewness formula to all columns
skewness_values <- map(segData %>% dplyr::select_if(., is.numeric), skewness) %>% as_tibble()
skewness_values[1,]

# using these values as a guide that variables can be prioritized for visualizing the distribution
# the basic R function hist or the histogram function in the lattice can be used to assess the shape of the distribution
var_hist <- hist(segData$AngleCh1)

# to determine which type of transformation should be used the MASS package contains the boxcox function
# this function estimates lambda but does not create the transformed variables
# a caret function BoxCoxTrans can find the appropriate transformation and apply that transofrmation to the data
Ch1AreaTrans <- BoxCoxTrans(segData$AreaCh1)
# Box-Cox Transformation
# 
# 1009 data points used to estimate Lambda
# 
# Input data summary:
#         Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 150.0   194.0   256.0   325.1   376.0  2186.0 
# 
# Largest/Smallest: 14.6 
# Sample Skewness: 3.53 
# 
# Estimated Lambda: -0.9 

# look at the calculation - original data
head(segData$AreaCh1)
# [1] 819 431 298 256 258 358

# after transformation
predict(Ch1AreaTrans, head(segData$AreaCh1))
# [1] 1.108458 1.106383 1.104520 1.103554 1.103607 1.105523

# here is the formula for the first entry
lambda_bc <- -.9
(819 ^ (lambda_bc) - 1) / (lambda_bc)
# [1] 1.108458


# another caret function PreProcess applies this transformation to a set of predictors
# the base R function prcomp can be used for PCA - remember to center and scale the data before applying princomps

# stating pca
pca_object <- prcomp(segData, center = T, scale. = T)

# calculate the cummulative percent of variance each PC accounts for
summary(pca_object)
# Importance of components:
#                           PC1    PC2    PC3     PC4     PC5     PC6     PC7     PC8     PC9
# Standard deviation     3.4827 3.1413 2.6257 2.11538 1.69572 1.54598 1.39673 1.37877 1.28844
# Proportion of Variance 0.2091 0.1701 0.1189 0.07715 0.04958 0.04121 0.03364 0.03278 0.02862
# Cumulative Proportion  0.2091 0.3793 0.4981 0.57528 0.62485 0.66606 0.69970 0.73247 0.76110

# the transformed predictor values are contained in x of the pca_object
head(pca_object$x[,1:5])
# PC1        PC2         PC3       PC4        PC5
# [1,]  5.0985749  4.5513804 -0.03345155 -2.640339  1.2783212
# [2,] -0.2546261  1.1980326 -1.02059569 -3.731079  0.9994635
# [3,]  1.2928941 -1.8639348 -1.25110461 -2.414857 -1.4914838
# [4,] -1.4646613 -1.5658327  0.46962088 -3.388716 -0.3302324
# [5,] -0.8762771 -1.2790055 -1.33794261 -3.516794  0.3936099
# [6,] -0.8615416 -0.3286842 -0.15546723 -2.206636  1.4731658

# another sub object is the rotation
# this stores the variable loadings
# rows correspond to predictor columns and are associatedc with the components
head(pca_object$rotation[,1:3])

# PC1         PC2          PC3
# AngleCh1     0.001213758 -0.01284461  0.006816473
# AreaCh1      0.229171873  0.16061734  0.089811727
# AvgIntenCh1 -0.102708778  0.17971332  0.067696745
# AvgIntenCh2 -0.154828672  0.16376018  0.073534399
# AvgIntenCh3 -0.058042158  0.11197704 -0.185473286
# AvgIntenCh4 -0.117343465  0.21039086 -0.105060977


# spatial sign
# the caret packages class spatial sign contains functionality for the spatial sign transformation
# the basic syntax is spatialSign(SegData)
segData_spatial <- spatialSign(segData)
head(segData_spatial[,1:3])
# AngleCh1     AreaCh1  AvgIntenCh1
# 1 0.0006213207 0.003804515 0.0001482938
# 2 0.0014981921 0.006054784 0.0003938958
# 3 0.0021602840 0.009309640 0.0006078177
# 4 0.0032589770 0.007624981 0.0005608105
# 5 0.0028767454 0.007117471 0.0004847288
# 6 0.0009676856 0.004441888 0.0005246344

# missing values
# we have no missing values in our data but we can use the impute package
# we can use impute.knn to use knn to esitmate the nearest data
# we can nest this method in the preProcess statement to apply the imputatin to our dataset

# to administer a series of transformations to multiple datasets we can use PreProcess
# this has a ability to transform, center, scale or impute values as well as spatial sign and feature extraction
# after calling the PreProcess function the predict method applies the results to a set of data
# here is the Box Cox transform, center, and scale the data then execute PCA in a PreProcess statement
trans <- preProcess(x = segData, method = c('center', 'scale', 'pca'))
# Created from 1009 samples and 58 variables
# 
# Pre-processing:
#         - centered (58)
# - ignored (0)
# - principal component signal extraction (58)
# - scaled (58)
# 
# PCA needed 22 components to capture 95 percent of the variance

# apply the transformations to the data!
transformed <- predict(trans, segData)
head(transformed[,1:3])
# PC1        PC2         PC3
# 1  5.0985749  4.5513804 -0.03345155
# 2 -0.2546261  1.1980326 -1.02059569
# 3  1.2928941 -1.8639348 -1.25110461
# 4 -1.4646613 -1.5658327  0.46962088
# 5 -0.8762771 -1.2790055 -1.33794261
# 6 -0.8615416 -0.3286842 -0.15546723


# the order in which the possible transformation are applied are:
# transformation, centering, scaling, imputation, feature extraction, then spatial sign
# many of the modeling functions have options to center and scale prior to modeling
# for example when using the train function there is an option to use preProcess prior to modeling within the resampling iteratations


## Filtering
# to filter near zero variance predictors the caret packages has nearZeroVar
# this will return the column numbers of any predictors that fulfill the conditions to have near zeroVar
nearZeroVar(segData)
# integer(0)

# if there are predictors to be removed we need to subset out the columns based on the nearZeroVar function call

# to filter for between-predictor correlations the cor function can calculate the correlations between predictor values
correlations <- cor(segData)
dim(correlations)
# [1] 58 58
correlations[1:4, 1:4]
# AngleCh1      AreaCh1 AvgIntenCh1 AvgIntenCh2
# AngleCh1     1.000000000 -0.002627172 -0.04300776 -0.01944681
# AreaCh1     -0.002627172  1.000000000 -0.02529739 -0.15330301
# AvgIntenCh1 -0.043007757 -0.025297394  1.00000000  0.52521711
# AvgIntenCh2 -0.019446810 -0.153303007  0.52521711  1.00000000

# to visually examine the structure of the data we can use the corrplot function and package
# input needs to be a correlation matrix
corrplot(correlations, order = 'hclust')

# to filter based on correlations we can use the FindCorrelations function
# for a given threshold of pairwise correlations the function returns column numbers to delete predictors
highCorr <- findCorrelation(correlations, cutoff = .7)
length(highCorr)
# [1] 33
head(highCorr)
# [1] 23 40 43 36  7 15

# filter out the high correlation predictors
filtered_segData <- segData[,-highCorr]
head(filtered_segData)
# # A tibble: 6 x 25
# AngleCh1 ConvexHullPerimRatioC… FiberAlign2Ch3 FiberAlign2Ch4 FiberWidthCh1 IntenCoocASMCh3
# <dbl>                  <dbl>          <dbl>          <dbl>         <dbl>           <dbl>
# 1    134                    0.797          0.488          0.352         13.2          0.0281 
# 2    107                    0.935          0.301          0.522         21.1          0.00686
# 3     69.2                  0.866          0.220          0.733          7.40         0.0310 
# 4    109                    0.920          0.364          0.481         12.1          0.108  
# 5    104                    0.931          0.359          0.244         10.2          0.0130 
# 6     78.0                  0.961          0.479          0.467         14.6          0.0250 
# # ... with 19 more variables: IntenCoocASMCh4 <dbl>, IntenCoocContrastCh3 <dbl>,
# #   IntenCoocContrastCh4 <dbl>, KurtIntenCh1 <dbl>, KurtIntenCh3 <dbl>, KurtIntenCh4 <dbl>,
# #   NeighborAvgDistCh1 <dbl>, NeighborMinDistCh1 <dbl>, ShapeBFRCh1 <dbl>, ShapeLWRCh1 <dbl>,
# #   SpotFiberCountCh3 <int>, SpotFiberCountCh4 <int>, TotalIntenCh2 <int>, VarIntenCh1 <dbl>,
# #   VarIntenCh3 <dbl>, VarIntenCh4 <dbl>, WidthCh1 <dbl>, XCentroid <int>, YCentroid <int>





## Creating Dummy Variables
# several methods exist for creating dummy varibales based on a particular model
# we can either select a subset of predictors to create dummy variables or apply to all variables
# for tree based models we should put all predictors into dummy variables
# this example will work through the cars dataset - we aim to predict the price of a car based on dimensions
data(cars)
head(cars)
# Price Mileage Cylinder Doors Cruise Sound Leather Buick Cadillac Chevy Pontiac Saab
# 1 22661.05   20105        6     4      1     0       0     1        0     0       0    0
# 2 21725.01   13457        6     2      1     1       0     0        0     1       0    0
# 3 29142.71   31655        4     2      1     1       1     0        0     0       0    1
# 4 30731.94   22479        4     2      1     0       0     0        0     0       0    1
# 5 33358.77   17590        4     2      1     1       1     0        0     0       0    1
# 6 30315.17   23635        4     2      1     0       0     0        0     0       0    1
# Saturn convertible coupe hatchback sedan wagon
# 1      0           0     0         0     1     0
# 2      0           0     1         0     0     0
# 3      0           1     0         0     0     0
# 4      0           1     0         0     0     0
# 5      0           1     0         0     0     0
# 6      0           1     0         0     0     0

# to model the price as a function of mileage and type of car we can use the function dummyVars
# our data is already dummy variabled
type <- c('convertible', 'coupe', 'hatchback', 'sedan', 'wagon')
cars$Type <- factor(apply(cars[, 14:18], 1, function(x) type[which(x == 1)]))

# cars subset data
cars_sub <- cars %>% dplyr::select(Price, Mileage, Type)

# check levels
levels(cars_sub$Type)
# [1] "convertible" "coupe"       "hatchback"   "sedan"       "wagon" 

# we can use dummy variables to encode the type predictor
simpleMod <- dummyVars(~Mileage + Type, data = cars_sub, levelsOnly = T)
# Dummy Variable Object
# 
# Formula: ~Mileage + Type
# 2 variables, 1 factors
# Factor variable names will be removed
# A less than full rank encoding is used

# to generate the dummy variables for the training set or any new samples the predict method is used with dummyVars
predict(simpleMod, head(cars_sub))
# Mileage convertible coupe hatchback sedan wagon
# 1   20105           0     0         0     1     0
# 2   13457           0     1         0     0     0
# 3   31655           1     0         0     0     0
# 4   22479           1     0         0     0     0
# 5   17590           1     0         0     0     0
# 6   23635           1     0         0     0     0

# we see that the type variable was expanded into five variables for five factor levels
# the model is simple because it assumes there is not interation between type and mileage
# to fit a more advanced model we could assume that there is a joint effect of mileage and car type
# this type of effect is called an interaction
# this will add another five predictors to our data frame
withInteraction <- dummyVars(~Mileage + Type + Mileage:Type,
                             data = cars_sub, levelsOnly = T)


# apply the interaction dummy variables to the actual data using predict
predict(withInteraction, head(cars_sub))
# Mileage convertible coupe hatchback sedan wagon Mileage:convertible Mileage:coupe
# 1   20105           0     0         0     1     0                   0             0
# 2   13457           0     1         0     0     0                   0         13457
# 3   31655           1     0         0     0     0               31655             0
# 4   22479           1     0         0     0     0               22479             0
# 5   17590           1     0         0     0     0               17590             0
# 6   23635           1     0         0     0     0               23635             0
# Mileage:hatchback Mileage:sedan Mileage:wagon
# 1                 0         20105             0
# 2                 0             0             0
# 3                 0             0             0
# 4                 0             0             0
# 5                 0             0             0
# 6                 0             0             0




## Exercises

## 3.1 UCI Glass Identification
# the data consists of 214 glass samples labeled as one of seven categories
# there are nine predictors 
# access the data:
library(mlbench)
data(Glass)
str(Glass)
# 'data.frame':	214 obs. of  10 variables:
# $ RI  : num  1.52 1.52 1.52 1.52 1.52 ...
# $ Na  : num  13.6 13.9 13.5 13.2 13.3 ...
# $ Mg  : num  4.49 3.6 3.55 3.69 3.62 3.61 3.6 3.61 3.58 3.6 ...
# $ Al  : num  1.1 1.36 1.54 1.29 1.24 1.62 1.14 1.05 1.37 1.36 ...
# $ Si  : num  71.8 72.7 73 72.6 73.1 ...
# $ K   : num  0.06 0.48 0.39 0.57 0.55 0.64 0.58 0.57 0.56 0.57 ...
# $ Ca  : num  8.75 7.83 7.78 8.22 8.07 8.07 8.17 8.24 8.3 8.4 ...
# $ Ba  : num  0 0 0 0 0 0 0 0 0 0 ...
# $ Fe  : num  0 0 0 0 0 0.26 0 0 0 0.11 ...
# $ Type: Factor w/ 6 levels "1","2","3","5",..: 1 1 1 1 1 1 1 1 1 1 ...

# using visualizations explore the predictor variables and understand thier distributions as well as relationships between predictors

# subset data for analysis
glass_labels <- Glass$Type
glass_p <- Glass %>% dplyr::select(., -Type)

# examine correlations and plot
glass_cor <- cor(glass_p)
corrplot(glass_cor, order = 'hclust')

# view distributions of highly correlated variables
hist(glass_p$K)
hist(glass_p$Al)

# find the correlated variables
high_corr <- findCorrelation(glass_cor, cutoff = .75)
glass_p %>% dplyr::select(7) %>% names(.)
# [1] "Ca"

# look at distribution of the removed variable
hist(glass_p$Ca)

# subset variables to remove Ca
glass_p <- glass_p %>% dplyr::select(., -7)

# view skewness
map(glass_p, skewness) %>% as_tibble()
# # A tibble: 1 x 8
# RI    Na    Mg    Al     Si     K    Ba    Fe
# <dbl> <dbl> <dbl> <dbl>  <dbl> <dbl> <dbl> <dbl>
# 1  1.60 0.448 -1.14 0.895 -0.720  6.46  3.37  1.73

# investigate the highest skewness values
hist(glass_p$K)
hist(glass_p$Ba)

# scale for outliers - look at columns with high variance
map(glass_p, var)

# plot the high variance columns
plot(glass_p$Al)
hist(glass_p$Al)
sort(glass_p$Al)

# outliers values
library(outliers)
a <- map(glass_p, mean) %>% as_tibble()
b <- map(glass_p, outlier) %>% as_tibble()
rbind(a, b)
# # A tibble: 2 x 8
# RI    Na    Mg    Al    Si     K    Ba     Fe
# <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>  <dbl>
# 1  1.52  13.4  2.68  1.44  72.7 0.497 0.175 0.0570
# 2  1.53  17.4  0     3.50  69.8 6.21  3.15  0.510


# what transformations should we apply if any?
trans <- preProcess(glass_p, method = c('BoxCox'))
# Created from 214 samples and 4 variables
# 
# Pre-processing:
#         - Box-Cox transformation (4)
# - ignored (0)
# 
# Lambda estimates for Box-Cox transformation:
#         -2, -0.1, 0.5, 2

glass_p <- predict(trans, glass_p)

# check for nearzero
near_zero <- nearZeroVar(glass_p)
# integer(0)

center_scale <- preProcess(glass_p, method = c('center', 'scale'))
# Created from 214 samples and 8 variables
# 
# Pre-processing:
#         - centered (8)
# - ignored (0)
# - scaled (8)

glass_p <- predict(center_scale, glass_p)

# final glass_p predictors ready for modeling
str(glass_p %>% as_tibble())
# Classes ‘tbl_df’, ‘tbl’ and 'data.frame':	214 obs. of  8 variables:
# $ RI: num  0.876 -0.247 -0.722 -0.231 -0.31 ...
# $ Na: num  0.313 0.613 0.18 -0.215 -0.14 ...
# $ Mg: num  1.252 0.635 0.6 0.697 0.649 ...
# $ Al: num  -0.6552 -0.0873 0.2745 -0.2344 -0.3419 ...
# $ Si: num  -1.1273 0.0972 0.4351 -0.0584 0.5524 ...
# $ K : num  -0.6701 -0.0262 -0.1641 0.1118 0.0812 ...
# $ Ba: num  -0.352 -0.352 -0.352 -0.352 -0.352 ...
# $ Fe: num  -0.585 -0.585 -0.585 -0.585 -0.585 ...








## 3.2 Soybeans UCI
# data collected to predict disease in soybeans
# the 35 predictors are mostly categorical and include information on enviroment conditions
# the labels consist of 19 distinct classes
# load the data
data("Soybean")
str(Soybean)
# 'data.frame':	683 obs. of  36 variables:
# $ Class          : Factor w/ 19 levels "2-4-d-injury",..: 11 11 11 11 11 11 11 11 11 11 ...
# $ date           : Factor w/ 7 levels "0","1","2","3",..: 7 5 4 4 7 6 6 5 7 5 ...
# $ plant.stand    : Ord.factor w/ 2 levels "0"<"1": 1 1 1 1 1 1 1 1 1 1 ...
# $ precip         : Ord.factor w/ 3 levels "0"<"1"<"2": 3 3 3 3 3 3 3 3 3 3 ...
# $ temp           : Ord.factor w/ 3 levels "0"<"1"<"2": 2 2 2 2 2 2 2 2 2 2 ...
# $ hail           : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 2 1 1 ...
# $ crop.hist      : Factor w/ 4 levels "0","1","2","3": 2 3 2 2 3 4 3 2 4 3 ...
# $ area.dam       : Factor w/ 4 levels "0","1","2","3": 2 1 1 1 1 1 1 1 1 1 ...
# $ sever          : Factor w/ 3 levels "0","1","2": 2 3 3 3 2 2 2 2 2 3 ...
# $ seed.tmt       : Factor w/ 3 levels "0","1","2": 1 2 2 1 1 1 2 1 2 1 ...
# $ germ           : Ord.factor w/ 3 levels "0"<"1"<"2": 1 2 3 2 3 2 1 3 2 3 ...
# $ plant.growth   : Factor w/ 2 levels "0","1": 2 2 2 2 2 2 2 2 2 2 ...
# $ leaves         : Factor w/ 2 levels "0","1": 2 2 2 2 2 2 2 2 2 2 ...
# $ leaf.halo      : Factor w/ 3 levels "0","1","2": 1 1 1 1 1 1 1 1 1 1 ...
# $ leaf.marg      : Factor w/ 3 levels "0","1","2": 3 3 3 3 3 3 3 3 3 3 ...
# $ leaf.size      : Ord.factor w/ 3 levels "0"<"1"<"2": 3 3 3 3 3 3 3 3 3 3 ...
# $ leaf.shread    : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
# $ leaf.malf      : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
# $ leaf.mild      : Factor w/ 3 levels "0","1","2": 1 1 1 1 1 1 1 1 1 1 ...
# $ stem           : Factor w/ 2 levels "0","1": 2 2 2 2 2 2 2 2 2 2 ...
# $ lodging        : Factor w/ 2 levels "0","1": 2 1 1 1 1 1 2 1 1 1 ...
# $ stem.cankers   : Factor w/ 4 levels "0","1","2","3": 4 4 4 4 4 4 4 4 4 4 ...
# $ canker.lesion  : Factor w/ 4 levels "0","1","2","3": 2 2 1 1 2 1 2 2 2 2 ...
# $ fruiting.bodies: Factor w/ 2 levels "0","1": 2 2 2 2 2 2 2 2 2 2 ...
# $ ext.decay      : Factor w/ 3 levels "0","1","2": 2 2 2 2 2 2 2 2 2 2 ...
# $ mycelium       : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
# $ int.discolor   : Factor w/ 3 levels "0","1","2": 1 1 1 1 1 1 1 1 1 1 ...
# $ sclerotia      : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
# $ fruit.pods     : Factor w/ 4 levels "0","1","2","3": 1 1 1 1 1 1 1 1 1 1 ...
# $ fruit.spots    : Factor w/ 4 levels "0","1","2","4": 4 4 4 4 4 4 4 4 4 4 ...
# $ seed           : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
# $ mold.growth    : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
# $ seed.discolor  : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
# $ seed.size      : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
# $ shriveling     : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
# $ roots          : Factor w/ 3 levels "0","1","2": 1 1 1 1 1 1 1 1 1 1 ...


# investigate the frequency distributions for the categorial predictors
freq_list <- map_if(Soybean, is.factor, table) %>% as.list()

sapply(freq_list, barplot)

# investigate missing data
is_missing <- purrr::map_df(Soybean, is.na)
percent_missing <- purrr::map_df(is_missing, mean) %>% gather(.) %>% arrange(-value)

# develop a strategy for handling the missing values
library(mice)

soybean_number <- Soybean %>% 
        mutate_if(is.factor, as.numeric)


knn_impute <- preProcess(soybean_number, method = "knnImpute")

soybean_number <- predict(knn_impute, soybean_number)

purrr::map_df(soybean_number, function(x) mean(is.na(x)))



## 3.3 QSAR modeling
# where characteristics of a chemical compound are used to predict other properties
# use the QSAR dataset within caret
# load the data
data(BloodBrain)
str(bbbDescr)

# are there any degenerate distributions?
skew_checks <- map_df(bbbDescr, skewness) %>% gather() %>% arrange(value)

# degnerate distributions
hist(bbbDescr$negative)
hist(bbbDescr$wnsa2)

trans <- preProcess(bbbDescr, method = c('BoxCox'))
# Created from 208 samples and 40 variables
# 
# Pre-processing:
# - Box-Cox transformation (40)
# - ignored (0)
# 
# Lambda estimates for Box-Cox transformation:
#         Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -2.000  -0.150   0.150   0.050   0.425   2.000 

# transforming the degenerate distributions
bbbdesc <- predict(trans, bbbDescr)

# are there any strong relationships with predictor data?
bbb_cor <- cor(bbbDescr)
corrplot(bbb_cor, order = 'hclust', type = 'upper', tl.cex = .3, tl.pos = 'td', diag = F, cl.pos = 'r')

corrs <- findCorrelation(bbbdesc)

bbbdesc <- bbbdesc[,-corrs]

pca_cols <- preProcess(bbbdesc, method = c('center', 'scale', 'pca'))
# Created from 208 samples and 19 variables
# 
# Pre-processing:
# - centered (19)
# - ignored (0)
# - principal component signal extraction (19)
# - scaled (19)
# 
# PCA needed 12 components to capture 95 percent of the variance

pca_bbb <- predict(pca_cols, bbbdesc)




# chapter 4 overfitting and model tuning ------------------------------------------------------

# many modern classification and regression models are highly adaptable
# they are capable of modeling complex relationships
# however they can vastly overemphasize patterns that are not reproducible
# without a methodological approach to evaluting models the modeler will not know about the problem until prediction happens

# overfitting has been discussed in many fields
# overfitting is a concern for any predictive model regardless of field of research
# this chapter aims to explain and illustrate key principles of laying a foundation onto which trustworthy models can be built
# we will describe strategies that enable us to have confidence that the model we build will predict new samples with a similiar degree of accuarcy
# without this confidence the model's predictions are useless

# all model building efforts are constrained by the existing data
# for many problems data may have a limited number of samples, may be bad quality, or may be unrepresentative of new samples
# working under these assumptions of "good data" and "good representation" we must use the data at hand to build a model
# almost all predictive modeling techniques have tuning parameters that enable the model to flex to find stucture in the data
# we must use the existing data to identify settings for the model's parameters that yeild the best and most rellistic performance
# this has been achieved by splitting the existing data into training and test sets
# the trianing set is used to build and tune the model and the test set is used to estiamte the model's performance

# modern approaches to model building split the data into multiple training and testing sets
# which have been shown to often find the more optimal tuning parameters and give a more accurate representation of the model's performance
# to avoid overfitting we propose a general model building approach that includes parameter tuning and model evaluation
# the ultimate goal is to find the reproducible structure in the data
# this approach entails splitting existing data into distinct sets for the purposes of tuning model parameters and evaluating performance
# the choice of splitting the data depends on characterisitics of the existing data such as its size a structure

## The Problem of Overfitting
# there now exist many techinques that can learn the strucutre of a set of data so well that when the model is applied to the data it was built on...
# it correctly predicts every sample
# in additon to learning the general patterns of the data, the model has also learned the characterisitics of each sample's unqiue noise
# this type of model is said to be over-fit and will usually have poor accuracy when predicting the new sample
# to illustrate over-fitting we will consider a classificaiton example
# there is a significant overlap between classes which is the case for most classification problems

# one objective for this data set would be to develop a model to classify new samples
# in this example we can represent classification rules by boundary lines
# we can have a flexible and a simply decision boundary line
# in the complex model - we can  draw circles around the classifiers to accurately model on the training dataset
# the simple model is determined by a simple line or connected line
# the complex model will look like it has great accuracy - but it will likely be overoptimistic
# the simple model could have some limits to the bias is reduces but it may generalize better
# estimating the utility of a model by re-predicting on the training set is apparent performance of the model
# in two dimensions it is not difficult to determine that the more flexible model is overfitting

## Model Tuning
# many models have important parameters which cannot be directly estimated from the data
# for example the K nearest neighbor classificaiton method a new sample is predicted based on its k closest neighbors ( by some distance metric)
# the quesiton is how many neighbors should we use - we cannot get this directly from our data
# a choice of too few neighbors would likely leave bias, too many neighbors would likely cause overfitting
# this type of model parameter is referred to as a tuning parameter becuase there is no analytical formula available to calculate the value

# several models discussed in this text have at least one tuning parameter
# since many of these parameters control the complexity of the model, poor choises for the values can result in over-fitting
# take SVM for example
# one of the tuning parameters in the svm algorithm is referred to as the cost parameter
# when the cost is large, the model will go to great lengths  to correctly label every point
# smaller values produce models that are not as aggressive

# once candidate set of paramters have been selected then we must obtain trustworthy estimates of model performance
# the performance on the hold out samples is aggregated into a performance profile
# this can be used to determine the final tuning parameters
# we might choose odd number Ks in a KNN method and model on the training set for each value!
# this will help us select the optimal case to then predict on to the final validation set!

# there are more complex ways to tune parameters such as genetic algorithms, and simplex search methods
# these procedures algorithmically determine the appropriate values for tuning pararmeters
# they will iterate until they land at the parameter settings with optimal performance
# they will evaluate a large number of candidate models and can be superior to a defined set of tuning parrameters

# a more difficult problem is obtaining trustworthy estimates of model performance for these candidate models
# the apparent error rate (training error rate) can produce extremely optimistic performance estiamtes
# a better approach is to test the model on samples that were not used for training
# evaluating a model on a test set is the obvious choice
# but to get reasonable precision of the performance values, the size of the test set may need to be large

# an alternative approach to evaluating a model on a single test set is to resample the training set
# this process uses several modified versions of the training set to build multiple models and then uses stats to provide honest estimates of model performance


## Data Splitting
# now that we have outlined the general procedure for finding the optimal tuning parameters we will look at actually splitting the data
# a few common steps in model building are:
        # pre-process the predictor data
        # estimating model parameters
        # selecting predictors for the model
        # evaluating model performance
        # fine tuning class prediction rules (ROC curves)
# given a fixed amount of data the modeler must decide how to "spend" thier data points to accomodate these activities

# one of the first decisions to make when modeling is to decide which samples will be used to evaluate performance
# ideally the model should be evaluated on samples that were not used to build or fine-tune the model
# this will result in an unbiased sense of model effectivess
# when a large amount of data is at hand a set of samples can be set aside to evaluate the final model
# the training data set is the general term for the samples used to create the model
# the test or validation data set is used to qualify performance

# when the number of samples is not large a strong case can be made that a test set should be avoided because every sample may be needed for model builing
# the size of the test set may not have sufficient power or precision to make reasonable judgements
# several researchers show that validation using a single test set can be a poor choice
# hold out samples of tolerable size do not match the cross-validation itself for reliability in assessing model fit 
# resampling methods such as crossvalidation can be used to produce appropriate estimates of model performance using the training set
# althought resampling techniques can be misapplied they often produce performance estiamtes superior to a single test set because they evaluate many alternate versions of the data
 
# if a test set is deemed necessary there are several methods to splitting the samples
# non-random approaches to splitting the data are sometimes appropriate
# a model may be created using certain patient sets and then tested on a different sample population to understand generalization
# when we want to predict 'new' events - we want a estimatation of prediction on a new set of data - build on the history and see how to translates to the future

# in most cases there is the desire to make the training and test sets as homogenous as possible
# random sampling methods can be used to create similiar datasets
# the simplest way to split the data into a training and test set is to take a simple random sample
# this does not control for any of the data attributes such as the percentage of data in each class
# when one class has a disproportionately small frequency compared to others, there is a chance the distribution of the outcomes may be substantially different between test and training

# to account for this outcome when splitting the data - stratified random sampling applies random smapling within subgroups (such as the classes)
# in this way there is a higher likelihood that the outcome distributions will match
# when the outcome is a number we can use a similiar strategy
# the numeric values are broken into smaller groups and randomization is executed within these groups


# data can also be split on the basis of the predictor values
# tis idea is called maximum dissimilarity sampling
# dissimiliarity between two samples can be measured in a number of ways
# the simplest method is to use the distance between the predictor values for two samples
# if the distance is small the points are close to each other
# larger distances between points are indicative of dissimiliarity
# suppose the test set is initialized with a single sample
# the dissim between this inital sample and the unallocted samples can be calculated
# the unallocated sample that is most dissim would then be added to the test set
# to allocate more samples to the test set a method is needed to determine the dissim between groups of points
# one approach is to use the average or minimum of dissimiliarity
# to measure the dissim between the two samples in the test set and a sginle unallocated point...
# we can determine the two dissims and average them
# the third point added to the test set would be chosen as having the maximum average dissim to the existing set
# this process would continue until the targeted test set size is achieved

## Resampling Techniques
# resampling techniques for esitmating model performance operate in a similiar fashion:
# a subset of samples are used to fit a model and the remaining samples are used to estimate the efficacy of the model
# this process is repeated multiple times and the results are aggregated and summarized
# the differences in techniques usually center around the mdethod in which subsamples are chosen

## K-Fold Cross Validation
# the samples are randomly paritioned into k sets of roughly equal size
# a model is fit using all samples excpet the first subset - or fold
# the held-out samples are predicted by this model and used to esitmate performance
# the first subset is returned to the training set and the procedure repeats with the second subset held out
# the k resampled esitmates of performance are summairized usually with mean standard error and used to understand the relationship between  the tuning parameters and model utility

# a slight variant to this mehtod is to select the k partitions in a way that makes the folds balanced with respect to the outcome
# stratified random sampling creates balance with respect to each outcome














