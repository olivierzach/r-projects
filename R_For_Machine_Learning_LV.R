###################################################################
## Code for "R for Machine Learning"
## Max Kuhn 2018

###################################################################
## Slide 19

load("Tayko.RData")
str(all_data)

###################################################################
## Slide 20

library(ggplot2)
ggplot(all_data, aes(x = first_activity, y= last_activity)) + 
  geom_point(alpha = .25) + coord_equal()

###################################################################
## Slide 21

all_data$activity_length <- all_data$first_activity - all_data$last_activity
all_data <- all_data[, -grep("_activity", names(all_data))]
predictors <- names(all_data)[names(all_data) != "purch"]
all_data <- subset(all_data, num_trans > 0)

all_data <- subset(all_data, num_trans > 0)
nrow(all_data)

###################################################################
## Slide 26

library(caret)

set.seed(8131)
in_train <- createDataPartition(all_data$purch, p = 3/4, list = FALSE)
head(in_train)
train_data <- all_data[ in_train,]
test_data  <- all_data[-in_train,]

###################################################################
## Slide 62

dummies <- dummyVars( ~ ., data = train_data[, predictors], 
                     fullRank = TRUE)
train_dummies <- predict(dummies, train_data[, predictors])
test_dummies  <- predict(dummies, test_data[, predictors])
colnames(train_dummies)

###################################################################
## Slide 65

pp_values <- preProcess(train_data[, predictors], 
                        method = c("center", "scale"))
pp_values

train_scaled <- predict(pp_values, newdata = train_data[, predictors])
test_scaled  <- predict(pp_values, newdata = test_data[, predictors])

###################################################################
## Data for page 69 

data(segmentationData)

segmentationData$Cell <- NULL
segmentationData$Class <- ifelse(segmentationData$Class == "PS", "One", "Two")
segmentationData <- segmentationData[, c("EqSphereAreaCh1", "PerimCh1", "Class", "Case")]
names(segmentationData)[1:2] <- paste0("Predictor", LETTERS[1:2])
example_train <- subset(segmentationData, Case == "Train")
example_test  <- subset(segmentationData, Case == "Test")

example_train$Case <- NULL
example_test$Case  <- NULL

###################################################################
## Slide 69

dim(example_train)
dim(example_test)
head(example_train)

###################################################################
## Slide 70

theme_set(theme_bw())
ggplot(example_train, aes(x = PredictorA, 
                      y = PredictorB,
                      color = Class)) +
  geom_point(alpha = .5, cex = 2.6) + 
  theme(legend.position = "top")

###################################################################
## Slide 71

library(reshape2)
melted <- melt(example_train, id.vars = "Class")
ggplot(melted, aes(x = factor(paste("Class", Class)), y = log(value))) +
  geom_boxplot() + 
  theme_bw() +
  facet_wrap(~variable, scales = "free_y") + 
  theme(legend.position = "top") + 
  xlab("") + ylab("log(value)") 

###################################################################
## Slide 72

pca_pp <- preProcess(example_train[, 1:2],
                     method = "pca") # also added "center" and "scale"
pca_pp
train_pc <- predict(pca_pp, example_train[, 1:2])
test_pc <- predict(pca_pp, example_test[, 1:2])
head(test_pc, 4)

###################################################################
## Slide 73

test_pc$Class <- example_test$Class
ggplot(test_pc, aes(x = PC1, 
                    y = PC2,
                    color = Class)) +
  geom_point(alpha = .5, cex = 2.6) + 
  theme(legend.position = "top")

###################################################################
## Slide 74

test_melt <- melt(test_pc, id.vars = "Class")
ggplot(test_melt, aes(x = factor(paste("Class", Class)), y = value)) +
  geom_boxplot() + 
  theme_bw() +
  facet_wrap(~variable) + 
  theme(legend.position = "top") + 
  ylim(c(-1, 1)) + 
  xlab("") + ylab("")

###################################################################
## Slide 88

glmn_grid <- expand.grid(alpha = seq(0, 1, by = .25), 
                         lambda = 10^seq(-3, -1, length = 20))

cv_ctrl <- trainControl(method = "cv", 
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)

set.seed(1735)
glmn_tune <- train(purch ~ ., data = train_data, 
                   method = "glmnet",
                   tuneGrid = glmn_grid,
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   trControl = cv_ctrl)

###################################################################
## Slide 90

glmn_tune

###################################################################
## Slide 92

ggplot(glmn_tune) + scale_x_log10() + theme_bw() + theme(legend.position = "top")

###################################################################
## Slide 94

glmn_pred <- predict(glmn_tune, newdata = test_data)
confusionMatrix(glmn_pred, test_data$purch)

###################################################################
## Slide 95

glmn_probs <- predict(glmn_tune, newdata = test_data, type = "prob")
head(glmn_probs, n = 4)

###################################################################
## Slide 96

library(pROC)
## The roc function assumes the *second* level is the one of
## interest, so we use the 'levels' argument to change the order.
glmn_roc <- roc(response = test_data$purch, predictor = glmn_probs[, "yes"], 
                 levels = rev(levels(test_data$purch)))


pROC::auc(glmn_roc)

###################################################################
## Slide 97

plot(glmn_roc, col = "#9E0142")

###################################################################
## Slide 98

predictors(glmn_tune)

###################################################################
## Slide 99

ggplot(varImp(glmn_tune, scale = FALSE))

###################################################################
## Slide 102

library(rpart)
rpart1 <- rpart(purch ~ .,
                data = train_data,
                control = rpart.control(maxdepth = 2))
rpart1

###################################################################
## Slide 103

library(partykit)
rpart1_plot <- as.party(rpart1)
## plot(rpart1_plot)

###################################################################
## Slide 104

plot(rpart1_plot)

###################################################################
## Slide 107

rpart_full <- rpart(purch ~ ., data = train_data)
rpart_full

###################################################################
## Slide 108

library(partykit)
rpart_full_plot <- as.party(rpart_full)
plot(rpart_full_plot)

###################################################################
## Slide 109

rpart_pred <- predict(rpart_full, newdata = test_data, type = "class")
confusionMatrix(data = rpart_pred, reference = test_data$purch)   # requires 2 factor vectors

###################################################################
## Slide 110

class_probs <- predict(rpart_full, newdata = test_data)
head(class_probs, 3)       

library(pROC)
rpart_roc <- roc(response = test_data$purch, predictor = class_probs[, "yes"], 
                 levels = rev(levels(test_data$purch)))
## Get the area under the ROC curve
pROC::auc(rpart_roc)

###################################################################
## Slide 111

plot(glmn_roc, col = "#9E0142")
plot(rpart_roc, add = TRUE, col = "grey")
legend(.4, .4, legend = c("glmnet", "rpart"), 
       lty = rep(1, 3),
       col = c("#9E0142", "grey"))

###################################################################
## Hands-on break #1

num_leaves <- function(x) {
  output <- capture.output(print(x))
  length(grep("\\*$", output))
}
num_leaves(rpart_full)

###################################################################
## Slide 119

library(xgboost)
# Requires its own data structure
train_object <- xgb.DMatrix(train_dummies, 
                            label = ifelse(train_data$purch == "yes", 1, 0))

set.seed(10)
mod <- xgb.train(data = train_object, 
                 nrounds = 50,                  # Boosting iterations
                 max_depth = 2,                 # How many splits in each tree
                 eta = 0.01,                    # Learning rate
                 silent = 1, nthread = 1, 
                 objective = "binary:logistic") # For classification 

###################################################################
## Slide 120

test_object <- xgb.DMatrix(test_dummies)
xgb_pred <- predict(mod, newdata = test_object)
head(xgb_pred)
xgb_pred <- factor(ifelse(xgb_pred > .5, "yes", "no"),
                   levels = c("yes", "no"))
head(xgb_pred)

###################################################################
## Slide 121

confusionMatrix(xgb_pred, test_data$purch)

###################################################################
## Slide 122

xgb_grid <- expand.grid(
  max_depth = seq(1, 7, by = 2),
  nrounds = seq(100, 1000, by = 50),
  eta = c(0.01, 0.1),
  # Other parameters. See ?xgboost
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 0.5
)

###################################################################
## Slide 123

set.seed(1735)
xgb_tune <- train(
  purch ~ ., data = train_data,
  method = "xgbTree",    
  # Use a custom grid of tuning parameters
  tuneGrid = xgb_grid, 
  trControl = cv_ctrl,
  metric = "ROC",
  # Remember the 'three dots' discussed in the bootcamp?
  # This options is directly passed to the xgb.train function.
  silent = 1
)  

###################################################################
## Slide 126

## library(doMC)          # on unix, linux or OS X
## ## library(doParallel) # windows and others
## registerDoMC(cores = 2)
## 
## # or
## 
## library(doParallel)
## cl <- makePSOCKcluster(parallel::detectCores(logical = FALSE))
## registerDoParallel(cl)

###################################################################
## Slide 129

xgb_tune

###################################################################
## Slide 130

ggplot(xgb_tune)

###################################################################
## Slide 131

xgb_pred <- predict(xgb_tune, newdata = test_data[, predictors]) # Magic!
confusionMatrix(xgb_pred, test_data$purch)

###################################################################
## Slide 132

xgb_probs <- predict(xgb_tune, 
                     newdata = test_data[, predictors], 
                     type = "prob")
head(xgb_probs)

###################################################################
## Slide 133

xgb_roc <- roc(response = test_data$purch, predictor = xgb_probs[, "yes"],
               levels = rev(levels(test_data$purch)))
pROC::auc(glmn_roc)
pROC::auc(rpart_roc)
pROC::auc(xgb_roc)

###################################################################
## Slide 134

plot(glmn_roc, col = "#9E0142")
plot(rpart_roc, col = "grey", legacy.axes = FALSE, add = TRUE)
plot(xgb_roc,   col = "#3288BD", legacy.axes = FALSE, add = TRUE)
legend(.4, .4, legend = c("glmnet", "rpart", "xgboost"), 
       lty = rep(1, 3),
       col = c("#9E0142", "grey", "#3288BD"))


###################################################################
## Slide 142

names(lrFuncs)
lrFuncs$fit

###################################################################
## Slide 143

## No tuning parameters for this model, so avoid an inner resampling loop
inner_ctrl <- trainControl(method = "none", 
                           classProbs = TRUE,
                           allowParallel = FALSE)

## Now the rfe control function. First, to get the ROC value
## we change the summary function as we did with train()
ourFunctions <- lrFuncs
ourFunctions$summary <- twoClassSummary

rfeCtrl <- rfeControl(functions = ourFunctions,
                      method = "cv",
                      verbose = TRUE)

###################################################################
## Slide 144

## We can pass the same options to train() through rfe()
set.seed(1735)
logistic_rfe <- rfe(
  x = train_dummies, 
  y = train_data$purch,
  sizes = 1:ncol(train_dummies),
  rfeControl = rfeCtrl,
  ## now the options that pass to train()
  method = "glm",
  preProc = c("center", "scale"),
  trControl = inner_ctrl,
  metric = "ROC"
  )

###################################################################
## Slide 145

logistic_rfe

###################################################################
## Slide 146

ggplot(logistic_rfe)

###################################################################
## Slide 147

rfe_test <- predict(logistic_rfe, test_dummies)
head(rfe_test)
roc(response = test_data$purch, predictor = rfe_test$yes, 
    levels = rev(levels(test_data$purch)))

###################################################################
## Slide 150

cv_values <- resamples(
  list(glmnet = glmn_tune, 
       xgboost = xgb_tune, logistic = logistic_rfe)
 )

###################################################################
## Slide 151

summary(cv_values, metric = "ROC")
glmn_tune$times$everything[3]/60
xgb_tune$times$everything[3]/60

###################################################################
## Slide 152

splom(cv_values, metric = "ROC", pscales = 0)

###################################################################
## Slide 153

dotplot(cv_values, metric = "ROC")

###################################################################
## Slide 154

roc_diffs <- diff(cv_values, metric = "ROC")
summary(roc_diffs)

###################################################################
## Slide 155

dotplot(roc_diffs, metric = "ROC")

###################################################################
## Slide 156

xyplot(cv_values, metric = "ROC", models = c("glmnet", "xgboost"), 
       what = "BlandAltman")

###################################################################
## Slide 161

day_values <- c("2015-05-10", "1970-11-04", "2002-03-04", "2006-01-13")
class(day_values)

library(lubridate)
days <- ymd(day_values)
str(days)

###################################################################
## Slide 162

day_of_week <- wday(days, label = TRUE)
day_of_week

year(days)
week(days)
month(days, label = TRUE)
yday(days)

###################################################################
## Slide 167

test_probs <- data.frame(purch = test_data$purch,
                         xgb = xgb_probs[, "yes"],
                         glmnet = glmn_probs[, "yes"])

head(test_probs)
lift_obj <- caret::lift(purch ~ glmnet + xgb, data = test_probs)
lift_obj
## plot(lift_obj, auto.key = list(lines = TRUE, points = FALSE))

###################################################################
## Slide 168

trellis.par.set(liftTheme)
plot(lift_obj,auto.key = list(columns = 2,
                               lines = TRUE,
                               points = FALSE))
