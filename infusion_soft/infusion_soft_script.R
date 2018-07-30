
# infusion_soft interview ---------------------------------------------------------------------

# title: "infusionsoft_interview"
# author: "Zach Olivier"
# date: "July 26, 2018"

# goal: import data, clean, pre-process, eda, model for inference, model for prediction




# workflow ------------------------------------------------------------------------------------

# load libraries
pacman::p_load(
  tidyverse, data.table, caret, 
  VIM, tictoc, modelr, GGally, readxl,
  xlsx, openxlsx, broom, corrplot, doParallel
)

options(scipen=999)


# import data ---------------------------------------------------------------------------------


# read in the large loan dataset from excel
# df <- openxlsx::read.xlsx(
#   "LoadData15_16.xlsx",
#   sheet = 1
# )


# save to Rds file for quicker recovery
# save(df, file = 'loan_data.Rds')


# load loan data frame 
load('loan_data.Rds')

# view response distribution
table(df$loan_status)


# read in data and format columns and response variable
(loan_df <- df %>% 
    as_tibble() %>% 
    filter(loan_status != 'Current') %>% 
    mutate( # combines default and charged off together = at risk customers
      response = ifelse(loan_status %in% c('Charged Off', 'Default'), 'Default','Paid')
    ) %>% 
    mutate_if(is.character, as.factor) %>%          
    select_if( # remove columns with close to  100% NAs
      function(x) {!mean(is.na(x)) > .9})
)

dim(loan_df)

# examine class balance
mean(loan_df$response == 'Default')
table(loan_df$response)


set.seed(45)

# pre-processing ------------------------------------------------------------------------------


# remove columns that have near zero variance
nzv <- nearZeroVar(loan_df)

loan_df <- loan_df[,-nzv]

dim(loan_df)



# handling missing values ---------------------------------------------------------------------


# profile missing values - will impute missing values less than 10% of column total
(missing <- sapply(loan_df, function(x) mean(is.na(x))) %>% 
   as.data.frame() %>% 
   rownames_to_column() %>% 
   filter(. > .11) %>% 
   arrange(-.)
)


# get index of missing high missing values columns
# too large to impute
# may be worth it to keep if correlated with response
check_missing <- which(colnames(loan_df) %in% as.list(missing$rowname)) %>% 
  as.list()

# empty list to store results of univarite models into
glm_bag <- list()


# define function to fit univariate models for each of high proportion of missing data
# estimate close to 0 = confident to drop column - else may omit rows but keep column
for (i in seq_along(check_missing)) {
  
  
  glm_df <- loan_df %>% 
    dplyr::select(check_missing[[i]], loan_status) %>% 
    na.omit() %>% 
    glm(
      loan_status ~ ., 
      data = ., 
      family = 'binomial'
    ) %>% 
    tidy() %>% 
    dplyr::select(term, estimate)
  
  # print(glm_df)
  
  glm_bag[[i]] <- glm_df
  
}


# variable total_bal_il has a large negative effect on the response
# this might be a variable worth keeping even if we have to cut down the number of rows
glm_bag[[8]]


# correlation check with the high missing value columns
cor_missing <- loan_df %>% 
  dplyr::select(
    which(colnames(loan_df) %in% as.list(missing$rowname))
  ) %>% 
  na.omit() %>% 
  cor()

# check correlation between the high missing columns
corrplot(cor_missing, type = 'upper', order = 'hclust', tl.col = 'black', tl.srt = 45, diag = F,
         title = "Correlation Plot of High Missing Value columns", tl.cex = .6)


# final data frame for EDA
loan_eda <- loan_df %>% 
  dplyr::select(
    .,
    -which(colnames(loan_df) %in% as.list(missing$rowname)),
    -zip_code,
    -emp_title,
    -loan_status
  ) 

dim(loan_df)

# data frame to be used for exploration - check missing - under 10% ok to impute for modeling
sapply(loan_eda, function(x) mean(is.na(x))) %>% tidy()

# clear workspace
rm(df, loan_df, glm_df, glm_bag, missing, cor_missing, check_missing)



# learning partitions -------------------------------------------------------------------------

set.seed(456)

# set up names and partition sizes
partition = c(train = .6, test = .2, validation = .2)

# split data set into train, test, validation
loan_splits = sample(cut(
  seq(nrow(loan_eda)), 
  nrow(loan_eda)*cumsum(c(0,partition)),
  labels = names(partition)
))

# fracture the loan df dataset
dfs <- split(loan_eda, loan_splits)

# check resuts 
sapply(dfs, nrow) / nrow(loan_eda)
dim(dfs$train)

# convert to data frame for modeling
# datasets will have response
dfs <- map(dfs, as.data.frame)


# correlation eda -----------------------------------------------------------------------------


# str(dfs$train)

# visually inspect correlation across all variables
GGally::ggcorr(dfs$train, size = 2, hjust = 1)

# extract higly correlated numeric variables
c_matrix <- cor(dfs$train %>% dplyr::select_if(is.numeric) %>% na.omit())

# caret's highly correlated function - gives index
high_cor <- findCorrelation(c_matrix, cutoff = .5)

# print highly correlated columns
names(dfs$train[,high_cor])


# LVQ model -----------------------------------------------------------------------------------

# cross validation for LVQ model
control <- trainControl(method = 'cv', number = 3)

# lvq model - needed to randomly sample train to cut down on size fore presentation - not accurate
# should be training model on the full training set 
lvq <- train(
  response ~ ., 
  data = dfs$train %>% na.omit() %>% sample_frac(., .1),
  method = 'lvq',
  trControl = control
)

# grab variable importance
(var_importance <- varImp(lvq))

# view results
plot(var_importance)


cores <- detectCores()
cl <- makeCluster(cores[1] - 1)
registerDoParallel(cl)



# model preprocessing -------------------------------------------------------------------------

set.seed(854)

# center, scale, and impute missing data for final model
process <- preProcess(dfs$train, method = c('scale', 'knnImpute'))


# apply transformations to both train and test datasets
dfs$train <- predict(process, dfs$train) 
dfs$test <- predict(process, dfs$test)
dfs$validation <- predict(process, dfs$validation)


# three fold cv to cut down on time for presentation purposes - real cv could be 10 fold repeated 
control <- trainControl(method = 'cv', number = 3, allowParallel=T)


# interpretable model -------------------------------------------------------------------------



set.seed(987)

# fit simple logistic regression on most important variables from EDA
glm_mod <- train(
  response ~
    last_pymnt_amnt + 
    total_rec_prncp +
    total_pymnt + 
    total_pymnt_inv + 
    collection_recovery_fee +
    recoveries + 
    int_rate    +
    last_pymnt_d +
    dti + 
    bc_open_to_buy  + 
    term +
    revol_util +
    bc_util +
    percent_bc_gt_75 +
    avg_cur_bal +
    tot_hi_cred_lim +
    mort_acc +
    num_rev_tl_bal_gt_0 +
    num_actv_rev_tl +
    total_rec_int,
  method = 'glm',
  family = 'binomial',
  metric = 'Kappa',
  data = dfs$train %>% na.omit() %>% sample_frac(., .1),
  trControl = control
)

# check cv summary of glm model 
glm_mod
summary(glm_mod$finalModel)



# apply model to test data - set classification threshold at .5
glm_pred <- predict(glm_mod, newdata = dfs$test %>% na.omit(), type = 'prob') %>% 
  mutate(pred = ifelse(Paid >= .5, 'Paid', 'Default')) %>% 
  mutate_if(is.character, as.factor)

# add predictions to the test data set
glm_test <- dfs$test %>% na.omit() %>% add_predictions(glm_mod)

# view confusion matrix
confusionMatrix(glm_test$pred, glm_test$response)


# predictive model ----------------------------------------------------------------------------

set.seed(987)

# fit flexible model for pure prediction power
# rf is non-parameteric and insensistive to scaling
rf_mod <- train(
  response ~
    last_pymnt_amnt + 
    total_rec_prncp +
    total_pymnt + 
    total_pymnt_inv + 
    collection_recovery_fee +
    recoveries + 
    int_rate    +
    last_pymnt_d +
    dti + 
    bc_open_to_buy  + 
    term +
    revol_util +
    bc_util +
    percent_bc_gt_75 +
    avg_cur_bal +
    tot_hi_cred_lim +
    mort_acc +
    num_rev_tl_bal_gt_0 +
    num_actv_rev_tl +
    total_rec_int,
  method = 'rf',
  metric = 'Kappa',
  data = dfs$train %>% na.omit() %>% sample_frac(., .1),
  trControl = control
)

# check cv summary of rf model including best tuned parameters
rf_mod$finalModel


# apply model to test data - set classification threshold at .5
rf_pred <- predict(rf_mod, newdata = dfs$test %>% na.omit(), type = 'prob') %>% 
  mutate(pred = ifelse(Paid >= .5, 'Paid', 'Default')) %>% 
  mutate_if(is.character, as.factor)

# add predictions to the test data set
rf_test <- dfs$test %>% na.omit() %>% add_predictions(rf_mod)

# view confusion matrix
confusionMatrix(rf_test$pred, rf_test$response)




