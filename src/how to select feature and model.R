##install and load caret package
install.packages("caret", dependencies = TRUE,
                 INSTALL_opts = '--no-lock')
library(caret)

#load dataset
rm(list = ls())
data <- read.csv("data/npcl.csv")
str(data)
data_raw <- data[,-1]
library(tidyverse)
data_raw <- data_raw %>% 
  mutate(loss_degree = loss_rate > 0.4) %>%
  select(-loss_rate) 
y <- ifelse (data_raw$loss_degree == TRUE, "serious", "normal")
data_use <- cbind(data_raw,y) %>% select(-loss_degree)

head(data_use)
write.csv(data_use, file = "data/data_use.csv")

#impute NA in the dataset
library(skimr) # for basic statistics
skimmed <- skim_to_wide(data_use)
skimmed[, 1:11]

preProcess_missingdata_model <- preProcess(data_r, #build a model
                            method='knnImpute')
preProcess_missingdata_model
library(RANN) # imputing NA algorithm
data_r_NA <- predict(preProcess_missingdata_model, 
                           newdata = data_r)
anyNA(data_r_NA)

#for one-hot code
dummies_model <- dummyVars(loss_rate ~ ., data= data_r_NA)# build a model
data_r_NA_dum_mat <- predict(dummies_model, 
                               newdata = data_r_NA)
data_r_NA_dum <- data.frame(data_r_NA_dum_mat)#rebuild a dataframe including target
loss_rate <- data_r_NA$loss_rate
data_clean <- cbind(loss_rate,data_r_NA_dum)
head(data_clean)

##split dataset
set.seed(100)
train_idx <- createDataPartition(data_use$y, p=0.8, list=FALSE)
train <- data_use[train_idx,]
test <- data_use[-train_idx,]

##estimate the importance of predictors
#visualize feature importance
x = as.matrix(train[, 1:11])
y = as.factor(train$y)

featurePlot(x, y, plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(draw=FALSE), 
                          y = list(relation="free")))

#importance of features
options(warn=-1)
set.seed(100)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x, y,rfeControl = ctrl)
lmProfile

##build a rf model and calculat feature importance
set.seed(100)
rf_fit <- train(y ~ ., 
                    data = train, 
                    method = "rf")
rf_fit
plot(rf_fit)

varimp <- varImp(rf_fit) # calculate feature importance
plot(varimp)

rf_pred <- predict(rf_fit, test)#test rf performance
rf_pred
confusionMatrix(reference = as.factor(test$y), 
                data = rf_pred,
                mode = "everything")

#set uneLength or tuneGrid for better model performance

fitControl <- trainControl(
              method = 'cv',                  
              number = 5,                     
              savePredictions = 'final',
              classProbs = T,                  
              summaryFunction=twoClassSummary) 

rf_fit <- train(y~ ., #optimize with tuneLength
                data = train, 
                method = "rf", 
                tuneLength = 5,
                trControl = fitControl,
                verbose = FALSE
                )
rf_fit


##comparison among different models
rpart_fit = train(y ~ .,
                 data=train, 
                 method='rpart', 
                 tuneLength=15, 
                 trControl = fitControl)

svm_fit = train(y ~ .,
                data=train, 
                method='svmRadial', 
                tuneLength=15, 
                trControl = fitControl)

models_compare <- resamples(list(rpart = rpart_fit, randomForest = rf_fit, SVM= svm_fit))
summary(models_compare)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)

library(caretEnsemble)

# Stacking Algorithms - Run multiple algos in one call.
trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

algorithmList <- c('rf', 'rpart', 'svmRadial')

set.seed(100)
models <- caretList(y ~ ., data=train, 
                    trControl=trainControl, 
                    methodList=algorithmList) 
results <- resamples(models)
summary(results)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)








