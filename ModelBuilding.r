##############################################################################################################################
# In this part, we build multiple logistic regression models to predict weather forecast. 
# Specifically, we intend to produce the following forecasts:
#   - tomorrow's rain forecast at 9am 
#   - tomorrow's rain forecast at 3pm
##############################################################################################################################

##############################################################################################################################
# LIBRARIES USED
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(gmodels))
suppressPackageStartupMessages(library(ROCR))
# install.packages('e1071', dependencies=TRUE) # uncomment this line if you are having this error
# install.packages('performance') # uncomment this line if you are having this error
##############################################################################################################################

##############################################################################################################################
# SPLIT DATA INTO 80% TRAIN AND 20% TEST AND CHECK BASELINE MODEL ACCURACY
set.seed(1023)
clean_dataset <- read.csv("clean_dataset.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)
colnames(clean_dataset)
nrow(clean_dataset)
# We check the balance of RainTomorrow Yes/No fractions in the training and testing datasets.
sum(clean_dataset["RainTomorrow"] == "Yes")
sum(clean_dataset["RainTomorrow"] == "No")
# The dataset is slightly unbalanced with respect to the label to be predicted.
# We can see that the majority label is No (268 records for No > 60 records for Yes)

# We split the data
train_rec <- createDataPartition(clean_dataset$RainTomorrow, p = 0.8, list = FALSE)
training <- clean_dataset[train_rec,]
testing <- clean_dataset[-train_rec,]

# We check baseline model accuracy on the primary whole dataset
baselineModelAccuracy <- 1 - (sum(clean_dataset["RainTomorrow"] == "Yes") / sum(clean_dataset["RainTomorrow"] == "No"))
# If we always predict No we achieve a 77.6% accuracy.

# We check if this accuracy insists on the partioned data
1- sum(training["RainTomorrow"] == "Yes")/sum(training["RainTomorrow"] == "No")
1 - sum(testing["RainTomorrow"] == "Yes")/sum(testing["RainTomorrow"] == "No")
# It does! Yet, this is not intelligent. We will use Machine Learning instead of always
# predicting No. Specifically we will use logistic regression models.
##############################################################################################################################

##############################################################################################################################
# 10 FOLD CROSS VALIDATION DIRECTIVE
# For all models, we are going to take advantage of a train control directive made available by the caret package
# which prescribes repeated k-flod cross-validation. 
# The k-fold cross validation method involves splitting the training dataset into k-subsets. 
# For each subset, one is held out while the model is trained on all other subsets. 
# This process is completed until accuracy is determined for each instance in the training dataset, 
# and an overall accuracy estimate is provided. 
# The process of splitting the data into k-folds can be repeated a number of times and this is called repeated k-fold cross validation. 
# The final model accuracy is taken as the mean from the number of repeats.
trControl <- trainControl(method = "repeatedcv",  repeats = 5, number = 10, verboseIter = FALSE)
##############################################################################################################################

##############################################################################################################################
# MODEL BUILDING
# By taking into account the results from exploratory analysis, we compare two models for 9AM prediction.
# One with all the respective predictors for 9am + Evaporation, WindGustDir, WindGustSpeed
# One with all the repsective predictors for 9am
predictors_9am_c1 <- c("Evaporation", "WindGustDir", "WindGustSpeed", "WindSpeed9am", "Cloud9am",  "Humidity9am", "Pressure9am", "Temp9am")
predictors_9am_c2 <- c("WindSpeed9am", "Cloud9am",  "Humidity9am", "Pressure9am", "Temp9am")

# We do the same procedure, but for 3pm
predictors_3pm_c1 <- c("Evaporation", "WindGustDir", "WindGustSpeed", "WindSpeed3pm", "Cloud3pm",  "Humidity3pm", "Pressure3pm", "Temp3pm")
predictors_3pm_c2 <- c("WindSpeed3pm", "Cloud3pm",  "Humidity3pm", "Pressure3pm", "Temp3pm")

# Create a formula which provides the scheme of class and predictors, to pass it later on in the train() caret function
formula_9am_c1 <- as.formula(paste("RainTomorrow", paste(predictors_9am_c1, collapse="+"), sep="~"))
formula_9am_c2 <- as.formula(paste("RainTomorrow", paste(predictors_9am_c2, collapse="+"), sep="~"))
formula_3pm_c1 <- as.formula(paste("RainTomorrow", paste(predictors_3pm_c1, collapse="+"), sep="~"))
formula_3pm_c2 <- as.formula(paste("RainTomorrow", paste(predictors_3pm_c2, collapse="+"), sep="~"))

# The trainControl is passed as a parameter to the train() caret function.
mod9am_c1_fit <- train(formula_9am_c1,  data=training, method="glm", 
                       family="binomial", trControl = trControl, metric = 'Accuracy')
mod9am_c2_fit <- train(formula_9am_c2,  data=training, method="glm", 
                       family="binomial", trControl = trControl, metric = 'Accuracy')
mod3pm_c1_fit <- train(formula_3pm_c1,  data=training, method="glm", 
                       family="binomial", trControl = trControl, metric = 'Accuracy')
mod3pm_c2_fit <- train(formula_3pm_c2,  data=training, method="glm", 
                       family="binomial", trControl = trControl, metric = 'Accuracy')

# Model comparison with Accuracy
mod9am_c1_fit$results$Accuracy
mod9am_c2_fit$results$Accuracy
mod3pm_c1_fit$results$Accuracy
mod3pm_c2_fit$results$Accuracy
# We can see that the simpler models (less predictors) behave better than the more complex
##############################################################################################################################

##############################################################################################################################
# WE PLOT THE ROC CURVE OF EACH MODEL ON THE TESTING SET
glm.perf.plot <- function (prediction) {
  perf <- performance(prediction, measure = "tpr", x.measure = "fpr")     
  par(mfrow=(c(1,2)))
  plot(perf, col="red")
  grid()
  auc_res <- performance(prediction, "auc")
  auc_res@y.values[[1]]
}

mod9am_c2_pred_prob <- predict(mod9am_c2_fit, testing, type="prob")
mod9am_c2_pred_resp <- prediction(mod9am_c2_pred_prob$Yes,  testing$RainTomorrow)
glm.perf.plot(mod9am_c2_pred_resp)

mod3pm_c2_pred_prob <- predict(mod3pm_c2_fit, testing, type="prob")
mod3pm_c2_pred_resp <- prediction(mod3pm_c2_pred_prob$Yes,  testing$RainTomorrow)
glm.perf.plot(mod3pm_c2_pred_resp)

