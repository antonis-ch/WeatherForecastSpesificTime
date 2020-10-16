##############################################################################################################################
# IN THIS PART WE PREPROCESS THE WEATHER DATASET AND OUTLINE ITS EXPLORATORY ANALYSIS
##############################################################################################################################

##############################################################################################################################
# LIBRARIES USED
suppressPackageStartupMessages(library(corrplot))
suppressPackageStartupMessages(library(knitr))
##############################################################################################################################

##############################################################################################################################
# FIRST LOOK IN THE DATA
# The dataset under analysis is publicly available at:
# https://www.biz.uiowa.edu/faculty/jledolter/datamining/dataexercises.html
# It contains daily observations from a single weather station.
set.seed(1023)
baseline_dataset <- read.csv(
  url("https://www.biz.uiowa.edu/faculty/jledolter/datamining/weather.csv"), header = TRUE, sep = ",", stringsAsFactors = TRUE)

# Take a look in the data (head and column names)
kable(head(baseline_dataset))
colnames(baseline_dataset)

# Column names description
# Date: The date of observation (a date object).
# Location: The common name of the location of the weather station
# MinTemp: The minimum temperature in degrees centigrade
# MaxTemp: The maximum temperature in degrees centigrade
# Rainfall: The amount of rainfall recorded for the day in millimeters.
# Evaporation: Class A pan evaporation (in millimeters) during 24 h
# Sunshine: The number of hours of bright sunshine in the day
# WindGustDir: The direction of the strongest wind gust in the 24 h to midnight
# WindGustSpeed: The speed (in kilometers per hour) of the strongest wind gust in the 24 h to midnight
# WindDir9am: The direction of the wind gust at 9 a.m.
# WindDir3pm: The direction of the wind gust at 3 p.m.
# WindSpeed9am: Wind speed (in kilometers per hour) averaged over 10 min before 9 a.m.
# WindSpeed3pm: Wind speed (in kilometers per hour) averaged over 10 min before 3 p.m.
# Humidity9am: Relative humidity (in percent) at 9 am
# Humidity3pm: Relative humidity (in percent) at 3 pm
# Pressure9am: Atmospheric pressure (hpa) reduced to mean sea level at 9 a.m.
# Pressure3pm: Atmospheric pressure (hpa) reduced to mean sea level at 3 p.m.
# Cloud9am: Fraction of sky obscured by cloud at 9 a.m. This is measured in "oktas," which are a unit of eighths. 
# Cloud3pm: Fraction of sky obscured by cloud at 3 p.m; see Cloud9am for a description of the values
# Temp9am: Temperature (degrees C) at 9 a.m.
# Temp3pm: Temperature (degrees C) at 3 p.m.
# RainToday: Integer 1 if precipitation (in millimeters) in the 24 h to 9 a.m. exceeds 1 mm, otherwise 0
# RISK_MM: The continuous target variable; the amount of rain recorded during the next day
# RainTomorrow: The binary target variable whether it rains or not during the next day

# We look at the data structure and discover the dataset includes a mix of numerical and categorical variables.
str(baseline_dataset)

# We have available 366 records ...
(n <- nrow(baseline_dataset))
# ... which spans the following timeline
c(as.character(baseline_dataset$Date[1]), as.character(baseline_dataset$Date[n]))

# We further notice that RISK_MM relation with the RainTomorrow variable is the following:
all.equal(baseline_dataset$RISK_MM > 1, baseline_dataset$RainTomorrow == "Yes")
# The Rainfall variable and the RainToday are equivalent according to the following relationship (as anticipated by Rainfall).
all.equal(baseline_dataset$Rainfall > 1, baseline_dataset$RainToday == "Yes")
##############################################################################################################################

##############################################################################################################################
# FIRST ACTIONS ON THE DATA
# We decide to remove the Date, Location features (since we want models independent from time and space).
# We also remove the RISK_MM variable since it's equivalent with the RainTomorrow.
# We finaly remove Rainfall and RainToday, since they are equivalent and describe the amount of today's rainfall 
# (but we want to predict tomorrows' rainfall).
dataset1 <- subset(baseline_dataset, 
                        select = -c(Date, Location, RISK_MM, Rainfall, RainToday))
colnames(dataset1)

# We count NA's in all columns
(cols_withNa <- apply(dataset1, 2, function(x) sum(is.na(x))))
(n <- nrow(dataset1))

# We remove records with NA values...
dataset2 <- dataset1[complete.cases(dataset1),]
# ...hence we have to have 0 sums of NA's in all columns
(cols_withNa <- apply(dataset2, 2, function(x) sum(is.na(x))))

# From 366 records we have 328 now
(n <- nrow(dataset2))
#######################################################

#######################################################
# CATEGORICAL VARIABLE ANALYSIS
# We will run a pvalue test on all categorical variables with the RainTomorrow variable.
# We will remove variables that reject the null hypothesis, hence they have a pvalue < 0.1.
categorical_vars <- names(which(sapply(dataset2, class) == "factor"))
categorical_vars
categorical_vars <- setdiff(categorical_vars, "RainTomorrow")
chisq_test <- lapply(categorical_vars, function(x) { 
  chisq.test(dataset2[,x], dataset2[, "RainTomorrow"], simulate.p.value = TRUE)
})
names(chisq_test) <- categorical_vars
chisq_test

# We reject the null-hypothesis for WindDir9am and WindDir3pm, hence we remove them from the dataset
dataset3 <- subset(dataset2, select = -c(WindDir9am, WindDir3pm))

# We have the following columns atm:
colnames(dataset3)
##############################################################################################################################

##############################################################################################################################
# NUMERICAL VARIABLE ANALYSIS
# To obtain the quantitative variables we calculate the difference of the columns and the categorical vars.
chisq_test <- names(which(sapply(dataset3, class) == "factor"))
numerical_vars <- setdiff(colnames(dataset3), chisq_test)
numerical_vars <- setdiff(numerical_vars, "RainTomorrow")
numerical_vars
numerical_vars_mat <- as.matrix(dataset3[, numerical_vars, drop=FALSE])

# We measure the correlation among predictors
numerical_vars_cor <- cor(numerical_vars_mat)
numerical_vars_cor
# By taking a look at the above correlation results, we can state that:
# - Temp9am strongly positive correlated with MinTemp
# - Temp9am strongly positive correlated with MaxTemp
# - Temp9am strongly positive correlated with Temp3pm
# - Temp3pm strongly positive correlated with MaxTemp
# - Pressure9am strongly positive correlated with Pressure3pm
# - Humidity3pm strongly negative correlated with Sunshine
# - Cloud9am strongly negative correlated with Sunshine
# - Cloud3pm strongly negative correlated with Sunshine

# We plot the correlation among predictors
corrplot(numerical_vars_cor)

# From the above analysis, we can remove:
# MinTemp, MaxTemp (they are correlated with Temp9am and since Temp9am is correlated with Temp3pm
#   we can create the two models [9am, 3pm] without these features).
# Sunshine (since it's correlated with Cloud9am and Cloud3pm - used in morning, afternoon prediction models)
dataset4 <- subset(dataset3, select = -c(MinTemp, MaxTemp, Sunshine))

# We end up with the following columns:
colnames(dataset4)
##############################################################################################################################

##############################################################################################################################
# SAVE CHANGES
# Write the final dataset in a file:
write.csv(dataset4, file="clean_dataset.csv", row.names=FALSE)
##############################################################################################################################
