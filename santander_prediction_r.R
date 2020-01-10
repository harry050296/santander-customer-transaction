library(tidyverse)
library(moments)
library(DataExplorer)
library(caret)
library(Matrix)
library(pdp)
library(mlbench)
library(caTools)
library(randomForest)
library(glmnet)
library(mlr)
library(vita)
library(rBayesianOptimization)
library(lightgbm)
library(pROC)
library(DMwR)
library(ROSE)
library(yardstick)

#loading the train data
train_df<-read.csv('train.csv')
head(train_df)

#Dimension of train data
dim(train_df)

#Summary of the dataset
str(train_df)

#convert to factor
train_df$target<-as.factor(train_df$target)

require(gridExtra)
#Count of target classes
table(train_df$target)
#Percenatge counts of target classes
table(train_df$target)/length(train_df$target)*100
#Bar plot for count of target classes
plot1<-ggplot(train_df,aes(target))+theme_bw()+geom_bar(stat='count',fill='lightgreen')
#Violin with jitter plots for target classes
plot2<-ggplot(train_df,aes(x=target,y=1:nrow(train_df)))+theme_bw()+geom_violin(fill='lightblue')+
  facet_grid(train_df$target)+geom_jitter(width=0.02)+labs(y='Index')
grid.arrange(plot1,plot2, ncol=2)

#Take aways:
  
#1. We have a unbalanced data,where 90% of the data is the data of number 
#   of customers those will not make a transaction and 10% of the data is 
#   those who will make a transaction.
#2. Look at the violin plots seems that their is no relationship between 
#   the target with the index of the train dataframe. This is more dominated 
#   by the zero targets then for the ones.
#3. Look at the jitter plots with violin plots. We can observed that targets 
#   looks uniformaly distributed over the indexs of the dataframe.

#Distribution of train attributes from 3 to 102
for (var in names(train_df)[c(3:102)]){
  target<-train_df$target
  plot<-ggplot(train_df, aes(x=train_df[[var]],fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}

#Distribution of train attributes from 103 to 202
for (var in names(train_df)[c(103:202)]){
  target<-train_df$target
  plot<-ggplot(train_df, aes(x=train_df[[var]], fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}


#Take aways:
  
#1. We can observed that their is a considerable number of features which are significantly have different distributions for two target variables. For example like var_0,var_1,var_9,var_198 var_180 etc.
#2. We can observed that their is a considerable number of features which are significantly have same distributions for two target variables. For example like var_3,var_7,var_10,var_171,var_185 etc.

#loading test data
test_df<-read.csv('test.csv')
head(test_df)

#Dimension of test dataset
dim(test_df)

#Distribution of test attributes from 2 to 101
plot_density(test_df[,c(2:101)], ggtheme = theme_classic(),geom_density_args = list(color='blue'))

#Distribution of test attributes from 102 to 201
plot_density(test_df[,c(102:201)], ggtheme = theme_classic(),geom_density_args = list(color='blue'))

#Take aways:
  
#1. We can observed that their is a considerable number of features which are 
#   significantly have different distributions. For example like 
#   var_0,var_1,var_9,var_180 var_198 etc.
#2. We can observed that their is a considerable number of features which are 
#   significantly have same distributions. For example like 
#   var_3,var_7,var_10,var_171,var_185,var_192 etc.

#Let us see distribution of mean values per row and column in train and test dataset

#Applying the function to find mean values per row in train and test data.
train_mean<-apply(train_df[,-c(1,2)],MARGIN=1,FUN=mean)
test_mean<-apply(test_df[,-c(1)],MARGIN=1,FUN=mean)
ggplot()+
  #Distribution of mean values per row in train data
  geom_density(data=train_df[,-c(1,2)],aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per row in test data
  geom_density(data=test_df[,-c(1)],aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per row',title="Distribution of mean values per row in train and test dataset")

#Applying the function to find mean values per column in train and test data.
train_mean<-apply(train_df[,-c(1,2)],MARGIN=2,FUN=mean)
test_mean<-apply(test_df[,-c(1)],MARGIN=2,FUN=mean)
ggplot()+
  #Distribution of mean values per column in train data
  geom_density(aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per column in test data
  geom_density(aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per column',title="Distribution of mean values per row in train and test dataset")


#Applying the function to find standard deviation values per row in train and test data.
train_sd<-apply(train_df[,-c(1,2)],MARGIN=1,FUN=sd)
test_sd<-apply(test_df[,-c(1)],MARGIN=1,FUN=sd)
ggplot()+
  #Distribution of sd values per row in train data
  geom_density(data=train_df[,-c(1,2)],aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of mean values per row in test data
  geom_density(data=test_df[,-c(1)],aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per row',title="Distribution of sd values per row in train and test dataset")

#Applying the function to find sd values per column in train and test data.
train_sd<-apply(train_df[,-c(1,2)],MARGIN=2,FUN=sd)
test_sd<-apply(test_df[,-c(1)],MARGIN=2,FUN=sd)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per column',title="Distribution of std values per column in train and test dataset")

#Applying the function to find skewness values per row in train and test data.
train_skew<-apply(train_df[,-c(1,2)],MARGIN=1,FUN=skewness)
test_skew<-apply(test_df[,-c(1)],MARGIN=1,FUN=skewness)
ggplot()+
  #Distribution of skewness values per row in train data
  geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per row',title="Distribution of skewness values per row in train and test dataset")

#Applying the function to find skewness values per column in train and test data.
train_skew<-apply(train_df[,-c(1,2)],MARGIN=2,FUN=skewness)
test_skew<-apply(test_df[,-c(1)],MARGIN=2,FUN=skewness)
ggplot()+
  #Distribution of skewness values per column in train data
  geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per column',title="Distribution of skewness values per column in train and test dataset")

#Applying the function to find kurtosis values per row in train and test data.
train_kurtosis<-apply(train_df[,-c(1,2)],MARGIN=1,FUN=kurtosis)
test_kurtosis<-apply(test_df[,-c(1)],MARGIN=1,FUN=kurtosis)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per row',title="Distribution of kurtosis values per row in train and test dataset")

#Applying the function to find kurtosis values per column in train and test data.
train_kurtosis<-apply(train_df[,-c(1,2)],MARGIN=2,FUN=kurtosis)
test_kurtosis<-apply(test_df[,-c(1)],MARGIN=2,FUN=kurtosis)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per column',title="Distribution of kurtosis values per column in train and test dataset")

#Let us do Missing value analysis

#Finding the missing values in train data
missing_val<-data.frame(missing_val=apply(train_df,2,function(x){sum(is.na(x))}))
missing_val<-sum(missing_val)
missing_val

#Finding the missing values in test data
missing_val<-data.frame(missing_val=apply(test_df,2,function(x){sum(is.na(x))}))
missing_val<-sum(missing_val)
missing_val

#No missing values are present in both train and test data.

#Let us see correlation between the attributes

#Correlations in train data

#Correlations in train data
#convert factor to int
train_df$target<-as.numeric(train_df$target)
train_correlations<-cor(train_df[,c(2:202)])
train_correlations

#We can observed that the correlation between the train attributes is very small.

#Correlations in test data

#Correlations in test data
test_correlations<-cor(test_df[,c(2:201)])
test_correlations

#We can observed that the correlation between the test attributes is very small.

#Feature engineering

#Let us do some feature engineering by using

#1. Permutation importance
#2. Partial dependence plots

#Variable importance

#Variable importance is used to see top features in dataset based on mean decreses gini.

#Let us build simple model to find features which are more important.

#Split the training data using simple random sampling
train_index<-sample(1:nrow(train_df),0.75*nrow(train_df))
#train data
train_data<-train_df[train_index,]
#validation data
valid_data<-train_df[-train_index,]
#dimension of train and validation data
dim(train_data)
dim(valid_data)

#Training the Random forest classifier
set.seed(2732)
#convert to int to factor
train_data$target<-as.factor(train_data$target)
#setting the mtry
mtry<-floor(sqrt(200))
#setting the tunegrid
tuneGrid<-expand.grid(.mtry=mtry)
#fitting the ranndom forest
rf<-randomForest(target~.,train_data[,-c(1)],mtry=mtry,ntree=10,importance=TRUE)

#Variable importance
VarImp<-importance(rf,type=2)
VarImp

#Take away:
  
  #We can observed that the top important features are var_12, var_26, 
  #var_22,v var_174, var_198 and so on based on Mean decrease gini.

#Partial dependence plots

#Partial dependence plot gives a graphical depiction of the marginal effect of a variable on the class probability or classification.While feature importance shows what variables most affect predictions, but partial dependence plots show how a feature affects predictions.

#Let us calculate partial dependence plots on random forest

#Let us see impact of the main features which are discovered in the previous section by using pdp package.

#We will plot "var_13"
par.var_13 <- partial(rf, pred.var = c("var_13"), chull = TRUE)
plot.var_13 <- autoplot(par.var_13, contour = TRUE)
plot.var_13

#**Take away:

#1. The y_axis does not show the predictor value instead how the value 
#   changing with the change in given predictor variable.
#2. The blue shaded area indicates the level of confidence of 'var_13'
#3. On y-axis having a positive value means for that particular value of 
#   predictor variable it is less likely to predict the correct class and 
#   having a positive value means it has positive impact on predicting the 
#   correct class.

#We will plot "var_34"
par.var_34 <- partial(rf, pred.var = c("var_34"), chull = TRUE)
plot.var_34 <- autoplot(par.var_34, contour = TRUE)
plot.var_34

#**Take away:

#1. The y_axis does not show the predictor value instead how the value 
#   changing with the change in given predictor variable.
#2. The blue shaded area indicates the level of confidence of 'var_34'
#3. On y-axis having a positive value means for that particular value of 
#   predictor variable it is less likely to predict the correct class and 
#   having a positive value means it has positive impact on predicting the 
#   correct class.

##########################################################################

#Handling of imbalanced data

#Now we are going to explore 5 different approaches for dealing with imbalanced datasets.

#1. Change the performance metric
#2. Oversample minority class
#3. Undersample majority class
#4. Synthetic Minority Oversampling Technique(SMOTE)
#5. LightGBM

#Now let us start with simple Logistic regression model.

#Split the data
set.seed(689)
train.index<-sample(1:nrow(train_df),0.8*nrow(train_df))
#train data
train.data<-train_df[train.index,]
#validation data
valid.data<-train_df[-train.index,]
#dimension of train data
dim(train.data)
#dimension of validation data
dim(valid.data)
#target classes in train data
table(train.data$target)
#target classes in validation data
table(valid.data$target)


#Logistic Regression model

#Training and validation dataset

#Training dataset
X_t<-as.matrix(train.data[,-c(1,2)])
y_t<-as.matrix(train.data$target)
#validation dataset
X_v<-as.matrix(valid.data[,-c(1,2)])
y_v<-as.matrix(valid.data$target)
#test dataset
test<-as.matrix(test_df[,-c(1)])

#Logistic regression model
set.seed(667) # to reproduce results
lr_model <-glmnet(X_t,y_t, family = "binomial")
summary(lr_model)

#Cross validation prediction
set.seed(8909)
cv_lr <- cv.glmnet(X_t,y_t,family = "binomial", type.measure = "class")
cv_lr

#Minimum lambda
cv_lr$lambda.min
#plot the auc score vs log(lambda)
plot(cv_lr)

#We can observed that miss classification error increases as increasing the log(Lambda).

#Model performance on validation dataset
set.seed(5363)
cv_predict.lr<-predict(cv_lr,X_v,s = "lambda.min", type = "class")
cv_predict.lr

#Accuracy of the model is not the best metric to use when evaluating the imbalanced datasets as it may be misleading. So, we are going to change the performance metric.

#Confusion matrix
set.seed(689)
#actual target variable
target<-valid.data$target
#convert to factor
target<-as.factor(target)
#predicted target variable
#convert to factor
cv_predict.lr<-as.factor(cv_predict.lr)
confusionMatrix(data=cv_predict.lr,reference=target)

#ROC_AUC score and curve
set.seed(892)
cv_predict.lr<-as.numeric(cv_predict.lr)
roc(data=valid.data[,-c(1,2)],response=target,predictor=cv_predict.lr,auc=TRUE,plot=TRUE)


#Random Oversampling Examples(ROSE)

#It creates a sample of synthetic data by enlarging the features space of minority and majority class examples.

#Random Oversampling Examples(ROSE)
set.seed(699)
train.rose <- ROSE(target~., data =train.data[,-c(1)],seed=32)$data
#target classes in balanced train data
table(train.rose$target)
valid.rose <- ROSE(target~., data =valid.data[,-c(1)],seed=42)$data
#target classes in balanced valid data
table(valid.rose$target)

#Let us see how baseline logistic regression model performs on synthetic data points.

#Logistic regression model
set.seed(462)
lr_rose <-glmnet(as.matrix(train.rose),as.matrix(train.rose$target), family = "binomial")
summary(lr_rose)

#Cross validation prediction
set.seed(473)
cv_rose = cv.glmnet(as.matrix(valid.rose),as.matrix(valid.rose$target),family = "binomial", type.measure = "class")
cv_rose

#Minimum lambda
cv_rose$lambda.min
#plot the auc score vs log(lambda)
plot(cv_rose)

#Model performance on validation dataset
set.seed(442)
cv_predict.rose<-predict(cv_rose,as.matrix(valid.rose),s = "lambda.min", type = "class")
cv_predict.rose

#Confusion matrix
set.seed(478)
#actual target variable
target<-valid.rose$target
#convert to factor
target<-as.factor(target)
#predicted target variable
#convert to factor
cv_predict.rose<-as.factor(cv_predict.rose)
#Confusion matrix
confusionMatrix(data=cv_predict.rose,reference=target)

#ROC_AUC score and curve
set.seed(843)
#convert to numeric
cv_predict.rose<-as.numeric(cv_predict.rose)
roc(data=valid.rose[,-c(1,2)],response=target,predictor=cv_predict.rose,auc=TRUE,plot=TRUE)

#predict the model
#set.seed(6543)
#rose_pred<-predict(lr_rose,test,type='class')

#We can observed that ROSE model is performing well on imbalance data compare to baseline logistic regression.

#LightGBM

#LightGBM is a gradient boosting framework that uses tree based learning algorithms. We are going to use LightGBM model.

#Let us build LightGBM model

#Convert data frame to matrix
set.seed(5432)
X_train<-as.matrix(train.data[,-c(1,2)])
y_train<-as.matrix(train.data$target)
X_valid<-as.matrix(valid.data[,-c(1,2)])
y_valid<-as.matrix(valid.data$target)
test_data<-as.matrix(test_df[,-c(1)])


install.packages("lightgbm")
library(lightgbm)
#training data
lgb.train <- lgb.Dataset(data=X_train, label=y_train)
#Validation data
lgb.valid <- lgb.Dataset(data=X_valid,label=y_valid)

#Selecting best hyperparameters
set.seed(653)
lgb.grid = list(objective = "binary",
                metric = "auc",
                boost='gbdt',
                max_depth=-1,
                boost_from_average='false',
                min_sum_hessian_in_leaf = 12,
                feature_fraction = 0.05,
                bagging_fraction = 0.45,
                bagging_freq = 5,
                learning_rate=0.02,
                tree_learner='serial',
                num_leaves=20,
                num_threads=5,
                min_data_in_bin=150,
                min_gain_to_split = 30,
                min_data_in_leaf = 90,
                verbosity=-1,
                is_unbalance = TRUE)

set.seed(7663)
lgbm.model <- lgb.train(params = lgb.grid, data = lgb.train, nrounds =10000,eval_freq =1000,
                        valids=list(val1=lgb.train,val2=lgb.valid),early_stopping_rounds = 5000)

#lgbm model performance on test data
set.seed(6532)
lgbm_pred_prob <- predict(lgbm.model,test_data)
print(lgbm_pred_prob)
#Convert to binary output (1 and 0) with threshold 0.5
lgbm_pred<-ifelse(lgbm_pred_prob>0.5,1,0)
print(lgbm_pred)

set.seed(6521)
#feature importance plot
tree_imp <- lgb.importance(lgbm.model, percentage = TRUE)
lgb.plot.importance(tree_imp, top_n = 50, measure = "Frequency", left_margin = 10)

#We tried model with logistic regression,ROSE and lightgbm. 
#But,lightgbm is performing well on imbalanced data compared to other 
#models based on scores of roc_auc_score.

sub_df<-data.frame(ID_code=test_df$ID_code,lgb_predict_prob=lgbm_pred_prob,lgb_predict=lgbm_pred)
write.csv(sub_df,'submission.CSV',row.names=F)
head(sub_df)