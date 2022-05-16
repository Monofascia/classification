rm(list=ls())

library(readr)
Algerian_forest_fires_dataset_UPDATE <- read_csv("C:/Users/Admin/Downloads/Algerian_forest_fires_dataset_UPDATE.csv", 
                                                 col_types = cols(Classes = col_factor(levels = c("fire", 
                                                                                                  "not fire")), DC = col_number(), 
                                                                  DMC = col_number(), FFMC = col_number(), 
                                                                  FWI = col_number(), RH = col_number(), 
                                                                  Rain = col_number(), Temperature = col_number(),
                                                                  ISI = col_number(), BUI = col_number(),
                                                                  Ws = col_number()))
library(knitr)
kable(head(Algerian_forest_fires_dataset_UPDATE))

X <- Algerian_forest_fires_dataset_UPDATE[,4:14]

kable(head(X))

X  <- na.omit(X)

kable(cor(X[,1:10]), digits = 3)


fireSet <- X[,-(5:9)]

str(fireSet)

kable(summary(fireSet[,1:5]))

library(caret)

set.seed(123)
training.samples = createDataPartition(fireSet$Classes, p = .8, list = FALSE)
train.data <- fireSet[training.samples, ]
test.data <- fireSet[-training.samples, ]

library(ggord)
library(psych)

ggplot(train.data, aes(Classes)) + geom_bar()

summary.factor(train.data$Classes)

summary(train.data[,1:5])

pairs.panels(train.data[1:5],gap=0, bg=c("blue","green")[train.data$Classes],pch=21)

ggp_Temp <- ggplot(train.data, aes(x = " ", y = Temperature)) +
  geom_boxplot() + 
  facet_grid(facets = ~train.data$Classes, as.table = TRUE)
ggp_Temp

ggp_RH <- ggplot(train.data, aes(x = " ", y = RH)) +
  geom_boxplot() + 
  facet_grid(facets = ~train.data$Classes, as.table = TRUE)
ggp_RH

ggp_Ws <- ggplot(train.data, aes(x = " ", y = Ws)) +
  geom_boxplot() + 
  facet_grid(facets = ~train.data$Classes, as.table = TRUE)
ggp_Ws

ggp_Rain <- ggplot(train.data, aes(x = " ", y = Rain)) +
  geom_boxplot() + 
  facet_grid(facets = ~train.data$Classes, as.table = TRUE)
ggp_Rain 

ggp_FWI <- ggplot(train.data, aes(x = " ", y = FWI)) +
  geom_boxplot() + 
  facet_grid(facets = ~train.data$Classes, as.table = TRUE)
ggp_FWI

library(MASS)

LDA <- lda(Classes~., data = train.data)
LDA



library(klaR)
partimat(train.data$Classes ~. , data=train.data[,1:5],method="lda")


predictions <- predict(LDA, test.data)
names(predictions)

mean(predictions$class==test.data$Classes)



attach(train.data)
control <- trainControl(method="repeatedcv", number=10, repeats=3)

# CART - classification and regression Trees
fit.cart <- train(Classes~., data=train.data, method="rpart", trControl=control)

# LDA
fit.lda <- train(Classes~., data=train.data, method="lda", trControl=control)

# SVM: support vector machine lineare
fit.svm <- train(Classes~., data=train.data, method="svmLinearWeights2", trControl=control)
#con kernel radiale
fit.svmR <- train(Classes~., data=train.data, method="svmRadial", trControl=control)

# kNN
fit.knn <- train(Classes~., data=train.data, method="knn", trControl=control)


# Random Forest - random decison trees and bagging
fit.rf <- train(Classes~., data=train.data, method="rf", trControl=control)

results <- resamples(list(CART=fit.cart, LDA=fit.lda, SVM_Linear=fit.svm, SVM_Radial=fit.svmR, KNN=fit.knn, RF=fit.rf))
summary(results)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)




# CART
pCART <- predict(fit.cart, test.data)
# LDA
pLDA <- predict(fit.lda, test.data)
# SVM Linear
pSVM_L <- predict(fit.svm, test.data)
# SVM Radial
pSVM_R <- predict(fit.svmR, test.data)
# KNN
pKNN <- predict(fit.knn, test.data)
# Random Forest
pRF <- predict(fit.rf, test.data)

# CART
cmCART <- confusionMatrix(test.data$Classes, pCART)
# LDA
cmLDA <- confusionMatrix(test.data$Classes, pLDA)
# SVM_Linear
cmSVM_L <- confusionMatrix(test.data$Classes, pSVM_L)
# SVM_Radial
cmSVM_R <- confusionMatrix(test.data$Classes, pSVM_R)
# KNN
cmKNN <- confusionMatrix(test.data$Classes, pKNN)
# RANDOM FOREST
cmRF <- confusionMatrix(test.data$Classes, pRF)



ModelType <- c( "CART", "LDA", "SVM_L", "SVM_R", "KNN", "Random forest")  

TrainAcc <- c(max(fit.cart$results$Accuracy), max(fit.lda$results$Accuracy), 
              max(fit.svm$results$Accuracy), max(fit.svmR$results$Accuracy), 
              max(fit.knn$results$Accuracy), max(fit.rf$results$Accuracy))



Train_misscl_Er <- 1 - TrainAcc

TestAcc <- c(cmCART$overall[1], cmLDA$overall[1], cmSVM_L$overall[1], 
              cmSVM_R$overall[1],
              cmKNN$overall[1],cmRF$overall[1])


Test_misscl_Er <- 1 - TestAcc

metrics <- data.frame(ModelType, TrainAcc, Train_misscl_Er, TestAcc, 
                      Test_misscl_Er)  # data frame with above metrics


kable(metrics, digits = 3)
