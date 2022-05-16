rm(list=ls())
library(car)
library(dplyr)
library(rattle)
library(MASS)
library(caret)
library(devtools)
# library(klaR)
library(psych)
library(mlbench)
library(caret)
library(AppliedPredictiveModeling)
library(readr)
library(ggord)
library(knitr)
library(LiblineaR)

##### Upload #####
setwd("C:/Users/Patrizio/Desktop/TUTTO/Ud'A/CLEBA/statistical learning/CLASSIFICATION")
Algerian_forest_fires<- read_delim("Algerian_forest_fires_dataset_UPDATE.csv",";", 
                                  escape_double = FALSE, 
                                  col_types = cols(Classes = col_factor(levels = c("not fire","fire"))), 
                                  trim_ws = TRUE)
sum(is.na(Algerian_forest_fires))

Algerian_forest_fires <- na.omit(Algerian_forest_fires)
attach(Algerian_forest_fires)
Algerian_forest_fires <- subset(Algerian_forest_fires, select = -c(day, month, year, FFMC, DMC, DC, ISI, BUI))
#Algerian_forest_fires = Algerian_forest_fires[-1:-3]
View(Algerian_forest_fires)
str(Algerian_forest_fires)
###### Explorative Analysis #########
ggplot(train, aes(Classes)) + geom_bar()
summary.factor(train$Classes)
pairs.panels(train[1:5],gap=0, bg=c("blue","green")[train$Classes],pch=21)
kable(summary(train[,1:5]))

###### Classification #########

##### Split dataset #######
training.samples = createDataPartition(Algerian_forest_fires$Classes, p = .8, list = FALSE)
train <- Algerian_forest_fires[training.samples, ]
test <- Algerian_forest_fires[-training.samples, ]

###### VIF #########
glm <- glm(train$Classes ~ ., data = train, family = binomial())
summary(glm)
kable(vif(glm))

###### LDA wine #########
lda.all <- lda(train$Classes ~  ., data = train) 
lda.all

predictions <- predict(lda.all, test)

ldahist(data = predictions$x[,1], g=Classes)

plot(predictions$x[,1])
text(predictions$x[,1], Classes, cex=0.7, pos =4, col='red')

mean(predictions$class==test$Classes)

lda.data <- cbind(train, predict(lda.all)$x)
class = predictions$class
ggplot(lda.data, aes(LD1, fill = class)+
  geom_point(aes(color = Type)))

ggplot(data = lda.data)+ geom_density(aes(LD1, fill = class), alpha = 0.1)


##### Compare classifiers ########
attach(train)
control <- trainControl(method="repeatedcv", number=10, repeats=3)

# CART - classification and regression Trees
fit.cart <- train(Classes~., data=train, method="rpart", trControl=control)

# LDA
fit.lda <- train(Classes~., data=train, method="lda", trControl=control)

# SVM: support vector machine lineare
fit.svm <- train(Classes~., data=train, method="svmLinearWeights2", trControl=control)
#con kernel radiale
fit.svmR <- train(Classes~., data=train, method="svmRadial", trControl=control)

# kNN
fit.knn <- train(Classes~., data=train, method="knn", trControl=control)
# k è un ordine di vicinato 

# Random Forest - random decison trees and bagging
fit.rf <- train(Classes~., data=train, method="rf", trControl=control)

results <- resamples(list(CART=fit.cart, LDA=fit.lda, SVM_Linear=fit.svm, SVM_Radial=fit.svmR, KNN=fit.knn, RF=fit.rf))
summary(results)
scales <- list(x=list(relation="free"), 
               y=list(relation="free"))
bwplot(results, scales=scales)
splom(results)

##### Capacità previsiva #######
# comparision

# CART
pCART <- predict(fit.cart, test)
# LDA
pLDA <- predict(fit.lda, test)
# SVM Linear
pSVM_L <- predict(fit.svm, test)
# SVM Radial
pSVM_R <- predict(fit.svmR, test)
# KNN
pKNN <- predict(fit.knn, test)
# Random Forest
pRF <- predict(fit.rf, test)

# CART
cmCART <- confusionMatrix(test$Classes, pCART)
# LDA
cmLDA <- confusionMatrix(test$Classes, pLDA)
# SVM_Linear
cmSVM_L <- confusionMatrix(test$Classes, pSVM_L)
# SVM_Radial
cmSVM_R <- confusionMatrix(test$Classes, pSVM_R)
# KNN
cmKNN <- confusionMatrix(test$Classes, pKNN)
# RANDOM FOREST
cmRF <- confusionMatrix(test$Classes, pRF)


# put all of this together in a table.
ModelType <- c( "CART", "LDA", "SVM_L", "SVM_R", "KNN", "Random forest")  
# Training classification accuracy
TrainAcc <- c(max(fit.cart$results$Accuracy), max(fit.lda$results$Accuracy), 
              max(fit.svm$results$Accuracy), max(fit.svmR$results$Accuracy), 
              max(fit.knn$results$Accuracy), max(fit.rf$results$Accuracy))
# confronto dei valori massimi di accuracy

# Test misclassification error
Train_misscl_Er <- 1 - TrainAcc
# 1 - ERRORE DI ACCURACY = ERRORE DI CLASSIFICAZIONE

# validation classification accuracy
ValidAcc <- c(cmCART$overall[1], cmLDA$overall[1], cmSVM_L$overall[1], 
              cmSVM_R$overall[1],
              cmKNN$overall[1],cmRF$overall[1])

# Validation misclassification error or out-of-sample-error
Valid_misscl_Er <- 1 - ValidAcc

metrics <- data.frame(ModelType, TrainAcc, Train_misscl_Er, ValidAcc, 
                      Valid_misscl_Er)  # data frame with above metrics

# print table using kable() from knitr package
knitr::kable(metrics, digits = 3)

# SVM_L
##### Trying different costs #######
tune.out <- tune(svm, Classes ~., data=train, kernel='linear', 
                 ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)), scale = FALSE)
summary(tune.out)
yhat <- predict(tune.out$best.model, test) # con bestmodel prenod il migliore
confusionMatrix(yhat, test$Classes)
# l'accuratezza è del 98%
# l'indice k è pari a ?
# il test di Mcnemar tende a vedere se questa classificazione dell'indice kappa è 
# coerente perchè tale indice valuta più la coerenza che la correttezza delle variabili
# discriminanti rispetto ai gruppi.
# la spiegazione dei successivi indici DEVE ESSERE INCLUSA NEL REPORT!!"!"
--------------------------------------------------------------------------------
  ##### ROUTINE DA ESEGUIRE PRIMA DI POTER LANCIARE IL COMANDO decisionplot() #####

decisionplot <- function(model, data, class = NULL, predict_type = "class",
                         resolution = 100, showgrid = TRUE, ...) {
  
  if(!is.null(class)) cl <- data[,class] else cl <- 1
  data <- data[,1:2]
  k <- length(unique(cl))
  
  plot(data, col = as.integer(cl)+1L, pch = as.integer(cl)+1L, ...)
  
  # make grid
  r <- sapply(data, range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as.data.frame(g)
  
  ### guess how to get class labels from predict
  ### (unfortunately not very consistent between models)
  p <- predict(model, g, type = predict_type)
  if(is.list(p)) p <- p$class
  p <- as.factor(p)
  
  if(showgrid) points(g, col = as.integer(p)+1L, pch = ".")
  
  z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
  contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
          lwd = 2, levels = (1:(k-1))+.5)
  
  invisible(z)
}
---------------------------------------------------------------------------------------
  

svm_L_cost <- svm(Classes ~., data=train, kernel='linear', cost=0.1, scale=FALSE)
decisionplot(svm_L_cost, test, class = "Classes", main = "SVM (linear)")
# questo comando si crea una griglia xy su cui andare a fare una previsione per
# migliorare la previsione
# non va il plot
yhatSVM.LIN <- predict(svm_L_cost, test)
confusionMatrix(test$Classes,yhatSVM.LIN)

####### end confusion ########
######### plot ##########

# plot accuracy on training data...............non serve
model_compare <- data.frame(Model=ModelType, Accuracy=TrainAcc)
ggplot(aes(x=Model, y=Accuracy), data=model_compare) +
  geom_bar(stat='identity', fill = 'light green') +
  ggtitle('Comparative Accuracy of Models on Cross-Validation Data') +
  xlab('Models') +
  ylab('Overall Accuracy')
