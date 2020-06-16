# Primer metodo : Regresion lineal

library(funModeling)
library(recipes)
library(dplyr)
library(ggplot2)
library(caret)
library(vip)

ames = AmesHousing::make_ames()

# Probando diferentes modelos 

set.seed(123)  
cv_model1 = train(form = Sale_Price ~ Gr_Liv_Area,  data = ames, 
                  method = "lm",trControl = trainControl(method = "cv", number = 10))

set.seed(123)
cv_model2 = train(Sale_Price ~ Gr_Liv_Area + Year_Built, data = ames, 
                  method = "lm",trControl = trainControl(method = "cv", number = 10))

set.seed(123)
cv_model3 = train(Sale_Price~., data = ames, 
                  method = "lm",
                  trControl = trainControl(method = "cv", number = 10,verboseIter = T),
                 )

summary(resamples(list(model1 = cv_model1, model2 = cv_model2, model3 = cv_model3)))

# 

