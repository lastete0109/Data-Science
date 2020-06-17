library(caret)
library(modeldata)
library(e1071)
library(visdat)
library(recipes)
library(themis)
library(funModeling)
library(ROCR)
library(dplyr)
library(vip)
library(pdp)

data("attrition")

df = attrition %>% mutate_if(is.ordered, factor, ordered = FALSE)

set.seed(123)
train_index = createDataPartition(df$Attrition,p=0.8,list=FALSE)
train = df[train_index,]
test = df[-train_index,]

receta = recipe(Attrition ~ ., data = train) %>%
         step_nzv(all_predictors()) %>%
         step_center(all_numeric()) %>%
         step_scale(all_numeric()) %>%
         step_dummy(all_nominal(),-all_outcomes(),one_hot=FALSE) %>%
         step_smote(Attrition,over_ratio=0.8,neighbors = 5) 

set.seed(123)
cv_model_log =train(receta, data = train, 
                    method = "glm", family = "binomial",
                    trControl = trainControl(method = "cv", number = 10,verboseIter = T))

cv_model_log$results

pred = predict(cv_model_log,train,type="prob")[,2] 
pred.obj = prediction(pred, ifelse(train$Attrition=="Yes",1,0))
perf = performance(pred.obj, "sens", "spec")

# Nos permite visualizar metricas en funcion a diferentes puntos de corte (Default = 0.5)
plot(perf,
     avg="threshold",
     colorize=TRUE,
     lwd=1,
     main="ROC Curve w/ Thresholds",
     print.cutoffs.at=seq(0, 1, by=0.05),
     text.adj=c(0.3, -0.5),
     text.cex=0.75)

pred = predict(cv_model_log,train)
confusionMatrix(data=relevel(pred,ref="Yes"),
                reference=relevel(train$Attrition,ref="Yes"))

# No information rate : % de 0 inicial -> Que pasaria si predecimos todo como 0?

# Importancia de variables
vip(cv_model_log,num_features = 20)

# Graficas de dependencia parcial:
# Se necesita una funcion para retornar la probabilidad
pred.fun = function(object, newdata) {
  Yes = mean(predict(object, newdata, type = "prob")$Yes)
  as.data.frame(Yes)}

# Una sola variable
partial(cv_model_log,"EnvironmentSatisfaction",pred.fun = pred.fun) %>% 
  autoplot(rug=T) + ylim(c(0,1))

# Mas de una variable: se observa la interaccion: ver objeto pd -> Mejorar con ggplot2
pd = partial(cv_model_log,pred.var=c("OverTime","TotalWorkingYears"),
             pred.fun = pred.fun)

plotPartial(pd)        
