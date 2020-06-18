library(recipes)
library(glmnet)
library(caret)
library(vip)
library(AmesHousing)

ames = AmesHousing::make_ames()
set.seed(123)
train_index = createDataPartition(ames$Sale_Price,p=0.8,list=FALSE)
train = ames[train_index,]
test = ames[-train_index,]

x = model.matrix(Sale_Price~.,train)[,-1]
y = log(train$Sale_Price)

set.seed(123)
cv_glmnet = train(x = x,y = y,method = "glmnet",
                  preProc = c("zv", "center", "scale"),
                  trControl = trainControl(method = "cv", number = 10),
                  tuneLength = 10)
cv_glmnet$bestTune

ggplot(cv_glmnet)

pred = predict(cv_glmnet, x)
RMSE(exp(pred), exp(y))

data("attrition")
df = attrition %>% mutate_if(is.ordered, factor, ordered = FALSE)

set.seed(123)
train_index = createDataPartition(df$Attrition,p=0.8,list=FALSE)
train = df[train_index,]
test = df[-train_index,]

set.seed(123)
penalized_mod = train(Attrition ~ ., data = train, method = "glmnet",
                      family = "binomial", preProc = c("zv", "center", "scale"),
                      trControl = trainControl(method = "cv", number = 10,verboseIter = T),
                      tuneLength = 10)

penalized_mod$bestTune
penalized_mod$results %>% filter(alpha==pull(penalized_mod$bestTune,"alpha") &
                                 lambda==pull(penalized_mod$bestTune,"lambda"))
pred = predict(penalized_mod,train)
confusionMatrix(pred,train$Attrition)


