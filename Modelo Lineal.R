# Primer metodo : Regresion lineal

library(funModeling)
library(recipes)
library(dplyr)
library(ggplot2)
library(caret)
library(vip)
library(minerva)
library(ggrepel)

ames = AmesHousing::make_ames()

train_index = createDataPartition(ames$Sale_Price,p=0.8,list=FALSE)
ames_train = ames[train_index,]
ames_test = ames[-train_index,]

# Preprocesamiento y observando la correlacion (variables mas importantes)

preproc = recipe(Sale_Price ~ ., data = ames) %>%
          step_log(all_outcomes()) %>%
          step_other(all_nominal(),threshold = 0.1,other = "Other") %>%
          step_nzv(all_nominal()) %>%
          step_integer(matches("Qual|Cond|QC|Qu")) %>%
          step_center(all_numeric(), -all_outcomes()) %>%
          step_scale(all_numeric(), -all_outcomes()) %>%
          step_dummy(all_nominal(), -all_outcomes(), one_hot = FALSE)

ames_preproc = prep(preproc,ames) %>% bake(ames)

mic = mine(ames_preproc,
           master = grep("Sale_Price",names(ames_preproc)))

df_cor = data.frame(variable=row.names(mic$MIC),
                    mic=as.numeric(mic$MIC[,1]) %>% round(4),
                    mic.r2=as.numeric(mic$MICR2[,1]) %>% round(4)) %>%
         arrange(-mic) %>% 
         filter(variable!="Sale_Price")

df_cor %>% filter(mic>0.1) %>%
ggplot(aes(x=reorder(variable, mic),y=mic, fill=variable)) + 
  geom_bar(stat='identity') + 
  coord_flip() + 
  xlab("") + 
  ylab("Importancia de variable (MIC)") + 
  guides(fill=FALSE)


# Regresion lineal multiple

set.seed(123)
cv_model_lm = train(Sale_Price~., data = ames_train, 
                  method = "lm",
                  trControl = trainControl(method = "cv", number = 10,verboseIter = T))

# Regresion de componentes principales (PCR)
# Objetivo: minimiza la cantidad de dimensiones y evita el problema de multicolinealidad

set.seed(123)
cv_model_pcr = train(Sale_Price ~ ., data = ames_train, method = "pcr",
                     trControl = trainControl(method = "cv", number = 10,verboseIter = T),
                     preProcess = c("zv", "center", "scale"),tuneLength = 100)

cv_model_pcr$bestTune

cv_model_pcr$results %>% filter(ncomp==pull(cv_model_pcr$bestTune))

ggplot(cv_model_pcr)
                    

# Regresion por minimos cuadrados parciales (PLS)
# Objetivo: Evita la multicolinealidad, similar a PCR, pero maximiza la relacion de
# las dimensiones creadas con la variable respuesta

set.seed(123)
cv_model_pls= train(Sale_Price ~ ., data = ames_train, method = "pls",
                     trControl = trainControl(method = "cv", number = 10,verboseIter = T),
                     preProcess = c("zv", "center", "scale"),tuneLength = 30)

cv_model_pls$bestTune

cv_model_pls$results %>%
  dplyr::filter(ncomp == pull(cv_model_pls$bestTune))

ggplot(cv_model_pls)

summary(resamples(list(model1=cv_model_lm,model2=cv_model_pcr,model3=cv_model_pls)))    

# Importancia relativa (%) de variables
vip(cv_model_pls,num_features = 20,method="model")

# Dependencia parcial de las variables: Efecto que tiene la variable x en las predicciones
library(pdp)
partial(cv_model_pls,"First_Flr_SF",plot=T)
