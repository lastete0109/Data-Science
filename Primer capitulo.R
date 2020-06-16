library(dplyr)
library(ggplot2)
library(ggthemes)
library(caret)
library(h2o)
library(rsample)
library(modeldata)
library(visdat)
library(funModeling)
library(tibble)
library(recipes)
library(tictoc)

ames = AmesHousing::ames_raw 
names(ames) = make.names(names(ames),unique = T)
ames = ames %>% select(-c(Order,PID))

# Paso fundamental:  Particionar la data en train y test
# No tocar el test por nada del mundo, solo aplicar el procesamiento de limpieza
# Consejo: Particionar estratificadamente en funcion de y

train_index = createDataPartition(ames$SalePrice,p=0.8,list=FALSE)
ames_train = ames[train_index,]
ames_test = ames[-train_index,]

####################################################################################

# Feature engineering

# Target Engineering: Transformar la variable respuesta (util para modelos parametricos)
# Opcion 1: normalizar (logaritmo)
# Si la respuesta tiene valores negativos pequenos: log1p = log + 1 (argumento offset en step_log) 
# Si la respuesta tiene valores negativos grandes : Yeo Johnson (step_YeoJohnson)
# Opcion 2: Box cox -> mas flexible (step_boxCox)
# Ojo: Debemos usar el mismo lambda de Box Cox del train en el test

ames_recipe = recipe(SalePrice ~ ., data = ames_train) %>%
              step_log(all_outcomes())

# Valores perdidos: Se trata dependiendo del tipo
# Informative missingness: Tiene una causa de porque no figura
# Missigness at random: No tiene razon alguna (imputar o eliminar)

# Manual
ames %>%
  is.na() %>%
  reshape2::melt() %>%
  ggplot(aes(Var2, Var1, fill=value)) + 
  geom_raster()+ 
  coord_flip() +
  scale_y_continuous(NULL, expand = c(0, 0)) +
  scale_fill_grey(name = "", 
                  labels = c("Present", 
                             "Missing")) +
  xlab("Observation") +
  theme(axis.text.y  = element_text(size = 6))

# Con librerias
vis_miss(ames,cluster=TRUE)

# Imputacion: 

# Estimador estadistico
ames_recipe %>%
  step_medianimpute(Gr_Liv_Area)

# Por KNN
ames_recipe %>%
  step_knnimpute(all_predictors(), neighbors = 6)

# Random forest
ames_recipe %>%
  step_bagimpute(all_predictors())

# Feature filtering: Eliminar variables no informativas
# En un principio variables con nada o muy poca varianza

nearZeroVar(ames_train,saveMetrics = T) %>%
  rownames_to_column() %>%
  filter(nzv)

# Numeric feature engineering: Normalizar y estandarizar
# No afecta el rendimiento de los modelos no parametricos

recipe(SalePrice ~ ., data = ames_train) %>%
    step_YeoJohnson(all_numeric()) 

ames_recipe %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes())

# Categorical feature engineering: 
# Lumping: Agrupar variables con pocas frecuencias

lumping = recipe(SalePrice ~ ., data = ames_train) %>%
          step_other(Neighborhood, threshold = 0.01, 
                     other = "other") %>%
          step_other(`Screen Porch`, threshold = 0.1, 
                     other = ">0")

apply_2_training = prep(lumping, training = ames_train) %>%
                   bake(ames_train)

# One Hot Encoding vs Dummy Encoding: problema de colinealidad

recipe(SalePrice ~ ., data = ames_train) %>%
  step_dummy(all_nominal(), one_hot = TRUE)

# En caso de tener muchas variabes categoricas con muchos niveles cada uno,
# esto puede aumentar el nivel de features demasiado. Considerar label encoding.

# Label/Ordinal Encoding: Util para variables con un orden establecido (ordinales)
names(ames_train) %>% .[grepl("Qual",.)]
count(ames_train,`Ov Qual`)

recipe(SalePrice ~ ., data = ames_train) %>%
  step_integer(`Overall Qual`) %>%
  prep(ames_train) %>%
  bake(ames_train) %>%
  count(`Overall Qual`)

# Algunas recomendaciones:
# 1. Si usamos alguna transformacion (log o Box Cox), no centremos la data primero o
# hagamos operaciones que hagan la data no positiva. Alt: Yeo Johnson
# 2. Estandariza tus variables numericas y luego haz el one hot encoding.
# 3. Si hacemos lumping, previo al one hot encoding.
# 4. Hacer PCA solo en variables numericas.

# Ejemplo

# Eliminamos las variables con gran % de NAs 
status_train = df_status(ames_train)
var_na = status_train %>% filter(p_na>10) %>% .$variable
var_imp = status_train %>% filter(p_na<10 & p_na>0) %>% .$variable
ames_train_na = ames_train %>% select(-var_na)


blueprint = recipe(SalePrice ~ ., data = ames_train_na) %>%
            step_string2factor(all_nominal()) %>%
            step_knnimpute(all_predictors(),k=5) %>%
            step_nzv(all_nominal()) %>%
            step_integer(matches("Qual|Cond|QC|Qu")) %>%
            step_center(all_numeric(), -all_outcomes()) %>%
            step_scale(all_numeric(), -all_outcomes()) %>%
            step_dummy(all_nominal(), -all_outcomes(), one_hot = FALSE)


cv = trainControl(method = "repeatedcv", number = 7, repeats=5, verboseIter = TRUE)

hyper_grid = expand.grid(k = seq(2, 20, by = 1))

tic()
knn_fit2 = train(blueprint, data = ames_train_na, method = "knn", 
                 trControl = cv, tuneGrid = hyper_grid, metric = "RMSE",verbose=T)
toc()

# Dura alrededor de 2 horas :d

ggplot(knn_fit2)
