# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting Interest In Cross-sell

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup
# MAGIC 
# MAGIC 1. Loading favourite R libraries of choice
# MAGIC 2. Connect to databricks spark cluster via `{sparklyr}`
# MAGIC 3. Variables instantiated for notebook metadata

# COMMAND ----------

# so we don't need to use mflow::install_mlflow()
Sys.setenv(MLFLOW_BIN = '/databricks/python/bin/mlflow')
Sys.setenv(MLFLOW_PYTHON_BIN = '/databricks/python/bin/python')

library(data.table)
library(tidyverse)
library(sparklyr)
library(mlflow)
library(caret)

# sparklyr connection to databricks
sc <- spark_connect(method = "databricks")

# COMMAND ----------

# Databrick's dbutils functions do not yet have full coverage within R notebooks, therefore manually specifying paths
# NOTE: alternative would be to switch to python and append them to spark conf as a medium to pass through to R session
username <- "zachary.davies@databricks.com"
folder_path <- paste0("/Users/", username, "/experiments/")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Creating An `{mlflow}` Experiment
# MAGIC 
# MAGIC Firstly, we need to create an mlflow `experiment`.  
# MAGIC Experiments contain multiple machine learning `runs` and are awesome for organisation & productivity.  
# MAGIC 
# MAGIC ![](https://databricks.com/wp-content/uploads/2019/10/model-registry-new.png)
# MAGIC 
# MAGIC Here is what we need to do:
# MAGIC - Run cell below 
# MAGIC - It will create a Machine Learning Experiment that will be located in `/cross-selling-insurance`
# MAGIC - Alternatively, click link the cell created.
# MAGIC - Observe empty Experiment. We will be pushing Machine Learning models to it soon! 

# COMMAND ----------

exp_path <- paste0(folder_path, "cross-selling-insurance")

mlflow_set_tracking_uri("databricks")

# if an experiment already exists at the path, get the experiment ID
# otherwise, create experiment
experiment_id <- tryCatch(
  mlflow_create_experiment(exp_path),
  error = function(e) mlflow_get_experiment(name = exp_path)$experiment_id
)

# COMMAND ----------

displayHTML(
  glue::glue(
    "<b>Experiment ID</b>: {experiment_id}<br>",
    "<b>Experiment Location</b>: {exp_path}<br>",
    "<b>Experiment URL</b>: <a href='/#mlflow/experiments/{experiment_id}'>/#mlflow/experiments/{experiment_id}</a>"
  )
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Loading Dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC USE zdav_claims_cross_sell;
# MAGIC SHOW TABLES;

# COMMAND ----------

# pull all data into memory for this example - not using spark to train models for this exercise
data <- tbl(sc, "zdav_claims_cross_sell.car_and_health_insurance_cross_sell") %>%
  collect() %>%
  mutate_if(is.character, as.factor)

# COMMAND ----------

# quickly checking data
data %>% glimpse()

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA

# COMMAND ----------

# MAGIC %md
# MAGIC #### Correlation Matrix

# COMMAND ----------

# How to calculate correlations using `{sparklyr}` via `ml_corr()` -> great for datasets that are bigger than this demo
# df_corrs <- tbl(sc, "zdav_claims_cross_sell.car_and_health_insurance_cross_sell") %>%
#   select_if(is.numeric) %>%
#   ml_corr() %>%
#   mutate(col1 = colnames(.), .before = 1)

# Using data in memory (no spark)
df_corrs <- data %>%
  mutate_if(is.factor, as.integer) %>%
  cor() %>%
  data.frame() %>%
  rownames_to_column(var = "col1") 


# COMMAND ----------

df_corrs %>%
  pivot_longer(-col1, names_to = "col2") %>%
  mutate(value = round(value, 2)) %>%
  ggplot(aes(x = col2, y = col1, fill = value)) +
  geom_tile() + 
  geom_text(aes(label = value), size = 2.5) + 
  scale_fill_viridis_c() + 
  guides(fill = guide_colorbar(title = NULL, barheight = 20)) +
  theme_void() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1),
    axis.text.y = element_text(hjust = 1),
    plot.title = element_text(hjust = 0.5)
  ) +
  labs(
    title = "Correlation Heatmap",
    x = NULL,
    y = NULL,
    legend = NULL
  )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Is the dataset balanced?
# MAGIC Looks like it is biased towards `0` as it has ~267k records vs `1` with ~37k records.

# COMMAND ----------

# check dataset balance
data %>%
  group_by(Response) %>%
  count() %>%
  display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Rebalance dataset
# MAGIC Lets use the larger cohort as a goal to resample up until for each with `group_by()` + `resample_n()`.  
# MAGIC This will leave the larger cohort (`0`) unchanged and adjust `1`.

# COMMAND ----------

# rebalance dataset, take under-represented outcome and sample to same size as other
df_balanced <- data %>%
  group_by(Response) %>%
  sample_n(replace = T, size = sum(data$Response == 0)) %>% # or `size = sum(df_train$Response == 0)`
  ungroup() %>%
  mutate_if(is.character, as.factor)

# COMMAND ----------

# confirm changes worked
df_balanced %>%
  group_by(Response) %>%
  count() %>%
  display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### How do features look after rebalancing?
# MAGIC Using notebook widgets we can make our exploration interactive.

# COMMAND ----------

# create a widget for a few features
correlated_features <- list("Age", "Policy_Sales_Channel", "Claims_Amount", "Claims_Num", "Previously_Insured")
dbutils.widgets.dropdown(name = "Correlated Features", defaultValue = correlated_features[[1]], choices = correlated_features)

# COMMAND ----------

# updates dynamically via widget!
# can use `dbutils.widgets.get()` to fetch selected value - updates on dropdown trigger code to run
df_balanced %>%
  sample_n(10000) %>%
  select(Response, x = dbutils.widgets.get("Correlated Features")) %>%
  mutate(Response = as.character(Response)) %>%
  ggplot(aes(x = x, group = Response, fill = Response)) + 
  geom_histogram() +
  labs(
    title = paste("Distribution of", dbutils.widgets.get("Correlated Features"), "by Response"),
    x = dbutils.widgets.get("Correlated Features")
  ) + 
  theme_minimal() + 
  theme(
    plot.title = element_text(hjust = 0.5)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initial Baseline Model
# MAGIC 
# MAGIC Will now create a baseline machine learning model to determine a performance baseline (good or bad).  
# MAGIC The task will be then to beat this benchmark with more advanced models/methods.  
# MAGIC 
# MAGIC Using `{mlflow}` we can ensure reproducilibty and track all our work.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating Training & Testing Splits
# MAGIC Splitting data into two portions:
# MAGIC 1. **Train:** 80% of data, used to train model
# MAGIC 2. **Test:** 20% of data, used for inference
# MAGIC 
# MAGIC Will be removing `id` from the data for modelling.

# COMMAND ----------

# set.seed for reproducibility
set.seed(6789)

df_balanced_adj <- df_balanced %>%
  select(-id) %>%
  mutate_if(is.factor, as.integer) %>%
  mutate(Response = as.factor(Response))

# using caret to create partitions of 80%/20%
train_idx <- createDataPartition(df_balanced_adj$Response, p = 0.8, times = 1, list = FALSE)
df_train <- df_balanced_adj[train_idx, ] 
df_test <- df_balanced_adj[-train_idx, ]

# COMMAND ----------

# dataset sizes
glue::glue(
  "<b>all records</b>: {nrow(df_balanced)}<br>",
  "<b>training records</b>: {nrow(df_train)}<br>",
  "<b>testing records</b>: {nrow(df_test)}"
) %>% displayHTML()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Training with `{mlflow}`

# COMMAND ----------

run_caret_model_mlflow <- function(train, test, algo, params, experiment_id) {
  
  with(mlflow_start_run(experiment_id = experiment_id), {
  
    print(glue::glue("starting run with...\n{paste0('\t', names(params), ': ', params, collapse = '\n')}"))

    print(glue::glue("\ttraining on {nrow(train)} samples")) 

    # log params
    imap(params, ~mlflow_log_param(key = .y, value = .x))
    mlflow_log_param("algorithm", algo)
    
    # pre-process model
    pprocModel <- preProcess(train, method = c("scale"))

    # pre-process train/test sets
    pprocd_df_train <- predict(pprocModel, train)
    pprocd_df_test <- predict(pprocModel, test)

    # create trainControl
    tcont <- trainControl(method = "none")

    # train model
    model <- train(Response ~ ., data = pprocd_df_train, method = algo, trControl = tcont, tuneGrid = params)

    print(glue::glue("\ttraining complete"))
    print(glue::glue("\ttesting on {nrow(test)} samples"))

    # store model using {carrier} package
    predictor <- carrier::crate(
      .fn = ~{
        preproc_df <- stats::predict(!!pprocModel, .x)
        caret::predict.train(!!model, preproc_df)
       },
      !!pprocModel,
      !!model
    )
    
    mlflow_log_model(predictor, "model")

    # inference
    y_pred_train <- predict(model, pprocd_df_train)
    y_pred_test <- predict(model, pprocd_df_test)

    f1_train <- F_meas(y_pred_train, train$Response)
    f1_test <- F_meas(y_pred_test, test$Response)

    # log metrics
    mlflow_log_metric("f1_train", f1_train)
    mlflow_log_metric("f1_test", f1_test)

    print(glue::glue("\tinference complete"))

    print(glue::glue("F1 Scores:\nTrain:{f1_train}\nTest:{f1_test}"))
  
    mlflow_end_run()
    return(list(model = model, preproc_model = pprocModel))
    
  })
  
}

# COMMAND ----------

params <- data.frame(alpha = 0.5, lambda = 0.5)
glmnet <- run_caret_model_mlflow(train = df_train, test = df_test, algo = "glmnet", params = params, experiment_id = experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Improving the Model
# MAGIC 
# MAGIC lets see if we can improve the model by trying  <img src="https://machinelearningapplied.com/wp-content/uploads/2019/10/xgboost_logo.png" alt="xgBoost" width="60", height="50"/>  and trying various parameters...

# COMMAND ----------

params <- data.frame(
  nrounds = 25,
  max_depth = 2,
  eta = 0.1,
  gamma = 0.01,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)
xgb_tree <- run_caret_model_mlflow(train = df_train, test = df_test, algo = "xgbTree", params = params, experiment_id = experiment_id)

# COMMAND ----------

params <- data.frame(
  nrounds = 100,
  eta = 0.1,
  lambda = 0.1,
  alpha = 0.1
)
xgb_linear <- run_caret_model_mlflow(train = df_train[, -1], test = df_test[, -1], algo = "xgbLinear", params = params, experiment_id = experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Fetch Experiment Runs

# COMMAND ----------

runs <- mlflow_search_runs(experiment_id = experiment_id, order_by = "metrics.f1_test desc")
runs

# COMMAND ----------

best_run <- as.list(runs[1, ])
best_run

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Best Model & Predict

# COMMAND ----------

best_model <- mlflow_load_model(paste0(best_run$artifact_uri, "/model"))

# COMMAND ----------

mlflow_predict(data = df_train, model = best_model)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cleanup

# COMMAND ----------

# remove widgets made during notebook
dbutils.widgets.removeAll()

# try to delete experiment
tryCatch(
  mlflow_delete_experiment(experiment_id = experiment_id),
  error = function(e) NULL
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Finished!
# MAGIC [Previous Notebook - Data Exploration](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#notebook/4341258892203669/command/4341258892203670)