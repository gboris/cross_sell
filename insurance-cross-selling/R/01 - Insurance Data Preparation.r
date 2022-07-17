# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Preparation

# COMMAND ----------

# MAGIC %md ##### Quick tips before we get started..
# MAGIC * Attach the notebook to your cluster
# MAGIC * To run a cell, click Ctrl + Enter
# MAGIC * Press Escape, and then **h** to see keyboard shortcuts
# MAGIC * Click on the cell dashboard icon to add a chart to the Dashboard

# COMMAND ----------

# MAGIC %md
# MAGIC #### Explore Data Using Multiple Languages
# MAGIC Databricks notebooks support [multiple languages](https://docs.databricks.com/notebooks/notebooks-use.html#mix-languages).  
# MAGIC Each notebook has a default language.  
# MAGIC Use built-in 'magic' commands to switch over to another language or command.  
# MAGIC 
# MAGIC | Magic command |                  Purpose                       |
# MAGIC |---------------|------------------------------------------------|
# MAGIC | %md           |       Document using Markdown                  |
# MAGIC | %sql          |       Run a SQL command                        |
# MAGIC | %python       |       Run python code                          |
# MAGIC | %r            |       Run R code                               |
# MAGIC | %scala        |       Run Scala code                           |
# MAGIC | %fs           |       Shortcut for [dbutils.fs()](https://docs.databricks.com/dev-tools/databricks-utils.html#file-system-utilities)                     |
# MAGIC | %sh           |       Run a Bash shell command                 |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading Libraries
# MAGIC Databricks Notebooks supports R out of the box and the [Databricks runtime](https://docs.databricks.com/runtime/index.html) already has many of the R libaries you may use day to day.
# MAGIC 
# MAGIC For those that aren't its [easy to install them on the cluster](https://docs.databricks.com/libraries/cluster-libraries.html).

# COMMAND ----------

library(data.table)
library(tidyverse)
library(SparkR)
library(sparklyr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading Data
# MAGIC Databricks notebooks use the standard R you are familar with and therefore you can choose how to read in your data, here we will use `fread` from `{data.table}`.  
# MAGIC Then we will have a quick `glimpse` using a function from the `{tidyverse}`.

# COMMAND ----------

# read in baseline data and have a quick glimpse
# source: https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction?select=train.csv
raw_data <- fread("/Workspace/Repos/zachary.davies@databricks.com/anz-sa-demos/insurance-cross-selling/train.csv")

# COMMAND ----------

glimpse(raw_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Adjustments
# MAGIC 
# MAGIC Adding some additional features, specifically regarding the annual claim amount and annual premiums.  
# MAGIC **NOTE**: These features do not come with the original data and are generated to help the demo skip some complexity

# COMMAND ----------

# functions to generate additional features

gen_annual_claim_ammount <- function(df, max_amount = 120000) {
  dist <- rnorm(mean = 0, sd = 1.75, n = nrow(df))
  dist <- dist * (max_amount / max(dist))
  dist <- pmax(dist, 10)
  dist <- dist + (dist * runif(min = -0.2, max = 0.2, n = length(dist)))
  mask <- df[["Response"]] == 1
  dist[mask] <- dist[mask] * runif(min = 1.8, max = 3.0, n = sum(mask))
  dplyr::mutate(df, Claims_Amount = as.integer(dist))
}

gen_num_of_claims <- function(df, claim_avg = 40) {
  nonzero_mask <- df[["Claims_Amount"]] > 0
  nonzero <- df[["Claims_Amount"]][nonzero_mask]
  num_claims <- (nonzero / claim_avg) * runif(min = 1/3, max = 3, n = length(nonzero))
  num_claims <- pmax(num_claims - min(num_claims), 1.0)
  dplyr::mutate(df, Claims_Num = as.integer(num_claims))
}

adjust_annual_premium <- function(df, premium_avg = 40 * 52) {
  m <- mean(df[["Annual_Premium"]])
  factor <- m / premium_avg
  dplyr::mutate(df, Annual_Premium = as.integer(Annual_Premium / factor))
}

# COMMAND ----------

# applying adjustments
adj_data <- raw_data %>%
  gen_annual_claim_ammount() %>%
  gen_num_of_claims() %>%
  adjust_annual_premium()

# COMMAND ----------

# review data adjustments
display(head(adj_data, 1000))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Storing Data
# MAGIC 
# MAGIC Databricks supports registering DataFrames as tables into databases.  
# MAGIC First a database must be created, here we can switch to a `%sql` chunk to do so.
# MAGIC 
# MAGIC Below we create the database and then show two methods to write our DataFrame:
# MAGIC 1. Using `{SparkR}` package
# MAGIC 2. Using `{sparklyr}` package

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP DATABASE IF EXISTS zdav_claims_cross_sell CASCADE;
# MAGIC CREATE DATABASE IF NOT EXISTS zdav_claims_cross_sell;

# COMMAND ----------

# write to database SparkR
# we are not using {arrow} here, but we should for free performance
# requires copying to spark cluster first via `SparkR::createDataFrame()`
SparkR::createDataFrame(data = adj_data) %>%
  SparkR::saveAsTable("zdav_claims_cross_sell.car_and_health_insurance_cross_sell", source = "delta", mode = "overwrite")

# COMMAND ----------

# connection established for sparklyr
sc <- spark_connect(method = "databricks")

# write to database sparklyr
# we are not using {arrow} here, but we should for free performance
# requires copying table to spark cluster first via `sparklyr::copy_to()`
sparklyr::copy_to(sc, adj_data, "adj_data", overwrite = TRUE)
sparklyr::spark_write_table(x = tbl(sc, "adj_data"), name = "zdav_claims_cross_sell.car_and_health_insurance_cross_sell", mode = "overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Next Steps
# MAGIC 
# MAGIC [Previous Notebook - Introduction](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#notebook/4341258892203619/command/4341258892203668)  
# MAGIC [Next Notebook - Data Exploration](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#notebook/4341258892203669/command/4341258892203670)