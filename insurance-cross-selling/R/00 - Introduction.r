# Databricks notebook source
# MAGIC %sh ls -la /Workspace/Repos

# COMMAND ----------

# MAGIC %md
# MAGIC # Cross-Sell Car Insurance to Health Insurance Customers
# MAGIC ### ETL, Machine Learning, and Model Management
# MAGIC 
# MAGIC This collection of notebooks walks through:
# MAGIC 1. [Extract-Tranform-Load (ETL) of source data](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#notebook/4341258892203358/command/4341258892203618)
# MAGIC 2. [Exploring, Analysing & Visualising Data](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#notebook/4341258892203669)
# MAGIC 3. [Machine Learning Model Development with MLFlow](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#notebook/4341258892203307/command/4341258892203308)
# MAGIC 
# MAGIC *Notebooks are translated to R by zachary.davies@databricks.com*  
# MAGIC *Original workshop series by yan.moiseev@databricks.com*

# COMMAND ----------

# MAGIC %md
# MAGIC ### Requirements
# MAGIC - DBR 7.5 ML 
# MAGIC - Install following from CRAN for Cluster (xgboost, mlflow, kableExtra, glmnet, e1071, carrier, tidyverse, data.table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset
# MAGIC 
# MAGIC We have highly anonymised & modified data for ~40 thousands customers of this specific insurer seeking to enter car insurance business
# MAGIC 
# MAGIC | **Variable**           | **Definition**                                                                                                                  |
# MAGIC |--------------------|-----------------------------------------------------------------------------------------------------------------------------|
# MAGIC | id                 | Unique ID for the customer                                                                                                  |
# MAGIC | Gender               | Gender of the customer                                                                                                      |
# MAGIC | Age                | Age of the customer                                                                                                         |
# MAGIC | Driving_License    | **0**: Customer does not have DL<br>**1**: Customer already has DL                                                                  |
# MAGIC | Region_Code        | Unique code for the region of the customer                                                                                  |
# MAGIC | Previously_Insured | **1**: Customer already has Vehicle Insurance<br>**0**: Customer doesn't have Vehicle Insurance                                     |
# MAGIC | Vehicle_Age        | Age of the Vehicle                                                                                                          |
# MAGIC | Vehicle_Damage     | **1**: Customer got his/her vehicle damaged in the past<br>**0**: Customer didn't get his/her vehicle damaged in the past.          |
# MAGIC | Annual_Premium     | The amount customer needs to pay as premium in the year                                                                     |
# MAGIC | PolicySalesChannel | Anonymized Code for the channel of outreaching to the customer<br>ie. Different Agents, Over Mail, Over Phone, In Person, etc. |
# MAGIC | Vintage            | Number of Days, Customer has been associated with the company                                                               |
# MAGIC | Claims_Amount*      | Amount of money customer has claimed in the last year                                                                       |
# MAGIC | Claims_Num*        | Number of claims customer has submitted throughout their lifetime                                                           |
# MAGIC | **Response**\**          | **1**: Customer is interested<br>**0**: Customer is not interested                                                                  |
# MAGIC 
# MAGIC **\*** Features are created in notebook 01 for this walkthrough.  
# MAGIC **\*\*** `Response` is our target variable, all other columns are feature.
# MAGIC 
# MAGIC **Source**: [Health Insurance Cross Sell Prediction (Kaggle Dataset)](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Objective
# MAGIC 
# MAGIC Given all of the feature columns, predict `Response` of the customer, whether they would be interested in car insurance or not.

# COMMAND ----------

# MAGIC %md
# MAGIC [Next Notebook - Data Preparation](./insurance-cross-selling/R/01 - Insurance Data Preparation)