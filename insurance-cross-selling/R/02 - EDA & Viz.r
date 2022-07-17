# Databricks notebook source
# MAGIC %md
# MAGIC ## Explore, Analyse, Visualise
# MAGIC 
# MAGIC Digging into the data briefly, demonstrating some of the visualisation options available within a Databricks notebook.

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

# MAGIC %sql
# MAGIC SHOW DATABASES

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL
# MAGIC We can begin analysis using SQL (using `%sql`) because we have already registered a database, this can be viewed via the 'data' menu on the left-hand side

# COMMAND ----------

# MAGIC %sql
# MAGIC USE zdav_claims_cross_sell;
# MAGIC SHOW TABLES;

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

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM zdav_claims_cross_sell.car_and_health_insurance_cross_sell
# MAGIC LIMIT 100;

# COMMAND ----------

# MAGIC %md
# MAGIC #### Is the annual premium influenced by age of the customer?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT annual_premium, age
# MAGIC FROM zdav_claims_cross_sell.car_and_health_insurance_cross_sell
# MAGIC WHERE annual_premium > 500

# COMMAND ----------

# MAGIC %md
# MAGIC #### Identify the cross-sell opportunity for each region

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW cross_sell_region_agg AS
# MAGIC SELECT
# MAGIC   region_code,
# MAGIC   count(*) AS region_prospects,
# MAGIC   sum(annual_premium) AS total_annual_premium,
# MAGIC   sum(claims_amount) AS total_claims_amount,
# MAGIC   sum(claims_num) AS total_claims_count
# MAGIC FROM zdav_claims_cross_sell.car_and_health_insurance_cross_sell
# MAGIC GROUP BY region_code
# MAGIC ORDER BY region_code

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   region_code,
# MAGIC   region_prospects
# MAGIC FROM cross_sell_region_agg
# MAGIC ORDER BY region_prospects

# COMMAND ----------

# MAGIC %md
# MAGIC ### Switching gear, lets start using R...
# MAGIC 
# MAGIC Choose your favourite data wrangling and visualisation libraries.  
# MAGIC If they are unavailable then install them to the attached cluster using these [instructions](https://docs.databricks.com/libraries/cluster-libraries.html#install-a-library-on-a-cluster).

# COMMAND ----------

# lets use sparklyr here
library(data.table)
library(tidyverse)
library(sparklyr)

# connect to databricks spark session
sc <- spark_connect(method = "databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Collect basic statistics for each numeric feature

# COMMAND ----------

cols <- tbl(sc, "zdav_claims_cross_sell.car_and_health_insurance_cross_sell") %>% colnames()
numerics_cols <- setdiff(c("id", "Gender", "Vehicle_Age", "Vehicle_Damage", "Region_Code", "Driving_License", "Previously_Insured", "Policy_Sales_Channel", "Response"), cols)

tbl(sc, "zdav_claims_cross_sell.car_and_health_insurance_cross_sell") %>%
  sdf_describe() %>%
  collect() %>%
  display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculate the percentage split for categorical features

# COMMAND ----------

# categorical columns
prc_cols <- c('Vehicle_Age', 'Gender', 'Vehicle_Damage', 'Driving_License', 'Previously_Insured')

# pivot longer and aggregate
prc_col_freqs <- tbl(sc, "zdav_claims_cross_sell.car_and_health_insurance_cross_sell") %>%
  select(prc_cols) %>%
  mutate_all(as.character) %>%
  pivot_longer(everything()) %>%
  group_by(name, value) %>%
  summarise(n = n()) %>%
  collect() %>%
  group_by(name) %>%
  mutate(prc = round(prop.table(n), 2)) %>%
  ungroup() %>%
  arrange(name, prc)

# COMMAND ----------

# we can use `displayHTML()` to with `knitr::kable()` as an alternative to `display()`
prc_col_freqs %>%
  knitr::kable(format = "html") %>%
  displayHTML()

# COMMAND ----------

# use ggplot as you normal would outside databricks notebooks
categories_plot <- ggplot(prc_col_freqs, aes(x = value, y = prc, fill = value)) + 
  geom_col() +
  geom_text(aes(label = paste0(prc, "%")), hjust = 0) + 
  facet_wrap(vars(name), scales = "free", ncol = 2) +
  coord_flip() + 
  scale_y_continuous(expand = expand_scale(0, c(0, 0.1))) +
  theme_bw() +
  theme(legend.position = "none") + 
  labs(y = "")

categories_plot

# COMMAND ----------

# MAGIC %md
# MAGIC #### `{ggplot2}` works without explicit `collect()`!

# COMMAND ----------

# how do claims vary by region
region_claims <- tbl(sc, "cross_sell_region_agg") %>%
  # no `collect()` step here!
  ggplot(aes(x = total_claims_count, y = total_annual_premium, size = region_prospects)) +
  geom_point(colour = "skyblue") +
  geom_text(aes(label = region_code), size = 3, vjust = -1) +
  scale_x_log10(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma, n.breaks = 10) + 
  scale_radius(range = c(1, 10)) + 
  labs(
    y = "Total Annual Premium",
    x = "Total Claims Count",
    title = "Region-wise Claims & Annual Premium"
   ) + 
  theme_minimal() + 
  theme(legend.position = "none")

region_claims

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cleanup

# COMMAND ----------

# MAGIC %sql
# MAGIC /* Remove view created at start of notebook */
# MAGIC DROP VIEW IF EXISTS cross_sell_region_agg

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Next Steps
# MAGIC [Previous Notebook - Data Preparation](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#notebook/4341258892203619/command/4341258892203668)  
# MAGIC [Next Notebook - Training A Model](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#notebook/4341258892203307/command/4341258892203308)