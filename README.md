Background

This repository contains code written for the "Cabspotting" data. 
The project was built on databricks using Pyspark on top of Spark framework.

Databricks the platform by Microsoft Azure comes on top of Azure cloud.
Apache Spark is built for efficient data processing and parallelising the ML processes on multiple clusters and multi-cores.

Only limitation comes in the process with usage of pandas dataframe and scikit learn based libraries which can not be parallelised.
Here, though we have used these libraries.
The author implore usage of MlLib library for Machine Learning projects with production ready codes.

Alternatively. containers for tensorflow can be set up on docker container and used efficiently.

Code structure

The code is comprised of three main files inside the code folder:
•Import_data_Db.py: Methods for loading, cleaning and pre-processing the tar.gz file to final text file with all Taxi's data
•data.py: Creating final dataset used for Visualisations, ML model and the questions. Data is for one major location Caltrain in SF
•Q2_ML_PRED.py: Linear regression based predictor model which is super efficient for taxi data and takes 1 min to create a basic model
•Q1_CO2_POT.py: Calculation of potential for CO2 emission reduction using current data

Getting started

This implementation is based on community databricks platform which provides 6 GB memory and 2 clusters.

Only standard python libraries available on databricks cloud are used.
Advantage being that code does not relies on non-standard libraries and is production ready.
