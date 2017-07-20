Project: Prediction of Accident prone locations
===============================================

Participants
============

Nandini, Goswami, FG-16-3007, goswamin, goswamin@iu.edu

Bhateja, Sarita, FG-16-3003, sbhateja, sbhateja@iu.edu

Guruprasad, Kavya, FG-16-3008, kavya_prasad, prasadk@iu.edu


Abstract
========

The paper discusses the prediction of road accidents. We have UK road accident dataset for year 2015. The problem is treated as a machine learning classification problem and the outcome is classified as severity of accident on a range of 1-3 where 1 is the least severe and 3 is the most severe. The results have been visualized using bar graphs, line chart, histogram etc. Also, the location of various accident prone areas have been plotted on map. The data related to latitude and longitude of such locations in the dataset helped to achieve it.

References
==========

https://gitlab.com/cloudmesh_fall2016/project-042/blob/master/report/report.bib

Deliverables
============
===================================================================
Instructions on how to import Databricks notebook and run the code

==================================================================

1) Go to https://community.cloud.databricks.com and Sign Up.
2) Select the Community Edition.
3) Sign Up for Databricks Community Edition by entering personal details.
4) Select Workspaces from the left-hand column, and under workspaces, choose shared and select import under the dropdown.
5) Create Clusters by choosing the clusters in the left-hand column, click on te create cluster tab and enter the cluster name and Apache Spark Version Spark 2.0(Auto-Updating, Scala 2.0)
 


Running the Web Application to obtain the Google Map Images
============================================================

1) Install python version 3.4 or higher.

2) Create a database in postgress sql application named “data”. Postgress sql is the database.

3) Activate the virtual environment using the command "source bin/activate" for Mac and Linux system and "source script/activate

4) Dependencies of the project are there in requirements.txt file. Use pip install -r requirements.txt commands to install those dependencies

5) To run our first migrations(This is to create schemas in our database), command is “python manage.py makemigrations”, second command is “python manage.py migrate”

6) You can start the server using this command- python manage.py runserver.

7) The application runs in this URL:- localhost:8000

8) Used python excel library to export data from Django to database.

9) Exporting the database should be done using the URL:- localhost:8000/import_sheet/ and the database should be in xls format.

We have used simple JQuery, JavaScript in the front-end of the application to show the desired output.

Please find the code for this in the code in the google_map file in code directory

Visualization
==============

We have included the code of histogram.py under the code directory. This histogram is used for visualisation.

The dataset and the code needs to be in the same folder for it to run.
The pre-requisites for this is numpy,pandas,matplotlib and pylab for running the code.
The code saves a histogram image in pdf format
 

