# Mathematical Modeling Project 1

Welcome to our first data science project for MAP 4103! The aim of this project was to learn some new machine learning techniques, such as classification, clustering, resampling, or deep learning methods, from the text "An Introduction to Statistical Learning." For this project we chose to focus on clustering and classification techniques. For the clustering technique, we chose to use the k-nearest neighbors (KNN) algorithm, and for the classification models we went with support vector machine (SVM), decision tree, and random forest models.

A secondary goal of this project was to learn how to collaborate and share code and ideas using GitHub. Although we primarily used GitHub’s most basic functions for this project, it provided a good introduction to version control and teamwork.

### Introduction

In a previous semester, I was in the market for a used car. As such, I spent a considerable amount of time reading and learning about used cars. I developed something of an intuition for the price of a car based on the mileage, and the make and model of the vehicle. So, I thought it may be interesting to see if I could create a predictive model for the price of a used car. For our project, we are using the data set "Used Car Listings: Features and Price Prediction" by Tugberk Karan on Kaggle. This dataset contains data from many cars.com listings including the mileage, brand, model and various other features which lent itself nicely to developing a predictive model for the price of a used car in the previous semester.

This semester, we hope to employ some different models to make other predictions about the data, evaluate the models' performance and compare them to each other

Here is a link to the dataset:

https://www.kaggle.com/datasets/tugberkkaran/used-car-listings-features-and-prices-carscom.

### Instructions

#### Running the Project  

To run this project on your own machine, follow the steps below:  

##### 1. Install Required Software  

Ensure you have the following installed:  

- R can be downloaded from [CRAN](https://cran.r-project.org/)
- Jupyter Notebook can be installed via pip in the command line 
- IRKernel is needed to run R scripts in Jupyter Notebooks
- Optionally you can install RStudio from https://posit.co/download/rstudio-desktop/.

###### Installing Jupyter Notebook  

If you don’t have Jupyter installed, install it using pip. On windows, open command prompt and type the following:  

```bash
pip install notebook
```

Then press enter.

##### 2. Install the R Kernel for Jupyter

To enable Jupyter to run R scripts, install the IRkernel package in R:

```r
install.packages("IRkernel")
IRkernel::installspec(user = FALSE)
```

##### 3. Install Required R Packages

This project may require additional R libraries. If a package is missing, install it using the following in R:

```r
install.packages("packageName", dependencies = TRUE)
```
Using dependencies = TRUE ensures that all required dependencies are installed, reducing troubleshooting issues.

##### 4. Download the Dataset

This project uses the dataset Used Car Listings: Features and Price Prediction from Kaggle.

- Download the dataset from Kaggle.
- Place the dataset in an appropriate directory on your computer.
- Update file paths in the scripts to match the location of the dataset on your machine.

##### 5. Running the Project  

Once everything is set up, follow the appropriate steps depending on the file type:  

###### Running Jupyter Notebook Files (`.ipynb`) 

- Open a terminal or command prompt.  
- Launch Jupyter Notebook:  

```bash
   jupyter notebook
```

Open the notebook file (.ipynb) and run the code cells in order to execute the analysis.

###### Running R Script Files (.R)
If the script is an R file (.R), you can run it using R or RStudio:

Using R in the terminal type:

```bash
Rscript script_name.R
```

Using RStudio:
- Open RStudio.
- Navigate to the directory where the script is stored.
- Open the .R file and click Run or execute it line by line to execute the analysis.


### Analysis

The analysis presented here compares the performance of three classification models—Decision Tree, Random Forest, and Logistic Regression which were used to predict the first_owner variable. The dataset includes several important features such as brand, year, mileage, engine_size, fuel_type, and others. We selected the features used in each model with emphasis on reducing computational issues related to high-dimensional data. For example we excluding the features model and transmission type due to there being too many factor levels. The models were evaluated using the confusion matrix, which shows the number of true positives, true negatives, false positives, and false negatives.

##### Decision Tree:

The Decision Tree model was fitted using the rpart function. The performance metrics derived from this matrix include accuracy, sensitivity, specificity, and Kappa, all of which were reported in the results. With an accuracy of 76.15%, we see that the Decision Tree has potential to correctly classify instances.

##### Random Forest:

The Random Forest model was fitted using the randomForest function. It achieved an accuracy of 75.68%, demonstrating solid performance in classifying instances.


##### Logistic Regression:

The Logistic Regression model was fitted using glmnet, with lambda selected via cross-validation. It achieved an accuracy of 74.31%, the lowest among the models, but still showed reasonable classification performance.


### Conclusions:

Random Forest performed the best overall, with a balance of high sensitivity, specificity and accuracy, making it the most reliable model. The Decision Tree performed well also but showed slightly lower sensitivity compared to Random Forest. Logistic Regression exhibited the lowest overall accuracy but had the highest specificity, making it useful for scenarios where detecting negative instances is more important.

##### Future Improvements:

If more time and resources were available, the following improvements could be made:

- Hyperparameter Tuning: For Random Forest, tuning parameters like the number of trees (ntree) and the maximum tree depth (max_depth) could improve model performance.
- Feature Engineering: Further exploration of the features could yield additional variables or transformations that better capture the relationships in the data. For example, polynomial features or interactions between variables like mileage and year could be tested.
- Cross-Validation: Applying k-fold cross-validation instead of a single training/test split could help in assessing model performance more robustly and reduce the risk of overfitting.


```R

```
