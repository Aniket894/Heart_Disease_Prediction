# Machine Learning Project Documentation 

## Project Title: Heart Disease Classification


## 1. Project Objective:

The main objective of this project is to develop a machine learning model that can accurately classify whether a patient has heart disease based on certain medical attributes.


## 2. Dataset:

For this project, the dataset used is the Heart Disease UCI dataset, which is commonly used for heart disease classification tasks. The dataset contains various attributes such as age, sex, cholesterol levels, blood pressure, etc., along with a target variable indicating the presence or absence of heart disease. It consists of a total of [insert dataset size] samples.

## 3. Problem Type:

This project involves a binary classification problem where the goal is to predict whether a patient has heart disease (1) or does not have heart disease (0) based on the given attributes.


## Dataset Features :

age: Age of the patient.

sex: Gender of the patient (0 = female, 1 = male).

chest_pain_type: Type of chest pain experienced by the patient (0, 1, 2, or 3).

resting_bp: Resting blood pressure of the patient (in mm Hg).

cholestoral: Serum cholesterol levels of the patient (in mg/dl).

fasting_blood_sugar: Fasting blood sugar level > 120 mg/dl (1 = true, 0 = false).

restecg: Resting electrocardiographic results (0, 1, or 2).

max_hr: Maximum heart rate achieved by the patient.

exang: Exercise induced angina (1 = yes, 0 = no).

oldpeak: ST depression induced by exercise relative to rest.

slope: Slope of the peak exercise ST segment.

num_major_vessels: Number of major vessels colored by fluoroscopy.

thal: Thalassemia (0, 1, 2, or 3).

target: Presence of heart disease (1 = yes, 0 = no).



# Exploratory Data Analysis (EDA):


## Library Importation:

To facilitate data manipulation, visualization, and analysis, essential libraries like pandas, numpy, matplotlib, and seaborn were imported.


## Elimination of 'Unnamed: 0' Column:

The 'Unnamed: 0' column, presumably serving as an index or identifier, was removed from the dataset due to its lack of meaningful information for analysis.


## Data Type Assessment:

Upon examination, it was noted that the 'cholesterol' and 'oldpeak' columns were of data type float64, indicating decimal values. In contrast, all other columns were of data type int64, suggesting integer values.


## Null Value Examination:

A thorough review of the dataset confirmed the absence of null values across all columns. This ensures the dataset's completeness and obviates the necessity for imputation strategies.


## Duplicate Verification:

It was ascertained that the dataset contained no duplicate records, indicating that each row represented a unique instance or observation.


## Descriptive Statistics:

A descriptive summary of the dataset's numerical attributes was generated, offering insights into central tendency, dispersion, and distribution characteristics. This summary typically includes statistics such as mean, median, standard deviation, minimum, maximum, and quartiles for each relevant column.




# Feature Engineering:


## Splitting the Dataset:

The dataset was divided into two main components: independent variables (features) and dependent variable (target).


## Independent Variables (Features):

The independent variables, denoted as X, represent the input features used for predicting the target variable. These features were extracted from the dataset excluding the target variable.


## Dependent Variable (Target):

The dependent variable, denoted as y, represents the target variable that the model seeks to predict. It was separated from the dataset and represents the outcome or response variable of interest.


## Train-Test Split:

To assess the model's performance and generalization ability, the dataset was further split into training and testing sets.


## X_train, X_test, y_train, y_test:

The dataset was partitioned into four subsets:

X_train: This subset contains the independent variables (features) used for training the model.

X_test: This subset contains the independent variables (features) used for testing the trained model's performance.

y_train: This subset contains the corresponding dependent variable (target) values for the training set.

y_test: This subset contains the corresponding dependent variable (target) values for the testing set.




# Model Training: 


## Model Selection:

Logistic Regression was selected as the model for training due to its suitability for binary classification tasks and its interpretability.


## Initialization:

The logistic regression model was initialized with default parameters. These parameters include regularization strength (C), penalty (default is L2 regularization), solver algorithm, and tolerance for stopping criteria, among others.


## Fitting the Model:

The initialized logistic regression model was trained on the training data (X_train, y_train) using the fit() method. During training, the model adjusted its parameters to minimize the logistic loss function, aiming to achieve optimal classification performance.


## Prediction:

After training, the trained logistic regression model was used to predict the target variable values for the testing data (X_test). The predict() method was applied to generate predictions based on the learned parameters.


## Default Parameters:

The model training and prediction were performed using the default parameters provided by the logistic regression implementation. 



# Model Evaluation:


## Accuracy Score:

The accuracy score of the logistic regression model was found to be approximately 0.7746, indicating that the model correctly predicted the target variable in approximately 77.46% of the cases.


## Classification Report:

Precision: Precision measures the proportion of correctly predicted positive cases among all predicted positive cases. In this case, the precision for class 0 (negative class) is 0.77, and for class 1 (positive class) is 0.78.

Recall: Recall (also known as sensitivity) measures the proportion of correctly predicted positive cases among all actual positive cases. The recall for class 0 is 0.77, and for class 1 is 0.78.

F1-score: The F1-score is the harmonic mean of precision and recall, providing a balance between the two metrics. The F1-score for both classes is approximately 0.77.

Support: Support indicates the number of actual occurrences of each class in the dataset.


## Confusion Matrix:

The confusion matrix provides a tabular summary of the model's performance, showing the number of true positive, false positive, true negative, and false negative predictions.


## In the confusion matrix provided:

True Positives (TP): 118 (model correctly predicted positive instances)

False Positives (FP): 35 (model incorrectly predicted negative instances as positive)

False Negatives (FN): 36 (model incorrectly predicted positive instances as negative)

True Negatives (TN): 126 (model correctly predicted negative instances)


## Interpretation:

The accuracy score indicates that the model performs relatively well in predicting the target variable.

The classification report demonstrates that the model achieves balanced precision, recall, and F1-score for both classes, suggesting that it doesn't favor one class over the other.

The confusion matrix reveals that the model has a tendency to make more false negative predictions (36) compared to false positive predictions (35). This suggests that the model may have slightly higher sensitivity (recall) than specificity.




# Hyperparameter Tuning:


## Parameter Grid:

A parameter grid was defined to explore different combinations of hyperparameters for the logistic regression model. The grid included options for penalty, regularization strength (C), solver algorithm, and maximum number of iterations.


## Grid Search CV:

GridSearchCV, a technique for hyperparameter tuning, was applied to systematically search through the parameter grid and find the combination that maximizes the model's performance. It utilized cross-validation (with 3 folds in this case) to evaluate each combination.


## Best Parameters:

After performing grid search, the best combination of hyperparameters was identified as {'C': 10, 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'}.



# Model Evaluation Again :


## Accuracy Score:

The accuracy score of the tuned logistic regression model is approximately 0.7873, indicating an improvement from the previous accuracy score.


## Confusion Matrix:


The confusion matrix for the tuned model shows:

True Positives (TP): 119

False Positives (FP): 33

False Negatives (FN): 34

True Negatives (TN): 129


Compared to the previous confusion matrix, there is a reduction in false positive and false negative predictions, indicating better performance in both precision and recall.



## Classification Report:

Precision, recall, and F1-score for both classes have improved slightly compared to the previous model.

Precision and recall for class 0 and class 1 are both higher, indicating better balance between correctly identifying positive and negative instances.



## Interpretation:


The improved accuracy score and enhanced precision, recall, and F1-score indicate that the model performs better after hyperparameter tuning.

The reduction in false positive and false negative predictions in the confusion matrix suggests that the model's ability to correctly classify instances has improved.

Overall, the new evaluation metrics demonstrate that the tuned logistic regression model offers better predictive performance compared to the model trained with default parameters.



# Create Pickle File

Create pickle file containing the trained logistic regression model with the optimized hyperparameters. This allows to save the model for later use without needing to retrain it.




# Flask API Creation:


## Folder Structure:

I organize the project with three main folders: dataset, models, and templates. Each folder serves a specific purpose: storing datasets, trained models, and HTML templates, respectively.


## Initialize Folders:

Within each folder (dataset, models, and templates), i add an __init__.py file. This file initializes the folder as a Python package, allowing me to import modules from these folders in my Flask application.


## HTML Templates:

I create two HTML templates: index.html and result.html in the templates folder. These templates define the structure and layout of the web pages that will be rendered to the user.


## CSS Styling:

Additionally, i create a styles.css file to define the styling rules for your HTML templates. This file will be used to enhance the visual presentation of your web pages.


## Flask Application:

In the app.py file, i import necessary modules from the Flask framework (Flask, render_template, request) to create a web application.


## Loading Model:

I load the logistic regression model (Log_reg_model) from the pickle file (log.pkl) stored in the models folder. This model will be used to make predictions on new data.


## Routes:

I define two routes in your Flask application:

The / route returns a simple welcome message to the user when they visit the homepage.

The /predict_data route handles both GET and POST requests. It renders the index.html template when the user navigates to the page or submits a form. When a POST request is made (i.e., form submission), it collects input data, makes predictions using the loaded model, and renders the result.html template with the prediction result.


## Form Submission Handling:

Within the /predict_data route, i handle form submissions using the request.form object. I extract the input data (age, sex, etc.) from the form and convert them to the appropriate data types for model prediction.


## Error Handling:

I wrap the model prediction logic in a try-except block to catch any potential errors that may occur during prediction. 


## Run Application:

Finally, i start the Flask application by calling the app.run() method. The host="0.0.0.0" argument ensures that the application is accessible from any IP address on the network.

This Flask application provides a user-friendly interface for users to input data related to heart disease risk factors, makes predictions using a logistic regression model, and displays the prediction results on the web page.
