Project Name : Machine Learning For All
(Project for the Berkeley MIDS W210 course)

Team : Arvin Sahni, Jacky Zhang, Jason Sanchez, Nandu Menon

How to run :
- cd to the 'flask' folder
- python run.py

Open this on the browser window:
localhost:5000/index


About the App:

Once the user uploads the 2 files, we do some initial checks to ensure that the 2 files differ only by one column. Even if the user were to mistakenly upload the files in the wrong order(testing file uploaded as train or vice versa), the app will recognise it and use the train/test data appropriately. Each of the files are hashed to check if the same files have been uploaded previously. If the files/datasets have already been previously uploaded, the app can intelligently identify this and not redo the model development. The user input files are converted to pandas dataframes for further processing. Once the initial processing is done, the dataframes are passed onto the modelling and visualisation stacks. The data visualisation stack uses Shiny with R. The visualisations are interactive to make the user expereince pleasant and informative while the data modelling happens in the background.
.
There are four main steps to the data modelling : problem identification, feature engineering, ensemble model creation, and model introspection.

In the problem identification step, we determine if the general problem type is a regression problem or a classification problem based on the distribution and type of labels. For each feature in the dataset, we determine if it is a numeric feature or a categorical feature. Categorical features are further subdivided into dense categorical features (i.e. ones that if one-hot encoded would result in a small number of columns being added to the dataset) and sparse categorical features (i.e. ones with many unique categories).

In the feature engineering step, we impute numeric missing values with the mean and categorical missing values with a "MISSING" category. Outliers are indirectly handled in the model step. Categorical features are encoded in a variety of ways. We create the following new features for sparse categorical data: Count encodings (Each column is replaced with the number of instances of that column), Length encoding (Each column is replaced with the character length of the column), and Sort encoding (Each column is replaced with a number that represents approximately where the category would fall in a dictionary).

In the ensemble model creation step, for regression problems, a Random Forest model and a Ridge Regression model are trained on the transformed data. The out of bag predictions from the Random Forest and the cross-validation predictions from the Ridge model become features used to train a Linear Model that combines the estimates of the two base models. For classification problems, we use a similar method except use regularized Logistic Regression instead of Ridge Regression and use standard Logistic Regression to ensemble the class probability predictions together.

We experimented with significantly more complex ensemble structures that were many layers deep with dozens of base models through the network and skip cells that allowed data to pass through different layers. We also experimented with enabling the model to self-optimize hyperparameters via Randomized Grid Search. Although the performance of the model improved, ultimately the training time took too long to be practical. The system we use is very fast and has exactly one parameter: time_to_compute. Making this parameter higher leads to more accurate predictions at the expense of higher compute time.

In the model introspection step, we estimate the generalizable error rate of the model via cross validated predictions across a variety of error metrics. We also calculate an overall model score. For regression problems we use R^2 and for classification problems we use zero-scaled version of AUC. A score of 0 means the model is terrible and a score of 100 means the model is perfect. The accuracy scores are rendered on the third pane along with an easy to understand description.




