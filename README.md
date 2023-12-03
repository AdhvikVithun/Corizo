# Corizo
This is my work for the Corizo Internship. 
These are 2 projects given by them to complete:
1) Cardiovascular Disease Prediction
2) Spotify Song Recommendation System

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1) Cardiovascular Disease Prediction:
   
I)Libraries Import:

pandas: Used for data manipulation and analysis.
numpy: Used for numerical operations.
seaborn and matplotlib.pyplot: Used for data visualization.
plotly.express: Another library for interactive data visualization.
StandardScaler from sklearn.preprocessing: Used for feature scaling.
train_test_split from sklearn.model_selection: Splits the dataset into training and testing sets.
LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, SVC from sklearn: Machine learning models for classification.
accuracy_score, classification_report, confusion_matrix from sklearn.metrics: Used for evaluating model performance.

II)Data Loading:

Reads a CSV file named cardio_train.csv into a DataFrame using pandas.
Displays the first few rows of the dataset.

III)Preprocessing:

Drops the 'id' column as it is unnecessary for the analysis.
Checks for missing values and displays the data summary.
Visualizes the correlation between features using a heatmap.
Displays the count of individuals with and without cardiovascular disease.
Uses StandardScaler to scale the feature variables.
Splits the data into training and testing sets.

IV)Model Training and Evaluation:

Trains a Logistic Regression model and evaluates its accuracy.
Trains a Decision Tree Classifier with hyperparameter tuning and evaluates its accuracy.
Trains a Random Forest Classifier with hyperparameter tuning and evaluates its accuracy.
Explores the optimal number of neighbors for a K-Nearest Neighbors Classifier.
Trains a Support Vector Machine (SVM) classifier with a radial basis function (RBF) kernel and evaluates its accuracy.

V)Model Comparison and Visualization:

Compares the accuracy scores of different models.
Visualizes the accuracy scores using a bar plot.

VI)User Input and Prediction:

Takes user input for age, gender, height, weight, blood pressure, cholesterol, glucose level, smoking, alcohol consumption, and physical activity.
Uses a trained Random Forest model to predict the likelihood of cardiovascular disease.
Prints the prediction result ("Have Cardio" or "Not Cardio").

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2) Spotify Song Recommendation System:

I)Libraries Import:

pandas: Used for data manipulation and analysis.
numpy: Used for numerical operations.
matplotlib.pyplot and seaborn: Used for data visualization.
StandardScaler from sklearn.preprocessing: Used for feature scaling.
KMeans from sklearn.cluster: K-means clustering algorithm.
PCA from sklearn.decomposition: Principal Component Analysis for dimensionality reduction.
train_test_split from sklearn.model_selection: Splits the dataset into training and testing sets.
RandomForestClassifier from sklearn.ensemble: Random Forest model for classification.
accuracy_score, confusion_matrix from sklearn.metrics: Used for evaluating model performance.
joblib: Used for saving and loading models.

II)Data Loading and Correlation Analysis:

Reads a CSV file named spotify dataset.csv into a DataFrame.
Displays the correlation matrix heatmap for selected audio features.

III)Clustering:

Uses K-means clustering to group songs based on selected audio features.
Implements the Elbow Method to find the optimal number of clusters.
Visualizes clusters for different pairs of audio features.

IV)Model Training:

Splits the data into training and testing sets.
Uses a Random Forest Classifier to predict clusters for songs.
Evaluates the model's accuracy and displays a confusion matrix.
Saves the trained model as 'spotify_recommendation_model.pkl'.

V)User Preferences and Recommendation:

Loads the saved model.
Defines a function recommend_songs that takes user preferences, scales them, predicts the cluster using the model, and recommends songs from that cluster.
Demonstrates the recommendation process with a set of user preferences.
Displays the top song recommendations for the user, excluding songs already in their playlist.

VI)User Preferences Explanation:

The user preferences array represents scaled features for a song recommendation:
Danceability, Energy, Loudness, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo.
A note below the array explains the meaning of each feature, providing context for the user preferences.

VII)User Playlist and Recommendations:

Defines a sample user playlist.
Scales the user preferences and uses the recommend_songs function to get song recommendations.
Prints and displays the top recommendations for the user.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
