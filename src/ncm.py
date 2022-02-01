import os
import numpy as np
import pandas as pd
from utils import train_test_split, get_classification_results, plot_prediction
from classifiers import NCM

""" Import Dataset
"""
path = os.path.join(os.path.dirname(__file__), '..', 'data')
train_df = pd.read_csv(os.path.join(path, 'train.csv'))
test_df = pd.read_csv(os.path.join(path, 'test.csv'))

""" Preprocessing
	
	- Replace missing values in Cholesterol.
	- One-hot encode categorical data
	- Reconstruct dataframe
	- Prepare dataset
"""
def preprocess(df):
	# 
	df['Cholesterol'] = df['Cholesterol'].replace(0, df['Cholesterol'].mean())
	# Standardize
	df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
	df['FastingBS'] = (df['FastingBS'] - df['FastingBS'].mean()) / df['FastingBS'].std()
	df['MaxHR'] = (df['MaxHR'] - df['MaxHR'].mean()) / df['MaxHR'].std()
	df['Oldpeak'] = (df['Oldpeak'] - df['Oldpeak'].mean()) / df['Oldpeak'].std()
	# Extract labels
	labels = df['HeartDisease']
	df = df.drop(columns=['HeartDisease'])
	# Onehot encode categorical data
	categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
	df['Sex_M'] = (df['Sex'] == 'M').astype(int)
	df['ChestPainType_ASY'] = (df['ChestPainType'] == 'ASY').astype(int)
	df['ExerciseAngina_Y'] = (df['ExerciseAngina'] == 'Y').astype(int)
	df['ST_Slope_Flat'] = (df['ST_Slope'] == 'Flat').astype(int)
	df = df.drop(columns=['RestingBP', 'Cholesterol']+categorical_cols)
	# Concatenate labels
	df['HeartDisease'] = labels
	return df

# Preprocess dataset
train_df = preprocess(train_df)
test_df = preprocess(test_df)
X_train = train_df.iloc[:, :-1].to_numpy()
y_train = train_df.iloc[:, -1].to_numpy()
X_test = test_df.iloc[:, :-1].to_numpy()
y_test = test_df.iloc[:, -1].to_numpy()

""" Training
	
	- Specify max k value
	- Generate classifier model and fit train data
	- Predict using the generated model
	- Get best k value
"""
model = NCM()
model.fit(X_train, y_train)

""" Evaluation
	
	- Generate model with best k value
	- Concatenate train and validation data to make new train set
	- Predict results for test data and evaluate
"""
pred = model.predict(X_test)
print("""
-----------------------------
         Evaluation
-----------------------------
""")
evaluation = get_classification_results(pred, y_test)
print(evaluation)
print("""
-----------------------------
""")
path = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(path, exist_ok = True)
plot_prediction(test_df, model, evaluation, 'Heart Disease Prediction w/ NCM', os.path.join(path, 'ncm.png'))