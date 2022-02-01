import os
import numpy as np
import pandas as pd
from utils import train_test_split, cross_validation, get_classification_results, plot_prediction, plot_history
from classifiers import MLP, Dense, ReLU, Softmax, CategoricalCrossEntropy, Adam, Categorical

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

train_df['NoHeartDisease'] = (train_df['HeartDisease'] == 0).astype(int)
train_df, val_df = train_test_split(train_df, 0.05)

X_train = train_df.iloc[:, :-2].to_numpy()
y_train = train_df.iloc[:, -2].to_numpy()
X_train_val = val_df.iloc[:, :-2].to_numpy()
y_train_val = val_df.iloc[:, -2].to_numpy()
X_test = test_df.iloc[:, :-1].to_numpy()
y_test = test_df.iloc[:, -1].to_numpy()


""" Training
	
	- Specify max k value
	- Generate classifier model and fit train data
	- Predict using the generated model
	- Get best k value
"""
model = MLP()

model.add(Dense(X_train.shape[1], 16))
model.add(ReLU())
model.add(Dense(16, 16))
model.add(ReLU())
model.add(Dense(16, 2))
model.add(Softmax())

model.set(
	loss = CategoricalCrossEntropy(),
	optimizer = Adam(decay = 1e-3),
	accuracy = Categorical()
)

model.configure()

# # Load parameters
params = np.load(file = os.path.join(os.path.dirname(__file__), '..', 'params', 'mlp_params.npy'), allow_pickle = True)
model.set_params(params)

# model.train(X_train, y_train, epochs = 300, batch_size = 16, summarize_every = 100, validation_data = (X_train_val, y_train_val))

# # Save parameters
# params = model.get_params()
# os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'params'), exist_ok = True)
# np.save(os.path.join(os.path.dirname(__file__), '..', 'params', 'mlp_params'), params)
# plot_history(model.train_history, model.val_history, os.path.join(os.path.dirname(__file__), '..', 'images', 'mlp_training.png'))

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
plot_prediction(test_df, model, evaluation, 'Heart Disease Prediction w/ MLP', os.path.join(path, 'mlp_st.png'), 'Age', 'ST_Slope_Flat')
# plot_prediction(test_df, model, evaluation, 'Heart Disease Prediction w/ MLP', os.path.join(path, 'mlp.png'))