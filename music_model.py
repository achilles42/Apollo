import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
score

# Dump the trained model output to joblib file so that we don't have to train the model again and again.
joblib.dump(model, 'music-recommender.joblib')


# How to load the joblib trained model data and use for predection
model = joblib.load('music-recommender.joblib')
predictions = model.predict([[21, 1]])
predictions

# create decision tree
tree.export_graphviz(model, 
                     out_file='music-recommender.dot', 
                     feature_names=['age', 'gender'], 
                     class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)
