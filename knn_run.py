import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from measure import plot_confusion_matrix
from knn import KNN_MDS


IRIS_DATA_PATH = './dataset/iris.csv'
iris_df = pd.read_csv(IRIS_DATA_PATH)

iris_df['species'] = iris_df['species'].map({
    'Iris-setosa': 0, 
    'Iris-versicolor': 1, 
    'Iris-virginica': 2
})

X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = iris_df['species'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234, shuffle=True, stratify=y)

labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

accuracy_scores = []

for i in range(1, 10):
    knn = KNN_MDS(k=i)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print("K = ", i)
    print(classification_report(
        y_test, predictions, target_names=labels))
    accuracy_scores.append(accuracy_score(y_test, predictions))

sns.lineplot(x=range(1, len(accuracy_scores) + 1), y=accuracy_scores)
plt.title('Knn accuracy vs K (Neighbours)')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()

