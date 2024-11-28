
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import classification_report
from joblib import dump


ds = pd.read_csv("data/processed_train.csv")

# Вибір ознак та цільової змінної
X = ds.drop(columns=['Species', 'Unnamed: 0', 'Sample Number', 'Individual ID', 'studyName', 'Comments'])
y = ds['Species']

# Поділ даних на тренувальні та тестові
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(C=1, penalty='l1', solver='liblinear')

f1_scorer = make_scorer(f1_score, average='weighted')
f1_scores = cross_val_score(model, X, y, cv=5, scoring=f1_scorer)

print("F1-міра на кожній ітерації крос-валідації:", f1_scores)
print("Середнє значення F1-міри:", np.mean(f1_scores))
print("Стандартне відхилення F1-міри:", np.std(f1_scores))
# Тренування моделі
model.fit(X_train, y_train)

# Оцінка на тестових даних
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Збереження моделі
dump(model, 'models/logistic_regression_model.pkl')
