import pandas as pd
import joblib
from sklearn.metrics import classification_report
# Завантаження нових даних
new_input_data = pd.read_csv('data/new_input.csv')

# Завантаження моделі
model = joblib.load('models/logistic_regression_model.pkl')

# Видалення непотрібних стовпців, які не використовувались під час тренування
X_new = new_input_data.drop(columns=['Species', 'Unnamed: 0', 'Sample Number', 'Individual ID', 'studyName', 'Comments'], errors='ignore')

# Перевірка на наявність пропущених значень
if X_new.isnull().any().any():
    print("У нових даних є пропущені значення!")
    # Можна додати код для обробки пропущених значень (наприклад, заповнення середнім значенням, тощо)

# Передбачення
predictions = model.predict(X_new)

# Збереження результатів
output = new_input_data.copy()
output['predictions'] = predictions
output.to_csv('data/predictions.csv', index=False)
print("Прогнози збережено у файлі data/predictions.csv")


real_labels = new_input_data['Species']
print(classification_report(real_labels, predictions))
