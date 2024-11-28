import pandas as pd
from sklearn.model_selection import train_test_split
import os


os.system('python src/processed.py')

# Завантаження даних
data = pd.read_csv('data/processed_train.csv')

# Розділення на train і new_input (90:10)
train_data, new_input = train_test_split(data, test_size=0.1, random_state=42)

# Збереження результатів
train_data.to_csv('data/train_split.csv', index=False)
new_input.to_csv('data/new_input.csv', index=False)

print("Дані успішно розділені та збережені!")
