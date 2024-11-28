import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('future.no_silent_downcasting', True)
ds = pd.read_csv('data/train.csv')
ds['Island'] = ds['Island'].fillna(ds['Island'].mode()[0])
ds['Stage'] = ds['Stage'].fillna('Adult, 1 Egg Stage')
existing_ids = ds['Individual ID'].dropna().unique()

def fill_with_random_existing_id(row):
    if pd.isnull(row['Individual ID']):
        return np.random.choice(existing_ids)
    return row['Individual ID']

ds['Individual ID'] = ds.apply(fill_with_random_existing_id, axis=1)
ds = ds.dropna(subset=['Clutch Completion'])
def fill_missing_genders(group):
    most_common_gender = group['Sex'].mode()[0]
    group['Sex'] = group['Sex'].fillna(most_common_gender)
    return group

ds['Sex'] = ds.groupby('Species', group_keys=False, as_index=False).apply(lambda g: fill_missing_genders(g[['Species', 'Sex']].copy())[['Sex']])
most_frequent_value = ds['Sex'].mode()[0]
ds['Sex'] = ds['Sex'].replace('.', most_frequent_value)
ds['Comments'] = ds['Comments'].fillna('No comments')


ds['studyName'] = ds['studyName'].astype(str)

studyName_mapping = {'PAL0708': 708, 'PAL0809': 809, 'PAL0910' :910}
ds['studyName'] = ds['studyName'].map(studyName_mapping)
ds['studyName'] = ds['studyName'].astype(int)


sex_mapping = {'MALE': 0}
ds['Sex'] = ds['Sex'].map(sex_mapping).fillna(1)
ds['Sex'] = ds['Sex'].astype(int)

species_mapping = {'Adelie Penguin (Pygoscelis adeliae)': 0, 'Chinstrap penguin (Pygoscelis antarctica)': 1}
ds['Species'] = ds['Species'].map(species_mapping).fillna(2)
ds['Species'] = ds['Species'].astype(int)

ds['Region'] = ds['Region'].replace({'Anvers': 0})
ds['Region'] = ds['Region'].astype(int)

sex_mapping = {'No': 0}
ds['Clutch Completion'] = ds['Clutch Completion'].map(sex_mapping).fillna(1)
ds['Clutch Completion'] = ds['Clutch Completion'].astype(int)

island_counts = ds['Island'].value_counts()
ds['Island'] = ds['Island'].map(island_counts)

ds['Stage'] = ds['Stage'].replace({'Adult, 1 Egg Stage': 1})
ds['Stage'] = ds['Stage'].astype(int)

ds['Date Egg'] = ds['Date Egg'].str.replace('/', '', regex=False)
ds['Date Egg'] = ds['Date Egg'].astype(int)

ds['Comments'] = ds['Comments'].str.len()

frequency = ds['Individual ID'].value_counts()
ds['Individual ID'] = ds['Individual ID'].map(frequency)


def find_skewed_boundaries(df, variable, distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary

Delta15N_upper_limit, Delta15N_lower_limit = find_skewed_boundaries(ds, 'Delta 15 N (o/oo)', 1.5)
Delta15N_upper_limit, Delta15N_lower_limit

outliers_Delta15N = (ds['Delta 15 N (o/oo)'] > Delta15N_upper_limit) | (ds['Delta 15 N (o/oo)'] < Delta15N_lower_limit)

ds_trimmed = ds.loc[~outliers_Delta15N, ]

ds.shape, ds_trimmed.shape


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(ds[['Culmen Length (mm)', 'Body Mass (g)']])

ds_scaled = scaler.transform(ds[['Culmen Length (mm)', 'Body Mass (g)']])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
columns = [
    'Island',
    'Culmen Length (mm)',
    'Culmen Depth (mm)',
    'Flipper Length (mm)',
    'Body Mass (g)',
    'Delta 15 N (o/oo)',
    'Delta 13 C (o/oo)',
    'Comments'
]

scaler.fit(ds[columns])
ds[columns] = scaler.transform(ds[columns])

ds.to_csv('data/processed_train.csv', index=False)
