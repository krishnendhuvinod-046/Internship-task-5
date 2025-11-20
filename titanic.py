import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv')


# print(df.head())

# print(df.info())

# print(df.describe())

df.drop('Cabin', axis=1, inplace=True)

median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace=True)

df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, 
inplace=True)

#print("Data Information After Cleaning")
#print(df.info())
#print("\nData Head After Cleaning")
#print(df.head())

# plt.figure(figsize=(6, 4))
# sns.countplot(x='Survived', data=df)
# plt.title('Survival Count (0=Died, 1=Survived)')
# plt.show()

# categorical_cols = ['Pclass', 'Sex', 'Embarked']

# plt.figure(figsize=(15, 5))
# for i, col in enumerate(categorical_cols, 1):
#     plt.subplot(1, 3, i)
#     sns.countplot(x=col, data=df)
#     plt.title(f'Distribution of {col}')

# plt.tight_layout()
# plt.show()

numerical_cols = ['Age', 'Fare']

# plt.figure(figsize=(15, 5))
# for i, col in enumerate(numerical_cols, 1):
#     plt.subplot(1, 4, i*2 - 1)
#     sns.histplot(df[col], kde=True)
#     plt.title(f'Distribution of {col}')

#     plt.subplot(1, 4, i*2)
#     sns.boxplot(y=df[col])
#     plt.title(f'Boxplot of {col}')

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# sns.barplot(x='Sex', y='Survived', data=df)

# plt.subplot(1, 2, 2)
# sns.barplot(x='Pclass', y='Survived', data=df)

# plt.show()

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# sns.boxplot(x='Survived', y='Age', data=df)

# plt.subplot(1, 2, 2)
# sns.boxplot(x='Survived', y='Fare', data=df)

# plt.show()


df_corr = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
corr_matrix = df_corr.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')

plt.show()