import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("before_clean_titanic.csv")

# 1. Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# 2. Fill missing Embarked with mode (most common value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 3. Drop columns with too many missing or irrelevant values
df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)

# 4. Encode Sex (male=0, female=1)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# 5. One-hot encode Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Save cleaned dataset
df.to_csv("titanic_cleaned.csv", index=False)

print(" Dataset cleaned and saved as titanic_cleaned.csv")
print(df.head())
