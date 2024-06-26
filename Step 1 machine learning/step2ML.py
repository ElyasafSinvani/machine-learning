﻿import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("NYPD_Shooting_Incident_Data__Historic_.csv")

# Selecting relevant columns
selected_columns = ['BORO', 'PRECINCT', 'JURISDICTION_CODE', 'VIC_AGE_GROUP', 'VIC_SEX', 'VIC_RACE', 'OCCUR_DATE', 'OCCUR_TIME']
df_selected = df[selected_columns].copy()  # Use .copy() to avoid SettingWithCopyWarning

# Correctly convert 'OCCUR_DATE' to datetime format
df_selected['OCCUR_DATE'] = pd.to_datetime(df_selected['OCCUR_DATE'])

# Now we can safely use the .dt accessor because 'OCCUR_DATE' is confirmed to be in datetime format
df_selected['DAY_OF_WEEK'] = df_selected['OCCUR_DATE'].dt.dayofweek
df_selected['MONTH'] = df_selected['OCCUR_DATE'].dt.month

# Drop 'OCCUR_DATE' and 'OCCUR_TIME'
df_selected = df_selected.drop(['OCCUR_DATE', 'OCCUR_TIME'], axis=1)

# Defining the transformations for the numeric and categorical columns
numeric_features = ['PRECINCT', 'JURISDICTION_CODE', 'DAY_OF_WEEK', 'MONTH']
numeric_transformer = SimpleImputer(strategy='median')

categorical_features = ['BORO', 'VIC_AGE_GROUP', 'VIC_SEX', 'VIC_RACE']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combining transformations into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Applying the ColumnTransformer
df_processed = preprocessor.fit_transform(df_selected)

# Standardizing the processed data
scaler = StandardScaler(with_mean=False)  # Use with_mean=False for compatibility with sparse output
df_scaled = scaler.fit_transform(df_processed)

# Performing dimensionality reduction with TruncatedSVD
svd = TruncatedSVD(n_components=2)
principalComponents = svd.fit_transform(df_scaled)

# Creating a DataFrame for the results visualization
principalDf = pd.DataFrame(data=principalComponents, columns=['Principal Component 1', 'Principal Component 2'])

# Visualizing the results
plt.figure(figsize=(8, 6))
plt.scatter(principalDf['Principal Component 1'], principalDf['Principal Component 2'], s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D projection using TruncatedSVD')
plt.show()
