import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("NYPD_Shooting_Incident_Data__Historic_.csv")

# Selecting relevant columns
selected_columns = ['BORO', 'PRECINCT', 'JURISDICTION_CODE', 'VIC_AGE_GROUP', 'VIC_SEX', 'VIC_RACE', 'OCCUR_DATE', 'OCCUR_TIME']
df_selected = df[selected_columns]

# Encoding categorical variables with pd.get_dummies
df_encoded = pd.get_dummies(df_selected, columns=['VIC_SEX', 'VIC_RACE'])

# Convert 'OCCUR_DATE' to datetime to extract day of the week and month
df_encoded['OCCUR_DATE'] = pd.to_datetime(df_encoded['OCCUR_DATE'])
df_encoded['DAY_OF_WEEK'] = df_encoded['OCCUR_DATE'].dt.dayofweek
df_encoded['MONTH'] = df_encoded['OCCUR_DATE'].dt.month

# Drop 'OCCUR_DATE' and 'OCCUR_TIME' as they are not needed for the numeric analysis
df_encoded.drop(['OCCUR_DATE', 'OCCUR_TIME'], axis=1, inplace=True)

# Separating numeric and non-numeric data
numeric_data = df_encoded.select_dtypes(include=[np.number])

# Apply SimpleImputer only to the numeric data
imputer = SimpleImputer(strategy='median')
numeric_data_imputed = imputer.fit_transform(numeric_data)

# Standardizing the imputed numeric features
scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_data_imputed)

# Performing PCA on the standardized numeric data
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(numeric_data_scaled)

# Creating a DataFrame for PCA results visualization
principalDf = pd.DataFrame(data=principalComponents, columns=['Principal Component 1', 'Principal Component 2'])

# Visualizing the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(principalDf['Principal Component 1'], principalDf['Principal Component 2'], s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of NYPD Shooting Incident Dataset')
plt.show()
