import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pdb

df = pd.read_excel('/Users/reubenbillian/cfhw3/PCA_Treasuries.xlsx')
df = df.set_index(df['Unnamed: 0'])
df.index.name = ""
df = df.drop(['Unnamed: 0', 'Unnamed: 9', 'Unnamed: 10'], axis=1)
df = df.iloc[0:8]

num_features = 8

# pdb.set_trace()
differences = []

for col in df.columns:
    deltas = []
    for i in range(len(df[col]) - 1):
        deltas.append(df[col][i+1] - df[col][i])
    differences.append(deltas)

df_diff = pd.DataFrame(differences)



pca = PCA(n_components=num_features)
pca.fit(df)

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8, 5))
plt.bar(range(1, num_features + 1), explained_variance, alpha=0.6, label="Individual Variance")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Principal Components")
plt.legend()
plt.show()
for i, var in enumerate(explained_variance, 1):
    print(f"PC{i} explains {var:.2%} of the variance.")
