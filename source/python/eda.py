## imports
# Importing required libraries for EDA
import pandas as pd
import numpy as np
import seaborn as sns # visualisation
import matplotlib.pyplot as plt # visualisation
#matplotlib inline 
sns.set(color_codes=True)

# additonal libraries
from pathlib import Path


## Durchführung
path = Path(__file__).resolve().parent.parent / "train.csv"
train=pd.read_csv(path)
print(train.head())

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.catplot(x='Survived',col='Sex',kind='count',data=train) # factorplot -> catplot
plt.show()
