# Importing required libraries.
import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
#matplotlib inline 
sns.set(color_codes=True)
df = pd.read_csv('data.csv')
# To display the top 5 rows
df.head(5)
# To display the bottom 5 rows
df.tail(5) 