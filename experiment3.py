import pandas as pd
import math

# Dataset
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes']
}

df = pd.DataFrame(data)

# Entropy calculation
def entropy(target_col):
    values = target_col.value_counts()
    ent = 0
    for count in values:
        p = count / sum(values)
        ent -= p * math.log2(p)
    return ent

# Information Gain
def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    values = df[attribute].unique()
    weighted_entropy = 0

    for value in values:
        subset = df[df[attribute] == value]
        weighted_entropy += (len(subset)/len(df)) * entropy(subset[target])

    return total_entropy - weighted_entropy

# Find best attribute
target = 'PlayTennis'
for col in df.columns[:-1]:
    print(col, ":", information_gain(df, col, target))
