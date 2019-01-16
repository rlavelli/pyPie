# ------------------------------------------------------------------------------------------------------------------------------
# Fill Missing values

# Create a groupby object: by_sex_class
by_sex_class = titanic.groupby(['sex', 'pclass']) # aggregate by groups

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median()) #only for missing

# Impute age and assign to titanic['age']
titanic.age = by_sex_class['age'].transform(impute_median) #impute median by groups

# Print the output of titanic.tail(10)
print(titanic.tail(10))

# ------------------------------------------------------------------------------------------------------------------------------
# corrplot function custom
import numpy as np
import seaborn as sns

def plot_correlation_heatmap(df):
    corr = df.corr()
    
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)


    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

columns = []

plot_correlation_heatmap(train[columns])

# ------------------------------------------------------------------------------------------------------------------------------
