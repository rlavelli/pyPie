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

