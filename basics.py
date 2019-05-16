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

# Function for plot learning curves for ML models

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown
    

from sklearn.pipeline import Pipeline # pipe multiple transformations before model fit

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

plot_learning_curves(polynomial_regression, X, y)

# ------------------------------------------------------------------------------------------------------------------------------

# Function Head&Tail

def head_tail(dataframe):
    '''
    Function that concats head&tail of DataFrame
    '''
    data_head = dataframe.head(2)
    data_tail = dataframe.tail(2)
    
    middle = np.repeat('...', len(data_head.columns))
    middle = pd.DataFrame(middle).T
    middle.columns = data_head.columns
    middle.index = pd.Index(['...'])
    
    return pd.concat([data_head, middle, data_tail], axis=0)
