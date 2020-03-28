#!/usr/bin/python

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


import sklearn
from sklearn.neighbors import KernelDensity


######################################################################
# gaussian based density estimation for the targer feature (SalePrice)
######################################################################
def display_sell_price():

    saleprice_values = df_numerical[target_feature].values.copy()

    # compute mean and var for saleprice_values
    min_saleprice = np.min(saleprice_values)
    range_saleprice = np.max(saleprice_values) - min_saleprice
    reduced_saleprice_values = (saleprice_values.copy()-min_saleprice)/range_saleprice

    reduced_saleprice_values = reduced_saleprice_values.reshape(-1,1)

    # density estimation for the value distibution
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(reduced_saleprice_values)

    proba = np.exp(kde.score_samples(reduced_saleprice_values))

    y_pos = 1 + proba*0.04*(np.random.normal(size = len(proba))-0.5)

    lin_values = np.linspace(np.min(reduced_saleprice_values)-0.5, np.max(reduced_saleprice_values)+0.5, 100)
    lin_values_2 = lin_values.copy()
    lin_values_2 = lin_values_2*range_saleprice + min_saleprice

    lin_values = lin_values.reshape(-1,1)
    lin_proba = np.exp(kde.score_samples(lin_values))*0.1

    plt.fill(np.concatenate((lin_values_2, lin_values_2[::-1])), np.concatenate((2-lin_proba, (2+lin_proba)[::-1]) ), facecolor='green', edgecolor='orangered', alpha=0.5, linewidth=1)
    plt.scatter(saleprice_values, y_pos, s=1, marker = '.', c='k' )
    plt.boxplot(saleprice_values, vert=False, positions = [3], widths = [0.5], sym = 'r+')

    plt.ylim(0,4)
    plt.xlim(0, np.max(saleprice_values))
    plt.title('SalePrice values')



######################################################################
# plot a column values vs the values of the target
######################################################################
def plot_col_vs_target(dataframe, col, target_feature):
    df_temp = dataframe[[col, target_feature]]
    df_temp = df_temp.dropna()
    plt.scatter(df_temp[col], df_temp[target_feature], marker = 'x')
    plt.title(col + ' vs SalePrice')


######################################################################
# display categorical variables in a plot
######################################################################
def display_categorical_feature(dataframe, cat_feature, target_feature):
    print('hello')

    # extract the two columns
    df_temp = dataframe[[cat_feature, target_feature]]
    
    values = df_temp[cat_feature].values
    
    # for i in range(len(values)):
        # if np.isnan(values[i]):
            # values[i] = 'NA'

    print(np.isnan(np.array(values)))

    values = np.unique(values)

    print(cat_feature)
    print(values)

# consider 0 as missing values: EnclosedPorch, BsmtFinSF2, 3SsnPorch, PoolArea, ScreenPorch, BsmtUnfSF (few), OpenPorchSF, 22ndFlrSF, WoodDeckSF, BsmtFinSF1, YearRemodAdd, TotalBsmtSF, GarageArea
# create categorical variables when few values: KitchenAbvGr, LowQualFinSF, MiscVal, BsmtHalfBath, BsmtFullBath, HalfBath, Fireplaces, FullBath
# categorical with a lot of values: OverallCond (~), YrSold, MoSold, BedroomAbvGr, TotRmsAbvGrd, GarageCars (~)

# YearRemodAdd: same as construction date if no remodal -> potential bias ?

# potential order in categorical variables: LandContour, LandSlope, OverallQual, OverallCond, ExterQual, ExterCond, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, HeatingQC, KitchenQual, Functional, FireplaceQu, GarageFinish, GarageQual, GarageCond, PoolQC, Fence (? two categories), LotShape

# Also: try some visualisation/clustering through UMAP !!!! :D

df = pd.read_csv('data/train.csv')

# remove the ID
df.set_index('Id', inplace=True)
df.index.rename(None, inplace=True)

# select numerical variables
df.dtypes

# display columns with na values

# list of features
features = df.columns.copy()

df_numerical = df.select_dtypes(include = 'number')
df_numerical.columns


# Hold on, 'MSSubClass' is also a categorical variable
df_numerical = df_numerical.drop('MSSubClass', axis=1)

# not sure for: OverallQual, OverallCond, MoSold, YrSold
df_numerical = df_numerical.drop(['OverallQual', 'OverallCond', 'MoSold', 'YrSold'], axis=1)


print('numerical variables: ')
print(df_numerical.columns)

print('categorical variables: ')
cat_features = [feature for feature in features if feature not in df_numerical.columns]
print(cat_features)

# isolate the target feature: 'SalePrice'
target_feature = df_numerical.columns[-1]
print('target feature: ' + target_feature)

# plot data for SalePrice only (boxplot and scatter plot)
plt.subplot(1,2,1)
display_sell_price()

# plot a categorical variable
plt.subplot(1,2,2)
# display_categorical_feature(df, cat_features[0], target_feature)
display_categorical_feature(df, 'Alley', target_feature)

plt.show()
