#!/usr/bin/python

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


import sklearn
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import umap

######################################################################
# gaussian based density estimation for the targer feature (SalePrice)
######################################################################
def display_sell_price(ax):

    # x axis in scientific notation
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)

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
    plt.scatter(saleprice_values, y_pos, s=10, marker = '.')
    plt.boxplot(saleprice_values, vert=False, positions = [3], widths = [0.5], sym = 'r+')

    plt.ylim(0,4)
    plt.xlim(0, np.max(saleprice_values))
    plt.title('SalePrice distribution')




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
def display_categorical_feature(ax, dataframe, cat_feature, target_feature, mean_sort = False):

     # x axis in scientific notation
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)

    # extract the two columns
    df_temp = dataframe[[cat_feature, target_feature]]

    # Check the presence of NA values
    df_na = df_temp[cat_feature].isna()
    has_na = df_na.any() # True iff there are na values in the dataframe 
    
    if has_na:
        df_temp[cat_feature].fillna('NA', inplace=True)
        
    values = df_temp[cat_feature].values
    values = np.unique(values)

    if has_na: # 'NA' must be seperated
        ind = np.where(values == 'NA')
        values[ind] = values[-1]
        values[-1] = 'NA'

    lim = ax.get_xlim()
    x_range = lim[1] - lim[0]

    # compute positions for each categories
    pos = [ind for ind in range(len(values))]
    for ind in range(len(pos)):
        pos[ind] = x_range/(len(pos))*(ind+0.5)

    # order from mean
    if mean_sort:
        values_means = [-1 for ind in range(len(values))]
        for ind in range(len(values)):
            target_values = df_temp.loc[df_temp[cat_feature] == values[ind]][target_feature].values
            values_means[ind] = np.median(target_values)
        indices = np.argsort(values_means)
        values = [values[ind] for ind in indices]

    for ind in range(len(values)):
        target_values = df_temp.loc[df_temp[cat_feature] == values[ind]][target_feature].values

        X = [pos[ind] for j in range(len(target_values))]
        # X += (np.random.rand(len(X)) - 0.5)*0.1

        X += np.random.normal(0., 0.01, len(X))

        ax.scatter(X, target_values, marker = '.', s = 5)
        ax.boxplot(target_values, vert=True, positions=[pos[ind]], widths=[0.05], showfliers=False)

    ax.set_xlim(lim)

    # set x labels
    print(pos)
    ax.set_xticks(pos)
    ax.set_xticklabels(values)

    ax.set_title(target_feature + ' vs ' + cat_feature)

    return

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
ax = plt.subplot(2,2,1)
display_sell_price(ax)

# plot a categorical variable
ax = plt.subplot(2,2,2)
# display_categorical_feature(df, cat_features[0], target_feature)
# display_categorical_feature(ax, df, 'Alley', target_feature)

display_categorical_feature(ax, df, cat_features[45], target_feature, mean_sort=True)

# compute a UMAP on numerical features
df_copy = df_numerical.copy()
df_copy.drop(target_feature, axis=1, inplace=True) # remove the target feature from the dataset
df_copy.dropna(inplace=True)

# perform normalization and PCA
scaler = StandardScaler()
X_std = scaler.fit_transform(df_copy.values)

pca = PCA(n_components=15)
X_pca = pca.fit(X_std).transform(X_std)

ax = plt.subplot(2,2,3)
val2 = c=df_numerical.dropna()[target_feature].values.copy()
# val2=val2.reshape(0,1)

# print(val2)
# ax.scatter(X_pca[:,0], X_pca[:,1], marker='x', s=10, c=val2)

reducer = umap.UMAP(min_dist=0.2,n_neighbors=30,spread=0.5)
embedding = reducer.fit_transform(X_pca)
ax.scatter(embedding[:,0], embedding[:,1], marker='x', s=10, c=val2)
ax.set_aspect('equal')
ax.set_adjustable('datalim')
ax.set_title('UMAP')

plt.show()
