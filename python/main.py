#!/usr/bin/python

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# clustering
from sklearn.cluster import KMeans

# regression models
from sklearn import tree
from sklearn import svm
from sklearn import linear_model

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
    ax.set_xticks(pos)
    ax.set_xticklabels(values)

    ax.set_title(target_feature + ' vs ' + cat_feature)

    return

######################################################################
# compute UMAP
######################################################################
def display_UMAP(ax, df_train, df_test, target_feature, numerical_features):

    df_copy = df_train.copy()
    df_copy.drop(target_feature, axis=1, inplace=True) # remove the target feature from the dataset

    # merge train and test data (without target feature)
    df_copy = df_copy.append(df_test[df_copy.columns])

    df_copy.dropna(inplace=True)

    # perform normalization and PCA
    X_std = preprocessing.scale(df_copy.values)

    # try some clustering algorithm
    km = KMeans(6, n_init=30, max_iter=1000)
    labels = km.fit_predict(X_std)

    # new_label = np.max(labels)+1
    # for i in range(len(df_test)):
    #     labels[-i] = new_label

    color_list = list(mcolors.TABLEAU_COLORS)
    # print(len(color_list))

    cluster_colors = [color_list[elt] for elt in labels]
    # special color for the test datapoint


    # perform PCA
    pca = PCA(n_components=15)
    X_pca = pca.fit(X_std).transform(X_std)

    # val2 = c=df_numerical.dropna()[target_feature].values.copy()
    # val2=val2.reshape(0,1)

    # print(val2)
    # ax.scatter(X_pca[:,0], X_pca[:,1], marker='x', s=10, c=val2)

    # compute a UMAP on numerical features
    reducer = umap.UMAP(min_dist=0.2,n_neighbors=30,spread=0.5)
    embedding = reducer.fit_transform(X_pca)
    ax.scatter(embedding[:,0], embedding[:,1], marker='.', s=15, c=cluster_colors)

    ax.set_aspect('equal')
    ax.set_adjustable('datalim')
    ax.set_title('UMAP')

    return


######################################################################
# compute Error: Root-Mean-Squared-Error (RMSE) between SalePrice log
######################################################################
def error(Y, Y_pred):
    return np.sqrt(np.mean(np.square(np.log(Y) - np.log(Y_pred))))


######################################################################
# compute Error: Root-Mean-Squared-Error (RMSE) between SalePrice log
######################################################################
def plot_prediction_error(ax, Y, Y_pred):

    ax.set_title('SalePrice: gt vs predicted')

    # x axis in scientific notation
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)

    ax.scatter(Y, Y_pred, marker = 'x')
    
    ax.plot(ax.get_xlim(), ax.get_ylim(), color='red')

    return


######################################################################
# train a model on a subset of data and predict on the rest
######################################################################
def train_predict(model, dataframe, target_feature, ratio):

    # df contains only the (numerical) features used for the training/prediction
    # and contains na values

    df = dataframe.copy()

    prediction_features = df.columns.drop(target_feature)

    # normalization of tha data
    normalized_df = df
    normalized_df[prediction_features]=(normalized_df[prediction_features]-normalized_df[prediction_features].mean())/normalized_df[prediction_features].std()
    df = normalized_df

    # drop na
    df = df.dropna()

    # split indexes into training/testing indexes
    indexes = df.index
    split_index = int(indexes.size*ratio)
    train_indexes = indexes[:split_index]
    test_indexes = indexes[split_index+1:]

    # train the model
    X_train = df[prediction_features].loc[train_indexes]
    Y_train = df[target_feature].loc[train_indexes]
    model.fit(X_train, Y_train)

    # predict on the testing set
    X_test = df[prediction_features].loc[test_indexes]
    Y_test = df[target_feature].loc[test_indexes]
    Y_test_predicted = model.predict(X_test)

    # compute the error
    test_error = error(Y_test, Y_test_predicted)
    print(test_error)

    # plot the prediction
    ax = plt.subplot(2,2,3)
    plot_prediction_error(ax, Y_test, Y_test_predicted)

    return


# consider 0 as missing values: EnclosedPorch, BsmtFinSF2, 3SsnPorch, PoolArea, ScreenPorch, BsmtUnfSF (few), OpenPorchSF, 22ndFlrSF, WoodDeckSF, BsmtFinSF1, YearRemodAdd, TotalBsmtSF, GarageArea
# create categorical variables when few values: KitchenAbvGr, LowQualFinSF, MiscVal, BsmtHalfBath, BsmtFullBath, HalfBath, Fireplaces, FullBath
# categorical with a lot of values: OverallCond (~), YrSold, MoSold, BedroomAbvGr, TotRmsAbvGrd, GarageCars (~)

# YearRemodAdd: same as construction date if no remodal -> potential bias ?

# potential order in categorical variables: LandContour, LandSlope, OverallQual, OverallCond, ExterQual, ExterCond, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, HeatingQC, KitchenQual, Functional, FireplaceQu, GarageFinish, GarageQual, GarageCond, PoolQC, Fence (? two categories), LotShape

# Also: try some visualisation/clustering through UMAP !!!! :D

# Interrogation: standardisation: dot it several time or only once on all data ? and std for target feature ?

df = pd.read_csv('data/train.csv')

# remove the ID
df.set_index('Id', inplace=True)
# df.index.rename(None, inplace=True)

print('train set size: ' + str(len(df)))

# select numerical variables
df.dtypes

# load test set
df_test = pd.read_csv('data/test.csv')
df_test.set_index('Id', inplace=True)
# df_test.index.rename(None, inplace=True)
# print(df_test.head())
print('test set size: ' + str(len(df_test)))

# display columns with na values

# list of features
features = df.columns.copy()

df_numerical = df.select_dtypes(include = 'number')
df_numerical.columns


# Hold on, 'MSSubClass' is also a categorical variable -> but prediction better on linear model when interpreted as numerical
df_numerical = df_numerical.drop('MSSubClass', axis=1)
# not sure for: OverallQual, OverallCond, MoSold, YrSold
df_numerical = df_numerical.drop(['OverallQual', 'OverallCond', 'MoSold', 'YrSold'], axis=1)
df_numerical = df_numerical.drop(['YearBuilt', 'YearRemodAdd'], axis=1)


# Numerical features
# print('numerical variables: ')
numerical_features = df_numerical.columns.copy()
# print(numerical_features)

# categorical variables
# print('categorical variables: ')
cat_features = [feature for feature in features if feature not in numerical_features]
# print(cat_features)

# isolate the target feature: 'SalePrice'
target_feature = df_numerical.columns[-1]
print('target feature: ' + target_feature)


# compute list of features sharing na values on the same rows
df_numerical_na = df_numerical.isna()
# print(df_numerical_na.head())
res = df_numerical_na.apply(lambda x : [col for col in x.index if x[col]], result_type='reduce', axis=1)
# print(set(res.values))
print(np.unique(res))


# plot data for SalePrice only (boxplot and scatter plot)
ax = plt.subplot(2,2,1)
display_sell_price(ax)


# plot a categorical variable
ax = plt.subplot(2,2,2)
# display_categorical_feature(ax, df, cat_features[1], target_feature, mean_sort=False)
display_categorical_feature(ax, df, 'MSSubClass', target_feature, mean_sort=False)

# display a UMAP
# ax = plt.subplot(2,2,3)
# display_UMAP(ax, df_numerical, df_test, target_feature, numerical_features)

# Decision tree
# clf = tree.DecisionTreeRegressor(criterion='mse')

# Support Vector Machine
# clf = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

# Linear model
clf = linear_model.LinearRegression()

# train and predict
train_predict(clf, df_numerical, target_feature, 0.5)

plt.show()
