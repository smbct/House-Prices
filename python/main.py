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
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

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
# plot a column values vs the values of the target feature
######################################################################
def plot_feature_vs_target(ax, dataframe, feature, target_feature):
    
    # y axis in scientific notation
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)

    df_temp = dataframe[[feature, target_feature]]
    df_temp = df_temp.dropna()
    ax.scatter(df_temp[feature], df_temp[target_feature], marker = 'x')
    ax.set_title(feature + ' vs SalePrice')


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

    # axis in scientific notation
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

    print(df.shape)

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

######################################################################
# train a model on a subset of data and predict on the rest
######################################################################
def train_predict_bis(model, dataframe, target_feature, ratio):

    # df contains only the (numerical) features used for the training/prediction
    # and contains na values

    df = dataframe.copy()

    prediction_features = df.columns.drop(target_feature)

    # normalization of tha data
    normalized_df = df
    normalized_df[prediction_features]=(normalized_df[prediction_features]-normalized_df[prediction_features].mean())/normalized_df[prediction_features].std()
    df = normalized_df

    # split indexes into training/testing indexes
    indexes = df.index
    split_index = int(indexes.size*ratio)
    train_indexes = indexes[:split_index]
    test_indexes = indexes[split_index+1:]

    # create a dataframe for the test data and for the ground truth
    df_gt = df[target_feature].loc[test_indexes]
    df_predicted = df_gt.copy()
    df_predicted.loc[:] = 0

    # compute nan features
    na_features = df.isna().apply(lambda x : ([col for col in x.index if x[col]], x.name), result_type='reduce', axis=1)
    # print(res.head())
    # print(set(res.values))
    paired_features = np.unique([elt[0] for elt in na_features])
    paired_indexes = [[] for _ in paired_features]
    for elt in na_features:
        ind = 0
        while paired_features[ind] != elt[0]:
            ind += 1
        paired_indexes[ind].append(elt[1])

    # print(paired_indexes)
    # print(paired_features)

    # compute train indexes per nan group features
    nan_test_indexes = [[] for _ in paired_features]

    for tuple_index in range(len(paired_features)):
        for ind in paired_indexes[tuple_index]:
            if ind in test_indexes:
                nan_test_indexes[tuple_index].append(ind)

    # learn different models given available features
    for tuple_ind in range(len(paired_features)):

        # print(paired_features[tuple_ind])

        if len(nan_test_indexes[tuple_ind]) == 0:
            continue

        # train a model on every features excepted the tupled features
        # test it on the test indexes recorded, where only these features are missing
        to_exclude = paired_features[tuple_ind]

        new_cols = df.columns.drop(paired_features[tuple_ind])

        # create a training dataframe
        df_train = df[new_cols].loc[train_indexes].dropna()

        # print(df_train.size)

        # fit the model
        X = df_train[new_cols.drop(target_feature)]
        Y = df_train[target_feature]
        model.fit(X,Y)

        # make the prediction for the corresponding test_indexes
        test_indexes = nan_test_indexes[tuple_ind]

        # prediction_features = df_temp.columns.drop(target_feature)

        # predict on the test set
        X_test = df[new_cols.drop(target_feature)].loc[test_indexes]
        df_predicted.loc[test_indexes] = model.predict(X_test) 

    # do not feed log with invalid values
    df_predicted = df_predicted.apply(np.abs)

    # compute the error
    test_error = error(df_gt, df_predicted)
    print(test_error)

    # plot the prediction
    ax = plt.subplot(2,2,3)
    plot_prediction_error(ax, df_gt, df_predicted)


    return


######################################################################
# train a model for the final submission
######################################################################
def train_predict_final(model, dataframe, target_feature, train_indexes, test_indexes):

    # print(dataframe)

    # df contains only the (numerical) features used for the training/prediction
    # and contains na values

    df = dataframe.copy()

    prediction_features = df.columns.drop(target_feature)

    # normalization of tha data
    normalized_df = df
    normalized_df[prediction_features]=(normalized_df[prediction_features]-normalized_df[prediction_features].mean())/normalized_df[prediction_features].std()
    df = normalized_df

    # compute nan features
    na_features = df[prediction_features].isna().apply(lambda x : ([col for col in x.index if x[col]], x.name), result_type='reduce', axis=1)
    # print(res.head())
    # print(set(res.values))
    paired_features = np.unique([elt[0] for elt in na_features])
    paired_indexes = [[] for _ in paired_features]
    for elt in na_features:
        ind = 0
        while paired_features[ind] != elt[0]:
            ind += 1
        paired_indexes[ind].append(elt[1])

    # print(paired_indexes)
    print(paired_features)

    # compute train indexes per nan group features
    nan_test_indexes = [[] for _ in paired_features]

    for tuple_index in range(len(paired_features)):
        for ind in paired_indexes[tuple_index]:
            if ind in test_indexes:
                nan_test_indexes[tuple_index].append(ind)

    # print(nan_test_indexes)
    print([len(elt) for elt in nan_test_indexes])

    # learn different models given available features
    for tuple_ind in range(len(paired_features)):

        # print(paired_features[tuple_ind])

        if len(nan_test_indexes[tuple_ind]) == 0:
            continue

        # train a model on every features excepted the tupled features
        # test it on the test indexes recorded, where only these features are missing
        to_exclude = paired_features[tuple_ind]

        new_cols = df.columns.drop(paired_features[tuple_ind])

        # create a training dataframe
        df_train = df[new_cols].loc[train_indexes].dropna()


        # fit the model
        X = df_train[new_cols.drop(target_feature)]
        Y = df_train[target_feature]
        model.fit(X,Y)

        # make the prediction for the corresponding test_indexes
        test_indexes_2 = nan_test_indexes[tuple_ind]

        # prediction_features = df_temp.columns.drop(target_feature)

        # predict on the test set
        X_test = df[new_cols.drop(target_feature)].loc[test_indexes_2]
        df[target_feature].loc[test_indexes_2] = model.predict(X_test) 

    # save the result
    df_res = df[target_feature][test_indexes]
    df_res.to_csv('prediction_gb.csv', header=True)

    return

# consider 0 as missing values: EnclosedPorch, BsmtFinSF2, 3SsnPorch, PoolArea, ScreenPorch, BsmtUnfSF (few), OpenPorchSF, 2ndFlrSF, WoodDeckSF, BsmtFinSF1, YearRemodAdd, TotalBsmtSF, GarageArea
# create categorical variables when few values: KitchenAbvGr, LowQualFinSF, MiscVal, BsmtHalfBath, BsmtFullBath, HalfBath, Fireplaces, FullBath
# categorical with a lot of values: OverallCond (~), YrSold, MoSold, BedroomAbvGr, TotRmsAbvGrd, GarageCars (~)

# YearRemodAdd: same as construction date if no remodal -> potential bias ?

# potential order in categorical variables: LandContour, LandSlope, OverallQual, OverallCond, ExterQual, ExterCond, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, HeatingQC, KitchenQual, Functional, FireplaceQu, GarageFinish, GarageQual, GarageCond, PoolQC, Fence (? two categories), LotShape

# Also: try some visualisation/clustering through UMAP !!!! :D

# Interrogation: standardisation: dot it several time or only once on all data ? and std for target feature ?

np.random.seed(42)

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

train_indexes = df.index
test_indexes = df_test.index
# df_complete = df.append(df_test, sort=False)

# list of features
features = df.columns.copy()

df_numerical = df.select_dtypes(include = 'number')
df_numerical.columns


# Hold on, 'MSSubClass' is also a categorical variable -> but prediction better on linear model when interpreted as numerical
df_numerical = df_numerical.drop('MSSubClass', axis=1)
# not sure for: OverallQual, OverallCond, MoSold, YrSold
df_numerical = df_numerical.drop(['OverallQual', 'OverallCond', 'MoSold', 'YrSold'], axis=1)
df_numerical = df_numerical.drop(['YearBuilt', 'YearRemodAdd'], axis=1)

# remove numerical features that have too many missing data
df_numerical = df_numerical.drop(['3SsnPorch', 'PoolArea', 'EnclosedPorch', 'ScreenPorch'], axis=1)

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




# how to proceed ?
# - learn all the features that can be learned when there is no missing data
# - when there are missing data, create an alternative dataset without the missing column, train on the dataset (where it is possible) and predict only these features
# - alternatively we couls remove the features from the whole training set and learn everything at once

# To start with: keep lot frontage (exclude from training indexes all lines with na on lot frontage if target feature is not lotfrontage)

# plot data for SalePrice only (boxplot and scatter plot)
ax = plt.subplot(2,2,1)
display_sell_price(ax)



# plot_feature_vs_target(ax, df, 'LotFrontage', target_feature)
zeros_num = ['BsmtFinSF2', 'BsmtUnfSF', 'OpenPorchSF', '2ndFlrSF', 'WoodDeckSF', 'BsmtFinSF1', 'TotalBsmtSF', 'GarageArea'] # 'YearRemodAdd'
# consider removing: '3SsnPorch', 'PoolArea'

to_remove = ['3SsnPorch', 'PoolArea', 'EnclosedPorch', 'ScreenPorch']
# col = df_numerical.columns.drop(to_remove)
# df_numerical = df_numerical[col]

# plot a numerical feature
ax = plt.subplot(2,2,2)
plot_feature_vs_target(ax, df, zeros_num[0], target_feature)

# when this is done, the training set is too small
# so select only the most relevant one
# for elt in zeros_num:
#   df_numerical[elt] = df_numerical[elt].apply(lambda x : np.nan if x <= 1e-4 else x)

# df_numerical[zeros_num[0]] = df_numerical[zeros_num[0]].apply(lambda x : np.nan if x <= 1e-4 else x)


# plot a categorical feature
ax = plt.subplot(2,2,4)
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
# clf = linear_model.LinearRegression()

# Gradient Boosting
clf = GradientBoostingRegressor(loss='ls', n_estimators=500, max_depth=2, learning_rate=.05)

# Neural Network
# clf = MLPRegressor(activation='tanh', solver='sgd', hidden_layer_sizes=(5, 10), max_iter=10000)

# train and predict
# train_predict(clf, df_numerical, target_feature, 0.5)
train_predict_bis(clf, df_numerical, target_feature, 0.5)
# train_predict_final(clf, df_complete[df_numerical.columns], target_feature, train_indexes, test_indexes)

plt.show()
