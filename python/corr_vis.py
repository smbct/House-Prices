#!/usr/bin/python

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.neighbors import KernelDensity

# import scikitlearn



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
    return

######################################################################
# plot a column values vs the values of the target
######################################################################
def plot_col_vs_target(dataframe, col, target_feature):
    df_temp = dataframe[[col, target_feature]]
    df_temp = df_temp.dropna()
    plt.scatter(df_temp[col], df_temp[target_feature], marker = 'x')
    plt.title(col + ' vs SalePrice')
    return

######################################################################
# compute correlations between all pairs of numerical features 
######################################################################
def compute_correlations(df_numerical):
    print('test')
    features = df_numerical.columns

    size = len(features)

    corr_matrix = np.ones((size, size))

    for ind in range(size):
        for ind2 in range(ind+1, size):
            df_temp = df_numerical[[features[ind], features[ind2]]]
            df_temp = df_temp.dropna()
            corr_matrix[ind][ind2] = np.corrcoef(df_temp[features[ind]], df_temp[features[ind2]])[0, 1]
            corr_matrix[ind2][ind] = corr_matrix[ind][ind2]

    return corr_matrix


plt.rcParams.update({'font.size': 5})

df = pd.read_csv('data/train.csv')

# remove the ID
df.set_index('Id', inplace=True)

df.index.rename(None, inplace=True)

# print(df)

# select numerical variables
df.dtypes

df_numerical = df.select_dtypes(include = 'number')
df_numerical.columns

# Hold on, 'MSSubClass' is also a categorical variable
df_numerical = df_numerical.drop('MSSubClass', axis=1)

print(df_numerical.columns)

# look for nan values, get corresponding column's names
df_2 = df_numerical.isna().any()

# print(df_2['LotFrontage'])
# print(df_2)

numerical_na_col = [elt for elt in df_2.index if df_2[elt]]
numerical_not_na_col = [elt for elt in df_2.index if not df_2[elt]]

# print(numerical_na_col)
# print(numerical_not_na_col)

target_feature = numerical_not_na_col[-1]
print('target feature: ' + target_feature)


# plot data for SalePrice only (boxplot and scatter plot)
# plt.subplot(2,2,1)
# display_sell_price()

numerical_not_na_col_2 = numerical_not_na_col[:len(numerical_not_na_col)-1].copy()
indexes = [ind for ind in range(len(numerical_not_na_col_2))]

# compute correlation with salePrice with features not containing na
corr = []
for col in numerical_not_na_col_2:
    corr.append(np.corrcoef(df_numerical[col], df_numerical[target_feature])[0, 1])

# sort
indexes = [x for _,x in sorted(zip(corr,indexes))]
print('n elt: ' + str(len(indexes)))

# consider 0 as missing values: EnclosedPorch, BsmtFinSF2, 3SsnPorch, PoolArea, ScreenPorch, BsmtUnfSF (few), OpenPorchSF, 22ndFlrSF, WoodDeckSF, BsmtFinSF1, YearRemodAdd, TotalBsmtSF, GarageArea
# create categorical variables when few values: KitchenAbvGr, LowQualFinSF, MiscVal, BsmtHalfBath, BsmtFullBath, HalfBath, Fireplaces, FullBath
# categorical with a lot of values: OverallCond (~), YrSold, MoSold, BedroomAbvGr, TotRmsAbvGrd, GarageCars (~)

# compute correlation with missing values
corr2 = []
indexes_2 = [ind for ind in range(len(numerical_na_col))]
for col in numerical_na_col:
    # print(col)

    df_temp = df_numerical[[col, target_feature]]
    df_temp = df_temp.dropna()
    # print(df_temp)
    corr2.append(np.corrcoef(df_temp[col], df_temp[target_feature])[0, 1])

# sort
indexes_2 = [x for _,x in sorted(zip(corr2,indexes_2))]

print(corr2)


# ncol = 6
# nrow = 3
# create a grid of plots for correlations
# for ind_plot in range(1,ncol*nrow+1):
#     selected_feature = numerical_not_na_col_2[indexes[-ind_plot]]
#     plt.subplot(nrow,ncol,ind_plot)
#     plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#     plot_col_vs_target(df_numerical, selected_feature, target_feature)

# https://matplotlib.org/3.1.3/gallery/images_contours_and_fields/image_annotated_heatmap.html

corr_matrix = compute_correlations(df_numerical)
# print(corr_matrix)

plt.rcParams.update({'font.size': 8})

fig, ax = plt.subplots()
ax.imshow(corr_matrix, interpolation='nearest')


numrows,numcols = corr_matrix.shape

# ax.format_coord = format_coord

# We want to show all ticks...
ax.set_xticks(np.arange(numrows))
ax.set_yticks(np.arange(numcols))
# ... and label them with the respective list entries
ax.set_xticklabels(df_numerical.columns)
ax.set_yticklabels(df_numerical.columns)

plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")


plt.show()

