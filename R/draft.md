## Result of the feature selection with wrapper method and support vector machine

```
[1] "Round  1  ; Selected feature:  36  ; CV error= 0.3993  ; std dev= 0.0286"
[1] "Round  2  ; Selected feature:  4  ; CV error= 0.2352  ; std dev= 0.0182"
[1] "Round  3  ; Selected feature:  16  ; CV error= 0.2031  ; std dev= 0.0175"
[1] "Round  4  ; Selected feature:  1  ; CV error= 0.182  ; std dev= 0.0131"
[1] "Round  5  ; Selected feature:  6  ; CV error= 0.1685  ; std dev= 0.0129"
[1] "Round  6  ; Selected feature:  5  ; CV error= 0.1603  ; std dev= 0.0156"
[1] "Round  7  ; Selected feature:  12  ; CV error= 0.1532  ; std dev= 0.0159"
[1] "Round  8  ; Selected feature:  11  ; CV error= 0.1469  ; std dev= 0.0163"
[1] "Round  9  ; Selected feature:  3  ; CV error= 0.1447  ; std dev= 0.0164"
[1] "Round  10  ; Selected feature:  24  ; CV error= 0.1409  ; std dev= 0.0166"
[1] "Round  11  ; Selected feature:  26  ; CV error= 0.1385  ; std dev= 0.0182"
[1] "Round  12  ; Selected feature:  21  ; CV error= 0.1363  ; std dev= 0.0187"
[1] "Round  13  ; Selected feature:  25  ; CV error= 0.1355  ; std dev= 0.0174"
[1] "Round  14  ; Selected feature:  13  ; CV error= 0.1352  ; std dev= 0.0179"
[1] "Round  15  ; Selected feature:  7  ; CV error= 0.1351  ; std dev= 0.0175"

[1] "Round  16  ; Selected feature:  9  ; CV error= 0.1347  ; std dev= 0.0173"

[1] "Round  17  ; Selected feature:  14  ; CV error= 0.1347  ; std dev= 0.0178"
[1] "Round  18  ; Selected feature:  20  ; CV error= 0.1352  ; std dev= 0.018"
[1] "Round  19  ; Selected feature:  22  ; CV error= 0.1359  ; std dev= 0.0188"
[1] "Round  20  ; Selected feature:  17  ; CV error= 0.1363  ; std dev= 0.0187"
[1] "Round  21  ; Selected feature:  27  ; CV error= 0.1367  ; std dev= 0.0183"
[1] "Round  22  ; Selected feature:  19  ; CV error= 0.1372  ; std dev= 0.0186"
[1] "Round  23  ; Selected feature:  23  ; CV error= 0.1378  ; std dev= 0.0184"
[1] "Round  24  ; Selected feature:  10  ; CV error= 0.1385  ; std dev= 0.0197"
[1] "Round  25  ; Selected feature:  28  ; CV error= 0.1394  ; std dev= 0.0189"
[1] "Round  26  ; Selected feature:  34  ; CV error= 0.1407  ; std dev= 0.0232"
[1] "Round  27  ; Selected feature:  18  ; CV error= 0.1424  ; std dev= 0.0246"
[1] "Round  28  ; Selected feature:  35  ; CV error= 0.1443  ; std dev= 0.0248"
[1] "Round  29  ; Selected feature:  31  ; CV error= 0.1463  ; std dev= 0.0232"
[1] "Round  30  ; Selected feature:  8  ; CV error= 0.1482  ; std dev= 0.0239"
[1] "Round  31  ; Selected feature:  33  ; CV error= 0.1497  ; std dev= 0.0243"
[1] "Round  32  ; Selected feature:  32  ; CV error= 0.1513  ; std dev= 0.0232"
[1] "Round  33  ; Selected feature:  15  ; CV error= 0.1528  ; std dev= 0.0219"
[1] "Round  34  ; Selected feature:  30  ; CV error= 0.1546  ; std dev= 0.0222"
[1] "Round  35  ; Selected feature:  2  ; CV error= 0.1567  ; std dev= 0.022"
[1] "Round  36  ; Selected feature:  29  ; CV error= 0.1588  ; std dev= 0.0264"

-> features selected: 36 ; 4 ; 16 ; 1 ; 6 ; 5 ; 12 ; 11 ; 3 ; 24 ; 26 ; 21 ; 25 ; 13 ; 7 ; 9

```

## Result of the feature selection with mRMR method and support vector machine

```

1] 1
 [1]  4 22 12 16 26  7 24  8 30  6 13 27 23 19  9  5 25 14 28  2 29 20  3 17 11  1 32 31 21 33 36 35 34 18 10 15
[1] 2
 [1]  4 22 12 16  6 30 26  7 24 27 13  8 19  5 23 25  9 28 14  2 29 20  3 17 11  1 15 32 18 21 36 33 31 35 34 10
[1] 3
 [1]  4 22 12 16 26  7 24  8 30  6 27 13 23 19  5 25  9 28 14 29  3  2 20 17 11  1 32 10 21 31 33 36 35 18 15 34
[1] 4
 [1]  4 22 12 16 26  7 24  8 27 30  6 13 23 19  9 25  5 14 28 29  2 20  3 17 11  1 32 21 36 33 35 31 15 10 34 18
[1] 5
 [1]  4 22 27 16  6 12  5 26 24 19 13  7 30  8 23 25  9 14 28  2 29 20  3 17 11 32  1 18 21 33 31 35 15 36 34 10
[1] 6
 [1]  4 22 12 16  6 30 26 24  7 27 13  8 23  9 19  5 25 28 14  2 29 17 20  3 11  1 32 36 33 35 21 31 18 15 34 10
[1] 7
 [1]  4 22 27 16 12  7 24 30  6 26 13 23  8 19  5 25  9 14 28  2 29  3 20 17 11  1 32 36 21 35 33 31 10 34 15 18
[1] 8
 [1]  4 22 12 16 26  7 24 30  6 27 13  8 23  5 19 25  9 28 14  2 29 20  3 17 11  1 32 15 36 21 33 31 35 34 10 18
[1] 9
 [1]  4 22 12 16  6 27 30  7 24 26 13 23  8  9 19 25  5 14  2 28 29  3 20 17 11  1 32 21 36 34 15 18 31 35 33 10
[1] 10
 [1]  4 22 27 16 12  7 24  6 30 26 23 13  8 19  5 25  9 14 28  2 29 20  3 17 11  1 32 21 34 36 33 35 18 15 31 10
 [1] "#Features:  1  ; CV error= 0.2304  ; std dev= 0.0177"  "#Features:  2  ; CV error= 0.2303  ; std dev= 0.0188"
 [3] "#Features:  3  ; CV error= 0.2059  ; std dev= 0.0203"  "#Features:  4  ; CV error= 0.1761  ; std dev= 0.0214"
 [5] "#Features:  5  ; CV error= 0.168  ; std dev= 0.0193"   "#Features:  6  ; CV error= 0.1617  ; std dev= 0.015"  
 [7] "#Features:  7  ; CV error= 0.1604  ; std dev= 0.0164"  "#Features:  8  ; CV error= 0.1591  ; std dev= 0.0163"
 [9] "#Features:  9  ; CV error= 0.1557  ; std dev= 0.0162"  "#Features:  10  ; CV error= 0.155  ; std dev= 0.0159"
[11] "#Features:  11  ; CV error= 0.1535  ; std dev= 0.0161" "#Features:  12  ; CV error= 0.1519  ; std dev= 0.0147"
[13] "#Features:  13  ; CV error= 0.1544  ; std dev= 0.0164" "#Features:  14  ; CV error= 0.1547  ; std dev= 0.0195"
[15] "#Features:  15  ; CV error= 0.151  ; std dev= 0.0195"  "#Features:  16  ; CV error= 0.1492  ; std dev= 0.0198"
[17] "#Features:  17  ; CV error= 0.1458  ; std dev= 0.0186" "#Features:  18  ; CV error= 0.1474  ; std dev= 0.0185"
[19] "#Features:  19  ; CV error= 0.1486  ; std dev= 0.0201" "#Features:  20  ; CV error= 0.1536  ; std dev= 0.0272"
[21] "#Features:  21  ; CV error= 0.1532  ; std dev= 0.0262" "#Features:  22  ; CV error= 0.1512  ; std dev= 0.0263"
[23] "#Features:  23  ; CV error= 0.1506  ; std dev= 0.026"  "#Features:  24  ; CV error= 0.1496  ; std dev= 0.0258"
[25] "#Features:  25  ; CV error= 0.1497  ; std dev= 0.0256" "#Features:  26  ; CV error= 0.1499  ; std dev= 0.0257"
[27] "#Features:  27  ; CV error= 0.1515  ; std dev= 0.0251" "#Features:  28  ; CV error= 0.152  ; std dev= 0.0257"
[29] "#Features:  29  ; CV error= 0.1522  ; std dev= 0.0255" "#Features:  30  ; CV error= 0.1526  ; std dev= 0.0251"
[31] "#Features:  31  ; CV error= 0.1526  ; std dev= 0.0257" "#Features:  32  ; CV error= 0.1529  ; std dev= 0.0256"
[33] "#Features:  33  ; CV error= 0.1535  ; std dev= 0.0242" "#Features:  34  ; CV error= 0.1534  ; std dev= 0.0242"
[35] "#Features:  35  ; CV error= 0.1572  ; std dev= 0.0268" "#Features:  36  ; CV error= 0.1588  ; std dev= 0.0264"

```

## Result of the feature selection with corr method and support vector machine

```
[1] "#Features:  1  ; CV error= 0.2304  ; std dev= 0.0177"  "#Features:  2  ; CV error= 0.1994  ; std dev= 0.0203"
[3] "#Features:  3  ; CV error= 0.1863  ; std dev= 0.0187"  "#Features:  4  ; CV error= 0.1827  ; std dev= 0.0154"
[5] "#Features:  5  ; CV error= 0.1702  ; std dev= 0.0195"  "#Features:  6  ; CV error= 0.1693  ; std dev= 0.0174"
[7] "#Features:  7  ; CV error= 0.1712  ; std dev= 0.0159"  "#Features:  8  ; CV error= 0.1699  ; std dev= 0.0189"
[9] "#Features:  9  ; CV error= 0.1635  ; std dev= 0.0158"  "#Features:  10  ; CV error= 0.1593  ; std dev= 0.0157"
[11] "#Features:  11  ; CV error= 0.1609  ; std dev= 0.0179" "#Features:  12  ; CV error= 0.1576  ; std dev= 0.0169"
[13] "#Features:  13  ; CV error= 0.1557  ; std dev= 0.018"  "#Features:  14  ; CV error= 0.1519  ; std dev= 0.0163"
[15] "#Features:  15  ; CV error= 0.1598  ; std dev= 0.0257" "#Features:  16  ; CV error= 0.1595  ; std dev= 0.0262"
[17] "#Features:  17  ; CV error= 0.1594  ; std dev= 0.0261" "#Features:  18  ; CV error= 0.1603  ; std dev= 0.0262"
[19] "#Features:  19  ; CV error= 0.1595  ; std dev= 0.0257" "#Features:  20  ; CV error= 0.1575  ; std dev= 0.0254"
[21] "#Features:  21  ; CV error= 0.1572  ; std dev= 0.0244" "#Features:  22  ; CV error= 0.1578  ; std dev= 0.0249"
[23] "#Features:  23  ; CV error= 0.1575  ; std dev= 0.0248" "#Features:  24  ; CV error= 0.1562  ; std dev= 0.0233"
[25] "#Features:  25  ; CV error= 0.1568  ; std dev= 0.0238" "#Features:  26  ; CV error= 0.157  ; std dev= 0.0241"
[27] "#Features:  27  ; CV error= 0.1558  ; std dev= 0.0247" "#Features:  28  ; CV error= 0.1538  ; std dev= 0.0253"
[29] "#Features:  29  ; CV error= 0.1494  ; std dev= 0.0255" "#Features:  30  ; CV error= 0.1518  ; std dev= 0.0262"
[31] "#Features:  31  ; CV error= 0.1542  ; std dev= 0.0253" "#Features:  32  ; CV error= 0.155  ; std dev= 0.0257"
[33] "#Features:  33  ; CV error= 0.1549  ; std dev= 0.0244" "#Features:  34  ; CV error= 0.1572  ; std dev= 0.0269"
[35] "#Features:  35  ; CV error= 0.1582  ; std dev= 0.0264" "#Features:  36  ; CV error= 0.1588  ; std dev= 0.0264"
```

## Result of the feature selection with PCA method and support vector machine

```
[1] "#Features:  1  ; CV error= 0.4326  ; std dev= 0.0257"  "#Features:  2  ; CV error= 0.5188  ; std dev= 0.0211"
[3] "#Features:  3  ; CV error= 0.5211  ; std dev= 0.0207"  "#Features:  4  ; CV error= 0.5245  ; std dev= 0.0201"
[5] "#Features:  5  ; CV error= 0.5249  ; std dev= 0.0198"  "#Features:  6  ; CV error= 0.5231  ; std dev= 0.0197"
[7] "#Features:  7  ; CV error= 0.522  ; std dev= 0.02"     "#Features:  8  ; CV error= 0.5218  ; std dev= 0.0195"
[9] "#Features:  9  ; CV error= 0.5212  ; std dev= 0.0192"  "#Features:  10  ; CV error= 0.5216  ; std dev= 0.0192"
[11] "#Features:  11  ; CV error= 0.5204  ; std dev= 0.0193" "#Features:  12  ; CV error= 0.5208  ; std dev= 0.0194"
[13] "#Features:  13  ; CV error= 0.5211  ; std dev= 0.0191" "#Features:  14  ; CV error= 0.521  ; std dev= 0.0191"
[15] "#Features:  15  ; CV error= 0.5213  ; std dev= 0.0197" "#Features:  16  ; CV error= 0.5209  ; std dev= 0.0197"
[17] "#Features:  17  ; CV error= 0.5225  ; std dev= 0.0195" "#Features:  18  ; CV error= 0.5228  ; std dev= 0.0199"
[19] "#Features:  19  ; CV error= 0.5232  ; std dev= 0.0198" "#Features:  20  ; CV error= 0.5238  ; std dev= 0.02"  
[21] "#Features:  21  ; CV error= 0.5249  ; std dev= 0.0199" "#Features:  22  ; CV error= 0.5241  ; std dev= 0.0202"
[23] "#Features:  23  ; CV error= 0.5247  ; std dev= 0.0201" "#Features:  24  ; CV error= 0.528  ; std dev= 0.0202"
[25] "#Features:  25  ; CV error= 0.5303  ; std dev= 0.0201" "#Features:  26  ; CV error= 0.5319  ; std dev= 0.02"  
[27] "#Features:  27  ; CV error= 0.5324  ; std dev= 0.0198" "#Features:  28  ; CV error= 0.5336  ; std dev= 0.0198"
[29] "#Features:  29  ; CV error= 0.5335  ; std dev= 0.0201" "#Features:  30  ; CV error= 0.5339  ; std dev= 0.0202"
[31] "#Features:  31  ; CV error= 0.535  ; std dev= 0.0201"  "#Features:  32  ; CV error= 0.5354  ; std dev= 0.0201"
[33] "#Features:  33  ; CV error= 0.5358  ; std dev= 0.0203" "#Features:  34  ; CV error= 0.5356  ; std dev= 0.0204"
[35] "#Features:  35  ; CV error= 0.5355  ; std dev= 0.0205" "#Features:  36  ; CV error= 0.5351  ; std dev= 0.0204"
```

## next step:

- ensemble of ensemble of different models
- taking into account class variables
- tune parameters of svm
- Radial Basis Functions in the ensemble (look at slides 46 of algosRegr)
- ensemble of models: take only a subset of features

## Results

                                lm          |       rpart       |       nnet        |       lazy        |     svm
cross_validation all feat: 0.1878 - 0.0412  | 0.2167 - 0.0203   |  0.155 - 0.0268   |  0.1508 - 0.0233  | 0.1574 - 0.0263
cross_validation sel feat: 0.1912 - 0.0426  | 0.2167 - 0.0203   |  0.145 - 0.0181   |  0.1462 - 0.0253  | 0.1343 - 0.0175
ensemble all feat:         0.1685 - 0.0225  | 0.1928 - 0.0218   |  0.1819- 0.0628   |  0.1452 - 0.0221  | 0.154 - 0.0211
ensemble sel feat:         0.1822 - 0.0356  | 0.192 - 0.0216    |  0.1582 - 0.0274  |  0.1721 - 0.0837  | 0.1428 - 0.0143

## factor variables

"MSZoning X
"Street" X
"Alley" X
"LotShape" X
"LandContour" X
"Utilities" X
"LotConfig"    X                                     
"LandSlope" X
"Neighborhood" X
"Condition1" X
"Condition2" X
"BldgType"  X
"HouseStyle" X
"RoofStyle" X                       
"RoofMatl" X
"Exterior1st" X
"Exterior2nd" X
"MasVnrType"  X
"ExterQual"  V
"ExterCond" X
"Foundation" X                
"BsmtQual" X   
"BsmtCond" X  
"BsmtExposure" X
"BsmtFinType1" X
"BsmtFinType2" X
"Heating"   X
"HeatingQC"   V
"CentralAir" X
"Electrical"   X
"KitchenQual" V
"Functional"  X
"FireplaceQu" X
"GarageType"  X
"GarageFinish" X           
"GarageQual" X
"GarageCond"  X
"PavedDrive" X
"PoolQC"   X
"Fence" X
"MiscFeature" X
"SaleType"     X
"SaleCondition" X

factor_sel <- c("MSSubClass", "HouseStyle", "ExterQual", "HeatingQC", "KitchenQual", "Functional")

## wrap features lm

[1] "Round  1  ; Selected feature:  73  ; CV error= 0.4069  ; std dev= 0.029"
[1] "Round  2  ; Selected feature:  46  ; CV error= 0.2623  ; std dev= 0.0157"
[1] "Round  3  ; Selected feature:  58  ; CV error= 0.2285  ; std dev= 0.0262"
[1] "Round  4  ; Selected feature:  55  ; CV error= 0.2132  ; std dev= 0.0224"
[1] "Round  5  ; Selected feature:  48  ; CV error= 0.1979  ; std dev= 0.0202"
[1] "Round  6  ; Selected feature:  33  ; CV error= 0.1913  ; std dev= 0.0203"
[1] "Round  7  ; Selected feature:  47  ; CV error= 0.1822  ; std dev= 0.0254"
[1] "Round  8  ; Selected feature:  27  ; CV error= 0.176  ; std dev= 0.0211"
[1] "Round  9  ; Selected feature:  28  ; CV error= 0.175  ; std dev= 0.021"
[1] "Round  10  ; Selected feature:  44  ; CV error= 0.1741  ; std dev= 0.0224"
[1] "Round  11  ; Selected feature:  19  ; CV error= 0.1738  ; std dev= 0.0227"
[1] "Round  12  ; Selected feature:  29  ; CV error= 0.1736  ; std dev= 0.0232"
[1] "Round  13  ; Selected feature:  54  ; CV error= 0.1711  ; std dev= 0.0234"
[1] "Round  14  ; Selected feature:  74  ; CV error= 0.1705  ; std dev= 0.0236"
[1] "Round  15  ; Selected feature:  67  ; CV error= 0.1702  ; std dev= 0.0236"
[1] "Round  16  ; Selected feature:  31  ; CV error= 0.1702  ; std dev= 0.0236"
[1] "Round  17  ; Selected feature:  22  ; CV error= 0.1702  ; std dev= 0.0237"
[1] "Round  18  ; Selected feature:  42  ; CV error= 0.1703  ; std dev= 0.0237"
[1] "Round  19  ; Selected feature:  9  ; CV error= 0.1703  ; std dev= 0.0237"
[1] "Round  20  ; Selected feature:  23  ; CV error= 0.1704  ; std dev= 0.0238"
[1] "Round  21  ; Selected feature:  25  ; CV error= 0.1705  ; std dev= 0.0237"
[1] "Round  22  ; Selected feature:  53  ; CV error= 0.1667  ; std dev= 0.0242"
[1] "Round  23  ; Selected feature:  16  ; CV error= 0.1663  ; std dev= 0.0243"
[1] "Round  24  ; Selected feature:  38  ; CV error= 0.166  ; std dev= 0.0241"
[1] "Round  25  ; Selected feature:  5  ; CV error= 0.1661  ; std dev= 0.0239"
[1] "Round  26  ; Selected feature:  39  ; CV error= 0.166  ; std dev= 0.024"
[1] "Round  27  ; Selected feature:  60  ; CV error= 0.1661  ; std dev= 0.024"
[1] "Round  28  ; Selected feature:  3  ; CV error= 0.1662  ; std dev= 0.024"
[1] "Round  29  ; Selected feature:  72  ; CV error= 0.1663  ; std dev= 0.024"
[1] "Round  30  ; Selected feature:  70  ; CV error= 0.1653  ; std dev= 0.0234"
[1] "Round  31  ; Selected feature:  71  ; CV error= 0.1655  ; std dev= 0.0234"
[1] "Round  32  ; Selected feature:  4  ; CV error= 0.1656  ; std dev= 0.0234"
[1] "Round  33  ; Selected feature:  17  ; CV error= 0.1657  ; std dev= 0.0234"
[1] "Round  34  ; Selected feature:  8  ; CV error= 0.166  ; std dev= 0.0233"
[1] "Round  35  ; Selected feature:  49  ; CV error= 0.1661  ; std dev= 0.0235"
[1] "Round  36  ; Selected feature:  61  ; CV error= 0.1664  ; std dev= 0.0237"
[1] "Round  37  ; Selected feature:  12  ; CV error= 0.1661  ; std dev= 0.023"
[1] "Round  38  ; Selected feature:  14  ; CV error= 0.1663  ; std dev= 0.0228"
[1] "Round  39  ; Selected feature:  10  ; CV error= 0.1664  ; std dev= 0.0228"
[1] "Round  40  ; Selected feature:  77  ; CV error= 0.1667  ; std dev= 0.0231"
[1] "Round  41  ; Selected feature:  65  ; CV error= 0.1672  ; std dev= 0.0233"
[1] "Round  42  ; Selected feature:  7  ; CV error= 0.1675  ; std dev= 0.0233"
[1] "Round  43  ; Selected feature:  37  ; CV error= 0.1748  ; std dev= 0.0428"
[1] "Round  44  ; Selected feature:  6  ; CV error= 0.1715  ; std dev= 0.0358"
[1] "Round  45  ; Selected feature:  57  ; CV error= 0.1703  ; std dev= 0.0321"
[1] "Round  46  ; Selected feature:  20  ; CV error= 0.1695  ; std dev= 0.0288"
[1] "Round  47  ; Selected feature:  51  ; CV error= 0.1691  ; std dev= 0.0284"
[1] "Round  48  ; Selected feature:  41  ; CV error= 0.1687  ; std dev= 0.028"
[1] "Round  49  ; Selected feature:  52  ; CV error= 0.1687  ; std dev= 0.028"
[1] "Round  50  ; Selected feature:  56  ; CV error= 0.1687  ; std dev= 0.028"
[1] "Round  51  ; Selected feature:  30  ; CV error= 0.1688  ; std dev= 0.0281"
[1] "Round  52  ; Selected feature:  32  ; CV error= 0.1688  ; std dev= 0.0281"
[1] "Round  53  ; Selected feature:  15  ; CV error= 0.169  ; std dev= 0.0282"
[1] "Round  54  ; Selected feature:  76  ; CV error= 0.1696  ; std dev= 0.0276"
[1] "Round  55  ; Selected feature:  75  ; CV error= 0.1702  ; std dev= 0.0281"
[1] "Round  56  ; Selected feature:  35  ; CV error= 0.17  ; std dev= 0.0278"
[1] "Round  57  ; Selected feature:  2  ; CV error= 0.1765  ; std dev= 0.0301"
[1] "Round  58  ; Selected feature:  24  ; CV error= 0.1709  ; std dev= 0.0245"
[1] "Round  59  ; Selected feature:  36  ; CV error= 0.1706  ; std dev= 0.0238"
[1] "Round  60  ; Selected feature:  26  ; CV error= 0.1706  ; std dev= 0.0238"
[1] "Round  61  ; Selected feature:  34  ; CV error= 0.1706  ; std dev= 0.0238"
[1] "Round  62  ; Selected feature:  63  ; CV error= 0.1713  ; std dev= 0.0235"
[1] "Round  63  ; Selected feature:  45  ; CV error= 0.1672  ; std dev= 0.0226"
[1] "Round  64  ; Selected feature:  69  ; CV error= 0.1645  ; std dev= 0.0233"
[1] "Round  65  ; Selected feature:  64  ; CV error= 0.1627  ; std dev= 0.0226"
[1] "Round  66  ; Selected feature:  50  ; CV error= 0.1616  ; std dev= 0.0224"
[1] "Round  67  ; Selected feature:  18  ; CV error= 0.1608  ; std dev= 0.0269"
[1] "Round  68  ; Selected feature:  66  ; CV error= 0.1595  ; std dev= 0.0256"
[1] "Round  69  ; Selected feature:  78  ; CV error= 0.1588  ; std dev= 0.0241"
[1] "Round  70  ; Selected feature:  43  ; CV error= 0.1585  ; std dev= 0.0235"
[1] "Round  71  ; Selected feature:  1  ; CV error= 0.1583  ; std dev= 0.0218"
[1] "Round  72  ; Selected feature:  21  ; CV error= 0.1583  ; std dev= 0.0218"
[1] "Round  73  ; Selected feature:  40  ; CV error= 0.1583  ; std dev= 0.0218"
[1] "Round  74  ; Selected feature:  62  ; CV error= 0.1586  ; std dev= 0.0217"
[1] "Round  75  ; Selected feature:  11  ; CV error= 0.1592  ; std dev= 0.021"
[1] "Round  76  ; Selected feature:  13  ; CV error= 0.1592  ; std dev= 0.021"
[1] "Round  77  ; Selected feature:  68  ; CV error= 0.1598  ; std dev= 0.0215"
[1] "Round  78  ; Selected feature:  59  ; CV error= 0.1736  ; std dev= 0.0446"

## wrap features svm

[1] "Round  1  ; Selected feature:  9  ; CV error= 0.3991  ; std dev= 0.0287"
[1] "Round  2  ; Selected feature:  46  ; CV error= 0.2296  ; std dev= 0.019"
[1] "Round  3  ; Selected feature:  58  ; CV error= 0.1977  ; std dev= 0.0195"
[1] "Round  4  ; Selected feature:  54  ; CV error= 0.176  ; std dev= 0.0212"
[1] "Round  5  ; Selected feature:  48  ; CV error= 0.1627  ; std dev= 0.0186"
[1] "Round  6  ; Selected feature:  47  ; CV error= 0.1524  ; std dev= 0.0191"
[1] "Round  7  ; Selected feature:  45  ; CV error= 0.1463  ; std dev= 0.0186"
[1] "Round  8  ; Selected feature:  53  ; CV error= 0.1424  ; std dev= 0.0176"
[1] "Round  9  ; Selected feature:  66  ; CV error= 0.1385  ; std dev= 0.0189"
[1] "Round  10  ; Selected feature:  68  ; CV error= 0.1365  ; std dev= 0.018"
[1] "Round  11  ; Selected feature:  6  ; CV error= 0.1342  ; std dev= 0.0183"
[1] "Round  12  ; Selected feature:  21  ; CV error= 0.1327  ; std dev= 0.0189"
[1] "Round  13  ; Selected feature:  10  ; CV error= 0.1314  ; std dev= 0.0187"
[1] "Round  14  ; Selected feature:  63  ; CV error= 0.1302  ; std dev= 0.0196"
[1] "Round  15  ; Selected feature:  28  ; CV error= 0.1298  ; std dev= 0.019"
[1] "Round  16  ; Selected feature:  14  ; CV error= 0.1296  ; std dev= 0.0188"
[1] "Round  17  ; Selected feature:  11  ; CV error= 0.1295  ; std dev= 0.0188"
[1] "Round  18  ; Selected feature:  13  ; CV error= 0.1295  ; std dev= 0.0202"
[1] "Round  19  ; Selected feature:  62  ; CV error= 0.1292  ; std dev= 0.0204"
[1] "Round  20  ; Selected feature:  49  ; CV error= 0.1289  ; std dev= 0.0203"
[1] "Round  21  ; Selected feature:  12  ; CV error= 0.1286  ; std dev= 0.0204"
[1] "Round  22  ; Selected feature:  55  ; CV error= 0.1286  ; std dev= 0.0197"
[1] "Round  23  ; Selected feature:  15  ; CV error= 0.1287  ; std dev= 0.0207"
[1] "Round  24  ; Selected feature:  7  ; CV error= 0.1284  ; std dev= 0.0211"

[1] "feature:  58 mean error:  0.4069 sd error:  0.029"
[1] "feature:  31 mean error:  0.2623 sd error:  0.0157"
[1] "feature:  43 mean error:  0.2285 sd error:  0.0262"
[1] "feature:  40 mean error:  0.2132 sd error:  0.0224"
[1] "feature:  33 mean error:  0.1979 sd error:  0.0202"
[1] "feature:  25 mean error:  0.1913 sd error:  0.0203"
[1] "feature:  32 mean error:  0.1822 sd error:  0.0254"
[1] "feature:  19 mean error:  0.176 sd error:  0.0211"
[1] "feature:  20 mean error:  0.175 sd error:  0.021"
[1] "feature:  29 mean error:  0.1741 sd error:  0.0224"
[1] "feature:  21 mean error:  0.174 sd error:  0.0229"
[1] "feature:  6 mean error:  0.172 sd error:  0.0218"
[1] "feature:  10 mean error:  0.172 sd error:  0.0216"
[1] "feature:  23 mean error:  0.1719 sd error:  0.0216"
[1] "feature:  5 mean error:  0.172 sd error:  0.0215"
[1] "feature:  59 mean error:  0.1714 sd error:  0.0218"
[1] "feature:  57 mean error:  0.1715 sd error:  0.0218"
[1] "feature:  27 mean error:  0.171 sd error:  0.0213"
[1] "feature:  52 mean error:  0.171 sd error:  0.0215"
[1] "feature:  9 mean error:  0.1711 sd error:  0.0216"
[1] "feature:  55 mean error:  0.1696 sd error:  0.0205"
[1] "feature:  56 mean error:  0.1697 sd error:  0.0205"
[1] "feature:  3 mean error:  0.1699 sd error:  0.0206"
[1] "feature:  8 mean error:  0.1701 sd error:  0.0206"
[1] "feature:  17 mean error:  0.1701 sd error:  0.0205"
[1] "feature:  38 mean error:  0.1698 sd error:  0.0214"
[1] "feature:  51 mean error:  0.1683 sd error:  0.022"
[1] "feature:  37 mean error:  0.1683 sd error:  0.0218"
[1] "feature:  45 mean error:  0.1684 sd error:  0.0218"
[1] "feature:  7 mean error:  0.1684 sd error:  0.0219"

58, 31, 43, 40, 33, 25, 32, 19, 20, 29, 21, 6, 10, 23, 5, 59, 57, 27, 52, 9, 55, 56, 3, 8, 17, 38, 51, 37, 45, 7
