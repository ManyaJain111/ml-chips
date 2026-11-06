| case     | model         |   degree |    alpha |   p95_err_pct |   median_err_pct |    fit_s |   max_features |   min_samples_leaf |   num_leaves |   learning_rate | hidden_layer_sizes   |
|:---------|:--------------|---------:|---------:|--------------:|-----------------:|---------:|---------------:|-------------------:|-------------:|----------------:|:---------------------|
| nmos_on  | poly_ridge    |        2 |   1e-06  |     3.744e+04 |           69.29  |  0.0469  |         nan    |                nan |          nan |          nan    | nan                  |
| nmos_on  | poly_ridge    |        2 |   0.0001 |     3.744e+04 |           69.25  |  0.05181 |         nan    |                nan |          nan |          nan    | nan                  |
| nmos_on  | poly_ridge    |        2 |   0.01   |     3.732e+04 |           69.32  |  0.05988 |         nan    |                nan |          nan |          nan    | nan                  |
| nmos_on  | poly_ridge    |        3 |   1e-06  |   463         |           43.44  |  0.7964  |         nan    |                nan |          nan |          nan    | nan                  |
| nmos_on  | poly_ridge    |        3 |   0.0001 |   460.4       |           43.34  |  0.6134  |         nan    |                nan |          nan |          nan    | nan                  |
| nmos_on  | poly_ridge    |        3 |   0.01   |   405.6       |           43.07  |  0.6383  |         nan    |                nan |          nan |          nan    | nan                  |
| nmos_on  | random_forest |      nan | nan      |   121.2       |           26.51  |  0.755   |           0.33 |                  1 |          nan |          nan    | nan                  |
| nmos_on  | random_forest |      nan | nan      |   132.4       |           28.29  |  0.6095  |           0.33 |                  3 |          nan |          nan    | nan                  |
| nmos_on  | random_forest |      nan | nan      |   115.9       |           23.92  |  1.281   |           0.66 |                  1 |          nan |          nan    | nan                  |
| nmos_on  | random_forest |      nan | nan      |   127.5       |           25.14  |  1.075   |           0.66 |                  3 |          nan |          nan    | nan                  |
| nmos_on  | random_forest |      nan | nan      |   125.5       |           24.12  |  1.864   |           1    |                  1 |          nan |          nan    | nan                  |
| nmos_on  | random_forest |      nan | nan      |   132         |           25.69  |  1.609   |           1    |                  3 |          nan |          nan    | nan                  |
| nmos_on  | gbdt          |      nan | nan      |    47.2       |            9.784 | 11.5     |         nan    |                nan |           15 |            0.03 | nan                  |
| nmos_on  | gbdt          |      nan | nan      |    44.15      |            9.803 | 11.22    |         nan    |                nan |           15 |            0.1  | nan                  |
| nmos_on  | gbdt          |      nan | nan      |    44.1       |            8.58  | 21.76    |         nan    |                nan |           31 |            0.03 | nan                  |
| nmos_on  | gbdt          |      nan | nan      |    49.66      |           10     | 21.5     |         nan    |                nan |           31 |            0.1  | nan                  |
| nmos_on  | gbdt          |      nan | nan      |    44.43      |            8.92  | 42.73    |         nan    |                nan |           63 |            0.03 | nan                  |
| nmos_on  | gbdt          |      nan | nan      |    50.79      |           10.69  | 40.11    |         nan    |                nan |           63 |            0.1  | nan                  |
| nmos_on  | mlp           |      nan |   1e-05  |   109.5       |           23.3   |  3.131   |         nan    |                nan |          nan |          nan    | (64, 64)             |
| nmos_on  | mlp           |      nan |   0.001  |   104.9       |           19.32  |  5.864   |         nan    |                nan |          nan |          nan    | (64, 64)             |
| nmos_on  | mlp           |      nan |   1e-05  |   109.6       |           22.2   |  5.446   |         nan    |                nan |          nan |          nan    | (128, 128)           |
| nmos_on  | mlp           |      nan |   0.001  |   101.1       |           21.18  |  4.604   |         nan    |                nan |          nan |          nan    | (128, 128)           |
| nmos_on  | mlp           |      nan |   1e-05  |    62.02      |           15.96  |  7.057   |         nan    |                nan |          nan |          nan    | (128, 128, 128)      |
| nmos_on  | mlp           |      nan |   0.001  |    62.47      |           16.3   |  8.443   |         nan    |                nan |          nan |          nan    | (128, 128, 128)      |
| nmos_off | poly_ridge    |        2 |   1e-06  |  1952         |           61.67  |  0.03475 |         nan    |                nan |          nan |          nan    | nan                  |
| nmos_off | poly_ridge    |        2 |   0.0001 |  1986         |           61.76  |  0.05066 |         nan    |                nan |          nan |          nan    | nan                  |
| nmos_off | poly_ridge    |        2 |   0.01   |  1761         |           61.59  |  0.04666 |         nan    |                nan |          nan |          nan    | nan                  |
| nmos_off | poly_ridge    |        3 |   1e-06  |  2233         |           51.73  |  0.9083  |         nan    |                nan |          nan |          nan    | nan                  |
| nmos_off | poly_ridge    |        3 |   0.0001 |  2070         |           51.94  |  0.6152  |         nan    |                nan |          nan |          nan    | nan                  |
| nmos_off | poly_ridge    |        3 |   0.01   |  1899         |           52.06  |  0.8847  |         nan    |                nan |          nan |          nan    | nan                  |
| nmos_off | random_forest |      nan | nan      |    77.68      |           10.63  |  0.8287  |           0.33 |                  1 |          nan |          nan    | nan                  |
| nmos_off | random_forest |      nan | nan      |    80.24      |           11.01  |  0.6704  |           0.33 |                  3 |          nan |          nan    | nan                  |
| nmos_off | random_forest |      nan | nan      |    77.23      |            9.472 |  1.442   |           0.66 |                  1 |          nan |          nan    | nan                  |
| nmos_off | random_forest |      nan | nan      |    80.19      |            9.567 |  1.22    |           0.66 |                  3 |          nan |          nan    | nan                  |
| nmos_off | random_forest |      nan | nan      |    80.66      |            8.967 |  2.171   |           1    |                  1 |          nan |          nan    | nan                  |
| nmos_off | random_forest |      nan | nan      |    81.34      |            9.694 |  1.855   |           1    |                  3 |          nan |          nan    | nan                  |
| nmos_off | gbdt          |      nan | nan      |    80.7       |           10.55  |  9.145   |         nan    |                nan |           15 |            0.03 | nan                  |
| nmos_off | gbdt          |      nan | nan      |   106.5       |           12.17  | 11.07    |         nan    |                nan |           15 |            0.1  | nan                  |
| nmos_off | gbdt          |      nan | nan      |    58.75      |            8.337 | 17.65    |         nan    |                nan |           31 |            0.03 | nan                  |
| nmos_off | gbdt          |      nan | nan      |    77.91      |            9.16  | 21.31    |         nan    |                nan |           31 |            0.1  | nan                  |
| nmos_off | gbdt          |      nan | nan      |    56.78      |            6.543 | 41.93    |         nan    |                nan |           63 |            0.03 | nan                  |
| nmos_off | gbdt          |      nan | nan      |    71.9       |            9.265 | 39.6     |         nan    |                nan |           63 |            0.1  | nan                  |
| nmos_off | mlp           |      nan |   1e-05  |    52.76      |           11.38  |  4.688   |         nan    |                nan |          nan |          nan    | (64, 64)             |
| nmos_off | mlp           |      nan |   0.001  |    61.7       |           11.86  |  4.126   |         nan    |                nan |          nan |          nan    | (64, 64)             |
| nmos_off | mlp           |      nan |   1e-05  |    55.14      |           14.49  |  6.941   |         nan    |                nan |          nan |          nan    | (128, 128)           |
| nmos_off | mlp           |      nan |   0.001  |   113         |           14.25  |  7.936   |         nan    |                nan |          nan |          nan    | (128, 128)           |
| nmos_off | mlp           |      nan |   1e-05  |    44.56      |            9.613 |  9.904   |         nan    |                nan |          nan |          nan    | (128, 128, 128)      |
| nmos_off | mlp           |      nan |   0.001  |    36.89      |            7.797 |  9.339   |         nan    |                nan |          nan |          nan    | (128, 128, 128)      |
| pmos_on  | poly_ridge    |        2 |   1e-06  |     1.793e+04 |           67.9   |  0.1065  |         nan    |                nan |          nan |          nan    | nan                  |
| pmos_on  | poly_ridge    |        2 |   0.0001 |     1.79e+04  |           67.9   |  0.05089 |         nan    |                nan |          nan |          nan    | nan                  |
| pmos_on  | poly_ridge    |        2 |   0.01   |     1.811e+04 |           67.82  |  0.07748 |         nan    |                nan |          nan |          nan    | nan                  |
| pmos_on  | poly_ridge    |        3 |   1e-06  |   440.8       |           45.17  |  0.7668  |         nan    |                nan |          nan |          nan    | nan                  |
| pmos_on  | poly_ridge    |        3 |   0.0001 |   421.7       |           45.24  |  0.5938  |         nan    |                nan |          nan |          nan    | nan                  |
| pmos_on  | poly_ridge    |        3 |   0.01   |   420.6       |           47.04  |  0.5512  |         nan    |                nan |          nan |          nan    | nan                  |
| pmos_on  | random_forest |      nan | nan      |   116.4       |           25.37  |  0.7773  |           0.33 |                  1 |          nan |          nan    | nan                  |
| pmos_on  | random_forest |      nan | nan      |   121.9       |           27.1   |  0.6258  |           0.33 |                  3 |          nan |          nan    | nan                  |
| pmos_on  | random_forest |      nan | nan      |   109.5       |           23.99  |  1.327   |           0.66 |                  1 |          nan |          nan    | nan                  |
| pmos_on  | random_forest |      nan | nan      |   119         |           25.65  |  1.122   |           0.66 |                  3 |          nan |          nan    | nan                  |
| pmos_on  | random_forest |      nan | nan      |   115.3       |           25.08  |  2       |           1    |                  1 |          nan |          nan    | nan                  |
| pmos_on  | random_forest |      nan | nan      |   121.8       |           26.2   |  1.692   |           1    |                  3 |          nan |          nan    | nan                  |
| pmos_on  | gbdt          |      nan | nan      |    45.11      |           10.22  | 11.7     |         nan    |                nan |           15 |            0.03 | nan                  |
| pmos_on  | gbdt          |      nan | nan      |    42.78      |            9.381 | 11.08    |         nan    |                nan |           15 |            0.1  | nan                  |
| pmos_on  | gbdt          |      nan | nan      |    40.71      |            8.412 | 19.57    |         nan    |                nan |           31 |            0.03 | nan                  |
| pmos_on  | gbdt          |      nan | nan      |    44.49      |            9.624 | 21.23    |         nan    |                nan |           31 |            0.1  | nan                  |
| pmos_on  | gbdt          |      nan | nan      |    42.82      |            8.269 | 42.05    |         nan    |                nan |           63 |            0.03 | nan                  |
| pmos_on  | gbdt          |      nan | nan      |    50.28      |            9.767 | 39.27    |         nan    |                nan |           63 |            0.1  | nan                  |
| pmos_on  | mlp           |      nan |   1e-05  |    95.52      |           22.57  |  3.604   |         nan    |                nan |          nan |          nan    | (64, 64)             |
| pmos_on  | mlp           |      nan |   0.001  |    66.14      |           16.7   |  5.546   |         nan    |                nan |          nan |          nan    | (64, 64)             |
| pmos_on  | mlp           |      nan |   1e-05  |   102.4       |           20.4   |  5.637   |         nan    |                nan |          nan |          nan    | (128, 128)           |
| pmos_on  | mlp           |      nan |   0.001  |    70.78      |           16.74  |  6.437   |         nan    |                nan |          nan |          nan    | (128, 128)           |
| pmos_on  | mlp           |      nan |   1e-05  |    41.22      |           10.11  | 13.37    |         nan    |                nan |          nan |          nan    | (128, 128, 128)      |
| pmos_on  | mlp           |      nan |   0.001  |    53.8       |           12.97  | 17.67    |         nan    |                nan |          nan |          nan    | (128, 128, 128)      |
| pmos_off | poly_ridge    |        2 |   1e-06  |  1653         |           57.72  |  0.0451  |         nan    |                nan |          nan |          nan    | nan                  |
| pmos_off | poly_ridge    |        2 |   0.0001 |  1690         |           57.8   |  0.04425 |         nan    |                nan |          nan |          nan    | nan                  |
| pmos_off | poly_ridge    |        2 |   0.01   |  1593         |           57.85  |  0.04898 |         nan    |                nan |          nan |          nan    | nan                  |
| pmos_off | poly_ridge    |        3 |   1e-06  |  1057         |           48.71  |  0.9137  |         nan    |                nan |          nan |          nan    | nan                  |
| pmos_off | poly_ridge    |        3 |   0.0001 |   951.6       |           48.85  |  0.6084  |         nan    |                nan |          nan |          nan    | nan                  |
| pmos_off | poly_ridge    |        3 |   0.01   |   946.6       |           49.48  |  0.7637  |         nan    |                nan |          nan |          nan    | nan                  |
| pmos_off | random_forest |      nan | nan      |    77.3       |            9.369 |  0.8592  |           0.33 |                  1 |          nan |          nan    | nan                  |
| pmos_off | random_forest |      nan | nan      |    78.11      |           10.03  |  0.7102  |           0.33 |                  3 |          nan |          nan    | nan                  |
| pmos_off | random_forest |      nan | nan      |    77.12      |            9.009 |  1.53    |           0.66 |                  1 |          nan |          nan    | nan                  |
| pmos_off | random_forest |      nan | nan      |    78.6       |            9.616 |  1.262   |           0.66 |                  3 |          nan |          nan    | nan                  |
| pmos_off | random_forest |      nan | nan      |    78.8       |            9.295 |  2.3     |           1    |                  1 |          nan |          nan    | nan                  |
| pmos_off | random_forest |      nan | nan      |    79.9       |            9.315 |  1.934   |           1    |                  3 |          nan |          nan    | nan                  |
| pmos_off | gbdt          |      nan | nan      |    64.48      |           10.05  | 10.72    |         nan    |                nan |           15 |            0.03 | nan                  |
| pmos_off | gbdt          |      nan | nan      |    72.99      |            9.664 | 10.83    |         nan    |                nan |           15 |            0.1  | nan                  |
| pmos_off | gbdt          |      nan | nan      |    57.8       |            8.046 | 21.3     |         nan    |                nan |           31 |            0.03 | nan                  |
| pmos_off | gbdt          |      nan | nan      |    67.68      |            8.394 | 21.01    |         nan    |                nan |           31 |            0.1  | nan                  |
| pmos_off | gbdt          |      nan | nan      |    57.69      |            6.895 | 40.83    |         nan    |                nan |           63 |            0.03 | nan                  |
| pmos_off | gbdt          |      nan | nan      |    70.48      |            8.365 | 41.4     |         nan    |                nan |           63 |            0.1  | nan                  |
| pmos_off | mlp           |      nan |   1e-05  |    75.86      |           15.13  |  4.555   |         nan    |                nan |          nan |          nan    | (64, 64)             |
| pmos_off | mlp           |      nan |   0.001  |    60.24      |           11.83  |  3.877   |         nan    |                nan |          nan |          nan    | (64, 64)             |
| pmos_off | mlp           |      nan |   1e-05  |    65.86      |           12.13  |  6.444   |         nan    |                nan |          nan |          nan    | (128, 128)           |
| pmos_off | mlp           |      nan |   0.001  |    53.48      |           12.16  |  5.681   |         nan    |                nan |          nan |          nan    | (128, 128)           |
| pmos_off | mlp           |      nan |   1e-05  |    33.98      |            6.741 | 12.33    |         nan    |                nan |          nan |          nan    | (128, 128, 128)      |
| pmos_off | mlp           |      nan |   0.001  |    41.39      |            7.714 | 18.05    |         nan    |                nan |          nan |          nan    | (128, 128, 128)      |
