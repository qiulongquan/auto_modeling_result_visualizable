### 自动模型构筑自动超参数优化，可视化精度比较程序

```
工具要集合 7 种超参数优化方法，和 8 种模型。
功能要求：

1.  可以自动进行模型构筑和学习，并获取 mse 指标。
2.  所有的模型和优化方法运行后，自动按照 mse 进行排列（显示那种模型，并使用了那种优化方法）。
3.  每种模型构筑的时候都自动保存模型，并记录实验数据。
4.  显示整体进行进度，并逐步输出结果记录（使用 logging）。


系统部署要求:
部署在AWS上面，可以使用GPU提速训练NN神经网络模型
可以使用jupyter导入本程序【模型自动训练模块】，加入简单的代码就可以完成整个测试和数据图表输出


未来加入功能
加入深度学习模型，可以使用AWS的GPU来加速训练，并获得更高的精度

```

### 程序文件结构

```
├── Figure_1.png
├── Figure_2.png
├── Figure_3.png
├── README.md
├── documents
├── log
├── model_and_record
│   ├── BO-GP
│   │   ├── ANN
│   │   │   └── ANN
│   │   │       ├── assets
│   │   │       ├── saved_model.pb
│   │   │       └── variables
│   │   │           ├── variables.data-00000-of-00001
│   │   │           └── variables.index
│   │   ├── KNN
│   │   │   └── KNN.pkl
│   │   ├── RandomForestRegressor
│   │   │   └── RandomForestRegressor.pkl
│   │   ├── SVR
│   │   │   └── SVR.pkl
│   │   └── bo_score_df.pkl
│   ├── BO-TPE
│   │   ├── ANN
│   │   │   └── ANN
│   │   │       ├── assets
│   │   │       ├── saved_model.pb
│   │   │       └── variables
│   │   │           ├── variables.data-00000-of-00001
│   │   │           └── variables.index
│   │   ├── BO_TPE_score_df.pkl
│   │   ├── KNN
│   │   │   └── KNN.pkl
│   │   ├── NGBoost
│   │   │   └── NGBoost.pkl
│   │   ├── RandomForestRegressor
│   │   │   └── RandomForestRegressor.pkl
│   │   └── SVR
│   │       └── SVR.pkl
│   ├── Optuna
│   │   ├── ANN
│   │   │   └── ANN
│   │   │       ├── assets
│   │   │       ├── saved_model.pb
│   │   │       └── variables
│   │   │           ├── variables.data-00000-of-00001
│   │   │           └── variables.index
│   │   ├── GradientBoostingRegressor
│   │   │   └── GradientBoostingRegressor.pkl
│   │   ├── KNN
│   │   │   └── KNN.pkl
│   │   ├── Optuna_score_df.pkl
│   │   ├── RandomForestRegressor
│   │   │   └── RandomForestRegressor.pkl
│   │   └── SVR
│   │       └── SVR.pkl
│   ├── all_result.csv
│   ├── all_result.pkl
│   ├── baseline
│   │   ├── ann
│   │   │   └── ann.pkl
│   │   ├── base_score_df.pkl
│   │   ├── knn
│   │   │   └── knn.pkl
│   │   ├── randomforest
│   │   │   └── randomforest.pkl
│   │   └── svr
│   │       └── svr.pkl
│   ├── gp_minimize
│   │   ├── KNN
│   │   │   └── KNN.pkl
│   │   ├── RandomForestRegressor
│   │   │   └── RandomForestRegressor.pkl
│   │   ├── SVR
│   │   │   └── SVR.pkl
│   │   └── gp_minimize_score_df.pkl
│   ├── grid_search
│   │   ├── ANN
│   │   │   └── ANN
│   │   │       ├── assets
│   │   │       ├── saved_model.pb
│   │   │       └── variables
│   │   │           ├── variables.data-00000-of-00001
│   │   │           └── variables.index
│   │   ├── KNN
│   │   │   └── KNN.pkl
│   │   ├── RandomForestRegressor
│   │   │   └── RandomForestRegressor.pkl
│   │   ├── SVR
│   │   │   └── SVR.pkl
│   │   └── gridsearch_score_df.pkl
│   └── random_search
│       ├── ANN
│       │   └── ANN
│       │       ├── assets
│       │       ├── saved_model.pb
│       │       └── variables
│       │           ├── variables.data-00000-of-00001
│       │           └── variables.index
│       ├── KNN
│       │   └── KNN.pkl
│       ├── RandomForestRegressor
│       │   └── RandomForestRegressor.pkl
│       ├── SVR
│       │   └── SVR.pkl
│       └── randomsearch_score_df.pkl
├── models_MSE_sort.png
├── models_MSE_sort_1.png
├── models_MSE_summary.png
├── requirements.txt
├── src
│   ├── ANN_model.py
│   ├── baseline.py
│   ├── basic_config.py
│   ├── bo_gp.py
│   ├── bo_tpe.py
│   ├── common
│   │   ├── collection_result_process.py
│   │   ├── get_logger_instance.py
│   │   ├── msyh.ttc
│   │   └── save_model_and_result_record.py
│   ├── gp_minimize.py
│   ├── grid_search.py
│   ├── hyper_optimize_main_process.py
│   ├── model_and_record
│   │   ├── baseline
│   │   │   ├── base_score_df.pkl
│   │   │   ├── knn
│   │   │   │   └── knn.pkl
│   │   │   ├── randomforest
│   │   │   │   └── randomforest.pkl
│   │   │   └── svr
│   │   │       └── svr.pkl
│   │   └── grid_search
│   │       ├── KNN
│   │       │   └── KNN.pkl
│   │       ├── RandomForestRegressor
│   │       │   └── RandomForestRegressor.pkl
│   │       ├── SVR
│   │       │   └── SVR.pkl
│   │       └── gridsearch_score_df.pkl
│   ├── optuna_optimizer.py
│   └── random_search.py
├── 结论.txt
└── 要件定義
    └── 機能仕様書.xlsx
```
