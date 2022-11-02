# Used Cars Price Prediction using Machine Learning

### 1. Abstract

Rapid price fluctuations caused by various economical factors like inflation and the supply chain crisis, as well as political factors like wars, have dramatically affected the prices of consumer goods. Machine learning has proven to be a game changer in price optimization with its ability to process massive amounts of data, consider a large number of variables and predict with high accuracy in comparison to traditional pricing methods. This project aims to experiment with various supervised machine learning algorithms and build highly accurate price predictors. The [“US Used Cars Dataset”](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) was used to train the algorithms. A thorough data cleaning process was carried out to impute missing values and trim certain features along with an exploratory data analysis. Following that, data preprocessing took place during which the categorical features were one-hot encoded and the numerical features were scaled using a Robust Scaler. Ten different regression learning algorithms were trained and evaluated, five of which were ensemble-based, and the other five were linear models. All of the models were evaluated based on their mean absolute error (MAE), root mean square error (RMSE) and r-squared (R2) scores. The ensemble methods performed better than the linear models in all three metrics. The best three models were then tuned using randomized search cross-validation. The tuned models were able to achieve better scores, with the CatBoost taking the lead followed closely by LightGBM. 

### 2. **Introduction**

Machine learning has become a beneficial tool in many business applications. Various businesses utilize machine learning to reduce costs, increase sales, and for other reasons. In this project, multiple supervised learning algorithms were utilized for an important business problem, which is pricing, and specifically the pricing of used cars. Pricing strategies can vary depending on the use case, and their complexity can also differ. Due to many factors like increased inflation, supply chain issues, and wars, pricing has become such an important yet difficult task. The cars industry is a great example of how pricing can drastically affect the business, and why it is important to enforce scientific and data driven pricing methods. Machine learning is used as one of the most effective tools for price optimization due to its ability to process large amounts of data and consider a wide range of features when predicting the price of a vehicle making it more advantageous than traditional pricing methods. [1]

The aim of this project is to assemble and train a powerful regression model to accurately predict a car’s price given some features. Ten supervised machine learning algorithms were trained and evaluated on a large dataset with around 400,000 rows of used cars data that is scraped off of craigslist [2], and the best three performing models were chosen for further hyperparameter tuning. The models were evaluated based on the mean absolute error (MAE), root mean square error (RMSE), and R squared (R2) scores. Two main types of learning algorithms were used, ensemble and linear models. Overall the ensemble models performed better than linear models in terms of efficiency and accuracy. The Catboost algorithm had the best performance although it was slightly memory intensive. A more memory friendly algorithm with very similar performance was the LightGBM.

The implementation of this project was entirely done on a jupyter notebook using CPUs only and the following Python libraries: Pandas, NumPy, Seaborn, Matplotlib, Plotly, MissingNo, Pickle, Time, Scikit-learn, Lightgbm, XGBoost and CatBoost.

The report begins by going briefly over similar works, then describing the data, going through the data cleaning phase and exploratory data analysis. It then goes through the splitting and preprocessing methods implemented, followed by the training and evaluation of the learning algorithms. After that the hyperparameter tuning approach is explained. The final section concludes with a brief recap and summary of the results.

### 3. **Related Work**

Similar works have been able to achieve varying results. One article [3] discusses the results achieved when training 5 different models that are random forest, linear regression, ridge regression, lasso, kNN, and XGBoost on the same dataset. The best model was the random forest with a MAE of 2047.74, and a RMSE of 3960.11. Another implementation [4] also on the same dataset experimented with a variety of different algorithms, and the best results were achieved with a linear regression model with an r-squared of 87% on train and test sets with faster training time compared to the other models. Also in their implementation, the tree based methods surprisingly performed worse unlike the results achieved in this project. The models used in this project outperform the results of the related works in terms of lower errors and higher r-squared scores, however, there is some overfitting that could be reduced with further tuning.

### 4. **Data Cleaning & EDA**

The US Used Cars dataset contains 426,880 rows and 26 columns with used cars data that is regularly scrapped off of craigslist. It includes recent car listings from all states in the United States and has features like year, manufacturer, model, condition, mileage, price and more. Most of the features are categorical with a few being numerical or text, one datetime feature, and two coordinate features (longitude and latitude). The table below describes all the features, their data types, the number of unique values and the ratio of null values in each.

| Feature # |  Feature Name  |        Data Type       | Number of Unique Values | Ratio of Null Values |
|:---------:|:--------------:|:----------------------:|:-----------------------:|:--------------------:|
| 0         | id             | Numerical (int)        | 426880                  | 0%                   |
| 1         | url            | Text                   | 426880                  | 0%                   |
| 2         | region         | Categorical            | 404                     | 0%                   |
| 3         | region_url     | Text                   | 413                     | 0%                   |
| 4         | price (target) | Numerical (int)        | 15655                   | 0%                   |
| 5         | year           | Numerical (float)      | 114                     | 0.3%                 |
| 6         | manufacturer   | Categorical            | 42                      | 4%                   |
| 7         | model          | Text                   | 29667                   | 1%                   |
| 8         | condition      | Categorical            | 6                       | 41%                  |
| 9         | cylinders      | Categorical            | 8                       | 42%                  |
| 10        | fuel           | Categorical            | 5                       | 0.7%                 |
| 11        | odometer       | Numerical (float)      | 104870                  | 1%                   |
| 12        | title_status   | Categorical            | 6                       | 2%                   |
| 13        | transmission   | Categorical            | 3                       | 0.6%                 |
| 14        | VIN            | Text                   | 118264                  | 38%                  |
| 15        | drive          | Categorical            | 3                       | 31%                  |
| 16        | size           | Categorical            | 4                       | 72%                  |
| 17        | type           | Categorical            | 13                      | 22%                  |
| 18        | paint_color    | Categorical            | 12                      | 31%                  |
| 19        | image_url      | Text                   | 241899                  | 0.02%                |
| 20        | description    | Text                   | 360911                  | 0.02%                |
| 21        | county         | Categorical            | 0                       | 100%                 |
| 22        | state          | Categorical            | 51                      | 0%                   |
| 23        | lat            | Numerical (coordinate) | 53181                   | 1.5%                 |
| 24        | long           | Numerical (coordinate) | 53772                   | 1.5%                 |
| 25        | posting_date   | Datetime               | 381536                  | 0.02%                |

The id, url, image_url, description and posting_date features have high cardinality that is above 50%. These features can be dropped due to the high cardinality as preprocessing these features would increase the sparsity of the data.

It is also important to look at the ratio of null values for each column. The county feature has only null values and the size feature has a lot of null values with 72% of its values being null. Both of these features can be dropped as they would not be useful.

It was now time to drop unnecessary features. Several features were dropped for a variety of reasons. The dropped columns and the reason for dropping them are explained as follows. The id and url features were not useful and had high cardinality. The region feature was also not useful, and the state feature would be kept instead. The region_url and the VIN features were not useful. The size feature had around 72% of missing values. The image_url and description features were also not very useful and had high cardinality. The county feature was also dropped because all of its values were null. Finally, the lat and long as well as the posting_date features were not needed.

After dropping these columns, there were 14 features remaining. Now, it was time to take a closer look at each of the kept features and clean them as needed. The cleaning focused on replacing missing values and trimming the data as needed. Given that this dataset is scrapped from classified listings, it is prone to errors and mistakes, and it is better to check each feature and try to clean as much as possible. Starting with the target feature, **price**; cars that are priced over 520,000 USD totaling 67 cars were removed. This was done because when looking at these cars, most of them were filled with arbitrary numbers and thus were not useful entries and would affect the learning algorithm. It was also found that there were over 36,000 cars priced with zero or very small numbers below 100 USD, and these were also removed. The number of rows was immediately reduced from 426,880 to 390,424. There were no null values in the price feature.

The **year** feature had values from 1900 to 2022, and for this feature, all cars with a year model that is older than 1960 (totalling 2308 cars) were dropped. This was done because such old cars are rare, and keeping them skewed the feature more than needed. The datatype of this feature was also changed from float to int and there were no null values in this feature.

The third feature, which is the **manufacturer**, had around 3% missing values. Given it was a relatively small portion of the data, all the rows with null values were dropped.

Fourthly, the **model** feature had many unique values and around 1% of its values were missing. Even though this feature has high cardinality, it is important to keep it as it is generally one of the most important characteristics of a car and is very important for the pricing. The missing values were dropped considering their low percentage.

The fifth feature, which is the **odometer**, is an important numeric feature. In the analysis, it can be seen that there are 273 cars with over 1,000,000 miles on the clock. All of these were dropped due to being most likely inaccurate inputs, as some were unreasonably large values in the billions.

Moving on to the sixth feature, which is the **condition** of the car; it had around 38% of its values missing. In order to impute these missing values, a custom imputation method was implemented inspired by a notebook [5] on kaggle. The average mileage of each condition was calculated and the missing values would be filled using these averages. Firstly, all cars with a model year of 2022 or newer, were filled with the ‘new’ condition regardless of their mileage. Secondly, all cars with a mileage less than or equal to the average mileage of cars with ‘like_new’ condition, were filled with ‘like_new’ condition. Similarly, cars with mileage higher than or equal to the average mileage of cars with fair condition, which is the second worst condition, were filled with ‘fair’ condition. With the current imputation, only 17% of the values remain missing. To impute these, the same approach was used, by filling the missing condition of cars with a mileage that is between the ‘like_new’ average mileage and the ‘excellent’ average mileage with ‘excellent’ condition. Likewise, the missing condition of cars with mileage that is between the ‘excellent’ average mileage and the ‘fair’ average mileage with ‘good’ condition. Now three were no missing values. ‘Good’ has a lower average mileage than the ‘like_new’ and ‘excellent’ conditions, which could be due to the fact that this data was scraped off of craigslist, where individuals list their cars for sale and set the condition of their car. Furthermore, the mileage could give an idea of the car's condition, but it will not always be accurate. Some cars might have a higher mileage and a better condition than others with lower mileage. This custom imputation provided more accurate data than other kinds of imputation like a most frequent imputation.

The **cylinders** feature also had a big percentage of missing values. As there was a large portion of missing values in this column (40%), dropping these rows was not preferable, and doing a most frequent imputation seemed like a viable way to fill the missing data. In this case, the missing values were replaced with '6 cylinders'. This definitely could have an effect on the learning algorithms, but it was a cost to deal with. Other ways of imputation like adding a new category did not seem like a good option considering it would be the new most common category.

The eighth feature is the **title status**. In this feature there were around 1% of missing values. However, instead of dropping these rows, the missing values were filled with one of the categories of this feature, which is ‘missing’.

The **transmission** and the **fuel** features had around 0.04% and 0.05% of missing values respectively, and the rows with missing values of both were all dropped due to being a very tiny portion of the data.

The body **type** feature had around 20% of missing values. To impute these, the missing values were filled with one of the feature’s categories–‘other’. Most frequent imputation was not suitable because there were many different types and the most common one (sedan) was closely followed by other types. Furthermore, forward/backward filling could produce inaccurate data. Adding a new category was possible, but there were already many categories for this feature. Lastly, dropping 20% of the data was not preferable.

Both of the **drive** and **paint color** features had around 30% of their values missing. To impute these, the best method seemed like the forward filling imputation, given that the average prices of categories were relatively close to each other. Dropping or creating a new category were also not preferable. Similarly, most frequent imputation would skew the data drastically.

The last feature **state** did not have any missing values, and was kept as is.

At this step, the data was cleaned and ready for the training. Also, here all of the features were visualized in terms of count plots, box plots or scatter plots in relation to the target feature. Some of these visuals can be seen below and all of the visualizations can be found on the jupyter notebook.

***Null values ratio:***

![null.png](https://github.com/mo-adi/used_cars_pricing/blob/master/figures/null.png)

***Average price of used cars for each state between 1960 and 2022:***

![map.png](https://github.com/mo-adi/used_cars_pricing/blob/master/figures/map.png)

***Count of cars by their paint color:***

![colors.png](https://github.com/mo-adi/used_cars_pricing/blob/master/figures/colors.png)

### 5. **Splitting & Preprocessing**

At this step, the data was ready to be split ahead of the training. Firstly, the target variable was separated and the data was split into train and test sets, with 80% and 20% split sizes respectively. As most of the features were categorical, encoding these features into a numeric representation was essential. The most common ways of encoding categorical data are label encoding and one hot encoding. Even though one hot encoding can be more expensive and would create sparse data when there is high cardinality, it was the categorical encoder of choice. Although some of the features can be considered ordinal and a label or ordinal encoder could be applied on these, a small external test was carried out during the implementation and the results of the models were better when one hot encoding was applied.

As for numerical features, given that they were not normally distributed, scaling was necessary and three scaling methods were compared and evaluated. These were Standard Scaler, Min Max Scaler and Robust Scaler. Given that the features were skewed and had outliers, the robust scaler seemed like a reasonable option as it is known to handle outliers better than other scaling methods, and thus was chosen as the numerical scaling method. A small external test was conducted to compare the performance of an ensemble algorithm and a linear model using all three scaling methods. No noticeable differences were found in the results.

Following that, two main preprocessing functions were created, the first one did the one-hot encoding for categorical features and scaled numerical features using a robust scaler. As some tree and ensemble methods do not necessarily require feature scaling like XGBoost, another function was created which only encoded the categorical features. In the next section, the training and evaluation of the learning algorithms will be discussed.

### 6. **Training & Evaluation**

As previously mentioned in the report, 10 supervised learning algorithms were trained and tested for this problem, 5 of which were ensemble-based and 5 were linear models. The ensemble-based algorithms were **Random** **Forest** **Regressor**, **Gradient** **Boosting** **Regressor**, **LightGBM**, **XGBoost**, and **CatBoost**; Whereas the linear models were **Linear** **Regressor**, **Ridge** **Regressor**, **Elastic** **Net**, **SGDRegressor**, and **Huber** **Regressor**. Initially, all models were trained with some selected hyperparameters that aimed to improve the data fit and reduce overfitting without being computationally expensive. All of the models were evaluated using 3 different metrics–**Mean Absolute Error**, **Root Mean Squared Error**, and **R2 Scores**. In addition, a plot showing the resulting prediction in comparison to the labels in the form of a model fit was generated for each model. Most of the algorithms achieved comparable results with the exception of the Elastic Net model, which performed quite poorly. Some of the models had the ability to take into consideration the categorical features and encode them directly, these were the LightGBM and CatBoost Regressors. Two LightGBM models were created, one which did not consider categorical features and took in the main preprocessed data, and the other which took in unprocessed data with the categorical features having the data type ‘category’. It is worth mentioning that both data splits were generated/split using the same random state to be able to reproduce the results. As for the CatBoost model, it can take the indices of the categorical features and encode them. For this a copy of the unprocessed data was also used. Most of the other algorithms were trained on the preprocessed data.

To summarize the results of the 10 baselines, the **random forest regressor** with 50 estimators, 4000 max features (around 20% of total preprocessed features), max depth of 30 and a minimum samples leaf of 3 was able to achieve low errors but with noticeable overfitting and took long to train due to the large dataset size. The **gradient boosting regressor** with 200 trees achieved higher errors but less overfitting and faster training than the random forest regressor. The **LightGBM** with a learning rate of 0.07, 1000 trees, and 10 minimum child samples was able to complete training in lightning fast time within 9 seconds, and with low errors and overfitting. The **categorical LightGBM** with a learning rate of 0.05, 5000 trees, max depth of 7 and 110 leaves achieved lower error scores but slightly higher overfitting. The **XGBoost** model achieved poor results when trained on the non-scaled data, however it did well with low error scores and some overfitting when trained on the fully preprocessed data with robust scaling. The **Catboost** was slow but had relatively low errors and some overfitting as well, overall a promising algorithm that used minimal preprocessing. The **Linear Regressor** achieved slightly high error scores and with some noticeable overfitting. The **Ridge Regressor** was very fast to train in under 5 seconds and it achieved average errors but lower overfitting than the Linear Regressor. The **Elastic Net** was the poorest performing model among the 10 models with the highest errors. The **SGDRegressor** and the **Huber** **Regressor** also did not perform too well with high errors but low overfitting.

Overall, the best 3 models were the **LightGBM/Categorical LightGBM**, **XGBoost** and **CatBoost**. The next section will go over the hyperparameter tuning process for each of the three learning algorithms.

### 7. **Hyperparameter Tuning**

For the hyperparameter tuning, scikit-learn’s RandomizedSearchCV was mainly utilized to try out different random combinations of hyperparameters. Given the dataset size, GridSearchCV would be much more expensive to compute than a random search. The randomized search does not guarantee the optimal hyperparameter combination, but it will provide a good combination with a high confidence rate and in considerably less amount of time when compared to a brute-force grid search. For all models, the number of cross validation folds was set to 3 given its a large dataset this number would be enough, and the number of iterations (parameter combinations) was 60, and all cpus were utilized to speed up the process. The number of iterations was inspired by an article [6], which stated that regardless of the grid size, 60 iterations can get the best 5% sets of parameters 95% of the time.

1. **LightGBM**

Given that the base LightGBM regressor was overfitting, hyperparameters that help reduce it were used in the random grid search. These were the number of estimators (trees), with different numbers of trees between 50 and 1000. The learning rate, which is essentially the boosting learning rate, with different learning rates between 0.01 and 1. The minimum data in leaf, which requires each leaf to include a certain number of data points to help the model generalize and not be too specific, numbers tried were 10, 20 and 30. The minimum child samples, which is the minimum number of data points needed in a child leaf, the numbers 10, 20 and 30 were tried. The max depth, which is the maximum tree depth, where lower values should reduce overfitting, values between 3 and 10 were used. Lastly, the number of leaves, which is the maximum tree leaves for base learners, numbers between 6 and 500 were used in the random search.

After about 4 minutes, the errors were improved (lower) but the overfitting was still somewhat existent, although still not too high. The same hyperparameter search was done for the categorical version of the LightGBM. In which, the results were very similar to the baseline categorical LightGBM and in fact a little worse. However, when compared to the other models, it still performed very well.

2. **XGBoost**

The baseline XGBoost regressor had low errors but was overfitting. Hyperparameters that help reduce it were considered. These were the number of estimators, with different numbers of trees between 50 and 1000. The learning rate, with different values between 0.01 and 1. The maximum tree depth, with values between 3 and 10. The minimum child weight, which is the minimum sum of instance weight needed in a leaf, where larger values should help reduce overfitting, numbers between 0 and 10 were tried. Lastly, the sub sample, which is the ratio of training rows used, where lower values should help reduce overfitting, values between 0 and 1 were tried in the random search.

After about 16 minutes, lower errors were achieved in comparison to the baseline XGBoost, and higher r-squared scores, but the overfitting had slightly increased. This means that the values chosen in the random search were not optimal and did not help in reducing the overfitting, and different values should be tried.

3. **CatBoost**

For the CatBoost regressor, using a randomized search cross-validation proved to be very costly. For that matter, different values for the hyperparameters were directly fed into the model and several combinations were tested.

The best combination of hyperparameters found were 1000 iterations, a learning rate of 0.05, a depth of 16, a rsm of 0.5, a l2_leaf_reg of 3, and 10 early stopping rounds. These hyperparameters aimed to improve accuracy and reduce overfitting, and this combination proved to have the lowest errors with least possible overfitting. These are by no means the best values of these hyperparameters for this model, but due to CatBoost being very memory inefficient taking more than 24 minutes to complete training, this was a faster way to get some decent results without doing a grid search. The results showed lower MAE scores, higher RMSE scores, higher r-squared and slightly increased overfitting. All of these results are presented in the next section along with a brief summary.

### 8. **Results & Conclusion**

Below are the MAE, RMSE and R2 scores for each of the baselines as well as the tuned models.

![mae_scores.png](https://github.com/mo-adi/used_cars_pricing/blob/master/figures/mae_scores.png)

![rmse_scores.png](https://github.com/mo-adi/used_cars_pricing/blob/master/figures/rmse_scores.png)

![r2_scores.png](https://github.com/mo-adi/used_cars_pricing/blob/master/figures/r2_scores.png)

The tuned models achieved the lowest MAE and RMSE scores, and similarly had the highest R2 scores. The lower the MAE and RMSE scores and the higher the r-squared, the better the model. Overfitting can be seen by the difference between the train and test scores. The larger the difference the more overfitting. Overall, some models had some noticeable overfitting but nothing too extreme.

The best model in terms of lowest MAE and RMSE errors, and highest r-squared is the Tuned Catboost. Due to the data set size, a brute force search of the best hyperparameters was not possible due to computation limitations, and a randomized search cross validation was used to tune the hyperparameters instead. Both the LightGBM and CatBoost had the ability to encode categorical features internally, which is an advantage to these models. The CatBoost algorithm was relatively poor in terms of memory efficiency and took longer to train. The LightGBM on the other hand was much more efficient and quick, with very similar results. There were two LightGBM algorithms used, a regular one which took in the preprocessed data, and a categorical one which took an unprocessed version of the data and did its own categorical preprocessing. Surprisingly, the untuned categorical LightGBM achieved very comparable results with the tuned models.

Using the weights of the tuned CatBoost regression model, a web application has been created where the features of a specific car are used as input, and an accurate prediction of the fair market value (FMV) of the car is given as output. The code for the web application is below:

### Code for the Webapp:

[https://github.com/mo-adi/fmv_webapp](https://github.com/mo-adi/fmv_webapp)

To conclude, the tuned LightGBM, XGBoost and CatBoost were the best models with very similar performances. Given that all of these models had similar overfitting, the CatBoost is the best overall model as it achieved the lowest MAE and RMSE scores and had the best data fit with the highest r-squared scores. Even though the overfitting in these models was not extreme, there is still room for improvement. Better results can be achieved by conducting a more rigorous hyperparameter tuning process.

### **References**

[1] Ramesh, Shashank Markapuram. “Price Optimization with Machine Learning: What Every Retailer Should Know.” *7Learnings*, 23 Aug. 2022, https://7learnings.com/blog/price-optimization-with-machine-learning-what-every-retailer-should-know/.

[2] Reese, Austin. “Used Cars Dataset.” Kaggle, 6 May 2021, [https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data).

[3] Gokce, Enes. “Predicting Used Car Prices with Machine Learning Techniques.” *Medium*, Towards Data Science, 10 Jan. 2020, https://towardsdatascience.com/predicting-used-car-prices-with-machine-learning-techniques-8a9d8313952.

[4] Kumbar, Kshitij, et al. “CS 229 Project Report: Predicting Used Car Prices - Stanford University.” *CS 229 Project Report*, 2019, https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26612934.pdf.

[5] Ghori, Mohammed. “Data Cleaning + EDA + Used Cars Prediction(86%).” *Kaggle*, Kaggle, 1 June 2020, https://www.kaggle.com/code/msagmj/data-cleaning-eda-used-cars-prediction-86.

[6] Weiran, Sun. “Hyper Parameter Tuning with Randomised Grid Search.” *Medium*, Towards Data Science, 4 Sept. 2019, https://towardsdatascience.com/hyper-parameter-tuning-with-randomised-grid-search-54f865d27926.
