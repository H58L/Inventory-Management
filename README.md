# Inventory-Management
<ol>
<li><h2>Problem Statement:</h2></li>
<p>Develop an accurate machine learning model that can predict future sales for a retail store. The objective of this project is to help the retail store optimize their inventory, pricing, and promotional strategies to maximize their profits. 
By accurately forecasting sales, the store can ensure that they have the right amount of stock to meet customer demand without overstocking or understocking. 
</p>
<br>
<li><h2>Abstract</h2></li>
<p>This machine learning model uses the BigMart sales data to predict sales of a product per outlet. It takes into account parameters like item weight, visibility, MRP, outlet establishment year, location, and outlet type. 
The project makes use of data cleaning, visualization and prediction techniques.
</p>
<br>
<li><h2> Module Wise Description: </h2></li>
<ol>
  <li><h3>Missing Values Handling</h3>
    <p>
      Checking for missing values. 
df.describe() gives the statistical analysis of numeric data. 
Missing Numeric data(Item_weight) is filled with the mean of the column. Missing categorical data(Outlet_size) is filled with the mode of the column.
    </p>
  </li>
 
  <li>
    <h3>Dimensionality Reduction:</h3>
    <p>
      Dropping irrelevant columns ie, Item_Identifier and Outlet_Identifier. 
Using df.drop() 
    </p>
  </li>
  <li>
    <h3> Visualising and Analysing Data(Exploratory Analysis)</h3>
          <ul>
            <li>
                Using DTale.
            </li>
            <li>
              Using seaborn to show heat map.
            </li>
            <li>
              Using klib(for visualisation of data):
            </li>
          </ul>
  </li>
  <li>
    <h3>Data Cleaning using Klib library: </h3>
  </li>
  <ul>
    <li>
      Standardise column name 
    </li>
    <li>
      Convert columns to the best possible data types 
    </li>
    <li>
      Drop missing values 
    </li>
  </ul>
  <li>
    <h3>Preprocessing before data modelling: </h3>
    <p>
      To convert categorical data into numerical data. 
    </p>
  </li>
  <li>
   <h3>Split data into train and test </h3> 
   <p>Split training data into test and train. </p>
  </li>
  <li>
    <h3>Split data into train and test </h3>
    <p>Split training data into test and train.</p>
  </li>
  <li>
    <h3>Standardisation. </h3>
    <p>Convert data into a similar scale.</p>
  </li>
</ol>
<li><h2>Model Building</h2></li>
<ol>
  <b>
    <li>Using Linear Regression</li>
    <li>Using Random forest regressor</li>
    <li>Using XGBRegressor</li>
    <li>using decision tree regressor</li>
  </b>
</ol>
</ol>
<hr noshade>
<h2>Conclusion: </h2><p>By comparing accuracy values, the model built using Random forest regressor has the highest r2_score of 0.599. <br>
  Hence random forest regressor is the most accurate model. 
  We use the random forest regressor model to find the product that has most demand ie most sales from the test.csv data. 
  </p>
<h2>
  Libraries:</h2>
  <ol>
    <li><b>Pandas:</b> For data manipulation and analysis. 
    </li>
    <li>
      <b>NumPy</b> Used for numerical computing. It provides tools for working with arrays and matrices. 
    </li>
    <li><b>Matplotlib:</b>Used for data visualization through charts, including line plots, bar charts, scatter plots, and histograms. 
    </li>
    <li>
      <b>Seaborn:</b> Used for statistical data visualization including regression plots, distribution plots, and categorical plots. 
    </li>
    <li><b>Dtale:</b>Used to create an interactive graphical interface
       for exploring data frames, performing data cleaning and manipulation, and creating visualizations.  </li>
       <li>
        <b>
          Pandas Profiling:
        </b>
        Used for generating data profiling reports from Pandas DataFrames.
       </li>
       <li>
        <b>
          YData Profiling:
        </b>
        Used for generating data profiling reports from a wide range of data sources, including SQL databases, CSV files, and Pandas DataFrames. 8.Klib: Used for data exploration and analysis. It provides tools for visualizing data, cleaning and preprocessing data, and performing 
        feature selection. It includes functions for generating summary statistics 
       </li>
       <li>
        <b>
          Joblib  
        </b>
        : Used for saving and loading the model. It provides 
          tools for running parallel computations on a single machine or across a cluster of machines.
       </li>
       <li>
        <b>
          Sklearn:
        </b>
        Used for data preprocessing, feature selection, model selection, and model evaluation. 
       </li>
       <li>
        <b>
          LabelEncoder
        </b>
        Used for encoding categorical variables into numerical values to enable the model to understand categorical data.
       </li>
       <li>
        <b>
          train_test_split
        </b>
        It randomly splits the data into two parts, with a specified ratio of training to testing data. 
       </li>
       <li>
        <b>
          StandardScaler:
        </b>
        Used for standardizing numerical variables. Scales the data to enable the model to perform better. 
       </li>
       <li>
        <b>
          LinearRegression:
        </b>
        Builds linear regression models. Predicts continuous numerical variables.
       </li>
       <li>
        <b>
          r2_score, mean_absolute_error, and mean_squared_error:
        </b>
        evaluation metrics used for measuring the performance of machine learning models.
        r2_score measures how well the model fits the data, with a score of 1 indicating a perfect fit. 
        mean_absolute_error and mean_squared_error measure the average difference between the predicted and actual values, with lower values indicating better performance. 
       </li>
       <li>
        <b>
          RandomForestRegressor :
        </b>
        Used for building random forest 
        regression models. It creates an ensemble of decision trees and uses them to make predictions.
       </li>
       <li>
        <b>
          RepeatedStratifiedKFold: 
        </b>
        Performs repeated
         stratified k-fold cross-validation. It is a technique for evaluating the performance of machine learning models.
       </li>
       <li>
        <b>
          GridSearchCV
        </b>
        Performs grid search cross-validation. It is a technique
         for finding the best hyperparameters for a machine learning model
       </li>
  </ol>

  <h2>
    References:
  </h2>
  <a href="https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data">https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data</a><br>
  <a href="https://neptune.ai/blog/google-colab-dealing-with-files ">https://neptune.ai/blog/google-colab-dealing-with-files </a><br>
  <a href="https://klib.readthedocs.io/en/latest/_modules/klib/preprocess.html">https://klib.readthedocs.io/en/latest/_modules/klib/preprocess.html</a><br>
  <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.ht ml ">https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.ht ml </a><br>
  <a href="https://www.analyticsvidhya.com/blog/2022/02/a-comprehensive-guide-on-hyperparameter-tunin g-and-its-techniques/ ">https://www.analyticsvidhya.com/blog/2022/02/a-comprehensive-guide-on-hyperparameter-tunin g-and-its-techniques/ </a><br>
  <a href="https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/ ">https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/ </a><br>
  <a href="https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-s tandardization/">https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-s tandardization/</a><br>
  <a href="https://www.analyticsvidhya.com/blog/2023/02/how-to-save-and-load-machine-learning-models-i n-python-using-joblib-library/">https://www.analyticsvidhya.com/blog/2023/02/how-to-save-and-load-machine-learning-models-i n-python-using-joblib-library/</a><br>
  <a href="https://developer.android.com/guide/components/aidl#Calling">https://developer.android.com/guide/components/aidl#Calling</a><br>
