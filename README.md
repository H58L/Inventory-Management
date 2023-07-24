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
  <br>
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
      
  </li>
</ol>
<ol>
