# Cryptocurrencies

## Project Overview
The client is seeking an examination of all the current cryptocurrencies on the market and some means of organizing these cryptocurrencies for classification purposes. Since there is no clear label for direct classification, an unsupervised machine learning model is developed to group the cryptocurrencies into discernable clusters. The results should be visibly clear for easy evaluation.

## Resources

### Data Sources

- crypto_data.csv
- crypto_clustering.ipynb

### Software
The unsupervised machine learning model is developed in Python within the machine learning environment (mlenv):

- Python 3.7.15
- Jupyter Notebook 6.5.2
- Pandas 1.3.5
- scikit-learn 1.0.2
- plotly 5.12.0
- hvplot 0.8.2

## Pre-Process the Data
The first thing to do when receiving the data set is to clean and prepare the data for the project. The imported DataFrame is filtered for cryptocurrencies that are currently being traded; the column involved is then dropped after serving its purpose. Any rows with null values and duplicated rows are identified and then removed. This cleaned DataFrame is filtered for cryptocurrencies that actually have coins mined. The names of the cryptocurrencies will not be used in the clustering algorithm but, so the column is moved to a separate DataFrame (and dropped from the main DataFrame). The `get_dummies()` method is used for the binary encoding of the non-numeric columns. The resulting DataFrame is then fitted into the Standard Scaler and transformed to center the data and scale to unit variance.

## Principal Component Analysis (PCA)
Having a DataFrame with many columns will not be helpful to a clustering algorithm, so the DataFrame is condensed to a more compact from with only the most relevant traits using Principal Component Analysis (PCA) for dimensionality reduction. The PCA is set to 3 components with random state 0. The fitted and transformed results are placed into a new DataFrame.

## Clustering with K-Means
With the new DataFrame ready for clustering, the next step is to determine the best number of clusters by observing an elbow curve. This is done by fitting the PCA DataFrame into the K-Means algorithm multiple times from one cluster to 10 clusters (all at random state 1). The inertia of each result is put into a list which is matched to the K-value in a DataFrame to be plotted into an elbow curve with `hvplot`.

![Elbow Curve](https://github.com/Owen-Wang1234/Cryptocurrencies/blob/main/ElbowCurve.png)

Looking at the curve, the best number is determined to be 4, so the PCA data is finally fitted into K-Means model with 4 clusters (random state 1) to yield a set of predictions. One clustered DataFrame is formed by combining the cleaned pre-encoded DataFrame with the PCA DataFrame and the names DataFrame and appending the list of class predictions.

## Visualizing the Results
The clustered DataFrame is immediately plotted into a 3D scatter plot with `plotly`. Each principal component is set to an axis, each predicted class is noted by color and shape of the data point, and hovering over the point shows the name of the cryptocurrency and the algorithm used.

![PCA Scatter Plot](https://github.com/Owen-Wang1234/Cryptocurrencies/blob/main/PCA3DScatter.png)

Hvplot is used to produce a table from the clustered DataFrame. A quick check shows that there are 532 tradable cryptocurrencies used throughout this project. The columns for the coin supply and the coins mined are fitted and transformed with a MinMaxScaler method to scale the values between zero and one. The results form a DataFrame with the name and predicted class to be used for plotting. The `hvplot` scatter plot sets the number of coins mined to the x-axis and the total coin supply to the y-axis, colors the data points by their predicted class, and shows the name of the cryptocurrency when hovering the cursor over the point.

![Cryptocurrency Scatter Plot](https://github.com/Owen-Wang1234/Cryptocurrencies/blob/main/CryptoScatter.png)