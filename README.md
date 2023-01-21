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