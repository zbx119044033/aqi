# aqi
Air quality forecasting model using pollutant records and meteorological data   
The pipline of data processing is as follows:    
scrawler -> preprocess -> exploring -> feature engineering -> modeling -> post-analysis

### Dependence: ###
python3.5   
scrapy   
pandas   
numpy   
xgboost   
sklearn   
pyfunctional   

### pollute.py ###
data scrawler based on Scrapy framework

### data_clean.py / readin.py ###
data preprocess module

### exploring.py ###
exploretory data analysis module

### feature_enggineering.py / feature_pm.py ###
feature generation module

### extraction.py ###
feature extraction module

### training.py ###
training part using different models

### analysis.py ###
data postprocess and metrics calculating
