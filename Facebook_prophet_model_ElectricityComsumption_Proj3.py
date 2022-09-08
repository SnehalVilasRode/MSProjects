#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[2]:


#from google.colab import files
#uploaded = files.upload()


# In[3]:


# df2 = pd.read_csv(io.BytesIO(uploaded['Electric_Consumption_And_Cost__2010_-_April_2020_.csv']))


# ### Introduction to Data Science Project 3
# 
# #### Name : Priyanka Nigade
# 

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
import io

from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from fbprophet.plot import add_changepoints_to_plot


# In[2]:


# Read data from csv file 

df = pd.read_csv('sample_data/Electric_Consumption_And_Cost__2010_-_April_2020_.csv')

#df = pd.read_csv(io.BytesIO(uploaded['Electric_Consumption_And_Cost__2010_-_April_2020_.csv']))


# In[3]:


df.shape


# In[332]:


df.head()


# 
# #### Prophet builds a model by finding a best smooth line which can be represented as a sum of the following components:
# #### y(t) = g(t) + s(t) + h(t) + ϵₜ
# #### Overall growth trend. g(t)
# #### Yearly seasonality. s(t)
# #### Weekly seasonality. s(t)
# #### Holidays effects h(t)
# 

# ### Daily data:

# In[333]:


# Using start date for daily data 

dailyTimeSeriesDf = df[['Service Start Date', 'Consumption (KW)']]
dailyTimeSeriesDf


# In[334]:


# covert start date to datetime

dailyTimeSeriesDf['Service Start Date'] = pd.to_datetime(dailyTimeSeriesDf['Service Start Date'], infer_datetime_format=True)


# In[335]:


dailyTimeSeriesDf


# In[336]:


# check for null values

dailyTimeSeriesDf.columns[dailyTimeSeriesDf.isnull().any()]


# In[337]:


dailyTimeSeriesDf = dailyTimeSeriesDf.dropna()


# In[338]:


dailyTimeSeriesDf.info()


# In[339]:


dailyTimeSeriesDf.columns = ['ds', 'y']


# In[340]:


dailyTimeSeriesDf.head()


# 
#    **FaceBook Prophet Model for daily data**
# 
#    **Predict the EC for 100/200/365 days into the future.**

# In[341]:


# Instantiate prophet model

prophetModel = Prophet(daily_seasonality=True)
prophetModel = prophetModel.fit(dailyTimeSeriesDf)


# In[342]:


# Predicting values for future 100 Days

futureHundredDays = prophetModel.make_future_dataframe(100, freq = 'D')
forecastHundredDays = prophetModel.predict(futureHundredDays)
forecastHundredDays.tail()


# In[343]:


# Predicting values for future 200 Days

futureTwoHundredDays = prophetModel.make_future_dataframe(200, freq = 'D')
forecastTwoHundredDays = prophetModel.predict(futureTwoHundredDays)
forecastTwoHundredDays.tail()


# In[344]:


# Predicting values for future 365 Days

future_365Days = prophetModel.make_future_dataframe(365, freq = 'D')
forecast_365Days = prophetModel.predict(future_365Days)
forecast_365Days.tail()


# In[345]:


#plot the model forecast chart for 100 days
forecastHundredDays[['ds', 'yhat', 'yhat_lower',	'yhat_upper']].head()


# In[346]:


fig = prophetModel.plot(forecastHundredDays)


# In[347]:


#plot the model forecast chart for 200 days
forecastTwoHundredDays[['ds', 'yhat', 'yhat_lower',	'yhat_upper']].head()


# In[348]:


fig = prophetModel.plot(forecastTwoHundredDays)


# In[349]:


#plot the model forecast chart for 365 days
forecast_365Days[['ds', 'yhat', 'yhat_lower',	'yhat_upper']].head()


# In[350]:


fig = prophetModel.plot(forecast_365Days)


# Tune your FBProphet model on the following parameters:
# 
# 
# 1.   Forecasting growth:
# 2.   Seasonality:
# 3.   Trend Changepoints:

# In[351]:


# import seaborn as sns
sns.boxplot(dailyTimeSeriesDf['y'])
plt.show()


# In[352]:


tuned_prophetModel = Prophet(growth='logistic',
            interval_width = 0.8,
            n_changepoints=10,
            changepoint_range=0.8,
            changepoint_prior_scale=1,
            seasonality_mode='additive')
tuned_prophetModel.add_seasonality(name= 'daily', period=30.5, fourier_order=5, prior_scale=0.02)
dailyTimeSeriesDf['cap'] = 10000
dailyTimeSeriesDf['floor'] = 0

tuned_prophetModel.fit(dailyTimeSeriesDf)


# In[353]:


futureHundredDaystuned = tuned_prophetModel.make_future_dataframe(periods = 100, freq = 'D')
futureHundredDaystuned['cap'] = 30000
futureHundredDaystuned['floor'] = 0
fcst_HundredDays = tuned_prophetModel.predict(futureHundredDaystuned)
fcst_HundredDays.tail()


# In[354]:


future_2HundredDaystuned = tuned_prophetModel.make_future_dataframe(periods = 200, freq = 'D')
future_2HundredDaystuned['cap'] = 30000
future_2HundredDaystuned['floor'] = 0
fcst_2HundredDays = tuned_prophetModel.predict(future_2HundredDaystuned)
fcst_2HundredDays.tail()


# In[355]:


future_365Daystuned = tuned_prophetModel.make_future_dataframe(periods = 365, freq = 'D')
future_365Daystuned['cap'] = 30000
future_365Daystuned['floor'] = 0
fcst_365Days = tuned_prophetModel.predict(future_365Daystuned)
fcst_365Days.tail()


# In[356]:


fig = tuned_prophetModel.plot(fcst_HundredDays)


# In[357]:


fig = tuned_prophetModel.plot_components(fcst_HundredDays)


# In[358]:


fig = tuned_prophetModel.plot(fcst_2HundredDays)


# In[359]:


fig = tuned_prophetModel.plot_components(fcst_2HundredDays)


# In[360]:


fig = tuned_prophetModel.plot(fcst_365Days)


# In[361]:


fig = tuned_prophetModel.plot_components(fcst_365Days)


# In[362]:


fig = tuned_prophetModel.plot(fcst_365Days)
a = add_changepoints_to_plot(fig.gca(), tuned_prophetModel, fcst_365Days)


# **Cross Validation** 
# 
# **Evaluation** 
# *   MAE (Mean Absolute Error) 
# *   MAPE (Mean Absolute Percentage Error)
# *   R^2 (use sklearn’s respective metrics)

# Cross validation for 100 Days

# In[363]:


#daily_100 = cross_validation(prophetModel, period='100 days', horizon='50 days', parallel = 'processes')
#daily_100.head()


# In[364]:


# Get the performance matrix
#daily_100pm = performance_metrics(daily_100)
#daily_100pm.head()


# In[365]:


# Get the _2 score

#r2_daily_score = r2_score(daily_100['y'],daily_100['yhat'])
#r2_daily_score


# In[366]:


# Merge dataframes to get y in the matrix

forecastHundredDays =  forecastHundredDays.merge(dailyTimeSeriesDf, on= 'ds')
forecastTwoHundredDays = forecastTwoHundredDays.merge(dailyTimeSeriesDf, on= 'ds')
forecast_365Days = forecast_365Days.merge(dailyTimeSeriesDf, on= 'ds')


fcst_HundredDays =  fcst_HundredDays.merge(dailyTimeSeriesDf, on= 'ds')
fcst_2HundredDays = fcst_2HundredDays.merge(dailyTimeSeriesDf, on= 'ds')
fcst_365Days = fcst_365Days.merge(dailyTimeSeriesDf, on= 'ds')


# In[367]:


# checking 100 days performance without parameter tuning

print('Mean absolute percentage error for 100 days forecasting:', mean_absolute_percentage_error(forecastHundredDays.y, forecastHundredDays.yhat))
print('Mean absolute error for 100 days forecasting:           ', mean_absolute_error(forecastHundredDays.y, forecastHundredDays.yhat))
print('R2 score for 100 days forecasting:                      ', r2_score(forecastHundredDays.y, forecastHundredDays.yhat))


# In[368]:


# checking 100 days performance with parameter tuning

print('Mean absolute percentage error for 100 days forecasting:', mean_absolute_percentage_error(fcst_HundredDays.y, fcst_HundredDays.yhat))
print('Mean absolute error for 100 days forecasting:           ', mean_absolute_error(fcst_HundredDays.y, fcst_HundredDays.yhat))
print('R2 score for 100 days forecasting:                      ', r2_score(fcst_HundredDays.y, fcst_HundredDays.yhat))


# Ploting Actual vs Predicted

# In[369]:


# plotting the actual and forecast values for 100 days

ax = (dailyTimeSeriesDf.plot(x='ds',y='y',figsize=(20,5),title='Actual Vs Forecast'))
fcst_HundredDays.plot(x='ds',y='yhat',figsize=(20,5),title='Actual vs Forecast', ax=ax)


# In[370]:


# plotting the actual and forecast values for 200 days

ax = (dailyTimeSeriesDf.plot(x='ds',y='y',figsize=(20,5),title='Actual Vs Forecast'))
fcst_2HundredDays.plot(x='ds',y='yhat',figsize=(20,5),title='Actual vs Forecast', ax=ax)


# In[371]:


# plotting the actual and forecast values for 365 days

ax = (dailyTimeSeriesDf.plot(x='ds',y='y',figsize=(20,5),title='Actual Vs Forecast'))
fcst_365Days.plot(x='ds',y='yhat',figsize=(20,5),title='Actual vs Forecast', ax=ax)


# ### Monthly Mean data:

# In[372]:


timeSeriesDf = df[['Revenue Month', 'Consumption (KW)']]
timeSeriesDf


# #### Check for null values

# In[373]:


timeSeriesDf.columns[timeSeriesDf.isnull().any()]


# In[374]:


timeSeriesDf.dtypes


# In[375]:


# Get the mean
timeSeriesDf = timeSeriesDf.groupby('Revenue Month', as_index=False)['Consumption (KW)'].mean()


# In[376]:


timeSeriesDf


# In[377]:


# Convert date

timeSeriesDf['Revenue Month'] = pd.to_datetime(timeSeriesDf['Revenue Month'], infer_datetime_format=True)


# In[378]:


timeSeriesDf


# In[379]:


timeSeriesDf.columns = ['ds', 'y']


# In[380]:


timeSeriesDf


# In[381]:


timeSeriesDf.describe()


# In[382]:


#timeSeriesDf.plot()
timeSeriesDf.head()


# In[383]:


def plot_df(df, x, y, title="", xlabel='Month', ylabel='Monthly Electricity consumption(KW)', dpi=100):
    plt.figure(figsize=(15,4), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    


# In[384]:


plot_df(timeSeriesDf, x=timeSeriesDf['ds'], y=timeSeriesDf['y'], title='Monthly Electricity consumption(KW) from (2010 - April 2020)')


# **FaceBook Prophet Model for monthly**
# 
# 
# 

# **Predict the EC for 1/6/9 months into the future.**

# In[385]:


### FaceBook Prophet

prophetModel = Prophet()
prophetModel = prophetModel.fit(timeSeriesDf)


# In[386]:



# Predicting values for future a month

futureOneMonth = prophetModel.make_future_dataframe(1, freq = 'M')
forecastOneMonth = prophetModel.predict(futureOneMonth)
forecastOneMonth.tail()


# In[386]:





# In[387]:


# Predicting values for future 6 months 

futureSixMonth = prophetModel.make_future_dataframe( 6, freq = 'M')
forecastSixMonth = prophetModel.predict(futureSixMonth)
forecastSixMonth.tail()


# In[388]:


# Predicting values for future 9 months 

futureNineMonths = prophetModel.make_future_dataframe(periods = 9, freq = 'M')
forecastNineMonths = prophetModel.predict(futureNineMonths)
forecastNineMonths.tail()


# In[389]:


#plot the model forecast chart 
forecastOneMonth[['ds', 'yhat', 'yhat_lower',	'yhat_upper']].head()


# In[390]:


#plot the model forecast chart 
forecastSixMonth[['ds', 'yhat', 'yhat_lower',	'yhat_upper']].head()


# In[391]:


#plot the model forecast chart 
forecastNineMonths[['ds', 'yhat', 'yhat_lower',	'yhat_upper']].head()


# In[392]:


fig = prophetModel.plot(forecastOneMonth)


# In[393]:


fig = prophetModel.plot(forecastSixMonth)


# In[394]:


fig1 =prophetModel.plot(forecastNineMonths)


# **Checking the trends in the data**

# In[395]:


fig = prophetModel.plot_components(forecastNineMonths)


# **Tune your FBProphet model on the following parameters:**
# 
# 
# *   Forecasting growth:
# *   Seasonality:
# *  Trend Changepoints: 
# 
# 
# 
# 

# In[396]:


sns.boxplot(timeSeriesDf['y'])
plt.show()


# In[397]:


tuned_prophetModel = Prophet(growth='logistic',
            interval_width = 0.8,
            n_changepoints=10,
            changepoint_range=0.8,
            changepoint_prior_scale=1,
            seasonality_mode='additive')
tuned_prophetModel.add_seasonality(name= 'monthly', period=30.5, fourier_order=5, prior_scale=0.02)
timeSeriesDf['cap'] = 70

tuned_prophetModel.fit(timeSeriesDf)


# In[398]:


futureOneMonthtuned = tuned_prophetModel.make_future_dataframe(periods = 9, freq = 'M')
futureOneMonthtuned['cap'] = 70
fcst_one_monthtuned = tuned_prophetModel.predict(futureOneMonthtuned)
fcst_one_monthtuned[['yhat','yhat_lower','yhat_upper']].tail()


# In[399]:


futureSixMonthstuned = tuned_prophetModel.make_future_dataframe(periods = 9, freq = 'M')
futureSixMonthstuned['cap'] = 70
fcstSixMonthstuned = tuned_prophetModel.predict(futureSixMonthstuned)
fcstSixMonthstuned[['yhat','yhat_lower','yhat_upper']].tail()


# In[400]:



futureNineMonthstuned = tuned_prophetModel.make_future_dataframe(periods = 9, freq = 'M')
futureNineMonthstuned['cap'] = 70
fcstNineMonthstuned = tuned_prophetModel.predict(futureNineMonthstuned)
fcstNineMonthstuned[['yhat','yhat_lower','yhat_upper']].tail()


# In[401]:


fig = tuned_prophetModel.plot(fcst_one_monthtuned)


# In[402]:


fig = tuned_prophetModel.plot_components(fcst_one_monthtuned)


# In[403]:


fig = tuned_prophetModel.plot(fcstSixMonthstuned)


# In[404]:


fig = tuned_prophetModel.plot_components(fcstSixMonthstuned)


# In[405]:


fig = tuned_prophetModel.plot(fcstNineMonthstuned)


# In[406]:


fig = tuned_prophetModel.plot_components(fcstNineMonthstuned)


# **Cross Validation** 
# 
# **Evaluation** 
# *   MAE (Mean Absolute Error) 
# *   MAPE (Mean Absolute Percentage Error)
# *   R^2 (use sklearn’s respective metrics)
# 
# 
# 
# 
# 

# In[406]:





# In[407]:


#def get_cross_validation(prophetModel):
#   estimated_consumption = cross_validation(prophetModel, initial='960 days',period='135 days', horizon='273 days', parallel = 'processes')
#   get_performance_metrics(estimated_consumption)
#  get_r2_score(estimated_consumption)
#  estimated_consumption.head() 


# In[408]:


#def get_performance_metrics(estimated_consumption):
#   performance_matrix = performance_metrics(estimated_consumption)
#  performance_matrix.head()


# In[409]:


#def get_r2_score(estimated_consumption):
#   r2_mn = r2_score(estimated_consumption['y'],estimated_consumption['yhat'])
#  print(r2_mn)


# In[410]:


# get_cross_validation(tuned_prophetModel)


# In[411]:


#monthly_pm = performance_metrics(monthly_ES)
#monthly_pm.head()


# In[412]:


#r2_mn = r2_score(monthly_ES['y'],monthly_ES['yhat'])
#r2_mn


# In[413]:


# Merge dataframes to get y in the matrix

forecastOneMonth =  forecastOneMonth.merge(timeSeriesDf, on= 'ds')
forecastSixMonth = forecastSixMonth.merge(timeSeriesDf, on= 'ds')
forecastNineMonths = forecastNineMonths.merge(timeSeriesDf, on= 'ds')

fcst_one_monthtuned = fcst_one_monthtuned.merge(timeSeriesDf, on= 'ds')
fcstSixMonthstuned = fcstSixMonthstuned.merge(timeSeriesDf, on= 'ds')
fcstNineMonthstuned = fcstNineMonthstuned.merge(timeSeriesDf, on= 'ds')


# In[414]:


# checking one month performance without parameter tuning

print('Mean absolute percentage error for 1 month forecasting:', mean_absolute_percentage_error(forecastOneMonth.y, forecastOneMonth.yhat))
print('Mean absolute error for 1 month forecasting:           ', mean_absolute_error(forecastOneMonth.y, forecastOneMonth.yhat))
print('R2 score for 1 month forecasting:                      ', r2_score(forecastOneMonth.y, forecastOneMonth.yhat))


# In[415]:


# checking one month performance with parameter tuning

print('Mean absolute percentage error for 1 month forecasting:', mean_absolute_percentage_error(fcst_one_monthtuned.y, fcst_one_monthtuned.yhat))
print('Mean absolute error for 1 month forecasting:           ', mean_absolute_error(fcst_one_monthtuned.y, fcst_one_monthtuned.yhat))
print('R2 score for 1 month forecasting:                      ', r2_score(fcst_one_monthtuned.y, fcst_one_monthtuned.yhat))


# In[416]:


# checking six months performance without parameter tuning

print('Mean absolute percentage error for six months forecasting:', mean_absolute_percentage_error(forecastSixMonth.y, forecastSixMonth.yhat))
print('Mean absolute error for six months forecasting:           ', mean_absolute_error(forecastSixMonth.y, forecastSixMonth.yhat))
print('R2 score for six months forecasting:                      ', r2_score(forecastSixMonth.y, forecastSixMonth.yhat))


# In[417]:


# checking six month performance with parameter tuning

print('Mean absolute percentage error for 6 months forecasting:', mean_absolute_percentage_error(fcstSixMonthstuned.y, fcst_one_monthtuned.yhat))
print('Mean absolute error for 6 months forecasting:           ', mean_absolute_error(fcstSixMonthstuned.y, fcstSixMonthstuned.yhat))
print('R2 score for 6 months forecasting:                      ', r2_score(fcstSixMonthstuned.y, fcstNineMonthstuned.yhat))


# In[418]:


# plotting the actual and forecast values for one month

ax = (timeSeriesDf.plot(x='ds',y='y',figsize=(20,5),title='Actual Vs Forecast'))
fcst_one_monthtuned.plot(x='ds',y='yhat',figsize=(20,5),title='Actual vs Forecast', ax=ax)


# In[419]:


# plotting the actual and forecast values for six months

ax = (timeSeriesDf.plot(x='ds',y='y',figsize=(20,5),title='Actual Vs Forecast'))
fcstSixMonthstuned.plot(x='ds',y='yhat',figsize=(20,5),title='Actual vs Forecast', ax=ax)


# In[420]:


# plotting the actual and forecast values for nine months

ax = (timeSeriesDf.plot(x='ds',y='y',figsize=(20,5),title='Actual Vs Forecast'))
fcstNineMonthstuned.plot(x='ds',y='yhat',figsize=(20,5),title='Actual vs Forecast', ax=ax)


# ### Yearly Mean data:

# In[150]:


yearlyTimeSeriesDf = df[['Revenue Month', 'Consumption (KW)']]
yearlyTimeSeriesDf


# In[151]:


# Get yearly mean data
yearlyTimeSeriesDf = yearlyTimeSeriesDf.groupby(pd.PeriodIndex(yearlyTimeSeriesDf['Revenue Month'], freq = 'Y'))['Consumption (KW)'].mean().reset_index()


# In[152]:


# Generate date for the years
yearlyTimeSeriesDf['Revenue Month'] = yearlyTimeSeriesDf['Revenue Month'].apply(lambda i: str(i).replace(' Q', '')).apply(lambda i: pd.to_datetime(i))


# In[153]:


yearlyTimeSeriesDf.shape


# In[154]:


yearlyTimeSeriesDf.columns = ['ds', 'y']
yearlyTimeSeriesDf.head()


# In[155]:


# Check for null values
yearlyTimeSeriesDf.columns[yearlyTimeSeriesDf.isnull().any()]


# **FaceBook Prophet Model for Yearly data**
# 
# **Predict the EC for 1/10/20 yeras into the future.**

# In[156]:


# Instantiate prophet model
prophetModel = Prophet()
prophetModel = prophetModel.fit(yearlyTimeSeriesDf)


# In[157]:



# Predicting values for future 1 year

futureOneYear = prophetModel.make_future_dataframe(1, freq = 'Y')
forecastOneYear = prophetModel.predict(futureOneYear)
forecastOneYear.tail()


# In[158]:


# Predicting values for future 10 year

futureTenYear = prophetModel.make_future_dataframe(10, freq = 'Y')
forecastTenYear = prophetModel.predict(futureTenYear)
forecastTenYear.tail()


# In[159]:


# Predicting values for future 20 year

futureTwentyYear = prophetModel.make_future_dataframe(20, freq = 'Y')
forecastTwentyYear = prophetModel.predict(futureTwentyYear)
forecastTwentyYear.tail()


# In[160]:


#plot the model forecast chart 
forecastOneYear[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# In[194]:


fig = prophetModel.plot(forecastOneYear)


# In[162]:


#plot the model forecast chart 
forecastTenYear[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# In[195]:


fig = prophetModel.plot(forecastTenYear)


# In[164]:


#plot the model forecast chart 
forecastTwentyYear[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# In[196]:


fig = prophetModel.plot(forecastTwentyYear)


# **Checking the trends in the data**

# In[197]:


# Trends for one year
fig = prophetModel.plot_components(forecastOneYear)


# In[198]:


# Trends for 10 years
fig =prophetModel.plot_components(forecastTenYear)


# In[223]:


# Trends for 20 years
fig = prophetModel.plot_components(forecastTwentyYear)


# Tune your FBProphet model on the following parameters:
# * Forecasting growth:
# * Seasonality:
# * Trend Changepoints:
# 

# In[172]:


# Getting outliers to decide cap value

sns.boxplot(yearlyTimeSeriesDf['y'])
plt.show()


# In[173]:


cap = 65


# In[174]:


tuned_prophetModel = Prophet(growth='logistic',
            interval_width = 0.8,
            n_changepoints=10,
            changepoint_range=0.8,
            changepoint_prior_scale=1,
            seasonality_mode='additive')
tuned_prophetModel.add_seasonality(name= 'yearly', period=30.5, fourier_order=5, prior_scale=0.02)
yearlyTimeSeriesDf['cap'] = cap


# In[175]:


tuned_prophetModel.fit(yearlyTimeSeriesDf)


# In[176]:


futureOneYearTuned = tuned_prophetModel.make_future_dataframe(periods = 1, freq = 'Y')
futureOneYearTuned['cap'] = cap
fcst_OneYear = tuned_prophetModel.predict(futureOneYearTuned)
fcst_OneYear.tail()


# In[178]:


futureTenYearsTuned = tuned_prophetModel.make_future_dataframe(periods = 10, freq = 'Y')
futureTenYearsTuned['cap'] = cap
fcst_TenYears = tuned_prophetModel.predict(futureTenYearsTuned)
fcst_TenYears.tail()


# In[179]:


futureTwentyYearsTuned = tuned_prophetModel.make_future_dataframe(periods = 10, freq = 'Y')
futureTwentyYearsTuned['cap'] = cap
fcst_TwentyYears = tuned_prophetModel.predict(futureTwentyYearsTuned)
fcst_TwentyYears.tail()


# In[180]:


fig = tuned_prophetModel.plot(fcst_OneYear)


# In[183]:


fig = tuned_prophetModel.plot(fcst_TenYears)


# In[182]:


fig = tuned_prophetModel.plot(fcst_TwentyYears)


# Changing the changepoint_prior_scale

# In[193]:


tuned_prophetModel = Prophet(changepoint_prior_scale=0.001)
tuned_prophetModel.fit(yearlyTimeSeriesDf)
#fig = m.plot(forecast)

future = tuned_prophetModel.make_future_dataframe(periods = 1, freq = 'Y')
future['cap'] = cap
fcst = tuned_prophetModel.predict(future)
fig = tuned_prophetModel.plot(fcst)


# In[231]:


fig = tuned_prophetModel.plot(fcst)
a = add_changepoints_to_plot(fig.gca(), tuned_prophetModel, fcst)


# Cross Validation
# Evaluation
# * MAE (Mean Absolute Error)
# * MAPE (Mean Absolute Percentage Error)
# * R^2 (use sklearn’s respective metrics)

# In[ ]:


#yearly_ES = cross_validation(prophetModel, initial='970 days',period='135 days', horizon='365 days', parallel = 'processes')
#yearly_ES.head()
#yearly_pm = performance_metrics(yearly_ES)
#yearly_pm.head()
#r2_mn = r2_score(yearly_ES['y'],yearly_ES['yhat'])
#r2_mn


# In[184]:


# Merge dataframes to get y in the matrix

forecastOneYear =  forecastOneYear.merge(yearlyTimeSeriesDf, on= 'ds')
forecastTenYear = forecastTenYear.merge(yearlyTimeSeriesDf, on= 'ds')
forecastTwentyYear = forecastTwentyYear.merge(yearlyTimeSeriesDf, on= 'ds')

fcst_OneYear = fcst_OneYear.merge(yearlyTimeSeriesDf, on= 'ds')
fcst_TenYears = fcst_TenYears.merge(yearlyTimeSeriesDf, on= 'ds')
fcst_TwentyYears = fcst_TwentyYears.merge(yearlyTimeSeriesDf, on= 'ds')


# In[185]:


# checking one year performance without parameter tuning

print('Mean absolute percentage error for one year forecasting:', mean_absolute_percentage_error(forecastOneMonth.y, forecastOneMonth.yhat))
print('Mean absolute error for one year forecasting:           ', mean_absolute_error(forecastOneMonth.y, forecastOneMonth.yhat))
print('R2 score for one year forecasting:                      ', r2_score(forecastOneMonth.y, forecastOneMonth.yhat))


# In[187]:


# checking one year performance with parameter tuning

print('Mean absolute percentage error for one year forecasting:', mean_absolute_percentage_error(fcst_OneYear.y, fcst_OneYear.yhat))
print('Mean absolute error for one year forecasting:           ', mean_absolute_error(fcst_OneYear.y, fcst_OneYear.yhat))
print('R2 score for one year forecasting:                      ', r2_score(fcst_OneYear.y, fcst_OneYear.yhat))


# In[188]:


# plotting the actual and forecast values - 1year

ax = (yearlyTimeSeriesDf.plot(x='ds',y='y',figsize=(20,5),title='Actual Vs Forecast'))
fcst_OneYear.plot(x='ds',y='yhat',figsize=(20,5),title='Actual vs Forecast', ax=ax)


# In[189]:


# plotting the actual and forecast values - 10 years

ax = (yearlyTimeSeriesDf.plot(x='ds',y='y',figsize=(20,5),title='Actual Vs Forecast'))
fcst_TenYears.plot(x='ds',y='yhat',figsize=(20,5),title='Actual vs Forecast', ax=ax)


# In[190]:


# plotting the actual and forecast values - 20 years

ax = (yearlyTimeSeriesDf.plot(x='ds',y='y',figsize=(20,5),title='Actual Vs Forecast'))
fcst_TwentyYears.plot(x='ds',y='yhat',figsize=(20,5),title='Actual vs Forecast', ax=ax)


# **Conclusion**:
# 
# Out of all the above model - model for 1 year forecasting with tuned parameter has good results with 
# 
# R2 score for one year forecasting: 0.8604296077455287
# 
# Parameters used:
# 
# tuned_prophetModel = Prophet(growth='logistic',
#             interval_width = 0.8,
#             n_changepoints=10,
#             changepoint_range=0.8,
#             changepoint_prior_scale=1,
#             seasonality_mode='additive')
# tuned_prophetModel.add_seasonality(name= 'yearly', period=30.5, fourier_order=5, prior_scale=0.02)
# yearlyTimeSeriesDf['cap'] = 75

# 
# ### Predict Electric Consumption for each of the 5 Boroughs (independently)!`
# 
# 

# In[232]:


#Lets get the distint boroughs
df['Borough'].value_counts()


# In[233]:


# create separate datasets for each borough

brooklynDf = df[df['Borough'] == 'BROOKLYN']
manhattanDf = df[df['Borough'] == 'MANHATTAN']
bronxDf = df[df['Borough'] == 'BRONX']
queensDf = df[df['Borough'] == 'QUEENS']
statenislandDf = df[df['Borough'] == 'STATEN ISLAND']


# BROOKLYN

# In[234]:


yearlyBrooklynTimeSeriesDf = brooklynDf[['Revenue Month', 'Consumption (KW)']]
yearlyBrooklynTimeSeriesDf


# In[235]:


# Get yearly mean data
yearlyBrooklynTimeSeriesDf = yearlyBrooklynTimeSeriesDf.groupby(pd.PeriodIndex(yearlyBrooklynTimeSeriesDf['Revenue Month'], freq = 'Y'))['Consumption (KW)'].mean().reset_index()


# In[236]:


# Generate date for the years
yearlyBrooklynTimeSeriesDf['Revenue Month'] = yearlyBrooklynTimeSeriesDf['Revenue Month'].apply(lambda i: str(i).replace(' Q', '')).apply(lambda i: pd.to_datetime(i))


# In[237]:


yearlyBrooklynTimeSeriesDf.columns = ['ds', 'y']
yearlyBrooklynTimeSeriesDf.head()


# In[238]:


# Check for null values
yearlyBrooklynTimeSeriesDf.columns[yearlyBrooklynTimeSeriesDf.isnull().any()]


# **FaceBook Prophet Model for Yearly data**
# 

# In[240]:


# Instantiate prophet model
brookprophetModel = Prophet()
brookprophetModel = brookprophetModel.fit(yearlyBrooklynTimeSeriesDf)


# In[242]:


# Predicting values for future 10 year

futureBrookTenYear = brookprophetModel.make_future_dataframe(10, freq = 'Y')
forecastBrookTenYear = brookprophetModel.predict(futureBrookTenYear)
forecastBrookTenYear[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[243]:


fig = brookprophetModel.plot(forecastBrookTenYear)


# Tune your FBProphet model on the following parameters:
# * Forecasting growth:
# * Seasonality:
# * Trend Changepoints:

# In[244]:


# Getting outliers to decide cap value

sns.boxplot(yearlyBrooklynTimeSeriesDf['y'])
plt.show()


# In[245]:


cap = 57


# In[246]:


tuned_brookprophetModel = Prophet(growth='logistic',
            interval_width = 0.8,
            n_changepoints= 5,
            changepoint_range=0.8,
            changepoint_prior_scale=1,
            seasonality_mode='additive')
tuned_brookprophetModel.add_seasonality(name= 'yearly', period=30.5, fourier_order=5, prior_scale=0.02)
yearlyBrooklynTimeSeriesDf['cap'] = cap


# In[247]:


tuned_brookprophetModel.fit(yearlyBrooklynTimeSeriesDf)


# In[248]:


futureBrookTenYearsTuned = tuned_brookprophetModel.make_future_dataframe(periods = 10, freq = 'Y')
futureBrookTenYearsTuned['cap'] = cap
fcst_brookTenYears = tuned_prophetModel.predict(futureBrookTenYearsTuned)
fcst_brookTenYears.tail()


# In[249]:



fig = tuned_brookprophetModel.plot(fcst_brookTenYears)


# Cross Validation Evaluation
# * MAE (Mean Absolute Error)
# * MAPE (Mean Absolute Percentage Error)
# * R^2 (use sklearn’s respective metrics)

# In[260]:


# Merge dataframes to get y in the matrix

forecastBrookTenYear =  forecastBrookTenYear.merge(yearlyBrooklynTimeSeriesDf, on= 'ds')
fcst_brookTenYears =  fcst_brookTenYears.merge(yearlyBrooklynTimeSeriesDf, on= 'ds')
fcst_brookTenYears


# In[258]:


# checking ten year performance without parameter tuning

print('Mean absolute percentage error for ten year forecasting:', mean_absolute_percentage_error(forecastBrookTenYear.y, forecastBrookTenYear.yhat))
print('Mean absolute error for ten year forecasting:           ', mean_absolute_error(forecastBrookTenYear.y, forecastBrookTenYear.yhat))
print('R2 score for one ten forecasting:                       ', r2_score(forecastBrookTenYear.y, forecastBrookTenYear.yhat))


# In[261]:


# checking ten year performance with parameter tuning

print('Mean absolute percentage error for ten year forecasting:', mean_absolute_percentage_error(fcst_brookTenYears.y, fcst_brookTenYears.yhat))
print('Mean absolute error for ten year forecasting:           ', mean_absolute_error(fcst_brookTenYears.y, fcst_brookTenYears.yhat))
print('R2 score for one ten forecasting:                       ', r2_score(fcst_brookTenYears.y, fcst_brookTenYears.yhat))


# MANHATTAN

# In[263]:


yearlyManhattanTimeSeriesDf = manhattanDf[['Revenue Month', 'Consumption (KW)']]
yearlyManhattanTimeSeriesDf


# In[264]:


# Get yearly mean data
yearlyManhattanTimeSeriesDf = yearlyManhattanTimeSeriesDf.groupby(pd.PeriodIndex(yearlyManhattanTimeSeriesDf['Revenue Month'], freq = 'Y'))['Consumption (KW)'].mean().reset_index()


# In[265]:


# Generate date for the years
yearlyManhattanTimeSeriesDf['Revenue Month'] = yearlyManhattanTimeSeriesDf['Revenue Month'].apply(lambda i: str(i).replace(' Q', '')).apply(lambda i: pd.to_datetime(i))


# In[266]:


yearlyManhattanTimeSeriesDf.columns = ['ds', 'y']
yearlyManhattanTimeSeriesDf.head()


# In[267]:


# Check for null values
yearlyManhattanTimeSeriesDf.columns[yearlyManhattanTimeSeriesDf.isnull().any()]


# **FaceBook Prophet Model for Yearly data**

# In[268]:


# Instantiate prophet model
manhattanProphetModel = Prophet()
manhattanProphetModel = manhattanProphetModel.fit(yearlyManhattanTimeSeriesDf)


# In[270]:


# Predicting values for future 10 year

futureManhattanTenYear = manhattanProphetModel.make_future_dataframe(10, freq = 'Y')
forecastManhattanTenYear = manhattanProphetModel.predict(futureManhattanTenYear)
forecastManhattanTenYear[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[272]:


#plot the model forecast chart 

fig = manhattanProphetModel.plot(forecastOneYear)


# Tune your FBProphet model on the following parameters:
# * Forecasting growth:
# * Seasonality:
# * Trend Changepoints:

# In[273]:


# Getting outliers to decide cap value

sns.boxplot(yearlyManhattanTimeSeriesDf['y'])
plt.show()


# In[274]:


cap = 72


# In[275]:


tuned_manhattanprophetModel = Prophet(growth='logistic',
            interval_width = 0.8,
            n_changepoints=10,
            changepoint_range=0.8,
            changepoint_prior_scale=1,
            seasonality_mode='additive')
tuned_manhattanprophetModel.add_seasonality(name= 'yearly', period=30.5, fourier_order=5, prior_scale=0.02)
yearlyManhattanTimeSeriesDf['cap'] = cap


# In[276]:


tuned_manhattanprophetModel.fit(yearlyManhattanTimeSeriesDf)


# In[277]:


futureManhattanTenYearsTuned = tuned_manhattanprophetModel.make_future_dataframe(periods = 10, freq = 'Y')
futureManhattanTenYearsTuned['cap'] = cap
fcst_manhattanTenYears = tuned_manhattanprophetModel.predict(futureManhattanTenYearsTuned)
fcst_manhattanTenYears.tail()


# In[278]:


fig = tuned_manhattanprophetModel.plot(fcst_manhattanTenYears)


# Cross Validation Evaluation
# * MAE (Mean Absolute Error)
# * MAPE (Mean Absolute Percentage Error)
# * R^2 (use sklearn’s respective metrics)

# In[289]:



# Merge dataframes to get y in the matrix

forecastManhattanTenYear =  forecastManhattanTenYear.merge(yearlyManhattanTimeSeriesDf, on= 'ds')
fcst_manhattanTenYears =  fcst_manhattanTenYears.merge(yearlyManhattanTimeSeriesDf, on= 'ds')

#forecastManhattanTenYear


# In[286]:


# checking 10 year performance without parameter tuning

print('Mean absolute percentage error for ten year forecasting:', mean_absolute_percentage_error(forecastManhattanTenYear.y, forecastManhattanTenYear.yhat))
print('Mean absolute error for ten year forecasting:           ', mean_absolute_error(forecastManhattanTenYear.y, forecastManhattanTenYear.yhat))
print('R2 score for ten year forecasting:                      ', r2_score(forecastManhattanTenYear.y, forecastManhattanTenYear.yhat))


# In[290]:


# checking 10 year performance with parameter tuning

print('Mean absolute percentage error for ten year forecasting:', mean_absolute_percentage_error(fcst_manhattanTenYears.y, fcst_manhattanTenYears.yhat))
print('Mean absolute error for ten year forecasting:           ', mean_absolute_error(fcst_manhattanTenYears.y, fcst_manhattanTenYears.yhat))
print('R2 score for ten year forecasting:                      ', r2_score(fcst_manhattanTenYears.y, fcst_manhattanTenYears.yhat))


# BRONX

# In[291]:


yearlyBronxTimeSeriesDf = bronxDf[['Revenue Month', 'Consumption (KW)']]
yearlyBronxTimeSeriesDf


# In[292]:


# Get yearly mean data
yearlyBronxTimeSeriesDf = yearlyBronxTimeSeriesDf.groupby(pd.PeriodIndex(yearlyBronxTimeSeriesDf['Revenue Month'], freq = 'Y'))['Consumption (KW)'].mean().reset_index()


# In[293]:


# Generate date for the years
yearlyBronxTimeSeriesDf['Revenue Month'] = yearlyBronxTimeSeriesDf['Revenue Month'].apply(lambda i: str(i).replace(' Q', '')).apply(lambda i: pd.to_datetime(i))


# In[294]:


yearlyBronxTimeSeriesDf.columns = ['ds', 'y']
yearlyBronxTimeSeriesDf.head()


# **FaceBook Prophet Model for Yearly data**

# In[295]:


# Instantiate prophet model
bronxprophetModel = Prophet()
bronxprophetModel = bronxprophetModel.fit(yearlyBronxTimeSeriesDf)


# In[296]:


# Predicting values for future 10 year

futurebronxTenYear = bronxprophetModel.make_future_dataframe(10, freq = 'Y')
forecastbronxTenYear = bronxprophetModel.predict(futurebronxTenYear)
forecastbronxTenYear[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[297]:


fig = bronxprophetModel.plot(forecastbronxTenYear)


# Cross Validation Evaluation
# * MAE (Mean Absolute Error)
# * MAPE (Mean Absolute Percentage Error)
# * R^2 (use sklearn’s respective metrics)

# In[298]:


forecastbronxTenYear =  forecastbronxTenYear.merge(yearlyBronxTimeSeriesDf, on= 'ds')


# In[299]:


# checking ten year performance without parameter tuning

print('Mean absolute percentage error for ten year forecasting:', mean_absolute_percentage_error(forecastbronxTenYear.y, forecastbronxTenYear.yhat))
print('Mean absolute error for ten year forecasting:           ', mean_absolute_error(forecastbronxTenYear.y, forecastbronxTenYear.yhat))
print('R2 score for ten year forecasting:                      ', r2_score(forecastbronxTenYear.y, forecastbronxTenYear.yhat))


# In[300]:


# plotting the actual and forecast values - 1year

ax = (yearlyBronxTimeSeriesDf.plot(x='ds',y='y',figsize=(20,5),title='Actual Vs Forecast'))
forecastbronxTenYear.plot(x='ds',y='yhat',figsize=(20,5),title='Actual vs Forecast', ax=ax)


# QUEENS

# In[301]:


yearlyQueensTimeSeriesDf = queensDf[['Revenue Month', 'Consumption (KW)']]
yearlyQueensTimeSeriesDf


# In[302]:


# Get yearly mean data
yearlyQueensTimeSeriesDf = yearlyQueensTimeSeriesDf.groupby(pd.PeriodIndex(yearlyQueensTimeSeriesDf['Revenue Month'], freq = 'Y'))['Consumption (KW)'].mean().reset_index()


# In[303]:


# Generate date for the years
yearlyQueensTimeSeriesDf['Revenue Month'] = yearlyQueensTimeSeriesDf['Revenue Month'].apply(lambda i: str(i).replace(' Q', '')).apply(lambda i: pd.to_datetime(i))


# In[304]:


yearlyQueensTimeSeriesDf.columns = ['ds', 'y']
yearlyQueensTimeSeriesDf.head()


# **FaceBook Prophet Model for Yearly data**

# In[305]:


# Instantiate prophet model
queensprophetModel = Prophet()
queensprophetModel = queensprophetModel.fit(yearlyQueensTimeSeriesDf)


# In[306]:


# Predicting values for future 10 year

futureQueensTenYear = queensprophetModel.make_future_dataframe(10, freq = 'Y')
forecastQueensTenYear = queensprophetModel.predict(futureQueensTenYear)
forecastQueensTenYear[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[307]:


fig = queensprophetModel.plot(forecastQueensTenYear)


# Cross Validation Evaluation
# * MAE (Mean Absolute Error)
# * MAPE (Mean Absolute Percentage Error)
# * R^2 (use sklearn’s respective metrics)

# In[309]:


# Merge dataframes to get y in the matrix

forecastQueensTenYear =  forecastQueensTenYear.merge(yearlyQueensTimeSeriesDf, on= 'ds')


# In[313]:


# checking ten year performance without parameter tuning

print('Mean absolute percentage error for ten years forecasting:', mean_absolute_percentage_error(forecastQueensTenYear.y_y, forecastQueensTenYear.yhat))
print('Mean absolute error for ten years forecasting:           ', mean_absolute_error(forecastQueensTenYear.y_y, forecastQueensTenYear.yhat))
print('R2 score for ten years forecasting:                      ', r2_score(forecastQueensTenYear.y_y, forecastQueensTenYear.yhat))


# In[315]:


# plotting the actual and forecast values - 10 years

ax = (yearlyQueensTimeSeriesDf.plot(x='ds',y='y',figsize=(20,5),title='Actual Vs Forecast'))
forecastQueensTenYear.plot(x='ds',y='yhat',figsize=(20,5),title='Actual vs Forecast', ax=ax)


# STATEN ISLAND 

# In[316]:


yearlystatenTimeSeriesDf = statenislandDf[['Revenue Month', 'Consumption (KW)']]
yearlystatenTimeSeriesDf


# In[317]:


# Get yearly mean data

yearlystatenTimeSeriesDf = yearlystatenTimeSeriesDf.groupby(pd.PeriodIndex(yearlystatenTimeSeriesDf['Revenue Month'], freq = 'Y'))['Consumption (KW)'].mean().reset_index()


# In[318]:


# Generate date for the years

yearlystatenTimeSeriesDf['Revenue Month'] = yearlystatenTimeSeriesDf['Revenue Month'].apply(lambda i: str(i).replace(' Q', '')).apply(lambda i: pd.to_datetime(i))


# In[319]:


yearlystatenTimeSeriesDf.columns = ['ds', 'y']
yearlystatenTimeSeriesDf.head()


# **FaceBook Prophet Model for Yearly data**

# In[320]:


# Instantiate prophet model
statenprophetModel = Prophet()
statenprophetModel = statenprophetModel.fit(yearlystatenTimeSeriesDf)


# In[322]:


# Predicting values for future 10 year

futurestatenTenYear = statenprophetModel.make_future_dataframe(10, freq = 'Y')
forecaststatenTenYear = statenprophetModel.predict(futurestatenTenYear)
forecaststatenTenYear[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[323]:


fig = statenprophetModel.plot(forecaststatenTenYear)


# Cross Validation Evaluation
# * MAE (Mean Absolute Error)
# * MAPE (Mean Absolute Percentage Error)
# * R^2 (use sklearn’s respective metrics)

# In[324]:


# Merge dataframes to get y in the matrix

forecaststatenTenYear =  forecaststatenTenYear.merge(yearlystatenTimeSeriesDf, on= 'ds')


# In[328]:


forecaststatenTenYear.head()


# In[326]:



# checking ten years performance without parameter tuning

print('Mean absolute percentage error for ten years forecasting:', mean_absolute_percentage_error(forecaststatenTenYear.y, forecaststatenTenYear.yhat))
print('Mean absolute error for ten years forecasting:           ', mean_absolute_error(forecaststatenTenYear.y, forecaststatenTenYear.yhat))
print('R2 score for ten years forecasting:                      ', r2_score(forecaststatenTenYear.y, forecaststatenTenYear.yhat))



# CONCLUSION
# 
# Staten island shows the higher consumprion of electricity in next 10 years
