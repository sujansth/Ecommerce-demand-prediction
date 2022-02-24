from datetime import datetime, timedelta,date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from numpy import sqrt
# import pandas_datareader as data
from keras.models import load_model
import streamlit as st

data = pd.read_csv('../data/final_data.csv', parse_dates=['date'])

st.title('Time Series Analysis of Top4Sport')

data = data.set_index('date')
data = data.resample('D').sum()
data.columns=['sales']
print(data.head())

# Visualization data
st.subheader('Total Revenue Vs Time Chart')
fig = plt.figure(figsize=(12,8))
plt.plot(data)
st.pyplot(fig)

st.subheader('Total Revenue Vs Time Chart with 100 Moving Average')
ma100 = data.sales.rolling(100).mean()
fig = plt.figure(figsize=(12,8))
plt.plot(data)
plt.plot(ma100, label='100MA')
plt.legend(loc='upper left')
st.pyplot(fig)

st.subheader('Total Revenue Vs Time Chart with 100MA & 200MA')
ma200 = data.sales.rolling(200).mean()
fig = plt.figure(figsize=(12,8))
plt.plot(data)
plt.plot(ma100, label='100 MA')
plt.plot(ma200, label='200 MA')
plt.legend(loc="upper left")
st.pyplot(fig)



#create a new dataframe to model the difference
df_diff = data.copy()

#add previous sales to the next row
df_diff['prev_sales'] = df_diff['sales'].shift(1)

#drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['sales'] - df_diff['prev_sales'])


#create dataframe for transformation from time series to supervised
df_supervised = df_diff.drop(['prev_sales'],axis=1)

#adding lags
for inc in range(1,90):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)
    
#drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)

# Import statsmodels.formula.api
import statsmodels.formula.api as smf

# Define the regression formula
model = smf.ols(formula='diff ~ lag_1', data=df_supervised)

# Fit the regression
model_fit = model.fit()

# Import statsmodels.formula.api
import statsmodels.formula.api as smf

# Define the regression formula
model = smf.ols(formula='diff ~ lag_1+lag_2+lag_3+lag_4+lag_5+lag_6+lag_7+lag_8+lag_9+lag_10+lag_11+lag_12+lag_13+lag_14+lag_15+lag_16+lag_17+lag_18+lag_19+lag_20+lag_21+lag_22+lag_23+lag_24+lag_25+lag_26+lag_27+lag_28+lag_29+lag_30+lag_31+lag_32+lag_33+lag_34+lag_35+lag_36+lag_37+lag_38+lag_39+lag_40+lag_41+lag_42+lag_43+lag_44+lag_45+lag_46+lag_47+lag_48+lag_49+lag_50', data=df_supervised)

# Fit the regression
model_fit = model.fit()


#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['sales'],axis=1)

#split train and test set
train_set, test_set = df_model[0:-150].values, df_model[-150:].values


#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)

# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)

# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)


X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])



# Finally load the model
model = load_model('lstm_model.h5')

# Let's do the prediction and see how the result looks like. 
y_pred=model.predict(X_test, batch_size=1)

#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))

#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])

#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

#create dataframe that shows the predicted sales

result_list = []
sales_dates = list(data[-151:].index)
act_sales = list(data[-151:].sales)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['date'] = sales_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)


daily_sales_pred = pd.merge(data, df_result,on='date',how='left')

# Final graph
st.subheader('Prediction Vs Original')
fig = plt.figure(figsize=(12,8))
plt.plot(daily_sales_pred.date, daily_sales_pred.sales, label='Actual sales')
plt.plot(daily_sales_pred.date, daily_sales_pred.pred_value, label='Predicted Sales')
plt.legend(loc="upper left")
st.pyplot(fig)


st.subheader('Prediction vs Original (Test data only)')
fig = plt.figure(figsize=(12,8))
plt.plot(daily_sales_pred.date[-150:], daily_sales_pred.sales[-150:], label='Actual sales')
plt.plot(daily_sales_pred.date[-150:], daily_sales_pred.pred_value[-150:], label='Predicted Sales')
plt.legend(loc="upper left")
st.pyplot(fig)
