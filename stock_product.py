import numpy as np
from flask import Flask, jsonify, make_response
from flask_restplus import Resource, Api, fields
from flask import request, stream_with_context, Response
from flask_cors import CORS
import json, csv
from werkzeug.utils import cached_property
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

app = Flask(__name__)
CORS(app)
api = Api(app)

name = api.model('name', {
        'name_id': fields.String(description='Enter Name', required=True, example='ABCD')
    })

# model_new = keras.models.load_model('my_model')

# @app.route('/')
# def home():
#     return render_template('index.html')

@api.route('/prediction')
class LargeNew(Resource):
    @api.expect(name)
    # def post(self):
    #     data = request.json
    #     tenant_id = data['tenant_id']
    #     with open('aggtrend_300_days.json') as f:
    #         output_dictionary = json.load(f)

    #     response = make_response(json.dumps(output_dictionary))
    #     response.headers['content-type'] = 'application/octet-stream'
    #     return response
    def post(self):
        data = request.json
        name_id = data['name_id']
        print(name_id)
        import pandas_datareader as pdr
        import pandas as pd
        msft = yf.Ticker(name_id)
        df_ori = msft.history('10y',interval='1d')
        df=df_ori.reset_index()
        df=df[0:2461]
        df.tail(3)

        df1=df[['Date','Close']]
        df1['Date']=pd.DatetimeIndex(df1['Date'])
        df1['Date2']=pd.DatetimeIndex(df1['Date']).date


        print(df1.shape)

        df1.head(3)

        del df1['Date']
        df1=df1.set_index('Date2')

        import pandas as pd
#         df=pd.read_csv('AAPL.csv')

#         df1=df.reset_index()['close']
        df=df1.copy()
    #creating dataframe
        data = df1.sort_index(ascending=True, axis=0)

        #creating train and test sets
        dataset = df1.values

        train = dataset[0:1800,:]
        valid = dataset[1800:,:]
        # dataset.shape

    #converting dataset into x_train and y_train
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        import numpy as np
        x_train, y_train = [], []
        for i in range(100,len(train)):
            x_train.append(scaled_data[i-100:i,0])
            y_train.append(scaled_data[i,0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=25))

        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=2)

        #predicting 246 values, using past 60 from the train data
        inputs = df1[len(df1) - len(valid) - 100:].values
        inputs = inputs.reshape(-1,1)
        inputs  = scaler.transform(inputs)

        X_test = []
        for i in range(100,inputs.shape[0]):
            X_test.append(inputs[i-100:i,0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)
        rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
        print('RMSE',rms)

        # rms=np.sqrt(np.mean(np.power((valid['Predictions']-valid['Close']),2)))
        # rms

        import numpy as np

        def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print('MAPE',mean_absolute_percentage_error(closing_price, valid))

        scaler = MinMaxScaler(feature_range=(0, 1))
        valid_scaled_data = scaler.fit_transform(valid)
        

        x_input=valid_scaled_data[valid_scaled_data.shape[0]-100:].reshape(1,-1)
        x_input.shape

        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
# demonstrate prediction for next 10 days
        from numpy import array

        lst_output=[]
        n_steps=100
        i=0
        while(i<7):

            if(len(temp_input)>100):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1


        print(lst_output)
        day_new=np.arange(1,101)
        day_pred=np.arange(101,108)


        day_pred

        import matplotlib.pyplot as plt

        # print(len(lst_output))


        # plt.plot(day_new,scaler.inverse_transform(df1[2362:]),color='red')
#         plt.plot(day_new,df1[2361:],color='red')
#         plt.plot(day_pred,scaler.inverse_transform(lst_output),color='blue')
        last=df_ori.index[-1].date()

        future_dates=pd.date_range(start=last, periods=7)
        future_dataset=pd.DataFrame()
        future_dataset=pd.DataFrame()
        future_dataset['date']=future_dates
        future_dataset['Close']=scaler.inverse_transform(lst_output)

        future_dataset['date']=pd.DatetimeIndex(future_dataset['date'])
        future_dataset['date']=pd.DatetimeIndex(future_dataset['date']).date
        future_dataset['weekday']=pd.DatetimeIndex(future_dataset['date']).weekday
        future_dataset=future_dataset[future_dataset['weekday'].isin([0,1,2,3,4])]
        future_dataset['date']=future_dataset['date'].astype('str')
        del future_dataset['weekday']

        future_dataset
        print(future_dataset.shape)
        print(future_dataset.head(3))
        print('out',json.dumps(future_dataset.to_dict(orient='records')))
        response = make_response(json.dumps(future_dataset.to_dict(orient='records')))
        response.headers['content-type'] = 'application/octet-stream'
        return response
        

if __name__ == "__main__":
    app.run(port=7001, debug=True)
