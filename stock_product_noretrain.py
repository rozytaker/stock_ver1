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

model_new = keras.models.load_model('my_model2')

# @app.route('/')
# def home():
#     return render_template('index.html')

@api.route('/prediction')
class LargeNew(Resource):
    @api.expect(name)
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
        print(df1.shape)

        import pandas as pd
#         df=pd.read_csv('AAPL.csv')

#         df1=df.reset_index()['close']
        df=df1.copy()
        train = df[0:1800]
        valid = df[1800:]
    #creating dataframe
        # training_size=int(len(df1)*0.65)
        # test_size=len(df1)-training_size
        # train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
        # print('train_data',train_data.shape)
        # print('test_data',test_data.shape)

        scaler = MinMaxScaler(feature_range=(0, 1))
        valid_scaled_data = scaler.fit_transform(valid)
        

        x_input=valid_scaled_data[valid_scaled_data.shape[0]-100:].reshape(1,-1)
        x_input.shape
        
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        print('tempinput',len(temp_input))
        # demonstrate prediction for next 10 days
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
                yhat = model_new.predict(x_input, verbose=0)
                print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model_new.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1


    #     print(lst_output)
    #     day_new=np.arange(1,101)
    #     day_pred=np.arange(101,131)
    #     df3=df1.tolist()
    #     df3.extend(lst_output)

    # #     df3=scaler.inverse_transform(df3).tolist()
    #     df3=scaler.inverse_transform(df3)
    #     pred=pd.DataFrame(df3)
    #     pred['Time']='2020-11-12'
    #     pred.columns=['pred','timestamp']
    #     print(pred.head(3))
    #     print('out',json.dumps(pred.to_dict(orient='records')))
    #     # pred.head(3)
    #     response = make_response(json.dumps(pred.to_dict(orient='records')))
    #     # response.headers['content-type'] = 'application/octet-stream'
    #     return response
        
#         plt.plot(day_pred,scaler.inverse_transform(lst_output),color='blue')
        # df1['Date']=pd.DatetimeIndex(df1['Date'])
        print('output',len(lst_output))
        last=df1.index[-1]

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
    app.run(port=6002,debug=True)
