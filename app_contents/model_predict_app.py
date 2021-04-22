import sys

if not (sys.version_info.major == 3 and sys.version_info.minor >= 5):
    print("This script requires Python 3.5 or higher!")
    print("You are using Python {}.{}.".format(sys.version_info.major, sys.version_info.minor))
    sys.exit(1)

import os
import tensorflow as tf

#import skimage.io
import json
import requests
from flask import Flask, render_template, send_from_directory, globals, request, Response
#from flask_socketio import SocketIO

#Added code by jkaspa
import numpy as np
import pandas as pd
import json
import io

#New libs
import urllib.request
import pickle

# for encoding using feature-engine
from feature_engine.categorical_encoders import OrdinalCategoricalEncoder

#Swagger UI
from flasgger import Swagger

import sys

if not (sys.version_info.major == 3 and sys.version_info.minor >= 5):
    print("This script requires Python 3.5 or higher!")
    print("You are using Python {}.{}.".format(sys.version_info.major, sys.version_info.minor))
    sys.exit(1)

#Getting required fields
def required_fields(df):
    df_imp_required = df [['COMPANY_CODE','PLANT','ORDER_TYPE',
                            'VENDOR_GROUP','VENDOR_STATUS',
                            'PURCHASING_ORGANIZATION','TOTAL_GR_QUANTITY','TOTAL_GR_VALUE',
                            'TOTAL_IR_QUANTITY','TOTAL_IR_VALUE','SINCE_LAST_GR',
                            'SINCE_LAST_IR','QUANTITY_DIFFERENCE','VALUE_DIFFERENCE']]
    return df_imp_required

#Fix data type formating
def fix_data_types(df):
    df["COMPANY_CODE"]=df["COMPANY_CODE"].astype(str)
    df["PLANT"]=df["PLANT"].astype(str)
    df["ORDER_TYPE"]=df["ORDER_TYPE"].astype(str)
    df["VENDOR_GROUP"]=df["VENDOR_GROUP"].astype(str)
    df["VENDOR_STATUS"]=df["VENDOR_STATUS"].astype(str)
    df["PURCHASING_ORGANIZATION"]=df["PURCHASING_ORGANIZATION"].astype(str)
    df["SCENARIO"]=df["SCENARIO"].astype(str)
    return df

#Fillin NA values or null values
def fill_num_mean(data, vars_with_num_na):
    """
    function to impute the missing numerical data with mode value
    Input parameters: data :  Dataframe , vars_with_num_na : list of numerical variables
    it returns a model value stored in a list and also a dictionary to persist the mode values 
    """
    mean_var_dict = {}
    for var in vars_with_num_na:

        if var == 'SINCE_LAST_GR' or var == 'SINCE_LAST_IR':
          mode_val = -999
        elif var == 'TOTAL_IR_QUANTITY':
          mode_val = 0
        elif var == 'TOTAL_GR_QUANTITY':
          mode_val = 0
        else:
          # calculate the mode
          mode_val = data[var].mean()
        
        # we persist the mean in the dictionary
        mean_var_dict[var] = mode_val
    return mean_var_dict
model=tf.keras.models.load_model('./model/1')
app = Flask(__name__)
swagger = Swagger(app)

# path in route should match deployed model name
@app.route('/grir_predict',methods=["POST"])
def do_inference_grir():
    """endpoint for GRIR JSON based predictions
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    if not request.is_json: #Checking for request data is JSON only
        return Response('404 Not Found: Only JSON input requests are allowed.', status=404)
    else:
        df_json = pd.DataFrame(request.json)
        
        #Get the required fields
        df_without_id = required_fields(df_json)
        
        #Caliculate the SCENARIO Column from data
        #Fix data types to string for encoding
        df_without_id['SCENARIO'] = df_without_id.apply(
                                            lambda row: 'GR' if row.TOTAL_IR_QUANTITY == 0 else 
                                                        ('IR' if row.TOTAL_GR_QUANTITY == 0 else
                                                        ('GR>IR' if row.TOTAL_GR_QUANTITY > row.TOTAL_IR_QUANTITY else
                                                        ( 'IR>GR' if row.TOTAL_IR_QUANTITY > row.TOTAL_GR_QUANTITY else
                                                         'GR=IR'))),axis=1)

        #Fix the data types for the fields
        df = fix_data_types(df_without_id)

        #Fill NA/Null values for intezer columns with their means
        numerical_var_list = ['SINCE_LAST_GR','SINCE_LAST_IR','TOTAL_GR_QUANTITY','TOTAL_IR_QUANTITY','QUANTITY_DIFFERENCE','VALUE_DIFFERENCE']
        
        #Getting mean for numerical columns
        mean_values = fill_num_mean(data=df, vars_with_num_na=numerical_var_list)
        
        #Filling Null values with mean values
        for var in numerical_var_list:
            mean_val = mean_values[var]
            df[var].fillna(mean_val,inplace=True)

        #Get the Ordinal Categorical Encoding dictionary from the link
        ordinal_enc = pickle.load( open('./pkl_files/ordinal_encoder.pkl', 'rb'))
        
        #Dictionary for categorical encoder
        dict_cat_enc = ordinal_enc.encoder_dict_
        
        #Updating the ordinal_enc dictionary for new values
        for column in dict_cat_enc:
            unq_val = df[column].unique() #get unique values of column
            max_enc_val = max(list(dict_cat_enc[column].values()))
            for val in unq_val:            
                if dict_cat_enc[column].get(val) == None: #Check if a value is not present in a dict
                    max_enc_val+=1
                    dict_cat_enc[column][val]=max_enc_val

        #Transform the categorical values
        df_enc=ordinal_enc.transform(df)

        #Get Scaler
        scaler = pickle.load(open('./pkl_files/Scaler.pkl', 'rb'))

        #Transform the ENCODED TEST DATA into scaled data
        df_enc_scaled = scaler.transform(df_enc)

        #Predict response
        pred = model.predict(df_enc_scaled)[:,1]
        
        #Convert the prediction outputs to output percentages
        df_pval=pd.DataFrame(pred,columns=['Status'])
        df_pval['Status_100'] = df_pval['Status']*100 #.map(int)#astype(float)
        df_pval['int'] = df_pval['Status_100'].astype(int)

        #copying the prediction interzer value to the json output data file
        df_json['prediction_%'] = df_pval['int']
        
        #Send the response to the request
    return Response(df_json.to_json(orient = "records"), status=200, mimetype='application/json')
port = os.getenv('PORT', 5000)
if __name__ == '__main__':
           app.run(host='0.0.0.0', port=int(port))

