from copyreg import pickle
from flask import Flask , render_template, request
import numpy as np
import pickle
import pandas as pd


app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
      return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [x for x in request.form.values()]
    features_value = [np.array(input_features)]

      
    features_name = [ "Item_Weight", "Item_Fat_Content","Item_Visibility","Item_Type", "Item_MRP", "Outlet_Size"
                       "Outlet_Location_Type", "Outlet_Type", "New_Item_Type"]
          
    df = pd.DataFrame(features_value, columns=[ "Item_Weight", "Item_Fat_Content","Item_Visibility","Item_Type", "Item_MRP", "Outlet_Size",
                       "Outlet_Location_Type", "Outlet_Type", "New_Item_Type"]
          )
    df['Item_Weight'] = df['Item_Weight'].astype(float)
    df['Item_Visibility'] = df['Item_Visibility'].astype(float)
    df['Item_MRP'] = df['Item_MRP'].astype(float)
   
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # dt['Outlet'] = le.fit_transform(dt['Outlet_Identifier'])
    cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
    for col in cat_col:
        df[col] = le.fit_transform(df[col])

    output = model.predict(df)
    
    
            
    return render_template('index.html', prediction_text='The predicted value is {}'.format(output))

if __name__ == "__main__":
      app.run(debug=True,port =8000)