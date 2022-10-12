import pandas as pd
import numpy as np
import pickle



dt = pd.read_csv('Big_mart_sales.csv')

item_weight_mean = dt.pivot_table(values = "Item_Weight", index = 'Item_Identifier')
miss_bool = dt['Item_Weight'].isnull()
for i, item in enumerate(dt['Item_Identifier']):
    if miss_bool[i]:
        if item in item_weight_mean:
            dt['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
        else:
            dt['Item_Weight'][i] = np.mean(dt['Item_Weight'])

outlet_size_mode = dt.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

miss_bool = dt['Outlet_Size'].isnull()
dt.loc[miss_bool, 'Outlet_Size'] = dt.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

dt.loc[:, 'Item_Visibility'].replace([0], [dt['Item_Visibility'].mean()], inplace=True)

dt['Item_Fat_Content'] = dt['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})

dt['New_Item_Type'] = dt['Item_Identifier'].apply(lambda x: x[:2])
dt['New_Item_Type'] = dt['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
dt.loc[dt['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# dt['Outlet'] = le.fit_transform(dt['Outlet_Identifier'])
cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
for col in cat_col:
    dt[col] = le.fit_transform(dt[col])


X = dt.drop(columns=[ 'Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales', 'Outlet_Establishment_Year'], axis=1)
y = dt['Item_Outlet_Sales']

X_copy=X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_copy.values, y.values, test_size=0.3, random_state=0)

from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train)

pickle.dump(regressor,open('model3.pkl','wb'))
model = pickle.load(open('model3.pkl','rb'))
