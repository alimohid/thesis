from numpy.core.fromnumeric import mean
import CoinGeckoAPI as cg
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

class Ridge_Regression:
    def __init__(self,dataframe,dataframe_for_prediction):
        self.dataframe =  dataframe
        self.dataframe_for_prediction = dataframe_for_prediction
    def train_for_prediction(self):
        x = self.dataframe[['TimeStamp','MAmonth','MAweek']]
        y = self.dataframe.iloc[:,3:].values
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x, y, test_size=0.30, random_state=40)
        self.ridge_reg = Ridge(alpha = 50,tol = 0.1)
        self.ridge_reg.fit(self.x_train,self.y_train)
    def predict_for_tomorrow(self):
        return mean(self.ridge_reg.predict(self.dataframe_for_prediction))
    def complete_prediction(self):
        prediction_list = []
        for i in self.dataframe.index:
            df = cg.convert_to_dataframe_for_prediction(self.dataframe["TimeStamp"][i],self.dataframe["MAmonth"][i],self.dataframe["MAweek"][i])
            prediction_list.append(mean(self.ridge_reg.predict(df)))
        return prediction_list

# x = cg.bitcoinDataFrame[['TimeStamp','MAmonth','MAweek']]
# y = cg.bitcoinDataFrame.iloc[:,3:].values

# x_train,x_test,y_train,y_test  = train_test_split(x, y, test_size=0.30, random_state=40)

# ridge_reg = Ridge(alpha = 50,tol = 0.1)
# ridge_reg.fit(x_train,y_train)
# print(mean(ridge_reg.predict(cg.bitcoinDataFrame_for_predicition)))