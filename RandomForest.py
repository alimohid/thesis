from numpy.core.fromnumeric import mean
from sklearn import metrics
import CoinGeckoAPI as cg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class Random_Forest:
    def __init__(self,dataframe,dataframe_for_prediction):
        self.dataframe  = dataframe
        self.dataframe_for_prediction =  dataframe_for_prediction
    def train_for_prediction(self):
        x = self.dataframe[['TimeStamp','MAmonth','MAweek']]
        y = self.dataframe.iloc[:,3:].values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y,test_size=0.3,random_state=0)
        self.RFreg = RandomForestRegressor(n_estimators=10,random_state=0)
        self.RFreg.fit(self.x_train,self.y_train)
        self.pred = self.RFreg.predict(self.x_test)
    def get_r2_score(self):
        return metrics.r2_score(self.y_test,self.pred)
    def get_mean_absolute_error(self):
        return  metrics.mean_absolute_error(self.y_test, self.pred)
    def get_mean_squared_error(self):
        return metrics.mean_squared_error(self.y_test, self.pred)
    def get_mean_squared_error(self):
        return np.sqrt(metrics.mean_squared_error(self.y_test, self.pred))
    def predict_for_tomorrow(self):
        return mean(self.RFreg.predict(self.dataframe_for_prediction))        


# x = cg.bitcoinDataFrame[['TimeStamp','MAmonth','MAweek']]
# y = cg.bitcoinDataFrame.iloc[:,3:].values

# x_train, x_test, y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=0)
# RFreg = RandomForestRegressor(n_estimators=10,random_state=0)
# RFreg.fit(x_train,y_train)
# y_predict = RFreg.predict(x_test)
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
# r2 = metrics.r2_score(y_test,y_predict)
# print(r2)
# print(mean(RFreg.predict(cg.bitcoinDataFrame_for_predicition)))


