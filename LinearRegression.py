from numpy.core.fromnumeric import mean
import CoinGeckoAPI as cg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class linear_regression:
    def __init__(self,dataframe,dataframe_for_prediction):
        self.dataframe = dataframe
        self.dataframe_for_prediction = dataframe_for_prediction
    def train_for_prediction(self):
        x = self.dataframe[['TimeStamp','MAmonth','MAweek']]
        y = self.dataframe.iloc[:,3:].values
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size = 0.2,random_state  = 0)
        self.regressor =  LinearRegression()
        self.regressor.fit(self.x_train,self.y_train)
        self.pred = self.regressor.predict(self.x_test)
    def get_r2_score(self):
        return r2_score(self.y_test,self.pred)
    def predict_for_tomorrow(self):
        return mean(self.regressor.predict(self.dataframe_for_prediction))
    def complete_prediction(self):
        prediction_list = []
        for i in self.dataframe.index:
            df = cg.convert_to_dataframe_for_prediction(self.dataframe["TimeStamp"][i],self.dataframe["MAmonth"][i],self.dataframe["MAweek"][i])
            prediction_list.append(mean(self.regressor.predict(df)))
        return prediction_list 
    def get_cofficients(self):
        return self.regressor.coef_
    def get_intercept(self):
        return self.regressor.intercept_
    




# x = cg.bitcoinDataFrame[['TimeStamp','MAmonth','MAweek']]
# y = cg.bitcoinDataFrame.iloc[:,3:].values
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state  = 0)
# regressor =  LinearRegression()
# regressor.fit(x_train,y_train)
# pred = regressor.predict(x_test)
# print(mean(regressor.predict(cg.bitcoinDataFrame_for_predicition)))

# print(r2_score(y_test,pred))