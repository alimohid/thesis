import LinearRegression as lr
import RandomForest as rr
import RidgeRegression as ridge
import CoinGeckoAPI as cg
import logging
from datetime import datetime
from numpy.core.fromnumeric import mean



logging.basicConfig(filename="file.log",format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info('Date and time for prediction: {}'.format(datetime.fromtimestamp(cg.timestamp_for_tomorrow/1000)))

bitcoin_prediction_list  = []
ethereum_prediction_list  = []
cardano_prediction_list = [] 
xrp_prediction_list  = []
dogecoin_prediction_list = []
ethereum_complete_prediction_list = []
cardano_complete_prediction_list = []
xrp_complete_prediction_list = []
dogecoin_complete_prediction_list = []
LR_Bitcoin = lr.linear_regression(cg.bitcoinDataFrame,cg.bitcoinDataFrame_for_predicition)
LR_Bitcoin.train_for_prediction()
#print(LR_Bitcoin.predict_for_tomorrow())
bitcoin_complete_prediction_list =  LR_Bitcoin.complete_prediction()
logger.info('Bitcoin Predicition by Linear Regression: {}'.format(LR_Bitcoin.predict_for_tomorrow()))
RR_Bitcoin = rr.Random_Forest(cg.bitcoinDataFrame,cg.bitcoinDataFrame_for_predicition)
RR_Bitcoin.train_for_prediction()
bitcoin_complete_prediction_list2 =  RR_Bitcoin.complete_prediction()
logger.info('Bitcoin Predicition by Random Forest: {}'.format(RR_Bitcoin.predict_for_tomorrow()))
Ridge_Bitcoin = ridge.Ridge_Regression(cg.bitcoinDataFrame,cg.bitcoinDataFrame_for_predicition)
Ridge_Bitcoin.train_for_prediction()
bitcoin_complete_prediction_list3 =  Ridge_Bitcoin.complete_prediction()
logger.info('Bitcoin Predicition by Ridge Regression: {}'.format(Ridge_Bitcoin.predict_for_tomorrow()))
logger.info('Mean Bitcoin prediction: {}'.format((LR_Bitcoin.predict_for_tomorrow()+RR_Bitcoin.predict_for_tomorrow()+Ridge_Bitcoin.predict_for_tomorrow())/3))
bitcoin_prediction_list.append(LR_Bitcoin.predict_for_tomorrow())
bitcoin_prediction_list.append(RR_Bitcoin.predict_for_tomorrow())
bitcoin_prediction_list.append(Ridge_Bitcoin.predict_for_tomorrow())
bitcoin_prediction_list.append((Ridge_Bitcoin.predict_for_tomorrow() + LR_Bitcoin.predict_for_tomorrow() + RR_Bitcoin.predict_for_tomorrow())/3)


LR_cardano = lr.linear_regression(cg.cardanoDataFrame,cg.cardanoDataFrame_for_prediction)
LR_cardano.train_for_prediction()
RR_cardano = rr.Random_Forest(cg.cardanoDataFrame,cg.cardanoDataFrame_for_prediction)
RR_cardano.train_for_prediction()
Ridge_cardano = ridge.Ridge_Regression(cg.cardanoDataFrame,cg.cardanoDataFrame_for_prediction)
Ridge_cardano.train_for_prediction()
logger.info('Cardano Predicition by Linear Regression: {}'.format(LR_cardano.predict_for_tomorrow()))
logger.info('Cradano Predicition by Random Forrest: {}'.format(RR_cardano.predict_for_tomorrow()))
logger.info('Cardano Predicition by Ridge Regression: {}'.format(Ridge_cardano.predict_for_tomorrow()))
logger.info('Mean cardano Predicition : {}'.format((LR_cardano.predict_for_tomorrow()+RR_cardano.predict_for_tomorrow()+Ridge_cardano.predict_for_tomorrow())/3))
cardano_prediction_list.append(LR_cardano.predict_for_tomorrow())
cardano_prediction_list.append(RR_cardano.predict_for_tomorrow())
cardano_prediction_list.append(Ridge_cardano.predict_for_tomorrow())
cardano_prediction_list.append((Ridge_cardano.predict_for_tomorrow() + LR_cardano.predict_for_tomorrow() + RR_cardano.predict_for_tomorrow())/3)
cardano_complete_prediction_list.append(LR_cardano.complete_prediction())
cardano_complete_prediction_list.append(RR_cardano.complete_prediction())
cardano_complete_prediction_list.append(Ridge_cardano.complete_prediction())

LR_ethereum = lr.linear_regression(cg.ethereumDataFrame,cg.ethereumDataFrame_for_predicition)
LR_ethereum.train_for_prediction()
RR_ethereum = rr.Random_Forest(cg.ethereumDataFrame,cg.ethereumDataFrame_for_predicition)
RR_ethereum.train_for_prediction()
Ridge_ethereum = ridge.Ridge_Regression(cg.ethereumDataFrame,cg.ethereumDataFrame_for_predicition)
Ridge_ethereum.train_for_prediction()
logger.info('Ethereum Predicition by Linear Regression: {}'.format(LR_ethereum.predict_for_tomorrow()))
logger.info('Ethereum Predicition by Random Forrest: {}'.format(RR_ethereum.predict_for_tomorrow()))
logger.info('Ethereum Predicition by Ridge Regression: {}'.format(Ridge_ethereum.predict_for_tomorrow()))
logger.info('Mean Ethereum Predicition : {}'.format((LR_ethereum.predict_for_tomorrow()+RR_ethereum.predict_for_tomorrow()+Ridge_ethereum.predict_for_tomorrow())/3))
ethereum_prediction_list.append(LR_ethereum.predict_for_tomorrow())
ethereum_prediction_list.append(RR_ethereum.predict_for_tomorrow())
ethereum_prediction_list.append(Ridge_ethereum.predict_for_tomorrow())
ethereum_prediction_list.append((Ridge_ethereum.predict_for_tomorrow() + LR_ethereum.predict_for_tomorrow() + RR_ethereum.predict_for_tomorrow())/3)
ethereum_complete_prediction_list.append(LR_ethereum.complete_prediction())
ethereum_complete_prediction_list.append(RR_ethereum.complete_prediction())
ethereum_complete_prediction_list.append(Ridge_ethereum.complete_prediction())

LR_dogecoin = lr.linear_regression(cg.dogecoinDataFrame,cg.dogecoinDataFrame_for_prediction)
LR_dogecoin.train_for_prediction()
RR_dogecoin = rr.Random_Forest(cg.dogecoinDataFrame,cg.dogecoinDataFrame_for_prediction)
RR_dogecoin.train_for_prediction()
Ridge_dogecoin = ridge.Ridge_Regression(cg.dogecoinDataFrame,cg.dogecoinDataFrame_for_prediction)
Ridge_dogecoin.train_for_prediction()
logger.info('Dogecoin Predicition by Linear Regression: {}'.format(LR_dogecoin.predict_for_tomorrow()))
logger.info('Dogecoin Predicition by Random Forrest: {}'.format(RR_dogecoin.predict_for_tomorrow()))
logger.info('Dogecoin Predicition by Ridge Regression: {}'.format(Ridge_dogecoin.predict_for_tomorrow()))
logger.info('Mean Dogecoin Predicition : {}'.format((LR_dogecoin.predict_for_tomorrow()+RR_dogecoin.predict_for_tomorrow()+Ridge_dogecoin.predict_for_tomorrow())/3))
dogecoin_prediction_list.append(LR_dogecoin.predict_for_tomorrow())
dogecoin_prediction_list.append(RR_dogecoin.predict_for_tomorrow())
dogecoin_prediction_list.append(Ridge_dogecoin.predict_for_tomorrow())
dogecoin_prediction_list.append((Ridge_dogecoin.predict_for_tomorrow() + LR_dogecoin.predict_for_tomorrow() + RR_dogecoin.predict_for_tomorrow())/3)
dogecoin_complete_prediction_list.append(LR_dogecoin.complete_prediction())
dogecoin_complete_prediction_list.append(RR_dogecoin.complete_prediction())
dogecoin_complete_prediction_list.append(Ridge_dogecoin.complete_prediction())

LR_xrp = lr.linear_regression(cg.xrpDataFrame,cg.xrpDataFrame_for_prediction)
LR_xrp.train_for_prediction()
RR_xrp = rr.Random_Forest(cg.xrpDataFrame,cg.xrpDataFrame_for_prediction)
RR_xrp.train_for_prediction()
Ridge_xrp = ridge.Ridge_Regression(cg.xrpDataFrame,cg.xrpDataFrame_for_prediction)
Ridge_xrp.train_for_prediction()
logger.info('Ripple Predicition by Linear Regression: {}'.format(LR_xrp.predict_for_tomorrow()))
logger.info('Ripple Predicition by Random Forrest: {}'.format(RR_xrp.predict_for_tomorrow()))
logger.info('Ripple Predicition by Ridge Regression: {}'.format(Ridge_xrp.predict_for_tomorrow()))
logger.info('Mean Ripple Predicition : {}'.format((LR_xrp.predict_for_tomorrow()+RR_xrp.predict_for_tomorrow()+Ridge_xrp.predict_for_tomorrow())/3))
xrp_prediction_list.append(LR_xrp.predict_for_tomorrow())
xrp_prediction_list.append(RR_xrp.predict_for_tomorrow())
xrp_prediction_list.append(Ridge_xrp.predict_for_tomorrow())
xrp_prediction_list.append((Ridge_xrp.predict_for_tomorrow() + LR_xrp.predict_for_tomorrow() + RR_xrp.predict_for_tomorrow())/3)
xrp_complete_prediction_list.append(LR_xrp.complete_prediction())
xrp_complete_prediction_list.append(RR_xrp.complete_prediction())
xrp_complete_prediction_list.append(Ridge_xrp.complete_prediction())