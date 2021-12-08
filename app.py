from flask import url_for
from flask import Flask
from flask import render_template
import main
import CoinGeckoAPI as cg
app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/bitcoin")
def bitcoin():
    labels = cg.bitcoinDataFrame['TimeStamp'].values.tolist()
    values = cg.bitcoinDataFrame['Price'].values.tolist()
    values2 = main.bitcoin_complete_prediction_list
    values3 = main.bitcoin_complete_prediction_list2
    values4 = main.bitcoin_complete_prediction_list3
    return render_template('bitcoin.html',title='Bitcoin',var = main.bitcoin_prediction_list,labels=labels,values=values,values2=values2,values3=values3,values4=values4)
@app.route("/ethereum")
def ethereum():
    labels = cg.ethereumDataFrame['TimeStamp'].values.tolist()
    values = cg.ethereumDataFrame['Price'].values.tolist()
    return render_template('ethereum.html',title='Ethereum', var = main.ethereum_prediction_list,labels=labels,values=values,values2=main.ethereum_complete_prediction_list[0],values3=main.ethereum_complete_prediction_list[1],values4=main.ethereum_complete_prediction_list[2] )
@app.route("/cardano")
def cardano():
    labels = cg.cardanoDataFrame['TimeStamp'].values.tolist()
    values = cg.cardanoDataFrame['Price'].values.tolist()
    return render_template('cardano.html',title='Cardano', var = main.cardano_prediction_list,labels=labels,values=values,values2=main.cardano_complete_prediction_list[0],values3=main.cardano_complete_prediction_list[1],values4=main.cardano_complete_prediction_list[2])
@app.route("/dogecoin")
def dogecoin():
    labels = cg.dogecoinDataFrame['TimeStamp'].values.tolist()
    values = cg.dogecoinDataFrame['Price'].values.tolist()
    return render_template('dogecoin.html',title='Dogecoin', var = main.dogecoin_prediction_list,labels=labels,values=values,values2=main.dogecoin_complete_prediction_list[0],values3=main.dogecoin_complete_prediction_list[1],values4=main.dogecoin_complete_prediction_list[2])
@app.route("/ripple")
def ripple():
    labels = cg.xrpDataFrame['TimeStamp'].values.tolist()
    values = cg.xrpDataFrame['Price'].values.tolist()
    return render_template('ripple.html',title='Ripple', var = main.xrp_prediction_list,labels=labels,values=values,values2=main.xrp_complete_prediction_list[0],values3=main.xrp_complete_prediction_list[1],values4=main.xrp_complete_prediction_list[2])    
if __name__ == '__main__':
    # from os import environ
    # app.run(host="0.0.0.0",debug=False, port=environ.get("PORT", 5000))
    app.run(debug=True)