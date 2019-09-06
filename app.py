from flask import Flask, url_for, render_template
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score

app = Flask(__name__)

data = pd.read_csv("data/reliance.csv")
data = data.dropna()
m, n = data.shape
data["Date"]= pd.to_datetime(data["Date"])
data['Date2num'] = data['Date'].apply(lambda x: mdates.date2num(x))
date = data.loc[:, ['Date']]

X = data["Date2num"].values.reshape(-1,1)
y = data["Close"].values.reshape(-1,1)

split = int(np.floor(m*0.7))
X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

date_train = date.loc[:split]
date_test = date.loc[split:]

future_date_ = "09-Sep-2019"
future_date2num = np.reshape(mdates.date2num(pd.to_datetime(future_date_)), (-1,1))

@app.route('/')
def index():
    head_tail = data.head()
    head_tail = head_tail.append(data.tail())
    return render_template("index.html", 
                    title="Home", company_name="Reliance Industries Limited", 
                    data=head_tail)

@app.route('/linear')
def linear():
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    plot_save_iamge('linear.png', y_pred)

    future_price_ = round(regressor.predict(future_date2num)[0,0],2)

    return render_template("linear.html", 
                            title="Linear Regression", 
                            future_date = future_date_,
                            future_price = future_price_,
                            r2_score = round(r2_score(y_test, y_pred),2))

@app.route('/ridge')
def ridge():
    regressor = linear_model.Ridge(alpha=0.05, normalize=True)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    plot_save_iamge('ridge.png', y_pred)

    future_price_ = round(regressor.predict(future_date2num)[0,0],2)
    return render_template("ridge.html", 
                            title="Ridge Regression", 
                            future_date = future_date_,
                            future_price = future_price_,
                            r2_score = round(r2_score(y_test, y_pred),2))

@app.route('/lasso')
def lasso():
    regressor = linear_model.Lasso(alpha=0.3, normalize=True)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    plot_save_iamge('lasso.png', y_pred)

    future_price_ = round(regressor.predict(future_date2num)[0],2)
    #future_price_ = regressor.predict(future_date2num)
    return render_template("lasso.html", 
                            title="Lasso Regression", 
                            future_date = future_date_,
                            future_price = future_price_,
                            r2_score = round(r2_score(y_test, y_pred),2))

def plot_save_iamge(img_name, y_pred):
    plt.xticks(rotation=45)
    plt.plot_date(date_test, y_test, fmt='b-', xdate=True, ydate=False, label='Real value')
    plt.plot_date(date_test, y_pred, fmt='r-', xdate=True, ydate=False, label='Predicted value')
    plt.legend(loc='upper center')
    plt.ylabel('Close prices')
    plt.title('Relinace Industries Limited')
    plt.grid()
    plt.savefig('static/images/' + img_name)

if __name__ == "__main__":
    app.run(debug=True)