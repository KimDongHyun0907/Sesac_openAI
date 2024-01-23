from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

url = 'http://companyinfo.stock.naver.com/v1/company/c1010001.aspx?cmp_cd=005930'
datas = pd.read_html(url, encoding='utf-8')[12]
datas = datas.to_dict(orient='records')

@app.route('/')
def index():
    return render_template('index_table.html', datas = datas)

@app.route('/stock', methods = ['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('stock.html')
    
    if request.method == 'POST':
        stock_name = request.form['stock_name']
        url = 'http://companyinfo.stock.naver.com/v1/company/c1010001.aspx?cmp_cd='+stock_name

        datas = pd.read_html(url, encoding='utf-8')[12]
        datas = datas.to_dict(orient='records')
        datainfo = pd.read_html(url, encoding='utf-8')[0]
        datainfo = datainfo.iloc[:, 0][0]

        return render_template('stock.html', datas = datas, datainfo = datainfo, stock_name = stock_name)
if __name__=="__main__":
    app.run(debug=True)