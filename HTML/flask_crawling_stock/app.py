from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

url = 'http://companyinfo.stock.naver.com/v1/company/c1010001.aspx?cmp_cd=005930'
datas = pd.read_html(url, encoding='utf-8')[12]
datas = datas.to_dict(orient='records')

@app.route('/')
def index():
    return render_template('index_table.html', datas = datas)

if __name__=="__main__":
    app.run(debug=True)