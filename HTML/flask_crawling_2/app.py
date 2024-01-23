from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    datas = [
        {'name': '반원', 'level': 60, 'point': 360, 'exp' : 45000},
        {'name': '반원2', 'level': 2, 'point': 20, 'exp' : 200},
        {'name': '반원3', 'level': 3, 'point': 30, 'exp' : 300}
    ]
    return render_template('index.html', datas = datas)

if __name__=="__main__":
    app.run(debug=True)
