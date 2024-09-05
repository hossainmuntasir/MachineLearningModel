from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model-comparison')
def model_comparison():
    return render_template('model_comparison.html')

@app.route('/building-1')
def building_1():
    return render_template('building_1.html')

@app.route('/building-2')
def building_2():
    return render_template('building_2.html')

@app.route('/building-3')
def building_3():
    return render_template('building_3.html')

@app.route('/about_us')
def about_us():
    return render_template('aboutus.html')

if __name__ == '__main__':
    app.run(debug=True)
