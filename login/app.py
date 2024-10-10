from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from dashboard import create_modelevaluation_dashboards, create_modelcomparison_dashboard
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
building_no = 3

create_modelevaluation_dashboards(app)
create_modelcomparison_dashboard(app, building_no)

app.secret_key = 'your secret key'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'logintest'

mysql = MySQL(app)


@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = % s', (username,))
        account = cursor.fetchone()
        if account and check_password_hash(account['password'], password):
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            msg = 'Logged in successfully!'
            return redirect('/index')
        else:
            msg = 'Incorrect username / password!'
    return render_template('login.html', msg=msg)



@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = % s', (username,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif len(password) < 8:
            msg = 'Password must be 8 characters long!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Hash the password before storing
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            cursor.execute('INSERT INTO accounts VALUES (NULL, % s, % s, % s)', (username, hashed_password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!, Please Login'
            return render_template('login.html', msg=msg)
    elif request.method == 'POST':
        msg = 'Please fill out the form!'
    return render_template('register.html', msg=msg)

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/model-comparison')
def model_comparison():
    if "username" not in session:
        return redirect('/login')
    return render_template('model_comparison.html')


@app.route('/building-1')
def building_1():
    if "username" not in session:
        return redirect('/login')
    return render_template('building_1.html')


@app.route('/building-2')
def building_2():
    if "username" not in session:
        return redirect('/login')
    return render_template('building_2.html')


@app.route('/building-3')
def building_3():
    if "username" not in session:
        return redirect('/login')
    return render_template('building_3.html')


@app.route('/about_us')
def about_us():
    return render_template('aboutus.html')


if __name__ == '__main__':
    app.run(debug=True)
