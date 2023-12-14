from flask import Flask, render_template, request, jsonify
import os
import subprocess
from system.paths import scripts
from system.error_handling import not_found_error, internal_error

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', scripts=scripts)

def get_data():
    data = {'message': 'Welcome to GloriosaAI!'}
    return jsonify(data)

current_script_dir = os.path.dirname(os.path.abspath(__file__))

@app.route('/execute', methods=['POST'])
def execute():
    command = request.form['command']
    
    try:
        result = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
        return render_template('result.html', result=result)
    except subprocess.CalledProcessError as e:
        return render_template('result.html', result=f"Error: {e.output}")

@app.errorhandler(404)
def not_found_handler(error):
    return not_found_error(error)

@app.errorhandler(500)
def internal_error_handler(error):
    return internal_error(error)

@app.route('/not_found_example')
def not_found_example():
    return "This resource is not found.", 404

@app.route('/internal_error_example')
def internal_error_example():
    
    raise Exception("Simulated internal server error")

if __name__ == "__main__":
    app.run(debug=True)

os.system("python main.py")