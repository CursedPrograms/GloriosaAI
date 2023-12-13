from flask import Flask, render_template

app = Flask(__name__)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.route('/not_found_example')
def not_found_example():
    return "This resource is not found.", 404

@app.route('/internal_error_example')
def internal_error_example():
    
    raise Exception("Simulated internal server error")

if __name__ == '__main__':
    app.run(debug=True)
