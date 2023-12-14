from flask import Flask, render_template, request
import os
import subprocess
from error_handling import not_found_error, internal_error

app = Flask(__name__)

scripts = {
    "1": {
        "name": "Run 'Trainer'",
        "description": "Train a Generative Adversarial Network (GAN)",
        "file_name": "trainer.py"
    },
    "2": {
        "name": "Run 'Video Encoder",
        "description": "Encode a video using GloriosaAI",
        "file_name": "video_encoder.py"
    },    
    "3": {
        "name": "Run 'ModelOut'",
        "description": "Output images from trained models with GloriosaAI",
        "file_name": "modelout.py"
    },
    "4": {
        "name": "Run 'Style Transfer'",
        "description": "Style an image with GloriosaAI",
        "file_name": "style_transfer/styles.py"
    },
    "00": {
        "name": "Run 'Install Dependencies'",
        "description": "Install necessary dependencies for GloriosaAI",
        "file_name": "install_dependencies.py"
    },
}

current_script_dir = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template('index.html', scripts=scripts)

@app.route('/run_script', methods=['POST'])
def run_script():
    script_choice = request.form['script_choice']

    if script_choice in scripts:
        selected_script = scripts[script_choice]
        script_file_name = selected_script["file_name"]
        script_file_path = os.path.join(current_script_dir, script_file_name)

        if os.path.exists(script_file_path):
            try:
                result = subprocess.check_output(["python", script_file_path], text=True, stderr=subprocess.STDOUT)
                return render_template('result.html', result=result)
            except subprocess.CalledProcessError as e:
                return render_template('result.html', result=f"Error: {e.output}")
        else:
            return render_template('result.html', result=f"Script file '{script_file_name}' does not exist.")
    else:
        return render_template('result.html', result="Invalid choice. Please select a valid script number.")

@app.errorhandler(404)
def not_found_handler(error):
    return not_found_error(error)

@app.errorhandler(500)
def internal_error_handler(error):
    return internal_error(error)

if __name__ == "__main__":
    app.run(debug=True)
