from flask import Flask, render_template, request
import os
import subprocess

app = Flask(__name__)

scripts = {
    "1": {
        "name": "Run 'Trainer'",
        "description": "Train a Generative Adversarial Network (GAN)",
        "file_name": "trainer.py"
    },
    "2": {
        "name": "Run 'Video Encoder'",
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
    user_choice = request.form.get('script_choice')

    if user_choice in scripts:
        selected_script = scripts[user_choice]
        script_file_name = selected_script["file_name"]
        script_file_path = os.path.join(current_script_dir, script_file_name)

        if os.path.exists(script_file_path):
            try:
                subprocess.run(["python", script_file_path])
                return f"Script '{selected_script['name']}' executed successfully."
            except Exception as e:
                return f"An error occurred while running the script: {e}"
        else:
            return f"Script file '{script_file_name}' does not exist."
    else:
        return "Invalid script choice."


if __name__ == "__main__":
    app.run(debug=True)