from flask import Flask, render_template, request
import os
import subprocess

app = Flask(__name__)

scripts = {
    "1": {
        "name": "Run 'Trainer'",
        "description": "Train a Generative Adversarial Network (GAN)",
        "file_name": "trainer.py",
        "template": "trainer.html"
    },
    "2": {
        "name": "Run 'Video Encoder'",
        "description": "Encode a video using GloriosaAI",
        "file_name": "video_encoder.py",
        "template": "video-encoder.html"
    },
    "3": {
        "name": "Run 'ModelOut'",
        "description": "Output images from trained models with GloriosaAI",
        "file_name": "modelout.py",
        "template": "modelout.html"
    },
    "4": {
        "name": "Run 'Style Transfer'",
        "description": "Style an image with GloriosaAI",
        "file_name": "style_transfer/styles.py",
        "template": "style-transfer.html"
    },
    "00": {
        "name": "Run 'Install Dependencies'",
        "description": "Install necessary dependencies for GloriosaAI",
        "file_name": "install_dependencies.py",
        "template": "install-dependencies.html"
    },
}

current_script_dir = os.path.dirname(os.path.abspath(__file__))

main_script_path = os.path.join(current_script_dir, "main.py")
if os.path.exists(main_script_path):
    subprocess.run(["python", main_script_path])

@app.route('/')
def index():
    return render_template('index.html', scripts=scripts)

def get_data():
    data = {'message': 'Hello from Flask!'}
    return jsonify(data)

@app.route('/run_script', methods=['POST'])
def run_script():
    user_choice = request.form.get('script_choice')

    if user_choice in scripts:
        selected_script = scripts[user_choice]
        script_file_name = selected_script["file_name"]
        script_file_path = os.path.join(current_script_dir, script_file_name)

        if os.path.exists(script_file_path):
            try:
                result = subprocess.run(["python", script_file_path], capture_output=True, text=True)
                output = result.stdout
                return render_template(selected_script['template'], script_name=selected_script['name'], output=output)
            except Exception as e:
                return f"An error occurred while running the script: {e}"
        else:
            return f"Script file '{script_file_name}' does not exist."
    else:
        return "Invalid script choice."

if __name__ == "__main__":
    app.run(debug=True)
