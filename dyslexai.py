import os      
scripts = {
    "1": {
        "name": "Run 'dyslexai_trainer.py'",
        "description": "Train a Generative Adversarial Network (GAN)",
        "file_name": "dyslexai_trainer.py"
    },
    "2": {
        "name": "Run 'dyslexai-video_encoder.py'",
        "description": "Encode a video using DyslexAI",
        "file_name": "dyslexai_video_encoder.py"
    },    
    "3": {
        "name": "Run 'dyslexai_modelmix.py'",
        "description": "Combine and mix trained models with DyslexAI",
        "file_name": "dyslexai_modelmix.py"
    },
	"4": {
        "name": "Run 'install_dependencies.py'",
        "description": "Install necessary dependencies for DyslexAI",
        "file_name": "install_dependencies.py"
    },
}

current_script_dir = os.path.dirname(os.path.abspath(__file__))

print("Available Scripts:")
for key, script_info in scripts.items():
    print(f"{key}: {script_info['name']} - {script_info['description']}")

user_choice = input("Enter the number of the script you want to run: ").strip()

if user_choice in scripts:
    selected_script = scripts[user_choice]
    script_file_name = selected_script["file_name"]
    script_file_path = os.path.join(current_script_dir, script_file_name)
    
    if os.path.exists(script_file_path):
        os.system(f"python {script_file_path}")
    else:
        print(f"Script file '{script_file_name}' does not exist.")
else:
    print("Invalid choice. Please select a valid script number.")