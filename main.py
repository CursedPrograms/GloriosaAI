import os
import subprocess
import json

def main():
    with open('config.json') as json_file:
        config_data = json.load(json_file)

    # Get the project name from the JSON data
    app_name = config_data.get('Config', {}).get('AppName', 'default_app')

    # Print the actual app name value
    print(app_name)

    scripts = {
        "1": {
            "name": "Run 'trainer.py'",
            "description": "Train a Generative Adversarial Network (GAN)",
            "file_name": "scripts/trainer.py"
        },
        "2": {
            "name": "Run 'video_encoder.py",
            "description": "Encode a video using GloriosaAI",
            "file_name": "scripts/video_encoder.py"
        },    
        "3": {
            "name": "Run 'modelout.py'",
            "description": "Output images from trained models with GloriosaAI",
            "file_name": "scripts/modelout.py"
        },
        "4": {
            "name": "Run 'image-processor.py'",
            "description": "Prepare images for GloriosaAI",
            "file_name": "scripts/image_processor.py"
        },
        "00": {
            "name": "Run 'install_dependencies.py'",
            "description": "Install necessary dependencies for GloriosaAI",
            "file_name": "scripts/install_dependencies.py"
        },
    }

    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    while True:
        print("\nAvailable Scripts:")
        for key, script_info in scripts.items():
            print(f"{key}: {script_info['name']} - {script_info['description']}")
        
        user_choice = input("Enter the number of the script you want to run (or 'q' to quit): ").strip()
        
        if user_choice == 'q':
            break
        
        if user_choice in scripts:
            selected_script = scripts[user_choice]
            script_file_name = selected_script["file_name"]
            script_file_path = os.path.join(current_script_dir, script_file_name)
            
            if os.path.exists(script_file_path):
                try:
                    subprocess.run(["python", script_file_path])
                except Exception as e:
                    print(f"An error occurred while running the script: {e}")
            else:
                print(f"Script file '{script_file_name}' does not exist.")
        else:
            print("Invalid choice. Please select a valid script number.")

if __name__ == "__main__":
    main()
