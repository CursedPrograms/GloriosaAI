import subprocess
import os

def install_dependencies():
    try:
        # Get the directory of the current script
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Specify the path to the requirements.txt file relative to the script's location
        requirements_file_path = os.path.join(script_directory, '../requirements.txt')

        # Activate the virtual environment if it exists
        venv_activate_path = os.path.join(script_directory, 'psdenv/Scripts/activate')
        if os.path.exists(venv_activate_path):
            subprocess.run([venv_activate_path], shell=True)

        # Install dependencies using pip
        subprocess.run(['pip', 'install', '-r', requirements_file_path])

        print("Dependencies installed successfully.")

    except Exception as e:
        print(f"Error installing dependencies: {e}")

if __name__ == "__main__":
    install_dependencies()