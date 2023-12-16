import os
import shutil

def clear_pycache(root_directory):
    for root, dirs, files in os.walk(root_directory):
        for directory in dirs:
            if directory == "__pycache__":
                pycache_path = os.path.join(root, directory)
                shutil.rmtree(pycache_path)
                print(f"Deleted: {pycache_path}")

if __name__ == "__main__":
    root_directory = "/path/to/your/project"
    clear_pycache(root_directory)
