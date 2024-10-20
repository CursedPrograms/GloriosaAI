import subprocess

def run_command(command):
    """Run a command in the shell and print the output."""
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        exit(1)

# Switch to a new orphan branch
run_command(["git", "checkout", "--orphan", "new_branch"])

# Stage all changes
run_command(["git", "add", "."])

# Commit changes
run_command(["git", "commit", "-m", "new_commit"])

# Delete the old main branch
run_command(["git", "branch", "-D", "main"])

# Rename the new branch to main
run_command(["git", "branch", "-m", "main"])

# Force push to the remote main branch
run_command(["git", "push", "-f", "origin", "main"])