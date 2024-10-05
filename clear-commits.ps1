# Switch to a new orphan branch
git checkout --orphan new_branch

# Stage all changes
git add .

# Commit changes
git commit -m "new_commit"

# Delete the old main branch
git branch -D main

# Rename the new branch to main
git branch -m main

# Force push to the remote main branch
git push -f origin main