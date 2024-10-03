@echo off
REM Switch to a new orphan branch
git checkout --orphan new_branch

REM Stage all changes
git add .

REM Commit changes
git commit -m "new_commit"

REM Delete the old main branch
git branch -D main

REM Rename the new branch to main
git branch -m main

REM Force push to the remote main branch
git push -f origin main
