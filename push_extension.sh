#!/bin/bash

# Set variables
REPO_URL="https://github.com/HeavenzFire/all-in-one-dev-tool.git"
COMMIT_MESSAGE="Initial commit"
EXTENSION_FOLDER="$HOME/Desktop/all-in-one-dev-tool"

# Check if the directory exists
if [ ! -d "$EXTENSION_FOLDER" ]; then
  echo "Directory $EXTENSION_FOLDER does not exist."
  exit 1
fi

# Navigate to the extension folder
cd "$EXTENSION_FOLDER"

# Initialize a new Git repository
git init

# Add the remote repository
git remote add origin $REPO_URL

# Add all files to the staging area
git add .

# Commit the changes
git commit -m "$COMMIT_MESSAGE"

# Push the changes to the remote repository
git push -u origin main

echo "Extension has been pushed to GitHub successfully!"
