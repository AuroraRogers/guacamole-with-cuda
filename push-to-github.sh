#!/bin/bash

# Script to push local guac-kalman-filter project to GitHub
# Created by Codegen

# Set colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting GitHub push process...${NC}"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed. Please install git first.${NC}"
    exit 1
fi

# Get the local directory path
read -p "Enter the full path to your local guac-kalman-filter directory: " LOCAL_DIR

# Validate directory exists
if [ ! -d "$LOCAL_DIR" ]; then
    echo -e "${RED}Error: Directory does not exist: $LOCAL_DIR${NC}"
    exit 1
fi

# Change to the directory
cd "$LOCAL_DIR" || { echo -e "${RED}Error: Could not change to directory $LOCAL_DIR${NC}"; exit 1; }

echo -e "${YELLOW}Checking if directory is already a git repository...${NC}"

# Check if it's already a git repository
if [ -d ".git" ]; then
    echo -e "${GREEN}Directory is already a git repository.${NC}"
    
    # Check if the remote already exists
    if git remote -v | grep -q "github.com/AuroraRogers/guac-kalman-filter-middleware"; then
        echo -e "${GREEN}Remote repository already configured.${NC}"
    else
        echo -e "${YELLOW}Adding remote repository...${NC}"
        git remote add origin https://github.com/AuroraRogers/guac-kalman-filter-middleware.git
        echo -e "${GREEN}Remote added successfully.${NC}"
    fi
else
    echo -e "${YELLOW}Initializing git repository...${NC}"
    git init
    echo -e "${GREEN}Git repository initialized.${NC}"
    
    echo -e "${YELLOW}Adding remote repository...${NC}"
    git remote add origin https://github.com/AuroraRogers/guac-kalman-filter-middleware.git
    echo -e "${GREEN}Remote added successfully.${NC}"
fi

# Add all files to git
echo -e "${YELLOW}Adding all files to git...${NC}"
git add .

# Commit changes
echo -e "${YELLOW}Committing changes...${NC}"
read -p "Enter a commit message (default: 'Initial commit'): " COMMIT_MSG
COMMIT_MSG=${COMMIT_MSG:-"Initial commit"}
git commit -m "$COMMIT_MSG"

# Push to GitHub
echo -e "${YELLOW}Pushing to GitHub...${NC}"
echo -e "${YELLOW}You may be prompted to enter your GitHub username and password/token.${NC}"

# Check if the repository has any branches
if git branch | grep -q "master"; then
    DEFAULT_BRANCH="master"
elif git branch | grep -q "main"; then
    DEFAULT_BRANCH="main"
else
    # Create a new branch if none exists
    DEFAULT_BRANCH="main"
    git checkout -b $DEFAULT_BRANCH
fi

# Push to GitHub
git push -u origin $DEFAULT_BRANCH

# Check if push was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully pushed to GitHub repository: https://github.com/AuroraRogers/guac-kalman-filter-middleware${NC}"
    echo -e "${GREEN}Your code is now available on GitHub!${NC}"
else
    echo -e "${RED}Push failed. Please check your GitHub credentials and try again.${NC}"
    echo -e "${YELLOW}If you're using HTTPS, you might need a personal access token instead of your password.${NC}"
    echo -e "${YELLOW}Visit https://github.com/settings/tokens to create a token.${NC}"
fi