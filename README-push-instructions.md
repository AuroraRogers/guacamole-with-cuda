# How to Push Your Local Project to GitHub

This guide will help you push your local `guac-kalman-filter` project to the GitHub repository `guac-kalman-filter-middleware`.

## Prerequisites

- Git installed on your system
- GitHub account with access to the repository
- Your local project files ready to be pushed

## Option 1: Using the Provided Script

I've created a script that automates the process for you. Here's how to use it:

1. Save the `push-to-github.sh` script to your computer
2. Make the script executable:
   ```bash
   chmod +x push-to-github.sh
   ```
3. Run the script:
   ```bash
   ./push-to-github.sh
   ```
4. Follow the prompts in the script:
   - Enter the full path to your local guac-kalman-filter directory
   - Provide a commit message (or use the default)
   - Enter your GitHub credentials when prompted

The script will:
- Initialize a git repository if needed
- Add the GitHub remote
- Add and commit your files
- Push to the GitHub repository

## Option 2: Manual Steps

If you prefer to do it manually, follow these steps:

1. Navigate to your local project directory:
   ```bash
   cd /path/to/your/guac-kalman-filter
   ```

2. Initialize a git repository (if not already done):
   ```bash
   git init
   ```

3. Add the GitHub repository as a remote:
   ```bash
   git remote add origin https://github.com/AuroraRogers/guac-kalman-filter-middleware.git
   ```

4. Add all your files to git:
   ```bash
   git add .
   ```

5. Commit your changes:
   ```bash
   git commit -m "Initial commit"
   ```

6. Create and checkout the main branch (if needed):
   ```bash
   git checkout -b main
   ```

7. Push to GitHub:
   ```bash
   git push -u origin main
   ```

## Authentication Notes

When pushing to GitHub, you'll need to authenticate:

- If you're using HTTPS (the default in the script), you'll be prompted for your GitHub username and password/token
- GitHub no longer accepts passwords for command-line operations, so you'll need to use a Personal Access Token
- You can create a token at: https://github.com/settings/tokens

## Troubleshooting

If you encounter issues:

1. **"Remote already exists" error**:
   ```bash
   git remote remove origin
   git remote add origin https://github.com/AuroraRogers/guac-kalman-filter-middleware.git
   ```

2. **Authentication failures**:
   - Make sure you're using a Personal Access Token, not your password
   - Check that your token has the correct permissions (at least `repo` scope)

3. **Conflicts when pushing**:
   If the remote repository already has content that conflicts with your local files:
   ```bash
   git pull --rebase origin main
   # Resolve any conflicts
   git push -u origin main
   ```

4. **Permission denied errors**:
   - Ensure you have write access to the repository
   - Check that your GitHub account is properly authenticated

For any other issues, please refer to the Git documentation or GitHub help.