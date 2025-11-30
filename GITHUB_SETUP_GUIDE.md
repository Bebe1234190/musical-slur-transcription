# GitHub Repository Setup Guide

This guide will help you set up your GitHub repository for the first time.

## Step 1: Create a GitHub Account and Repository

1. **Create a GitHub account** (if you don't have one):
   - Go to https://github.com
   - Sign up for a free account

2. **Create a new repository**:
   - Click the "+" icon in the top right corner
   - Select "New repository"
   - Choose a repository name (e.g., `musical-slur-transcription` or `piano-slur-ai`)
   - Add a description: "Transformer-based machine learning system for predicting musical slur annotations from MIDI files"
   - Choose **Public** (so others can see your work) or **Private** (if you want to keep it private)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

## Step 2: Install Git (if not already installed)

**On macOS:**
```bash
# Check if git is installed
git --version

# If not installed, install via Homebrew:
brew install git

# Or download from: https://git-scm.com/download/mac
```

**On Windows:**
- Download from: https://git-scm.com/download/win

**On Linux:**
```bash
sudo apt-get install git  # Ubuntu/Debian
# or
sudo yum install git      # CentOS/RHEL
```

## Step 3: Configure Git (First Time Only)

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email (use the email associated with your GitHub account)
git config --global user.email "your.email@example.com"
```

## Step 4: Initialize Git in Your Project

Open Terminal/Command Prompt and navigate to your project directory:

```bash
cd "/Users/esadmelikzade/Desktop/Slur Transcription Project"
```

Initialize git:
```bash
git init
```

## Step 5: Add Files to Git

Add all the files you want to include:

```bash
# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status
```

## Step 6: Make Your First Commit

```bash
git commit -m "Initial commit: Musical slur transcription system with transformer model"
```

## Step 7: Connect to GitHub

1. **Get your repository URL** from GitHub (it will look like):
   - `https://github.com/yourusername/repository-name.git`
   - Or: `git@github.com:yourusername/repository-name.git`

2. **Add the remote repository**:
```bash
git remote add origin https://github.com/yourusername/repository-name.git
```

3. **Push your code to GitHub**:
```bash
# Rename default branch to 'main' (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

You may be prompted for your GitHub username and password. For password, you'll need to use a **Personal Access Token** (not your regular password):

### Creating a Personal Access Token:

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name (e.g., "My Computer")
4. Select scopes: check `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again!)
7. Use this token as your password when pushing

## Step 8: Verify Upload

Go to your GitHub repository page and verify all files are there!

## Files Included in Repository

### ✅ **Included:**
- All Python source code (`src/` directory)
- Main entry point (`main.py`)
- Requirements files (`requirements.txt`, `requirements_minimal.txt`)
- README.md
- Documentation (`docs/` directory)
- Dataset:
  - 4 MIDI files (from `output/` and `Slur Training Dataset/`)
  - 4 completed annotation CSV files
  - Metadata and pedal CSV files
- `.gitignore` file

### ❌ **Excluded (via .gitignore):**
- Model checkpoint files (`*.pt`)
- Large numpy arrays (`*.npy`)
- Log files (`*.log`)
- Python cache (`__pycache__/`)
- Temporary files

## Future Updates

When you make changes and want to update GitHub:

```bash
# Check what changed
git status

# Add changed files
git add .

# Commit with a message
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## Repository Structure

Your repository will have this structure:

```
musical-slur-transcription/
├── .gitignore
├── README.md
├── main.py
├── requirements.txt
├── requirements_minimal.txt
├── src/
│   ├── complete_midi_processor.py
│   ├── main_ml.py
│   ├── ml_chunked_pipeline.py
│   ├── ml_chunked_train.py
│   ├── ml_data_pipeline.py
│   ├── ml_train.py
│   ├── ml_transformer_model.py
│   ├── run_multi_trial_training.py
│   ├── run_training_experiments.py
│   ├── slur_annotation_tool.py
│   └── [other source files]
├── docs/
│   ├── MODEL_ARCHITECTURE_SECTION.md
│   ├── PROJECT_COMPREHENSIVE_DOCUMENTATION_DECEMBER_2025.md
│   └── [other documentation]
├── output/
│   ├── *.mid (MIDI files)
│   ├── *_slur_annotation_completed.csv (annotations)
│   ├── *_metadata.txt
│   └── *_pedal.csv
└── Slur Training Dataset/
    └── *.mid (additional MIDI files)
```

## Troubleshooting

**If you get "fatal: not a git repository":**
- Make sure you're in the project directory
- Run `git init` first

**If push is rejected:**
- Make sure you've committed your changes first (`git commit`)
- Check that you've added the remote correctly (`git remote -v`)

**If you need to update .gitignore later:**
- Edit `.gitignore`
- Run `git rm -r --cached .` (removes all files from git cache)
- Run `git add .` (re-adds files respecting new .gitignore)
- Run `git commit -m "Update .gitignore"`

## Next Steps

1. Add a license file (optional but recommended):
   - Go to your repository on GitHub
   - Click "Add file" → "Create new file"
   - Name it `LICENSE`
   - GitHub will suggest common licenses (MIT, Apache, etc.)

2. Add topics/tags to your repository:
   - Go to repository settings
   - Add topics like: `machine-learning`, `music`, `transformer`, `midi`, `pytorch`

3. Consider adding:
   - A `CONTRIBUTING.md` file (if you want others to contribute)
   - Issue templates
   - A project description on the repository page

