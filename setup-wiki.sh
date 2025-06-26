#!/bin/bash

# UniversalCompute Wiki Setup Script
# This script helps set up the GitHub wiki with comprehensive documentation

echo "🚀 UniversalCompute Wiki Setup"
echo "==============================="
echo

# Check if gh CLI is available
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed. Please install it first."
    echo "   Visit: https://cli.github.com/"
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "wiki-docs" ]; then
    echo "❌ wiki-docs directory not found. Please run this script from the project root."
    exit 1
fi

echo "✅ Found wiki documentation files:"
ls -la wiki-docs/
echo

# Try to clone the wiki repository
echo "📖 Setting up GitHub Wiki..."
REPO_NAME="mivertowski/UniversalCompute"
WIKI_REPO="${REPO_NAME}.wiki"

# Remove existing wiki directory if it exists
if [ -d "wiki-repo" ]; then
    echo "🧹 Removing existing wiki directory..."
    rm -rf wiki-repo
fi

# Try to clone the wiki repository
echo "🔄 Attempting to clone wiki repository..."
if git clone "https://github.com/${WIKI_REPO}.git" wiki-repo 2>/dev/null; then
    echo "✅ Wiki repository cloned successfully"
    cd wiki-repo
else
    echo "📝 Wiki repository doesn't exist yet. Creating initial wiki..."
    
    # Create a temporary directory for the wiki
    mkdir -p wiki-repo
    cd wiki-repo
    git init
    
    # Create a simple initial page to establish the wiki
    echo "# UniversalCompute Wiki

Welcome to the UniversalCompute documentation wiki!

This wiki is being set up with comprehensive documentation. Please check back soon for complete content.

## Quick Links
- [GitHub Repository](https://github.com/mivertowski/UniversalCompute)
- [NuGet Package](https://www.nuget.org/packages/UniversalCompute/)

---
*This wiki is automatically generated and maintained.*" > Home.md
    
    git add Home.md
    git commit -m "Initial wiki setup"
    
    # Try to push to create the wiki repository
    if git remote add origin "https://github.com/${WIKI_REPO}.git" 2>/dev/null; then
        echo "🔄 Pushing initial wiki content..."
        if git push -u origin master 2>/dev/null || git push -u origin main 2>/dev/null; then
            echo "✅ Wiki repository created successfully"
        else
            echo "⚠️  Could not push to wiki repository. You may need to:"
            echo "   1. Go to https://github.com/${REPO_NAME}/wiki"
            echo "   2. Create the first wiki page manually"
            echo "   3. Then run this script again"
            echo
            echo "📋 Wiki content is ready in the wiki-docs/ directory"
            cd ..
            exit 0
        fi
    else
        echo "⚠️  Could not set up wiki repository automatically."
    fi
fi

# Copy our documentation files to the wiki repository
echo "📄 Copying documentation files..."
cp ../wiki-docs/*.md .

# Check what files we have
echo "📚 Wiki files to be uploaded:"
ls -la *.md

# Add and commit all files
git add .
git status

echo
echo "🎯 Ready to publish wiki content!"
echo "Would you like to commit and push the wiki documentation? (y/N)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    # Commit changes
    git commit -m "📚 Add comprehensive UniversalCompute documentation

- Add complete API reference documentation
- Add quick start tutorial with examples
- Add hardware accelerators guide (CPU, GPU, NPU, AMX, ANE)
- Add installation guide with troubleshooting
- Add extensive examples gallery covering:
  * Basic vector operations and matrix multiplication
  * Scientific computing (FFT, Monte Carlo)
  * Machine learning (neural networks)
  * Image processing and computer vision
  * Financial modeling and risk calculation
  * Multi-GPU workload distribution
  * Performance testing and benchmarking

📊 Total documentation: 5 comprehensive wiki pages
🎯 Target audience: Developers, researchers, data scientists
⚡ Covers all UniversalCompute v1.0.0-alpha1 features

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

    # Push to GitHub
    echo "🚀 Publishing wiki to GitHub..."
    if git push origin HEAD 2>/dev/null; then
        echo "✅ Wiki documentation published successfully!"
        echo
        echo "🌐 Your wiki is now available at:"
        echo "   https://github.com/${REPO_NAME}/wiki"
        echo
        echo "📖 Available Pages:"
        echo "   • Home - Welcome and overview"
        echo "   • Installation Guide - Complete setup instructions"  
        echo "   • Quick Start Tutorial - 15-minute getting started guide"
        echo "   • API Reference - Complete API documentation"
        echo "   • Hardware Accelerators - Specialized hardware support"
        echo "   • Examples Gallery - Real-world code examples"
        echo
        echo "🎉 Wiki setup completed successfully!"
    else
        echo "❌ Failed to push wiki content."
        echo "   You may need to manually push the changes or check repository permissions."
    fi
else
    echo "📋 Wiki content is ready but not published."
    echo "   To publish manually:"
    echo "   1. cd wiki-repo"
    echo "   2. git add ."
    echo "   3. git commit -m 'Add comprehensive documentation'"
    echo "   4. git push origin HEAD"
fi

cd ..

echo
echo "📁 Wiki documentation files are available in:"
echo "   • wiki-docs/ - Original documentation files"
echo "   • wiki-repo/ - Git repository for the wiki"
echo
echo "✨ Thank you for using UniversalCompute!"