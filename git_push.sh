#!/bin/bash
set -e

echo "======================================"
echo " GitHub Push Script (Codespaces Safe) "
echo "======================================"

BRANCH=$(git branch --show-current)
echo "Current branch: $BRANCH"

echo ""
echo "Git status:"
git status

echo ""
echo "Adding all files..."
git add .

if git diff --cached --quiet; then
    echo "No changes to commit."
else
    git commit -m "Add HDBSCAN declustering, XGB cross-verification, plots & analysis"
fi

echo ""
echo "Pulling latest changes from origin/$BRANCH..."
git pull origin "$BRANCH" --rebase

echo ""
echo "Pushing to origin/$BRANCH..."
git push origin "$BRANCH"

echo ""
echo "✅ Push completed successfully."
