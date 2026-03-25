#!/bin/bash

echo "Adding files..."
git add .

echo "Committing..."
git commit -m "Update: thesis work + models + results"

echo "Pushing to GitHub..."
git push origin main --force

echo "Done ✅"