#!/bin/bash

# Copy contents
mkdir gh-pages
# cp -r ./docs/build/html/. gh-pages
# touch .nojekyll
# Create gh-pages branch
cd ./docs/build/html/. || exit

git init
git config --local user.email "samy.khelifi@ign.fr"
git config --local user.name "samysung action"
git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git checkout -b gh-pages

# Deploy
git add .
git commit -m "Publish docs" || true
git push origin gh-pages --force
