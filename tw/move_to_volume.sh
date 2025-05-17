#!/bin/bash
# Script to move data to Fly.io volume

# Create needed directories on the volume
mkdir -p /data/article_cache
mkdir -p /data/datasets

# Run this on your local machine to copy files to the Fly volume:
# flyctl ssh sftp shell -a your-app-name
# mkdir -p /data/article_cache
# mkdir -p /data/datasets
# cd /data/article_cache
# put -r article_cache/* .
# cd /data/datasets
# put -r datasets/* .

# If you're running this script directly on the Fly.io instance:
# Check if article_cache exists in the current directory
if [ -d "./article_cache" ]; then
  echo "Copying article_cache to volume..."
  cp -rv ./article_cache/* /data/article_cache/
  echo "Done copying article_cache."
fi

# Check if datasets exists in the current directory
if [ -d "./datasets" ]; then
  echo "Copying datasets to volume..."
  cp -rv ./datasets/* /data/datasets/
  echo "Done copying datasets."
fi

echo "All files copied to volume."
echo "Remember that data in the /data volume persists between deployments." 