#!/bin/bash

# Setup environment variables for local development with Vercel Blob Storage

# Export environment variables
export BLOB_READ_WRITE_TOKEN="vercel_blob_rw_MzCMzRmJaiRlp3km_L5RVXS9InB9rTT1Aov2ZI4kzQFoT5S"
export BLOB_URL="https://mzcmzrmjairlp3km.public.blob.vercel-storage.com"

echo "Environment variables set:"
echo "BLOB_READ_WRITE_TOKEN: $BLOB_READ_WRITE_TOKEN"
echo "BLOB_URL: $BLOB_URL"

# Create/update a .env file (for applications that support it)
cat > .env << EOL
BLOB_READ_WRITE_TOKEN=vercel_blob_rw_MzCMzRmJaiRlp3km_L5RVXS9InB9rTT1Aov2ZI4kzQFoT5S
BLOB_URL=https://mzcmzrmjairlp3km.public.blob.vercel-storage.com
EOL

echo -e "\nCreated/updated .env file"

# Run a quick test to verify Blob Storage access
if [ -f "tools/test-blob.js" ]; then
  echo -e "\nTesting Vercel Blob access:"
  node tools/test-blob.js
else
  echo -e "\nSkipping Blob Storage test (test-blob.js not found)"
fi

echo -e "\nRun the following command to use these environment variables in your shell:"
echo "source setup-env.sh" 