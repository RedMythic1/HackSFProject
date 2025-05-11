#!/usr/bin/env node

// Script to migrate local articles to Vercel Blob Storage
const fs = require('fs');
const path = require('path');
const { put, list, del } = require('@vercel/blob');
const { promisify } = require('util');
const readFile = promisify(fs.readFile);
const readDir = promisify(fs.readdir);
const stat = promisify(fs.stat);

// Define storage paths
const CACHE_DIR = path.resolve(process.cwd(), '.cache');
const DIST_DIR = path.resolve(process.cwd(), 'dist');
const ARTICLES_DIR = path.resolve(process.cwd(), 'articles');

// Define blob prefixes
const BLOB_PREFIX = 'articles/';
const BLOB_ARTICLE_PREFIX = 'articles/final_article_';
const BLOB_SUMMARY_PREFIX = 'articles/summary_';
const BLOB_SEARCH_PREFIX = 'articles/search_';

// Check for token
if (!process.env.BLOB_READ_WRITE_TOKEN) {
  console.error('Error: BLOB_READ_WRITE_TOKEN environment variable is not set');
  console.error('Please set the token with: export BLOB_READ_WRITE_TOKEN="your_token_here"');
  console.error('Or run: source setup-env.sh');
  process.exit(1);
}

// Helper function to get blob key for a file
function getBlobKey(filePath) {
  const filename = path.basename(filePath);
  
  if (filename.startsWith('final_article_')) {
    return `${BLOB_ARTICLE_PREFIX}${filename.replace('final_article_', '')}`;
  } else if (filename.startsWith('summary_')) {
    return `${BLOB_SUMMARY_PREFIX}${filename.replace('summary_', '')}`;
  } else if (filename.startsWith('search_')) {
    return `${BLOB_SEARCH_PREFIX}${filename.replace('search_', '')}`;
  } else {
    return `${BLOB_PREFIX}${filename}`;
  }
}

// Helper function to upload a file to blob storage
async function uploadToBlob(filePath) {
  try {
    const blobKey = getBlobKey(filePath);
    
    // Read the file
    const content = await readFile(filePath, 'utf-8');
    
    // Try to parse as JSON to ensure it's valid
    try {
      JSON.parse(content);
    } catch (e) {
      console.warn(`Skipping ${filePath} - not valid JSON`);
      return { success: false, reason: 'invalid-json' };
    }
    
    // Upload to blob storage
    console.log(`Uploading ${filePath} to ${blobKey}...`);
    const { url } = await put(blobKey, content, { 
      access: 'public',
      contentType: 'application/json'
    });
    
    console.log(`Successfully uploaded to ${url}`);
    return { success: true, url };
  } catch (err) {
    console.error(`Error uploading ${filePath}:`, err);
    return { success: false, reason: err.message };
  }
}

// Helper function to find JSON files recursively
async function findJsonFiles(dir) {
  const files = [];
  
  try {
    // Check if the directory exists
    try {
      await stat(dir);
    } catch (e) {
      console.log(`Directory ${dir} does not exist, skipping`);
      return files;
    }
    
    // Read all files in the directory
    const entries = await readDir(dir, { withFileTypes: true });
    
    // Process each entry
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory()) {
        // Recursively search subdirectories
        const subFiles = await findJsonFiles(fullPath);
        files.push(...subFiles);
      } else if (entry.isFile() && entry.name.endsWith('.json')) {
        // Check if it's a relevant file (article, summary, or search)
        if (
          entry.name.startsWith('final_article_') || 
          entry.name.startsWith('summary_') || 
          entry.name.startsWith('search_')
        ) {
          files.push(fullPath);
        }
      }
    }
  } catch (err) {
    console.error(`Error reading directory ${dir}:`, err);
  }
  
  return files;
}

// Main function
async function main() {
  console.log('Starting migration to Vercel Blob Storage');
  
  // Get existing blobs to avoid duplicates
  console.log('Checking existing blobs...');
  const { blobs } = await list({ prefix: BLOB_PREFIX });
  const existingBlobKeys = new Set(blobs.map(b => b.pathname));
  console.log(`Found ${blobs.length} existing blobs`);
  
  // Find all JSON files
  console.log('Finding JSON files to migrate...');
  const cacheFiles = await findJsonFiles(CACHE_DIR);
  const distFiles = await findJsonFiles(DIST_DIR);
  const articleFiles = await findJsonFiles(ARTICLES_DIR);
  
  const allFiles = [...cacheFiles, ...distFiles, ...articleFiles];
  console.log(`Found ${allFiles.length} JSON files to process`);
  
  // Track statistics
  const stats = {
    total: allFiles.length,
    uploaded: 0,
    skipped: 0,
    failed: 0
  };
  
  // Process files in batches
  const BATCH_SIZE = 10;
  for (let i = 0; i < allFiles.length; i += BATCH_SIZE) {
    const batch = allFiles.slice(i, i + BATCH_SIZE);
    console.log(`Processing batch ${Math.floor(i/BATCH_SIZE) + 1} of ${Math.ceil(allFiles.length/BATCH_SIZE)}`);
    
    // Process each file in the batch concurrently
    const results = await Promise.all(batch.map(async (file) => {
      const blobKey = getBlobKey(file);
      
      // Skip if already exists
      if (existingBlobKeys.has(blobKey)) {
        console.log(`Skipping ${file} - already exists in blob storage`);
        stats.skipped++;
        return { file, status: 'skipped' };
      }
      
      // Upload to blob storage
      const result = await uploadToBlob(file);
      
      if (result.success) {
        stats.uploaded++;
        return { file, status: 'uploaded', url: result.url };
      } else {
        stats.failed++;
        return { file, status: 'failed', reason: result.reason };
      }
    }));
    
    // Log batch results
    results.forEach(result => {
      console.log(`${result.file}: ${result.status}${result.reason ? ` (${result.reason})` : ''}`);
    });
    
    // Add a small delay between batches
    if (i + BATCH_SIZE < allFiles.length) {
      console.log('Waiting 1 second before next batch...');
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  // Log final statistics
  console.log('\nMigration complete!');
  console.log(`Total files: ${stats.total}`);
  console.log(`Uploaded: ${stats.uploaded}`);
  console.log(`Skipped (already exist): ${stats.skipped}`);
  console.log(`Failed: ${stats.failed}`);
}

// Run the script
main().catch(err => {
  console.error('Migration failed:', err);
  process.exit(1);
}); 