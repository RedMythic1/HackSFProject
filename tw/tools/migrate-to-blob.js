// Script to migrate existing local cache files to Vercel Blob Storage
const fs = require('fs');
const path = require('path');
const { put, list } = require('@vercel/blob');
const { promisify } = require('util');
const { glob } = require('glob');
const readFile = promisify(fs.readFile);

// Blob storage prefixes
const BLOB_PREFIX = 'articles/';
const BLOB_ARTICLE_PREFIX = 'articles/final_article_';
const BLOB_SUMMARY_PREFIX = 'articles/summary_';
const BLOB_SEARCH_PREFIX = 'articles/search_';

// Cache directories
const CACHE_DIR = path.join(__dirname, '..', '.cache');
const LOCAL_CACHE_DIR = path.join(__dirname, '..', 'local_cache');

// Check if token is set
if (!process.env.BLOB_READ_WRITE_TOKEN) {
  console.error('Error: BLOB_READ_WRITE_TOKEN environment variable not set');
  console.error('Please run: source setup-env.sh');
  process.exit(1);
}

// Helper function to convert file path to blob key
function getBlobKeyFromPath(filePath) {
  const basename = path.basename(filePath);
  
  if (basename.startsWith('final_article_')) {
    return BLOB_ARTICLE_PREFIX + basename.replace('final_article_', '');
  } else if (basename.startsWith('summary_')) {
    return BLOB_SUMMARY_PREFIX + basename.replace('summary_', '');
  } else if (basename.startsWith('search_')) {
    return BLOB_SEARCH_PREFIX + basename.replace('search_', '');
  } else {
    return BLOB_PREFIX + basename;
  }
}

// Function to upload a file to blob storage
async function uploadFileToBlob(filePath) {
  try {
    const blobKey = getBlobKeyFromPath(filePath);
    const content = await readFile(filePath, 'utf-8');
    
    // Try to parse as JSON to validate
    try {
      JSON.parse(content);
      console.log(`Uploading ${filePath} to ${blobKey}...`);
      
      const { url } = await put(blobKey, content, { 
        access: 'public',
        contentType: 'application/json'
      });
      
      console.log(`Successfully uploaded ${filePath} to ${url}`);
      return { success: true, filePath, blobKey, url };
    } catch (jsonError) {
      console.warn(`Skipping ${filePath} - not valid JSON`);
      return { success: false, filePath, error: 'Not valid JSON' };
    }
  } catch (error) {
    console.error(`Error uploading ${filePath}: ${error.message}`);
    return { success: false, filePath, error: error.message };
  }
}

// Main function to migrate files
async function migrateToBlob() {
  console.log('Starting migration of local cache files to Vercel Blob Storage...');
  
  // Check if dirs exist
  const cacheExists = fs.existsSync(CACHE_DIR);
  const localCacheExists = fs.existsSync(LOCAL_CACHE_DIR);
  
  if (!cacheExists && !localCacheExists) {
    console.warn('No cache directories found. Nothing to migrate.');
    return;
  }
  
  let files = [];
  
  // Find JSON files in main cache
  if (cacheExists) {
    const cacheFiles = await glob('**/*.json', { cwd: CACHE_DIR });
    files = [...files, ...cacheFiles.map(f => path.join(CACHE_DIR, f))];
    console.log(`Found ${cacheFiles.length} files in main cache`);
  }
  
  // Find JSON files in local cache
  if (localCacheExists) {
    const localCacheFiles = await glob('**/*.json', { cwd: LOCAL_CACHE_DIR });
    files = [...files, ...localCacheFiles.map(f => path.join(LOCAL_CACHE_DIR, f))];
    console.log(`Found ${localCacheFiles.length} files in local cache`);
  }
  
  if (files.length === 0) {
    console.log('No JSON files found to migrate');
    return;
  }
  
  console.log(`Found a total of ${files.length} files to migrate`);
  
  // Get existing blob list to avoid duplicates
  const { blobs } = await list({ prefix: BLOB_PREFIX });
  const existingBlobPaths = new Set(blobs.map(b => b.pathname));
  console.log(`Found ${blobs.length} existing blobs in storage`);
  
  // Process files in batches to avoid overwhelming the API
  const batchSize = 10;
  const results = { success: 0, skipped: 0, failed: 0 };
  
  for (let i = 0; i < files.length; i += batchSize) {
    const batch = files.slice(i, i + batchSize);
    console.log(`Processing batch ${i/batchSize + 1} of ${Math.ceil(files.length/batchSize)}...`);
    
    const batchPromises = batch.map(file => {
      const blobKey = getBlobKeyFromPath(file);
      
      // Skip if already exists
      if (existingBlobPaths.has(blobKey)) {
        console.log(`Skipping ${file} - already exists in blob storage`);
        results.skipped++;
        return Promise.resolve({ success: false, skipped: true });
      }
      
      return uploadFileToBlob(file);
    });
    
    const batchResults = await Promise.all(batchPromises);
    
    // Count results
    for (const result of batchResults) {
      if (result.skipped) continue;
      if (result.success) results.success++;
      else results.failed++;
    }
    
    // Small delay to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  
  console.log('\nMigration completed!');
  console.log(`Successfully uploaded: ${results.success} files`);
  console.log(`Skipped (already exist): ${results.skipped} files`);
  console.log(`Failed: ${results.failed} files`);
}

// Run the migration
migrateToBlob().catch(err => {
  console.error('Migration failed:', err);
  process.exit(1);
}); 