#!/usr/bin/env node

/**
 * Tool to upload all files from local_cache to Vercel Blob Storage
 * 
 * Usage:
 *   node upload-local-cache.js
 */

const fs = require('fs');
const path = require('path');
const { put } = require('@vercel/blob');

// Define blob storage prefixes
const BLOB_PREFIX = 'articles/';
const BLOB_ARTICLE_PREFIX = 'articles/final_article_';
const BLOB_SUMMARY_PREFIX = 'articles/summary_';
const BLOB_SEARCH_PREFIX = 'articles/search_';

// Helper function to convert file path to blob key
function getBlobKeyFromPath(filePath) {
  if (!filePath) return null;
  
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

// Helper function to write to blob storage
async function writeBlob(blobKey, data) {
  try {
    console.log(`Writing blob: ${blobKey}`);
    const { url } = await put(blobKey, JSON.stringify(data, null, 2), {
      access: 'public',
      contentType: 'application/json'
    });
    
    console.log(`Successfully wrote blob: ${url}`);
    return { success: true, url };
  } catch (error) {
    console.error(`Error writing blob ${blobKey}:`, error);
    return { success: false, error: error.message };
  }
}

// Main function to upload all files
async function uploadAllFiles() {
  try {
    // Define cache directory path
    const cacheDir = path.join(process.cwd(), 'local_cache');
    console.log(`Scanning directory: ${cacheDir}`);
    
    // Read all files in the cache directory
    const files = fs.readdirSync(cacheDir);
    console.log(`Found ${files.length} files`);
    
    // Track upload results
    const results = {
      total: files.length,
      success: 0,
      failed: 0,
      skipped: 0,
      details: []
    };
    
    // Upload each file
    for (const filename of files) {
      try {
        // Skip directories and non-JSON files
        const filePath = path.join(cacheDir, filename);
        if (fs.statSync(filePath).isDirectory() || !filename.endsWith('.json')) {
          console.log(`Skipping non-JSON file or directory: ${filename}`);
          results.skipped++;
          results.details.push({
            file: filename,
            status: 'skipped',
            reason: 'Not a JSON file or is a directory'
          });
          continue;
        }
        
        // Read file content
        const fileContent = fs.readFileSync(filePath, 'utf8');
        let data;
        
        try {
          data = JSON.parse(fileContent);
        } catch (parseError) {
          console.error(`Error parsing JSON for ${filename}:`, parseError);
          results.failed++;
          results.details.push({
            file: filename,
            status: 'failed',
            reason: 'Invalid JSON'
          });
          continue;
        }
        
        // Upload to Vercel Blob Storage
        const blobKey = getBlobKeyFromPath(filePath);
        const uploadResult = await writeBlob(blobKey, data);
        
        if (uploadResult.success) {
          results.success++;
          results.details.push({
            file: filename,
            status: 'success',
            blobKey,
            url: uploadResult.url
          });
        } else {
          results.failed++;
          results.details.push({
            file: filename,
            status: 'failed',
            blobKey,
            error: uploadResult.error
          });
        }
      } catch (fileError) {
        console.error(`Error processing file ${filename}:`, fileError);
        results.failed++;
        results.details.push({
          file: filename,
          status: 'failed',
          error: fileError.message
        });
      }
    }
    
    console.log('\nUpload Summary:');
    console.log(`Total files: ${results.total}`);
    console.log(`Successfully uploaded: ${results.success}`);
    console.log(`Failed: ${results.failed}`);
    console.log(`Skipped: ${results.skipped}`);
    
    // Write results to a log file
    fs.writeFileSync('upload-results.json', JSON.stringify(results, null, 2));
    console.log('\nDetailed results written to upload-results.json');
    
    return results;
  } catch (error) {
    console.error('Error uploading files:', error);
    return { error: error.message };
  }
}

// Run the upload process
uploadAllFiles().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
}); 