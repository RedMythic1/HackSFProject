// Script to fix issues with corrupted or problematic article files
const fs = require('fs');
const path = require('path');
const { glob } = require('glob');
const { put, list, head, del } = require('@vercel/blob');

// Blob storage prefixes
const BLOB_PREFIX = 'articles/';
const BLOB_ARTICLE_PREFIX = 'articles/final_article_';
const BLOB_SUMMARY_PREFIX = 'articles/summary_';
const BLOB_SEARCH_PREFIX = 'articles/search_';

// Check if token is set
if (!process.env.BLOB_READ_WRITE_TOKEN) {
  console.error('Error: BLOB_READ_WRITE_TOKEN environment variable not set');
  console.error('Please run: source setup-env.sh');
  process.exit(1);
}

// Function to check if the JSON is valid
function isValidJSON(str) {
  try {
    JSON.parse(str);
    return true;
  } catch (e) {
    return false;
  }
}

// Function to check and fix blob issues
async function fixBlobIssues() {
  console.log('Checking for article file issues...');
  
  try {
    // 1. List all article blobs
    console.log('Listing article blobs...');
    const { blobs } = await list({ prefix: BLOB_ARTICLE_PREFIX });
    console.log(`Found ${blobs.length} article blobs`);
    
    // Keep track of issues
    const issues = {
      invalid: [],
      fixed: [],
      deleted: []
    };
    
    // 2. Check each blob for validity
    for (const blob of blobs) {
      console.log(`Checking blob: ${blob.pathname}`);
      
      try {
        // Get the content
        const response = await fetch(blob.url);
        if (!response.ok) {
          console.error(`Failed to fetch blob content: ${response.status}`);
          issues.invalid.push({ path: blob.pathname, reason: `HTTP error: ${response.status}` });
          continue;
        }
        
        const content = await response.text();
        
        // Check if it's valid JSON
        if (!isValidJSON(content)) {
          console.error(`Invalid JSON in blob: ${blob.pathname}`);
          issues.invalid.push({ path: blob.pathname, reason: 'Invalid JSON' });
          
          // Delete the invalid blob
          console.log(`Deleting invalid blob: ${blob.pathname}`);
          await del(blob.pathname);
          issues.deleted.push(blob.pathname);
        } else {
          const parsed = JSON.parse(content);
          
          // Check if it has the required fields
          if (!parsed.content) {
            console.warn(`Blob ${blob.pathname} is missing content field`);
            issues.invalid.push({ path: blob.pathname, reason: 'Missing content field' });
            
            // If it's the problematic PLAttice file, delete it
            if (blob.pathname.includes('PLAttice')) {
              console.log(`Deleting problematic PLAttice blob: ${blob.pathname}`);
              await del(blob.pathname);
              issues.deleted.push(blob.pathname);
            }
          }
        }
      } catch (error) {
        console.error(`Error processing blob ${blob.pathname}: ${error.message}`);
        issues.invalid.push({ path: blob.pathname, reason: error.message });
      }
    }
    
    // Print summary
    console.log('\nIssue resolution summary:');
    console.log(`Found ${issues.invalid.length} invalid blobs`);
    console.log(`Fixed ${issues.fixed.length} blobs`);
    console.log(`Deleted ${issues.deleted.length} blobs`);
    
    if (issues.invalid.length > 0) {
      console.log('\nInvalid blobs:');
      issues.invalid.forEach(issue => {
        console.log(`- ${issue.path}: ${issue.reason}`);
      });
    }
    
    if (issues.deleted.length > 0) {
      console.log('\nDeleted blobs:');
      issues.deleted.forEach(path => {
        console.log(`- ${path}`);
      });
    }
    
    console.log('\nFixed completed!');
    return true;
  } catch (error) {
    console.error('Error checking blob issues:', error);
    return false;
  }
}

// Run the fix function
fixBlobIssues().then(success => {
  process.exit(success ? 0 : 1);
}); 