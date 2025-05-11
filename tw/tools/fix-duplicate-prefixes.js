// Script to fix issues with duplicated prefixes in blob keys
const { put, list, head } = require('@vercel/blob');
const path = require('path');

// Blob storage prefixes
const BLOB_PREFIX = 'articles/';
const BLOB_ARTICLE_PREFIX = 'articles/final_article_';

// Check if token is set
if (!process.env.BLOB_READ_WRITE_TOKEN) {
  console.error('Error: BLOB_READ_WRITE_TOKEN environment variable not set');
  console.error('Please run: source setup-env.sh');
  process.exit(1);
}

// Function to check and fix duplicated prefixes in blob keys
async function fixDuplicatePrefixes() {
  console.log('Checking for blob keys with duplicated prefixes...');
  
  try {
    // List all article blobs
    console.log('Listing article blobs...');
    const { blobs } = await list({ prefix: BLOB_ARTICLE_PREFIX });
    console.log(`Found ${blobs.length} article blobs`);
    
    // Keep track of issues
    const issues = {
      duplicated: [],
      fixed: []
    };
    
    // Check each blob for duplicated prefixes
    for (const blob of blobs) {
      // Check if the pathname has a duplicated 'final_article_' prefix
      if (blob.pathname.includes('final_article_final_article_')) {
        console.log(`Found blob with duplicated prefix: ${blob.pathname}`);
        issues.duplicated.push(blob.pathname);
        
        try {
          // Get the content
          const response = await fetch(blob.url);
          if (!response.ok) {
            throw new Error(`Failed to fetch blob content: ${response.status}`);
          }
          let content = await response.text();
          
          // Check if the content is JSON and parse it to fix any duplicate prefixes within
          try {
            const contentObj = JSON.parse(content);
            // Look for duplicated prefixes in fields like "filename" or "filepath" that might exist
            if (contentObj.filepath && contentObj.filepath.includes('final_article_final_article_')) {
              contentObj.filepath = contentObj.filepath.replace('final_article_final_article_', 'final_article_');
              console.log(`Fixed duplicated prefix in content filepath: ${contentObj.filepath}`);
            }
            if (contentObj.filename && contentObj.filename.includes('final_article_final_article_')) {
              contentObj.filename = contentObj.filename.replace('final_article_final_article_', 'final_article_');
              console.log(`Fixed duplicated prefix in content filename: ${contentObj.filename}`);
            }
            // Stringify content back if we made changes
            content = JSON.stringify(contentObj, null, 2);
          } catch (jsonError) {
            // Not JSON or other parsing error, continue with original content
            console.log(`Content is not JSON or could not be parsed: ${jsonError.message}`);
          }
          
          // Create a new normalized key
          const normalizedKey = blob.pathname.replace('final_article_final_article_', 'final_article_');
          console.log(`Normalized key: ${normalizedKey}`);
          
          // Upload to the new key
          console.log(`Uploading content to normalized key: ${normalizedKey}`);
          const { url } = await put(normalizedKey, content, { 
            access: 'public',
            contentType: 'application/json'
          });
          
          // Delete the original blob using fetch API + DELETE method
          console.log(`Deleting original blob: ${blob.pathname}`);
          const deleteResponse = await fetch(`https://blob.vercel-storage.com/${blob.pathname}`, {
            method: 'DELETE',
            headers: {
              'Authorization': `Bearer ${process.env.BLOB_READ_WRITE_TOKEN}`
            }
          });
          
          if (!deleteResponse.ok) {
            console.warn(`Could not delete original blob: ${deleteResponse.status}`);
          }
          
          issues.fixed.push({
            original: blob.pathname,
            fixed: normalizedKey,
            url
          });
          
          console.log(`Successfully fixed: ${blob.pathname} → ${normalizedKey}`);
        } catch (error) {
          console.error(`Error fixing blob ${blob.pathname}: ${error.message}`);
        }
      }
    }
    
    // Special case: Check for specific article IDs known to have issues
    const problematicIds = ['1746915102_Xenolab', '1746915309_US_vs_Google_amicus_curiae_brief_of_Y_Combinator_in_support_of_plaintiffs'];
    
    for (const problemId of problematicIds) {
      // Look for both regular and duplicated keys in the blob list
      console.log(`Searching for problematic article ID: ${problemId}`);
      
      // Search in the blobs list
      const regularBlob = blobs.find(blob => blob.pathname === `articles/final_article_${problemId}.json`);
      const duplicatedBlob = blobs.find(blob => blob.pathname === `articles/final_article_final_article_${problemId}.json`);
      
      if (regularBlob) {
        console.log(`Found article with regular key: ${regularBlob.pathname}`);
        
        try {
          // Create a backup copy
          const response = await fetch(regularBlob.url);
          if (!response.ok) {
            throw new Error(`Failed to fetch blob content: ${response.status}`);
          }
          
          let content = await response.text();
          console.log(`Successfully read content from ${regularBlob.pathname}`);
          
          // Re-upload to ensure it's valid
          const { url } = await put(regularBlob.pathname, content, { 
            access: 'public',
            contentType: 'application/json',
            allowOverwrite: true
          });
          
          console.log(`Successfully backed up article to ${url}`);
          issues.fixed.push({
            original: regularBlob.pathname,
            fixed: regularBlob.pathname,
            url
          });
        } catch (error) {
          console.error(`Error backing up article ${regularBlob.pathname}: ${error.message}`);
        }
      }
      
      if (duplicatedBlob) {
        console.log(`Found article with duplicated key: ${duplicatedBlob.pathname}`);
        
        try {
          // Get content
          const response = await fetch(duplicatedBlob.url);
          if (!response.ok) {
            throw new Error(`Failed to fetch blob content: ${response.status}`);
          }
          
          let content = await response.text();
          
          // Create a normalized key
          const normalizedKey = `articles/final_article_${problemId}.json`;
          console.log(`Creating normalized key: ${normalizedKey}`);
          
          // Upload to the normalized key
          const { url } = await put(normalizedKey, content, { 
            access: 'public',
            contentType: 'application/json',
            allowOverwrite: true
          });
          
          // Delete the duplicated blob
          console.log(`Trying to delete duplicated blob: ${duplicatedBlob.pathname}`);
          const deleteResponse = await fetch(`https://blob.vercel-storage.com/${duplicatedBlob.pathname}`, {
            method: 'DELETE',
            headers: {
              'Authorization': `Bearer ${process.env.BLOB_READ_WRITE_TOKEN}`
            }
          });
          
          if (!deleteResponse.ok) {
            console.warn(`Could not delete duplicated blob: ${deleteResponse.status}`);
          }
          
          issues.fixed.push({
            original: duplicatedBlob.pathname,
            fixed: normalizedKey,
            url
          });
          
          console.log(`Successfully fixed: ${duplicatedBlob.pathname} → ${normalizedKey}`);
        } catch (error) {
          console.error(`Error fixing duplicated article ${duplicatedBlob.pathname}: ${error.message}`);
        }
      }
      
      if (!regularBlob && !duplicatedBlob) {
        console.log(`No blob found for article ID: ${problemId}`);
      }
    }
    
    // Print summary
    console.log('\nIssue resolution summary:');
    console.log(`Found ${issues.duplicated.length} blobs with duplicated prefixes`);
    console.log(`Fixed ${issues.fixed.length} blobs`);
    
    if (issues.duplicated.length > 0) {
      console.log('\nDuplicated prefixes found:');
      issues.duplicated.forEach(path => {
        console.log(`- ${path}`);
      });
    }
    
    if (issues.fixed.length > 0) {
      console.log('\nFixed blobs:');
      issues.fixed.forEach(fix => {
        console.log(`- ${fix.original} → ${fix.fixed}`);
      });
    }
    
    if (issues.duplicated.length === 0 && issues.fixed.length === 0) {
      console.log('\nNo duplicated prefixes found!');
    }
    
    console.log('\nFix completed!');
    return true;
  } catch (error) {
    console.error('Error checking for duplicated prefixes:', error);
    return false;
  }
}

// Run the fix function
fixDuplicatePrefixes().then(success => {
  process.exit(success ? 0 : 1);
}); 