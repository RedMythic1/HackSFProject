// Script to fix issues with duplicated prefixes in blob keys
const { put, list, get, del } = require('@vercel/blob');

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
          const originalBlob = await get(blob.pathname);
          const content = await originalBlob.text();
          
          // Create a new normalized key
          const normalizedKey = blob.pathname.replace('final_article_final_article_', 'final_article_');
          console.log(`Normalized key: ${normalizedKey}`);
          
          // Upload to the new key
          console.log(`Uploading content to normalized key: ${normalizedKey}`);
          const { url } = await put(normalizedKey, content, { 
            access: 'public',
            contentType: 'application/json'
          });
          
          // Delete the original blob with duplicated prefix
          console.log(`Deleting original blob: ${blob.pathname}`);
          await del(blob.pathname);
          
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
    
    if (issues.duplicated.length === 0) {
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