// Script to test Vercel Blob Storage access
const { put, list, head, del } = require('@vercel/blob');

// Check if token is set
if (!process.env.BLOB_READ_WRITE_TOKEN) {
  console.error('Error: BLOB_READ_WRITE_TOKEN environment variable not set');
  console.error('Please run: source setup-env.sh');
  process.exit(1);
}

console.log(`Vercel Blob token: ${process.env.BLOB_READ_WRITE_TOKEN.substring(0, 10)}...`);
console.log(`Vercel Blob URL: ${process.env.BLOB_URL || 'Not set'}`);

// Test function for Blob Storage
async function testBlobStorage() {
  console.log('Testing Vercel Blob Storage...');
  
  try {
    // 1. Create a test blob
    const testContent = JSON.stringify({ test: true, timestamp: Date.now() });
    const testBlobKey = 'test/blob-test-' + Date.now() + '.json';
    
    console.log(`Uploading test blob to ${testBlobKey}...`);
    const { url } = await put(testBlobKey, testContent, { 
      access: 'public',
      contentType: 'application/json'
    });
    
    console.log(`Test blob uploaded successfully: ${url}`);
    
    // 2. List blobs to verify
    console.log('Listing blobs with test/ prefix...');
    const { blobs } = await list({ prefix: 'test/' });
    console.log(`Found ${blobs.length} test blobs`);
    
    // 3. Read back the blob
    console.log('Reading test blob metadata...');
    const metadata = await head(testBlobKey);
    console.log('Blob metadata:', metadata);
    
    // 4. Fetch blob content
    console.log('Fetching blob content...');
    const response = await fetch(metadata.url);
    if (!response.ok) {
      throw new Error(`Failed to fetch blob content: ${response.status}`);
    }
    const content = await response.text();
    console.log('Blob content:', content);
    
    // 5. Delete the test blob to clean up
    console.log('Deleting test blob...');
    await del(testBlobKey);
    console.log('Test blob deleted successfully');
    
    console.log('\n✅ All Vercel Blob tests PASSED');
    return true;
  } catch (error) {
    console.error(`\n❌ Vercel Blob test FAILED: ${error.message}`);
    console.error(error);
    return false;
  }
}

// Run the test
testBlobStorage().then(success => {
  process.exit(success ? 0 : 1);
}); 