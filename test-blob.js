// Simple test script for Vercel blob API
const { put, list, get } = require('@vercel/blob');

// Log available functions
console.log('Vercel Blob functions:');
console.log('put:', typeof put);
console.log('list:', typeof list);
console.log('get:', typeof get);

async function testBlobAccess() {
  try {
    // Generate test data
    const testData = { 
      message: 'Test Data',
      timestamp: Date.now()
    };
    
    // Test write operation
    console.log('\nTesting put operation...');
    try {
      const blobResult = await put('test/data.json', JSON.stringify(testData), {
        access: 'public',
        contentType: 'application/json'
      });
      console.log('Put result:', blobResult);
    } catch (error) {
      console.error('Put error:', error.message);
    }
    
    // Test read operation
    console.log('\nTesting get operation...');
    try {
      const blob = await get('test/data.json');
      if (blob) {
        const content = await blob.text();
        console.log('Get result:', content);
      } else {
        console.log('Blob not found');
      }
    } catch (error) {
      console.error('Get error:', error.message);
    }
    
    // Test list operation
    console.log('\nTesting list operation...');
    try {
      const listResult = await list({ prefix: 'test/' });
      console.log('List result:', listResult);
    } catch (error) {
      console.error('List error:', error.message);
    }
  } catch (error) {
    console.error('Test failed:', error);
  }
}

testBlobAccess().catch(console.error); 