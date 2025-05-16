// Test script to check if vercel.js can correctly import from server.js
console.log('Testing vercel.js imports from server.js...');

try {
  // First make sure the server module exports are correct
  const serverModule = require('./server');
  console.log('Server module exports:', Object.keys(serverModule));
  
  // Now simulate what vercel.js does
  console.log('\nSimulating vercel.js importing the server module...');
  
  // Test process_articles_endpoint
  if (typeof serverModule.process_articles_endpoint === 'function') {
    console.log('Calling process_articles_endpoint with empty query...');
    serverModule.process_articles_endpoint({}).then(result => {
      console.log('process_articles_endpoint completed with type:', typeof result);
      console.log('Result is array:', Array.isArray(result));
      if (Array.isArray(result)) {
        console.log('Array length:', result.length);
      }
    }).catch(error => {
      console.error('Error calling process_articles_endpoint:', error);
    });
  } else {
    console.error('ERROR: process_articles_endpoint is not a function!');
  }
  
  console.log('Test completed');
} catch (error) {
  console.error('Error in test:', error);
} 