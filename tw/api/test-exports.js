// Test script to check if server.js is properly exporting functions
console.log('Testing server module exports...');

try {
  const serverModule = require('./server');
  console.log('Successfully imported server module');
  
  console.log('Available exports:', Object.keys(serverModule));
  
  // Check for expected functions
  const functionsToCheck = [
    'process_articles_endpoint',
    'get_article_endpoint',
    'analyze_interests_endpoint'
  ];
  
  functionsToCheck.forEach(funcName => {
    const isFunction = typeof serverModule[funcName] === 'function';
    console.log(`${funcName}: ${isFunction ? 'FOUND ✓' : 'MISSING ✗'}`);
  });
  
  // Check if Express app is exported
  const isApp = serverModule.app && typeof serverModule.app.use === 'function';
  console.log(`Express app: ${isApp ? 'FOUND ✓' : 'MISSING ✗'}`);
  
  console.log('Test completed successfully');
} catch (error) {
  console.error('Error testing server module:', error);
} 