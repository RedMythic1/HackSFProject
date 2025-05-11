// Script to check Vercel environment variables by making an API request
const axios = require('axios');

// Get the site URL from command line args or use default
const site = process.argv[2] || 'https://hack-sf-project-qlx0ubf8a-redmythic1s-projects.vercel.app';

async function checkVercelEnv() {
  console.log(`Checking Vercel environment variables for: ${site}`);
  
  try {
    // First try the health check endpoint
    const healthEndpoint = `${site}/api/health`;
    console.log(`Making health check request to: ${healthEndpoint}`);
    
    const healthResponse = await axios.get(healthEndpoint);
    console.log(`Health check status: ${healthResponse.status}`);
    
    // Use the dedicated env-check endpoint
    const endpoint = `${site}/api/env-check`;
    console.log(`Making request to: ${endpoint}`);
    
    const response = await axios.get(endpoint);
    console.log('Response status:', response.status);
    console.log('Response data:', response.data);
    
    // Check blob token configuration
    if (response.data.blob && response.data.blob.token_configured) {
      console.log('✅ Blob token is configured');
      
      if (response.data.blob.token_length < 20) {
        console.warn('⚠️ Warning: Blob token length seems too short. It might be incorrect.');
      }
    } else {
      console.error('❌ Blob token is NOT configured');
    }
    
    // Check for blob URL
    if (response.data.blob && response.data.blob.url_configured) {
      console.log('✅ Blob URL is configured');
    } else {
      console.error('❌ Blob URL is NOT configured');
    }
    
    return true;
  } catch (error) {
    console.error('Error checking Vercel environment:');
    
    if (error.response) {
      console.error(`Status: ${error.response.status}`);
      console.error('Response data:', error.response.data);
    } else {
      console.error(error.message);
    }
    
    return false;
  }
}

// Run the check
checkVercelEnv().then(success => {
  process.exit(success ? 0 : 1);
}); 