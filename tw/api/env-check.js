// Dedicated endpoint for checking environment settings without authentication
const express = require('express');
const cors = require('cors');
const app = express();

// Enable CORS
app.use(cors());

// Public endpoint to check environment settings
app.get('/api/env-check', (req, res) => {
  console.log('[env-check.js] GET /api/env-check');
  
  // Check if Blob token is set, but don't reveal it
  const hasBlobToken = !!process.env.BLOB_READ_WRITE_TOKEN;
  const hasBlobUrl = !!process.env.BLOB_URL;
  
  res.json({
    status: 'success',
    blob: {
      token_configured: hasBlobToken,
      url_configured: hasBlobUrl,
      token_length: hasBlobToken ? process.env.BLOB_READ_WRITE_TOKEN.length : 0,
      vercel_env: process.env.VERCEL || 'not set'
    },
    timestamp: Date.now()
  });
});

// Basic health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: Date.now() });
});

// Export the server
module.exports = app;

// Start the server if not being used as a module
if (require.main === module) {
  const port = process.env.PORT || 3030;
  app.listen(port, () => {
    console.log(`Environment check server listening on port ${port}`);
  });
} 