// Handle server API requests for serverless deployment
const express = require('express');
const path = require('path');
const fs = require('fs');
const { put, list, get, del } = require('@vercel/blob');

// Create Express router
const router = express.Router();

// More robust environment detection
const isVercel = process.env.VERCEL === '1' || process.env.VERCEL === 'true' || process.cwd().includes('/var/task');
console.log(`Current working directory: ${process.cwd()}`);
console.log(`VERCEL env var: ${process.env.VERCEL}`);
console.log(`Running in ${isVercel ? 'Vercel' : 'local'} environment`);

// Define blob storage prefixes
const BLOB_PREFIX = 'articles/';
const BLOB_ARTICLE_PREFIX = 'articles/final_article_';
const BLOB_SUMMARY_PREFIX = 'articles/summary_';
const BLOB_SEARCH_PREFIX = 'articles/search_';

// Log the actual paths being used
console.log(`Server handler initialized with:
  Using Vercel Blob Storage: Yes
  Current working directory: ${process.cwd()}`);

// Define API endpoints
router.get('/articles', async (req, res) => {
  try {
    // Import server module only when needed
    const { get_homepage_articles_endpoint } = require('./server');
    const result = await get_homepage_articles_endpoint();
    res.json(result);
  } catch (error) {
    console.error('Error fetching articles:', error);
    res.status(500).json({ 
      status: 'error', 
      message: `Server error: ${error.message}` 
    });
  }
});

router.get('/article/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    // Import server module only when needed
    const { get_article_endpoint } = require('./server');
    const result = await get_article_endpoint(id);
    
    if (result.status === 'error') {
      return res.status(404).json(result);
    }
    
    res.json(result);
  } catch (error) {
    console.error('Error fetching article:', error);
    res.status(500).json({ 
      status: 'error', 
      message: `Server error: ${error.message}` 
    });
  }
});

router.post('/process-articles', async (req, res) => {
  try {
    const { interests } = req.body;
    
    // Import server module only when needed
    const { process_articles_endpoint } = require('./server');
    const result = await process_articles_endpoint({ interests });
    
    res.json(result);
  } catch (error) {
    console.error('Error processing articles:', error);
    res.status(500).json({ 
      status: 'error', 
      message: `Server error: ${error.message}` 
    });
  }
});

router.post('/analyze-interests', async (req, res) => {
  try {
    const { interests } = req.body;
    
    // Import server module only when needed
    const { analyze_interests_endpoint } = require('./server');
    const result = await analyze_interests_endpoint(interests);
    
    res.json(result);
  } catch (error) {
    console.error('Error analyzing interests:', error);
    res.status(500).json({ 
      status: 'error', 
      message: `Server error: ${error.message}` 
    });
  }
});

// API route to get a cached file
router.get('/get-cached-file', async (req, res) => {
  try {
    const { file } = req.query;
    
    if (!file) {
      return res.status(400).json({
        status: 'error',
        message: 'File parameter is required'
      });
    }
    
    // Determine the blob key
    let blobKey;
    if (file.startsWith('final_article_')) {
      blobKey = `${BLOB_ARTICLE_PREFIX}${file.replace('final_article_', '')}`;
    } else if (file.startsWith('summary_')) {
      blobKey = `${BLOB_SUMMARY_PREFIX}${file.replace('summary_', '')}`;
    } else if (file.startsWith('search_')) {
      blobKey = `${BLOB_SEARCH_PREFIX}${file.replace('search_', '')}`;
    } else {
      blobKey = `${BLOB_PREFIX}${file}`;
    }
    
    console.log(`Looking for file in blob storage: ${blobKey}`);
    
    try {
      const blob = await get(blobKey);
      if (blob) {
        const content = await blob.text();
        return res.json({
          status: 'success',
          data: JSON.parse(content),
          source: 'blob'
        });
      }
    } catch (error) {
      console.warn(`Error reading from blob storage: ${error.message}`);
    }
    
    return res.status(404).json({
      status: 'error',
      message: 'File not found'
    });
  } catch (error) {
    console.error('Error getting cached file:', error);
    res.status(500).json({ 
      status: 'error', 
      message: `Server error: ${error.message}` 
    });
  }
});

// API route to sync cache
router.post('/sync-cache', async (req, res) => {
  try {
    // Just return success since all operations are now using Vercel Blob Storage
    res.json({
      status: 'success',
      message: 'All operations now use Vercel Blob Storage',
      stats: {
        added: 0,
        updated: 0,
        skipped: 0,
        errors: 0,
        totalLocal: 0
      }
    });
  } catch (error) {
    console.error('Error syncing cache:', error);
    res.status(500).json({ 
      status: 'error', 
      message: `Server error: ${error.message}` 
    });
  }
});

// Determine which Python command to use based on available versions
function getPythonCommand() {
  try {
    // Try different Python commands in order of preference
    const commands = ['python3.9', 'python3.8', 'python3', 'python'];
    const { execSync } = require('child_process');
    
    for (const cmd of commands) {
      try {
        execSync(`${cmd} --version`, { stdio: 'ignore' });
        console.log(`Found Python command: ${cmd}`);
        return cmd;
      } catch (e) {
        // Command not found, try next one
        console.log(`Command ${cmd} not available, trying next option...`);
      }
    }
    
    // If we get here, none of the commands worked
    console.warn('No Python command found, defaulting to python3');
    return 'python3';
  } catch (error) {
    console.warn('Error determining Python command:', error.message);
    return 'python3'; // Default to python3
  }
}

// Export the Python command for use in server.js
module.exports = {
  pythonCommand: getPythonCommand()
};

module.exports = router;