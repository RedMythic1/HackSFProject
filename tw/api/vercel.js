// Special handler for Vercel deployments
// This file is used as a fallback if other approaches fail

const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { put, list, get, del, head } = require('@vercel/blob');

// Create Express app
const app = express();

// Enable CORS and JSON parsing
app.use(cors({ origin: true, credentials: true }));
app.use(express.json());

// Environment detection
const isVercel = process.env.VERCEL === '1' || process.env.VERCEL === 'true';
console.log(`Running in ${isVercel ? 'Vercel' : 'local'} environment`);

// Define blob storage prefixes
const BLOB_PREFIX = 'articles/';
const BLOB_ARTICLE_PREFIX = 'articles/final_article_';
const BLOB_SUMMARY_PREFIX = 'articles/summary_';
const BLOB_SEARCH_PREFIX = 'articles/search_';

// Always use /tmp for Vercel - guaranteed to be writable
const CACHE_DIR = '/tmp/cache';
const LOCAL_CACHE_DIR = '/tmp/local_cache';

console.log(`[vercel.js] Using storage:
  Vercel Blob Storage enabled
  Fallback CACHE_DIR: ${CACHE_DIR}
  Fallback LOCAL_CACHE_DIR: ${LOCAL_CACHE_DIR}
  CWD: ${process.cwd()}`);

// Ensure tmp directories exist for local caching
try {
  if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
  }
  if (!fs.existsSync(LOCAL_CACHE_DIR)) {
    fs.mkdirSync(LOCAL_CACHE_DIR, { recursive: true });
  }
  console.log('[vercel.js] Temp directories created successfully');
} catch (error) {
  console.warn(`[vercel.js] Error creating directories: ${error.message}`);
}

// Demo data
const DEMO_ARTICLES = [
  {
    id: 'demo1',
    title: 'Introduction to Machine Learning',
    subject: 'A comprehensive guide to understanding the basics of Machine Learning and its applications in the modern world.',
    score: 0,
    timestamp: Date.now() - 3600000, // 1 hour ago
    link: 'https://example.com/article/1'
  },
  {
    id: 'demo2',
    title: 'The Future of Web Development',
    subject: 'Exploring emerging trends in web development including WebAssembly, Progressive Web Apps, and more.',
    score: 0,
    timestamp: Date.now() - 7200000, // 2 hours ago
    link: 'https://example.com/article/2'
  },
  {
    id: 'demo3',
    title: 'Using Vercel Blob Storage',
    subject: 'Learn how to use Vercel Blob Storage in serverless applications instead of the ephemeral filesystem.',
    score: 0,
    timestamp: Date.now() - 10800000, // 3 hours ago
    link: 'https://example.com/article/3'
  },
  {
    id: 'demo4',
    title: 'Artificial Intelligence Ethics',
    subject: 'Examining the ethical considerations in AI development and implementation in society.',
    score: 0,
    timestamp: Date.now() - 14400000, // 4 hours ago
    link: 'https://example.com/article/4'
  },
  {
    id: 'demo5',
    title: 'Cloud Computing Fundamentals',
    subject: 'An overview of cloud computing services, models, and best practices for businesses.',
    score: 0,
    timestamp: Date.now() - 18000000, // 5 hours ago
    link: 'https://example.com/article/5'
  },
  {
    id: 'demo6',
    title: 'Data Science for Beginners',
    subject: 'Getting started with data science: tools, techniques, and essential knowledge.',
    score: 0,
    timestamp: Date.now() - 21600000, // 6 hours ago
    link: 'https://example.com/article/6'
  }
];

// Fallback route handlers
app.get('/api/articles', async (req, res) => {
  try {
    console.log('[vercel.js] Handling GET /api/articles');
    
    // Import the server module with exported functions
    const serverModule = require('./server');
    console.log('Available server functions:', Object.keys(serverModule));
    
    // Call the process_articles_endpoint function
    const result = await serverModule.process_articles_endpoint(req.query);
    
    // Ensure we return the array format the frontend expects
    if (Array.isArray(result)) {
      console.log(`Returning ${result.length} articles from process_articles_endpoint`);
      return res.json(result);
    } else if (result && result.articles && Array.isArray(result.articles)) {
      console.log(`Returning ${result.articles.length} articles from result object`);
      return res.json(result.articles);
    } else {
      console.log('No articles found or invalid format, returning demo articles');
      return res.json(DEMO_ARTICLES);
    }
  } catch (error) {
    console.error('[vercel.js] Error in /api/articles:', error);
    return res.json(DEMO_ARTICLES); // Fallback to demo articles on error
  }
});

app.get('/api/article/:id', async (req, res) => {
  try {
    const { id } = req.params;
    console.log(`[vercel.js] Handling GET /api/article/${id}`);
    
    // Check for demo articles first (quick fallback)
    if (id && id.startsWith('demo')) {
      const demoArticle = DEMO_ARTICLES.find(a => a.id === id);
      if (demoArticle) {
        return res.json({
          status: 'success',
          article: {
            id: demoArticle.id,
            title: demoArticle.title,
            link: demoArticle.link,
            summary: demoArticle.subject,
            content: `# ${demoArticle.title}\n\n${demoArticle.subject}\n\nThis is a demo article provided as a fallback in Vercel deployment.`,
            timestamp: demoArticle.timestamp
          }
        });
      }
    }
    
    // Import the server module with exported functions
    const serverModule = require('./server');
    console.log('Available server functions:', Object.keys(serverModule));
    
    // Call the get_article_endpoint function
    const result = await serverModule.get_article_endpoint(id);
    
    if (result && (result.status === 'success' || result.article)) {
      return res.json(result);
    }
    
    // No article found
    return res.status(404).json({
      status: 'error',
      message: 'Article not found'
    });
  } catch (error) {
    console.error(`[vercel.js] Error in /api/article/${req.params.id}:`, error);
    return res.status(500).json({ 
      status: 'error', 
      message: error.message || 'Failed to fetch article'
    });
  }
});

app.post('/api/analyze-interests', async (req, res) => {
  try {
    const { interests } = req.body;
    console.log(`[vercel.js] Handling POST /api/analyze-interests with ${interests || 'no'} interests`);
    
    if (!interests) {
      return res.status(400).json({
        status: 'error',
        message: 'No interests provided'
      });
    }
    
    // Import the server module with exported functions
    const serverModule = require('./server');
    console.log('Available server functions:', Object.keys(serverModule));
    
    // Call the analyze_interests_endpoint function
    const result = await serverModule.analyze_interests_endpoint(interests);
    
    if (result && (result.status === 'success' || result.articles)) {
      return res.json(result);
    }
    
    // Simple fallback scoring based on interests if function fails
    const interestsList = interests.split(',').map(i => i.trim().toLowerCase());
    const scoredArticles = DEMO_ARTICLES.map(article => {
      let score = 50; // Base score
      interestsList.forEach(interest => {
        if (article.title.toLowerCase().includes(interest) || 
            article.subject.toLowerCase().includes(interest)) {
          score += 10;
        }
      });
      return { ...article, score: Math.min(score, 100) };
    });
    
    // Sort by score
    scoredArticles.sort((a, b) => b.score - a.score);
    
    return res.json(scoredArticles);
  } catch (error) {
    console.error('[vercel.js] Error in /api/analyze-interests:', error);
    return res.status(500).json({ 
      status: 'error', 
      message: error.message || 'Failed to analyze interests'
    });
  }
});

// Helper function to convert file path to blob key
function getBlobKeyFromPath(filePath) {
  if (!filePath) return null;
  
  const basename = filePath.split('/').pop();
  
  if (basename.startsWith('final_article_')) {
    return BLOB_ARTICLE_PREFIX + basename.replace('final_article_', '');
  } else if (basename.startsWith('summary_')) {
    return BLOB_SUMMARY_PREFIX + basename.replace('summary_', '');
  } else if (basename.startsWith('search_')) {
    return BLOB_SEARCH_PREFIX + basename.replace('search_', '');
  } else {
    return BLOB_PREFIX + basename;
  }
}

// Helper function to read from blob storage
async function readBlob(blobKey, defaultValue = null) {
  try {
    console.log(`Reading blob: ${blobKey}`);
    const blob = await get(blobKey);
    
    if (!blob) {
      console.log(`Blob not found: ${blobKey}`);
      return defaultValue;
    }
    
    const content = await blob.text();
    return JSON.parse(content);
  } catch (error) {
    console.error(`Error reading blob ${blobKey}:`, error);
    return defaultValue;
  }
}

// Helper function to write to blob storage
async function writeBlob(blobKey, data) {
  try {
    console.log(`Writing blob: ${blobKey}`);
    const { url } = await put(blobKey, JSON.stringify(data, null, 2), {
      access: 'public',
      contentType: 'application/json'
    });
    
    console.log(`Successfully wrote blob: ${url}`);
    return { success: true, url };
  } catch (error) {
    console.error(`Error writing blob ${blobKey}:`, error);
    return { success: false, error: error.message };
  }
}

// Helper function to list blobs with a prefix
async function listBlobs(prefix = BLOB_PREFIX) {
  try {
    console.log(`Listing blobs with prefix: ${prefix}`);
    const { blobs } = await list({ prefix });
    return { success: true, blobs };
  } catch (error) {
    console.error(`Error listing blobs with prefix ${prefix}:`, error);
    return { success: false, blobs: [], error: error.message };
  }
}

// Endpoint to list all blobs
app.get('/api/list-blobs', async (req, res) => {
  try {
    console.log('Listing all blobs in Vercel Blob Storage');
    const { blobs } = await list();
    
    // Format for easier viewing
    const formattedBlobs = blobs.map(blob => ({
      pathname: blob.pathname,
      downloadUrl: blob.url,
      size: blob.size,
      uploadedAt: blob.uploadedAt
    }));
    
    res.json({ 
      count: blobs.length,
      blobs: formattedBlobs 
    });
  } catch (error) {
    console.error('Error listing blobs:', error);
    res.status(500).json({ error: error.message });
  }
});

// Endpoint to view a specific blob
app.get('/api/blob/:path(*)', async (req, res) => {
  try {
    const blobPath = req.params.path;
    console.log(`Getting blob: ${blobPath}`);
    
    if (!blobPath) {
      return res.status(400).json({ error: 'Blob path is required' });
    }
    
    const blob = await get(blobPath);
    
    if (!blob) {
      return res.status(404).json({ error: 'Blob not found' });
    }
    
    const content = await blob.text();
    
    try {
      // Try to parse as JSON
      const jsonData = JSON.parse(content);
      res.json(jsonData);
    } catch (e) {
      // Return as text if not valid JSON
      res.send(content);
    }
  } catch (error) {
    console.error('Error getting blob:', error);
    res.status(500).json({ error: error.message });
  }
});

// Blob test route
app.get('/api/blob-test', async (req, res) => {
  console.log('[vercel.js] GET /api/blob-test');
  
  try {
    // Verify environment variables
    const blobToken = process.env.BLOB_READ_WRITE_TOKEN;
    const blobUrl = process.env.BLOB_URL;
    
    // Don't return the full token, just a masked version
    const maskedToken = blobToken ? 
      `${blobToken.substring(0, 10)}...${blobToken.substring(blobToken.length - 4)}` : 
      'not set';
    
    // Test blob access
    let blobAccessSuccessful = false;
    let blobErrorMessage = null;
    
    try {
      // Create a test blob
      const testBlobKey = `test/blob-test-${Date.now()}.json`;
      const { url } = await put(testBlobKey, JSON.stringify({ test: true, timestamp: Date.now() }), { 
        access: 'public', 
        contentType: 'application/json' 
      });
      
      // Get the blob metadata to confirm it worked
      const metadata = await head(testBlobKey);
      
      // Delete the test blob to clean up
      await del(testBlobKey);
      
      blobAccessSuccessful = true;
    } catch (blobError) {
      blobErrorMessage = blobError.message;
    }
    
    // Return the environment status
    res.json({
      status: 'success',
      environment: {
        NODE_ENV: process.env.NODE_ENV || 'not set',
        VERCEL: process.env.VERCEL || 'not set',
        BLOB_READ_WRITE_TOKEN: maskedToken,
        BLOB_URL: blobUrl || 'not set',
        DEPLOYMENT_URL: process.env.VERCEL_URL || 'not set',
        REGION: process.env.VERCEL_REGION || 'not set',
      },
      blob_access: {
        successful: blobAccessSuccessful,
        error: blobErrorMessage
      }
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

// Endpoint to upload local cache files to Vercel Blob Storage
app.post('/api/upload-local-cache', async (req, res) => {
  try {
    const { files } = req.body;
    
    if (!files || !Array.isArray(files)) {
      return res.status(400).json({ error: 'Files array is required' });
    }
    
    const results = [];
    
    for (const file of files) {
      try {
        const blobKey = getBlobKeyFromPath(file.path);
        const result = await writeBlob(blobKey, file.data);
        results.push({
          path: file.path,
          blobKey,
          success: result.success,
          url: result.url
        });
      } catch (error) {
        results.push({
          path: file.path,
          success: false,
          error: error.message
        });
      }
    }
    
    res.json({
      success: true,
      results
    });
  } catch (error) {
    console.error('Error uploading files:', error);
    res.status(500).json({ error: error.message });
  }
});

// Export helper functions for use in other modules
app.blobHelpers = {
  getBlobKeyFromPath,
  readBlob,
  writeBlob,
  listBlobs
};

// Catch-all for other API routes
app.all('/api/*', (req, res) => {
  console.log(`[vercel.js] Fallback: Unknown route ${req.method} ${req.path}`);
  return res.status(404).json({
    status: 'error',
    message: 'API endpoint not found (fallback handler)',
    path: req.path
  });
});

// Export the Express app for Vercel
module.exports = app; 