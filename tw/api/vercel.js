// Special handler for Vercel deployments
// This file is used as a fallback if other approaches fail

const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { put, list, get, del } = require('@vercel/blob');

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
  console.log('[vercel.js] Fallback: GET /api/articles');
  
  // Try to get articles from blob storage first
  try {
    const { blobs } = await list({ prefix: BLOB_ARTICLE_PREFIX });
    console.log(`[vercel.js] Found ${blobs.length} articles in blob storage`);
    
    if (blobs.length > 0) {
      const articles = [];
      
      for (const blob of blobs.slice(0, 10)) { // Limit to 10 articles for performance
        try {
          const content = await blob.text();
          const data = JSON.parse(content);
          const id = path.basename(blob.pathname).replace(`${BLOB_ARTICLE_PREFIX}`, '').replace('.json', '');
          
          // Extract title and subject from content
          let title = 'Untitled Article';
          let subject = '';
          
          if (data.content) {
            const lines = data.content.split('\n');
            if (lines[0]) {
              title = lines[0].startsWith('# ') ? lines[0].substring(2) : lines[0];
            }
            
            // Get first non-title paragraph as subject
            for (let i = 1; i < lines.length; i++) {
              if (lines[i] && !lines[i].startsWith('#') && lines[i].length > 10) {
                subject = lines[i].length > 100 ? lines[i].substring(0, 100) + '...' : lines[i];
                break;
              }
            }
          }
          
          articles.push({
            id,
            title,
            subject,
            score: 0,
            timestamp: data.timestamp || Date.now(),
            link: `https://news.ycombinator.com/item?id=${id}`
          });
        } catch (error) {
          console.warn(`[vercel.js] Error processing blob: ${error.message}`);
        }
      }
      
      if (articles.length > 0) {
        articles.sort((a, b) => b.timestamp - a.timestamp);
        return res.json(articles);
      }
    }
  } catch (error) {
    console.warn(`[vercel.js] Error accessing blob storage: ${error.message}`);
  }
  
  // Fall back to demo articles
  res.json(DEMO_ARTICLES);
});

app.get('/api/article/:id', async (req, res) => {
  const { id } = req.params;
  console.log(`[vercel.js] Fallback: GET /api/article/${id}`);
  
  // Check for demo articles first
  if (id.startsWith('demo')) {
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
  
  // Try to get from blob storage
  try {
    const blobKey = `${BLOB_ARTICLE_PREFIX}${id}.json`;
    console.log(`[vercel.js] Looking for article in blob storage: ${blobKey}`);
    
    const blob = await get(blobKey);
    if (blob) {
      const content = await blob.text();
      const data = JSON.parse(content);
      
      // Extract title and summary
      let title = 'Untitled Article';
      let summary = '';
      
      if (data.content) {
        const lines = data.content.split('\n');
        if (lines[0]) {
          title = lines[0].startsWith('# ') ? lines[0].substring(2) : lines[0];
        }
        
        // Get first non-title paragraph as summary
        for (let i = 1; i < lines.length; i++) {
          if (lines[i] && !lines[i].startsWith('#') && lines[i].length > 10) {
            summary = lines[i].length > 200 ? lines[i].substring(0, 200) + '...' : lines[i];
            break;
          }
        }
      }
      
      return res.json({
        status: 'success',
        article: {
          id,
          title,
          link: data.url || `https://news.ycombinator.com/item?id=${id}`,
          summary,
          content: data.content || '',
          timestamp: data.timestamp || Date.now()
        }
      });
    }
  } catch (error) {
    console.warn(`[vercel.js] Error getting article from blob storage: ${error.message}`);
  }
  
  // No article found
  return res.status(404).json({
    status: 'error',
    message: 'Article not found'
  });
});

app.post('/api/analyze-interests', (req, res) => {
  const { interests } = req.body;
  console.log(`[vercel.js] Fallback: POST /api/analyze-interests with ${interests || 'no'} interests`);
  
  if (!interests) {
    return res.status(400).json({
      status: 'error',
      message: 'No interests provided'
    });
  }
  
  // Simple scoring based on interests
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

// Blob test endpoint for verification
app.get('/api/blob-test', async (req, res) => {
  try {
    // Create a test blob
    const testData = {
      message: 'This is a test blob',
      timestamp: Date.now(),
      environment: process.env.VERCEL ? 'Vercel' : 'Local'
    };
    
    // Upload the test blob
    const { url } = await put('test-blob.json', JSON.stringify(testData, null, 2), {
      access: 'public',
      contentType: 'application/json'
    });
    
    res.json({
      success: true,
      message: 'Test blob created successfully',
      url,
      data: testData
    });
  } catch (error) {
    console.error('Error creating test blob:', error);
    res.status(500).json({ error: error.message });
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