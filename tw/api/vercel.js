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

// Define blob storage prefixes
const BLOB_PREFIX = 'articles/';
const BLOB_ARTICLE_PREFIX = 'articles/final_article_';
const BLOB_SUMMARY_PREFIX = 'articles/summary_';

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

// Helper endpoint to test blob storage
app.get('/api/blob-test', async (req, res) => {
  try {
    // Create a test blob
    const testData = {
      content: "# Test Article\n\nThis is a test article to verify Vercel Blob Storage is working.",
      timestamp: Date.now()
    };
    
    const { url } = await put('articles/test-article.json', JSON.stringify(testData), {
      access: 'public',
      contentType: 'application/json'
    });
    
    return res.json({
      status: 'success',
      message: 'Successfully wrote test blob',
      url
    });
  } catch (error) {
    return res.status(500).json({
      status: 'error',
      message: `Error testing blob storage: ${error.message}`
    });
  }
});

// Helper endpoint to list blobs
app.get('/api/list-blobs', async (req, res) => {
  try {
    const { blobs } = await list({ prefix: BLOB_PREFIX });
    return res.json({
      status: 'success',
      count: blobs.length,
      blobs: blobs.map(b => ({
        pathname: b.pathname,
        size: b.size,
        uploadedAt: b.uploadedAt
      }))
    });
  } catch (error) {
    return res.status(500).json({
      status: 'error',
      message: `Error listing blobs: ${error.message}`
    });
  }
});

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