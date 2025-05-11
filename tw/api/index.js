// Main API handler for Vercel serverless functions
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { createHash } = require('crypto');
const { promisify } = require('util');
const { glob } = require('glob');
const axios = require('axios');
const { JSDOM } = require('jsdom');

// Setup Express app
const app = express();

// Enable CORS with credentials support
app.use(cors({ origin: true, credentials: true }));
app.use(express.json());

// Environment detection
const isVercel = process.env.VERCEL === '1';
console.log(`Running in ${isVercel ? 'Vercel' : 'local'} environment`);

// Constants - always use /tmp for Vercel
const CACHE_DIR = isVercel ? '/tmp/cache' : path.join(process.cwd(), '.vercel', 'cache');
const LOCAL_CACHE_DIR = isVercel ? '/tmp/local_cache' : path.join(process.cwd(), 'local_cache');

// Detailed logging of directories
console.log(`API Server initialized with:
  CACHE_DIR: ${CACHE_DIR}
  LOCAL_CACHE_DIR: ${LOCAL_CACHE_DIR}
  Current working directory: ${process.cwd()}
  Vercel environment: ${isVercel ? 'Yes' : 'No'}`);

// Cache for articles and summaries
const CACHE = {
  articles: [],
  summaries: {},
  interests: {}
};

// Ensure cache directories exist
let cacheDirsAvailable = false;
try {
  if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    console.log(`Created CACHE_DIR: ${CACHE_DIR}`);
  }
  if (!fs.existsSync(LOCAL_CACHE_DIR)) {
    fs.mkdirSync(LOCAL_CACHE_DIR, { recursive: true });
    console.log(`Created LOCAL_CACHE_DIR: ${LOCAL_CACHE_DIR}`);
  }
  cacheDirsAvailable = true;
  console.log(`Cache directories initialized`);
} catch (error) {
  console.warn('Error creating cache directories (normal in serverless environments):', error.message);
}

// Safe file read function with memory cache fallback
function safeReadFile(filePath, defaultValue = null) {
  try {
    if (cacheDirsAvailable && fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, 'utf8'));
    }
  } catch (error) {
    console.warn(`Error reading file ${filePath}: ${error.message}`);
  }
  return defaultValue;
}

// Safe file write function with memory cache fallback
function safeWriteFile(filePath, data) {
  try {
    if (cacheDirsAvailable) {
      fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
      return true;
    }
  } catch (error) {
    console.warn(`Error writing file ${filePath}: ${error.message}`);
  }
  return false;
}

// Helper function to generate cache keys
function generateCacheKey(input) {
  return createHash('md5').update(input).digest('hex');
}

// Helper function to fetch HN articles
async function fetchArticles() {
  try {
    const response = await axios.get('https://news.ycombinator.com/');
    const dom = new JSDOM(response.data);
    const document = dom.window.document;
    
    const articles = [];
    const rows = document.querySelectorAll('.athing');
    
    rows.forEach(row => {
      const titleElement = row.querySelector('.titleline > a');
      if (!titleElement) return;
      
      const title = titleElement.textContent.trim();
      const itemId = row.getAttribute('id');
      
      if (itemId) {
        const commentLink = `https://news.ycombinator.com/item?id=${itemId}`;
        articles.push([title, commentLink]);
      }
    });
    
    CACHE.articles = articles;
    return articles;
  } catch (error) {
    console.error('Error fetching articles:', error);
    return [];
  }
}

// Helper function to extract content from article
async function extractContent(url) {
  try {
    const response = await axios.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      }
    });
    
    const dom = new JSDOM(response.data);
    const document = dom.window.document;
    
    // Extract title
    const title = document.querySelector('title')?.textContent || 'No title found';
    
    // Extract content from paragraphs
    const paragraphs = document.querySelectorAll('p');
    const content = Array.from(paragraphs)
      .map(p => p.textContent.trim())
      .filter(text => text.length > 30)
      .join('\n\n');
      
    return `Title: ${title}\n\n${content || 'No content found'}`;
  } catch (error) {
    console.error('Error extracting content:', error);
    return null;
  }
}

// Helper function to generate a simple summary
function summarizeContent(content, maxLength = 1000) {
  if (!content) return null;
  
  // Simple summarization: take first few paragraphs
  const paragraphs = content.split('\n\n');
  const summary = paragraphs.slice(0, 3).join('\n\n');
  
  return summary.length > maxLength ? summary.substring(0, maxLength) + '...' : summary;
}

// Basic scoring function for articles based on interests
function scoreArticle(article, interests) {
  if (!interests || !interests.length) return 50; // Default score
  
  const [title, _] = article;
  const interestTerms = interests.split(',').map(term => term.trim().toLowerCase());
  
  let score = 50; // Base score
  
  // Simple scoring: +10 for each interest keyword in the title
  interestTerms.forEach(term => {
    if (title.toLowerCase().includes(term)) {
      score += 10;
    }
  });
  
  // Cap score at 100
  return Math.min(score, 100);
}

// Route: Log messages
app.post('/api/log', (req, res) => {
  try {
    const { message, level = 'info', source = 'unknown' } = req.body;
    
    if (!message) {
      return res.status(400).json({ status: 'error', message: 'Missing log message' });
    }
    
    // In Vercel, we'll log to console instead of files
    console.log(`[${level.toUpperCase()}] [${source}] ${message}`);
    
    return res.json({ status: 'success' });
  } catch (error) {
    console.error('Error handling log message:', error);
    return res.status(500).json({ status: 'error', message: error.message });
  }
});

// Route: Check cache status
app.get('/api/check-cache', async (req, res) => {
  try {
    console.log('Checking cache status');
    
    // Find summary files
    const summaryFiles = await glob(`${CACHE_DIR}/summary_*.json`);
    const localSummaryFiles = await glob(`${LOCAL_CACHE_DIR}/summary_*.json`);
    const articleCount = summaryFiles.length + localSummaryFiles.length;
    
    // Find final article files
    const finalArticles = await glob(`${CACHE_DIR}/final_article_*.json`);
    const localFinalArticles = await glob(`${LOCAL_CACHE_DIR}/final_article_*.json`);
    const finalArticleCount = finalArticles.length + localFinalArticles.length;
    
    console.log(`Found ${articleCount} summary files and ${finalArticleCount} final article files`);
    
    // Count unique articles based on title
    const uniqueTitles = new Set();
    let validArticleCount = 0;
    
    const allFinalArticles = [...finalArticles, ...localFinalArticles];
    if (allFinalArticles.length > 0) {
      for (const articlePath of allFinalArticles) {
        try {
          const fileData = fs.readFileSync(articlePath, 'utf-8');
          const data = JSON.parse(fileData);
          
          if (data.content) {
            const contentLines = data.content.split('\n');
            let title = contentLines[0] || 'Unknown Title';
            if (title.startsWith('# ')) {
              title = title.substring(2); // Remove Markdown heading
            }
            
            // Normalize title (remove arrow notation)
            if (title.includes('->')) {
              title = title.split('->')[0].trim();
            }
            
            uniqueTitles.add(title);
            validArticleCount++;
          }
        } catch (error) {
          console.error(`Error reading article file ${articlePath}:`, error);
        }
      }
    }
    
    // Return cache status
    return res.json({
      status: 'success',
      cached: articleCount > 0,
      article_count: articleCount,
      valid_article_count: validArticleCount,
      final_article_count: uniqueTitles.size,
      unique_titles: Array.from(uniqueTitles)
    });
  } catch (error) {
    console.error('Error checking cache:', error);
    return res.status(500).json({ status: 'error', message: error.message });
  }
});

// Route: List final articles
app.get('/final-articles', async (req, res) => {
  try {
    // Get final article files
    const finalArticleGlob = await glob(`${CACHE_DIR}/final_article_*.json`);
    const localFinalArticleGlob = await glob(`${LOCAL_CACHE_DIR}/final_article_*.json`);
    
    // Combine and process article info
    const articles = [...finalArticleGlob, ...localFinalArticleGlob].map(filePath => {
      const filename = path.basename(filePath);
      const id = filename.replace('final_article_', '').replace('.json', '');
      return { id, filename };
    });
    
    return res.json({
      status: 'success',
      articles
    });
  } catch (error) {
    console.error('Error listing final articles:', error);
    return res.status(500).json({ status: 'error', message: error.message });
  }
});

// Route: Get final article by filename
app.get('/final-article/:filename', (req, res) => {
  try {
    const { filename } = req.params;
    
    if (!filename || !filename.startsWith('final_article_') || !filename.endsWith('.json')) {
      return res.status(400).json({ status: 'error', message: 'Invalid article filename' });
    }
    
    // Try local cache first, then main cache
    let articlePath = path.join(LOCAL_CACHE_DIR, filename);
    let source = 'local';
    
    if (!fs.existsSync(articlePath)) {
      articlePath = path.join(CACHE_DIR, filename);
      source = 'main';
      
      if (!fs.existsSync(articlePath)) {
        return res.status(404).json({ status: 'error', message: 'Article not found' });
      }
    }
    
    // Read the article file
    const fileContent = fs.readFileSync(articlePath, 'utf-8');
    const articleData = JSON.parse(fileContent);
    
    return res.json({
      ...articleData,
      source
    });
  } catch (error) {
    console.error('Error getting final article:', error);
    return res.status(500).json({ status: 'error', message: error.message });
  }
});

// Route: Get final article by ID
app.get('/get-final-article/:id', (req, res) => {
  try {
    const { id } = req.params;
    
    if (!id) {
      return res.status(400).json({ status: 'error', message: 'Missing article ID' });
    }
    
    // Try local cache first, then main cache
    const filename = `final_article_${id}.json`;
    let articlePath = path.join(LOCAL_CACHE_DIR, filename);
    let source = 'local';
    
    if (!fs.existsSync(articlePath)) {
      articlePath = path.join(CACHE_DIR, filename);
      source = 'main';
      
      if (!fs.existsSync(articlePath)) {
        return res.status(404).json({ status: 'error', message: 'Article not found' });
      }
    }
    
    // Read the article file
    const fileContent = fs.readFileSync(articlePath, 'utf-8');
    const articleData = JSON.parse(fileContent);
    
    return res.json({
      status: 'success',
      article: {
        ...articleData,
        id,
        source
      }
    });
  } catch (error) {
    console.error('Error getting final article by ID:', error);
    return res.status(500).json({ status: 'error', message: error.message });
  }
});

// Route: Analyze interests (simplified version)
app.post('/analyze-interests', async (req, res) => {
  try {
    const { interests } = req.body;
    
    if (!interests) {
      return res.status(400).json({ status: 'error', message: 'Missing interests' });
    }
    
    console.log(`Analyzing interests: ${interests}`);
    
    // Get all final articles for analysis
    const finalArticleGlob = await glob(`${CACHE_DIR}/final_article_*.json`);
    const localFinalArticleGlob = await glob(`${LOCAL_CACHE_DIR}/final_article_*.json`);
    const allArticlePaths = [...finalArticleGlob, ...localFinalArticleGlob];
    
    // Process articles (simplified without Vector DB)
    const articles = [];
    for (const filePath of allArticlePaths) {
      try {
        const fileContent = fs.readFileSync(filePath, 'utf-8');
        const data = JSON.parse(fileContent);
        const filename = path.basename(filePath);
        const id = filename.replace('final_article_', '').replace('.json', '');
        
        // Extract title from content
        let title = 'Untitled Article';
        if (data.content) {
          const contentLines = data.content.split('\n');
          if (contentLines[0] && contentLines[0].startsWith('# ')) {
            title = contentLines[0].substring(2);
          }
        }
        
        articles.push({
          id,
          title,
          filename
        });
      } catch (error) {
        console.error(`Error processing article ${filePath}:`, error);
      }
    }
    
    return res.json({
      status: 'success',
      articles
    });
  } catch (error) {
    console.error('Error analyzing interests:', error);
    return res.status(500).json({ status: 'error', message: error.message });
  }
});

// Cache sync API routes
app.post('/api/sync-cache', async (req, res) => {
  try {
    console.log('Synchronizing cache');
    
    // In Vercel, we'll have a simplified caching mechanism
    // Just ensure directories exist
    if (!fs.existsSync(LOCAL_CACHE_DIR)) {
      fs.mkdirSync(LOCAL_CACHE_DIR, { recursive: true });
    }
    
    return res.json({
      status: 'success',
      message: 'Cache synchronized in Vercel environment',
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
    return res.status(500).json({ status: 'error', message: error.message });
  }
});

// Route: Get cached file
app.get('/api/get-cached-file', (req, res) => {
  try {
    const { file } = req.query;
    
    if (!file) {
      return res.status(400).json({ status: 'error', message: 'Missing file parameter' });
    }
    
    // Try local cache first, then main cache
    let filePath = path.join(LOCAL_CACHE_DIR, file);
    let source = 'local';
    
    if (!fs.existsSync(filePath)) {
      filePath = path.join(CACHE_DIR, file);
      source = 'main';
      
      if (!fs.existsSync(filePath)) {
        return res.status(404).json({ status: 'error', message: 'File not found in cache' });
      }
    }
    
    // Read the file
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    
    try {
      // Try to parse as JSON
      const data = JSON.parse(fileContent);
      return res.json({
        status: 'success',
        data,
        source
      });
    } catch (error) {
      // Return as text if not valid JSON
      return res.json({
        status: 'success',
        data: fileContent,
        source
      });
    }
  } catch (error) {
    console.error('Error getting cached file:', error);
    return res.status(500).json({ status: 'error', message: error.message });
  }
});

// Route: Get article summary
app.get('/api/get-summary', (req, res) => {
  try {
    const { id } = req.query;
    
    if (!id) {
      return res.status(400).json({ status: 'error', message: 'Missing ID parameter' });
    }
    
    // Generate cache key
    const cacheKey = generateCacheKey(id);
    const filename = `summary_${cacheKey}.json`;
    
    // Try local cache first, then main cache
    let filePath = path.join(LOCAL_CACHE_DIR, filename);
    let source = 'local';
    
    if (!fs.existsSync(filePath)) {
      filePath = path.join(CACHE_DIR, filename);
      source = 'main';
      
      if (!fs.existsSync(filePath)) {
        return res.status(404).json({ status: 'error', message: 'Summary not found in cache' });
      }
    }
    
    // Read the file
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const data = JSON.parse(fileContent);
    
    return res.json({
      status: 'success',
      data,
      source
    });
  } catch (error) {
    console.error('Error getting summary:', error);
    return res.status(500).json({ status: 'error', message: error.message });
  }
});

// Route: Get article
app.get('/api/get-article', (req, res) => {
  try {
    const { key } = req.query;
    
    if (!key) {
      return res.status(400).json({ status: 'error', message: 'Missing key parameter' });
    }
    
    // Format the key for filename compatibility
    const safeKey = key.replace(/[^a-zA-Z0-9_]/g, '_');
    
    // Try to find articles matching this key
    const patternRegex = new RegExp(`final_article_\\d+_${safeKey}.*\\.json`);
    
    // Check local cache
    let found = false;
    let articleData = null;
    let source = 'local';
    
    // Check files in local cache
    const localFiles = fs.readdirSync(LOCAL_CACHE_DIR);
    for (const file of localFiles) {
      if (patternRegex.test(file)) {
        const filePath = path.join(LOCAL_CACHE_DIR, file);
        const fileContent = fs.readFileSync(filePath, 'utf-8');
        articleData = JSON.parse(fileContent);
        found = true;
        break;
      }
    }
    
    // Check main cache if not found locally
    if (!found) {
      source = 'main';
      const mainFiles = fs.readdirSync(CACHE_DIR);
      for (const file of mainFiles) {
        if (patternRegex.test(file)) {
          const filePath = path.join(CACHE_DIR, file);
          const fileContent = fs.readFileSync(filePath, 'utf-8');
          articleData = JSON.parse(fileContent);
          found = true;
          break;
        }
      }
    }
    
    if (!found) {
      return res.status(404).json({ status: 'error', message: 'Article not found in cache' });
    }
    
    return res.json({
      status: 'success',
      data: articleData,
      source
    });
  } catch (error) {
    console.error('Error getting article:', error);
    return res.status(500).json({ status: 'error', message: error.message });
  }
});

// Route: Verify email
app.post('/verify-email', (req, res) => {
  try {
    const { email } = req.body;
    
    if (!email) {
      return res.status(400).json({ status: 'error', message: 'Missing email address' });
    }
    
    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({ status: 'error', message: 'Invalid email format' });
    }
    
    // In Vercel version, we'll simplify this to just save a cookie
    res.cookie('user_email', email, {
      maxAge: 24 * 60 * 60 * 1000, // 24 hours
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax'
    });
    
    return res.json({
      status: 'success',
      message: 'Email verified successfully'
    });
  } catch (error) {
    console.error('Error verifying email:', error);
    return res.status(500).json({ status: 'error', message: error.message });
  }
});

// Route: Get articles
app.get('/api/articles', async (req, res) => {
  try {
    console.log('GET /api/articles - Processing request with query:', req.query);
    
    // Load the server module
    const serverFunctions = require('./server');
    const result = serverFunctions.process_articles_endpoint(req.query);
    
    // Ensure we return an array format the frontend expects
    if (Array.isArray(result)) {
      console.log(`Returning ${result.length} articles`);
      console.log('Sample article:', result.length > 0 ? JSON.stringify(result[0]).substring(0, 200) + '...' : 'No articles');
      return res.json(result);
    } else if (result && result.articles && Array.isArray(result.articles)) {
      console.log(`Returning ${result.articles.length} articles (from object)`);
      console.log('Sample article:', result.articles.length > 0 ? JSON.stringify(result.articles[0]).substring(0, 200) + '...' : 'No articles');
      return res.json(result.articles);
    } else {
      console.log('No articles found or invalid format');
      return res.json([]);
    }
  } catch (error) {
    console.error('Error getting articles:', error);
    return res.status(500).json([]);
  }
});

app.get('/api/article/:id', async (req, res) => {
  try {
    const articleId = req.params.id;
    
    if (!articleId) {
      return res.status(404).json({ 
        status: 'error', 
        message: 'Article ID is required'
      });
    }
    
    console.log(`GET /api/article/${articleId} - Getting article details`);
    
    // Use our server.js module
    const serverFunctions = require('./server');
    const result = serverFunctions.get_article_endpoint(articleId);
    
    console.log(`Article retrieval result status: ${result.status || 'unknown'}`);
    
    if (result.status === 'error') {
      return res.status(404).json(result);
    }
    
    return res.json(result);
  } catch (error) {
    console.error('Error retrieving article:', error);
    return res.status(500).json({ 
      status: 'error', 
      message: error.message || 'An unexpected error occurred'
    });
  }
});

app.post('/api/analyze-interests', (req, res) => {
  try {
    const { interests } = req.body;
    
    if (!interests) {
      return res.status(400).json({ status: 'error', message: 'No interests provided' });
    }
    
    // Use our server.js module
    const serverFunctions = require('./server');
    const result = serverFunctions.analyze_interests_endpoint(interests);
    
    return res.json(result);
  } catch (error) {
    console.error('Error analyzing interests:', error);
    return res.status(500).json({ status: 'error', message: error.message });
  }
});

// Handle both '/log' and '/api/log' routes
app.post('/log', (req, res) => {
  // Redirect to the main handler
  app.handle(req, { ...req, url: '/api/log' }, res);
});

// Create a catch-all route for API requests
app.all('/api/*', (req, res, next) => {
  // Check if this route matches any of our defined routes
  const routes = app._router.stack
    .filter(layer => layer.route)
    .map(layer => ({ path: layer.route.path, methods: layer.route.methods }));
  
  const matchedRoute = routes.find(route => {
    // Check if the path matches any of our defined routes
    if (req.path === route.path) {
      // Check if the HTTP method is supported
      return route.methods[req.method.toLowerCase()];
    }
    return false;
  });
  
  if (matchedRoute) {
    // If we found a matching route, continue to the next middleware
    return next();
  }
  
  if (req.method === 'OPTIONS') {
    // Handle CORS preflight requests
    return res.status(200).end();
  }
  
  // Return a 404 for any API route not explicitly defined
  console.warn(`API route not found: ${req.method} ${req.path}`);
  return res.status(404).json({
    status: 'error',
    message: 'API endpoint not found',
    path: req.path
  });
});

// Add error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    status: 'error',
    message: 'Internal Server Error',
    error: process.env.NODE_ENV === 'production' ? undefined : err.message
  });
});

// Handle requests at both root level and with /api prefix
app.use((req, res, next) => {
  console.log(`Request received: ${req.method} ${req.url}`);
  next();
});

// Start the server if not being used as a module
if (!module.parent) {
  const port = process.env.PORT || 3000;
  app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
  });
}

// Special route for Vercel serverless deployment
module.exports = app; 