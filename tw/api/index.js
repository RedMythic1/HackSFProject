// Main API handler for Vercel serverless functions
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { createHash } = require('crypto');
const { promisify } = require('util');
const { glob } = require('glob');

// Setup Express app
const app = express();

// Enable CORS with credentials support
app.use(cors({ origin: true, credentials: true }));
app.use(express.json());

// Constants
const CACHE_DIR = path.join(process.cwd(), '.vercel', 'cache');
const LOCAL_CACHE_DIR = path.join(process.cwd(), 'local_cache');

// Ensure cache directories exist
try {
  if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
  }
  if (!fs.existsSync(LOCAL_CACHE_DIR)) {
    fs.mkdirSync(LOCAL_CACHE_DIR, { recursive: true });
  }
  console.log(`Cache directories initialized: 
    CACHE_DIR: ${CACHE_DIR}
    LOCAL_CACHE_DIR: ${LOCAL_CACHE_DIR}`);
} catch (error) {
  console.error('Error creating cache directories:', error);
}

// Helper function to generate cache keys
function generateCacheKey(input) {
  return createHash('md5').update(input).digest('hex');
}

// Route: Log messages
app.post('/log', (req, res) => {
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
app.get('/check-cache', async (req, res) => {
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

// Special route for Vercel serverless deployment
module.exports = app; 