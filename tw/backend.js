const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const glob = require('glob');

const app = express();
const PORT = 5001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Constants
const CACHE_DIR = path.join(__dirname, '.cache');
const HTML_DIR = path.join(__dirname, 'public', 'articles');

// Helper function to normalize article title
function normalizeArticleTitle(title) {
  if (title.includes("->")) {
    return title.split("->")[0].trim();
  }
  return title;
}

// Extract article summary from content
function extractArticleSummary(content) {
  try {
    // First try to find a section explicitly labeled as "Summary"
    const summaryPattern = /## Summary\s+([\s\S]+?)(?=##|$)/;
    const match = summaryPattern.exec(content);
    
    if (match) {
      return match[1].trim();
    }
    
    // If no explicit summary section, generate one from the beginning of the article
    const lines = content.split('\n');
    
    // Skip the title if present
    let startIdx = 0;
    if (lines.length && lines[0].startsWith('# ')) {
      startIdx = 1;
    }
    
    // Collect text for summary (up to ~500 characters)
    let summaryText = "";
    let currentLength = 0;
    const targetLength = 500;
    
    for (let i = startIdx; i < lines.length; i++) {
      const line = lines[i].trim();
      // Skip headings and empty lines
      if (line.startsWith('#') || !line) {
        continue;
      }
      
      // Add this line to the summary
      summaryText += line + " ";
      currentLength += line.length;
      
      // Stop if we've reached target length
      if (currentLength >= targetLength) {
        summaryText += "...";
        break;
      }
    }
    
    return summaryText.trim() || "No summary available.";
  } catch (error) {
    console.error(`Error generating summary: ${error}`);
    return "Summary unavailable due to an error.";
  }
}

// Routes
app.get('/check-cache', (req, res) => {
  try {
    // Find all summary_*.json files
    const summaryFiles = glob.sync(path.join(CACHE_DIR, 'summary_*.json'));
    
    // Find all final_article_*.json files
    const finalArticles = glob.sync(path.join(CACHE_DIR, 'final_article_*.json'));
    
    // Count unique articles based on title
    const uniqueTitles = new Set();
    let validArticleCount = 0;
    
    finalArticles.forEach(articlePath => {
      try {
        const data = JSON.parse(fs.readFileSync(articlePath, 'utf-8'));
        const content = data.content || '';
        
        if (content) {
          const titleLines = content.split('\n');
          const title = titleLines[0].startsWith('# ') ? titleLines[0].substring(2) : titleLines[0];
          
          // Normalize title
          const normalizedTitle = normalizeArticleTitle(title);
          uniqueTitles.add(normalizedTitle);
          validArticleCount++;
        }
      } catch (error) {
        console.error(`Error reading article ${articlePath}: ${error}`);
      }
    });
    
    const uniqueArticleCount = uniqueTitles.size;
    
    console.log(`Found ${summaryFiles.length} cached article summaries and ${validArticleCount} valid cached final articles (${uniqueArticleCount} unique)`);
    
    return res.json({
      status: "success",
      message: "Cache check successful",
      cached: summaryFiles.length > 0 || validArticleCount > 0,
      article_count: summaryFiles.length,
      final_article_count: uniqueArticleCount,
      valid_article_count: validArticleCount
    });
  } catch (error) {
    console.error(`Exception checking cache: ${error}`);
    return res.status(500).json({
      status: "error",
      message: `Exception: ${error.toString()}`
    });
  }
});

app.get('/get-final-articles', (req, res) => {
  try {
    // Find all final_article_*.json files
    const finalArticles = glob.sync(path.join(CACHE_DIR, 'final_article_*.json'));
    
    // Extract and load article data
    const articleData = [];
    const invalidFiles = [];
    
    finalArticles.forEach(articlePath => {
      try {
        const data = JSON.parse(fs.readFileSync(articlePath, 'utf-8'));
        
        // Extract filename
        const filename = path.basename(articlePath);
        
        // Extract timestamp
        const timestamp = filename.replace('final_article_', '').replace('.json', '');
        
        // Get the first line as the title
        const content = data.content || '';
        
        if (!content) {
          console.warn(`Article has no content: ${articlePath}`);
          invalidFiles.push(articlePath);
          return;
        }
        
        const titleLines = content.split('\n');
        const title = titleLines[0].startsWith('# ') ? titleLines[0].substring(2) : titleLines[0];
        
        // Normalize title
        const normalizedTitle = normalizeArticleTitle(title);
        
        articleData.push({
          id: timestamp,
          title: normalizedTitle,
          timestamp: data.timestamp || 0,
          filename: filename
        });
      } catch (error) {
        console.error(`Error loading article ${articlePath}: ${error}`);
        invalidFiles.push(articlePath);
      }
    });
    
    // Sort by timestamp (newest first)
    articleData.sort((a, b) => b.timestamp - a.timestamp);
    
    // Remove duplicates based on title (keeping the newest version of each article)
    const uniqueTitles = new Set();
    const uniqueArticles = [];
    
    for (const article of articleData) {
      if (!uniqueTitles.has(article.title)) {
        uniqueTitles.add(article.title);
        uniqueArticles.push(article);
      }
    }
    
    console.log(`Found ${articleData.length} cached final articles, ${uniqueArticles.length} unique, removed ${invalidFiles.length} invalid files`);
    
    return res.json({
      status: "success",
      message: "Final articles retrieved successfully",
      articles: uniqueArticles,
      total_count: articleData.length,
      unique_count: uniqueTitles.size,
      invalid_count: invalidFiles.length
    });
  } catch (error) {
    console.error(`Exception getting final articles: ${error}`);
    return res.status(500).json({
      status: "error",
      message: `Exception: ${error.toString()}`
    });
  }
});

app.get('/get-final-article/:article_id', (req, res) => {
  try {
    const articleId = req.params.article_id;
    
    // Create the expected filename
    const filename = `final_article_${articleId}.json`;
    const articlePath = path.join(CACHE_DIR, filename);
    
    if (!fs.existsSync(articlePath)) {
      console.error(`Article with ID ${articleId} not found`);
      return res.status(404).json({
        status: "error",
        message: `Article with ID ${articleId} not found`
      });
    }
    
    // Load the article content
    const data = JSON.parse(fs.readFileSync(articlePath, 'utf-8'));
    const content = data.content || '';
    
    // Get the title (first line)
    const titleLines = content.split('\n');
    const title = titleLines[0].startsWith('# ') ? titleLines[0].substring(2) : titleLines[0];
    
    // Normalize title
    const normalizedTitle = normalizeArticleTitle(title);
    
    // Extract or generate a summary
    const summary = extractArticleSummary(content);
    
    console.log(`Retrieved article: ${normalizedTitle}`);
    
    return res.json({
      status: "success",
      message: "Article retrieved successfully",
      article: {
        id: articleId,
        title: normalizedTitle,
        content: content,
        summary: summary,
        timestamp: data.timestamp || 0
      }
    });
  } catch (error) {
    console.error(`Exception getting article ${req.params.article_id}: ${error}`);
    return res.status(500).json({
      status: "error",
      message: `Exception: ${error.toString()}`
    });
  }
});

app.post('/verify-email', (req, res) => {
  try {
    console.log("/verify-email endpoint called");
    console.log(`/verify-email received data: ${JSON.stringify(req.body)}`);
    
    const email = req.body.email ? req.body.email.trim() : '';
    
    if (!email) {
      return res.status(400).json({
        status: "error",
        message: "Email is required"
      });
    }
    
    // Improved email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[a-zA-Z0-9]{2,}$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({
        status: "error",
        message: "Invalid email format. Please use a valid email address (e.g. user@example.com)."
      });
    }
    
    // Instead of setting a cookie, we'll save the email in a file
    const userData = { email, verified: true, timestamp: Date.now() };
    
    // Make sure the user_data directory exists
    const userDataDir = path.join(__dirname, 'user_data');
    if (!fs.existsSync(userDataDir)) {
      fs.mkdirSync(userDataDir, { recursive: true });
    }
    
    // Save the user data
    fs.writeFileSync(path.join(userDataDir, 'user.json'), JSON.stringify(userData));
    
    return res.json({
      status: "success",
      message: "Email verified successfully"
    });
  } catch (error) {
    console.error(`Error verifying email: ${error}`);
    return res.status(500).json({
      status: "error",
      message: `An error occurred while verifying email: ${error.toString()}`
    });
  }
});

app.get('/check-email-verification', (req, res) => {
  try {
    const userDataPath = path.join(__dirname, 'user_data', 'user.json');
    
    if (fs.existsSync(userDataPath)) {
      const userData = JSON.parse(fs.readFileSync(userDataPath, 'utf-8'));
      
      if (userData.email && userData.verified) {
        return res.json({
          status: "success",
          verified: true,
          email: userData.email
        });
      }
    }
    
    return res.json({
      status: "success",
      verified: false
    });
  } catch (error) {
    console.error(`Error checking email verification: ${error}`);
    return res.status(500).json({
      status: "error",
      message: "An error occurred while checking email verification"
    });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Node.js backend server running on http://localhost:${PORT}`);
}); 