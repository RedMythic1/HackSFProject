// JavaScript implementation of server.py for Vercel deployment
// No Python dependencies should be used

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { glob } = require('glob');
const axios = require('axios');
const { JSDOM } = require('jsdom');

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

// --- Helper Functions ---

/**
 * Generate a cache key from a string
 * @param {string} input - Input string to hash
 * @returns {string} MD5 hash of the input
 */
function generateCacheKey(input) {
  return crypto.createHash('md5').update(input).digest('hex');
}

/**
 * Clean and normalize article title
 * @param {string} title - Article title to normalize
 * @returns {string} Normalized title
 */
function normalizeArticleTitle(title) {
  // Remove arrow notation (-> text) from titles
  if (title.includes("->")) {
    title = title.split("->")[0].trim();
  }
  return title;
}

/**
 * Extract or generate a summary from article content
 * @param {string} content - Article content
 * @returns {string} Summary text
 */
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
    if (lines.length > 0 && lines[0].startsWith('# ')) {
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

// --- API Endpoints Implementation ---

/**
 * Process articles endpoint implementation
 * @param {Object} query - Query parameters
 * @returns {Object} Response object with articles
 */
function process_articles_endpoint(query) {
  try {
    // Find all final article files in the cache
    const finalArticles = glob.sync(`${CACHE_DIR}/final_article_*.json`);
    const localFinalArticles = glob.sync(`${LOCAL_CACHE_DIR}/final_article_*.json`);
    const allFinalArticles = [...finalArticles, ...localFinalArticles];
    
    if (allFinalArticles.length === 0) {
      return {
        status: "success",
        message: "No articles found",
        articles: []
      };
    }
    
    // Extract and load article data
    const articleData = [];
    const uniqueTitles = new Set();
    
    for (const articlePath of allFinalArticles) {
      try {
        const data = JSON.parse(fs.readFileSync(articlePath, 'utf-8'));
        
        // Extract filename
        const filename = path.basename(articlePath);
        
        // Extract timestamp
        const articleId = filename.replace('final_article_', '').replace('.json', '');
        
        // Get the first line as the title
        const content = data.content || '';
        if (!content) {
          console.warn(`Article has no content: ${articlePath}`);
          continue;
        }
        
        const lines = content.split('\n');
        let title = lines.length > 0 ? lines[0] : 'Unknown Title';
        if (title.startsWith('# ')) {
          title = title.substring(2); // Remove Markdown heading marker
        }
        
        // Normalize title
        title = normalizeArticleTitle(title);
        
        if (!uniqueTitles.has(title)) {
          uniqueTitles.add(title);
          articleData.push({
            id: articleId,
            title: title,
            timestamp: data.timestamp || 0,
            filename: filename
          });
        }
      } catch (error) {
        console.error(`Error loading article ${articlePath}: ${error}`);
      }
    }
    
    // Sort by timestamp (newest first)
    articleData.sort((a, b) => b.timestamp - a.timestamp);
    
    return {
      status: "success",
      message: "Articles retrieved successfully",
      articles: articleData,
      total_count: articleData.length,
      unique_count: uniqueTitles.size
    };
  } catch (error) {
    console.error(`Error getting articles: ${error}`);
    return {
      status: "error",
      message: `Error retrieving articles: ${error.message}`,
      articles: []
    };
  }
}

/**
 * Get article by ID endpoint implementation
 * @param {string} articleId - Article ID
 * @returns {Object} Response object with article data
 */
function get_article_endpoint(articleId) {
  try {
    // Sanitize article_id to ensure it doesn't contain path traversal
    if (!articleId.match(/^[a-zA-Z0-9_]+$/)) {
      return {
        status: "error",
        message: "Invalid article ID format"
      };
    }
    
    // Lookup the article in the cache
    const articlePath = path.join(CACHE_DIR, `final_article_${articleId}.json`);
    const localArticlePath = path.join(LOCAL_CACHE_DIR, `final_article_${articleId}.json`);
    
    let articleData;
    
    if (fs.existsSync(articlePath)) {
      articleData = JSON.parse(fs.readFileSync(articlePath, 'utf-8'));
    } else if (fs.existsSync(localArticlePath)) {
      articleData = JSON.parse(fs.readFileSync(localArticlePath, 'utf-8'));
    } else {
      return {
        status: "error",
        message: "Article not found"
      };
    }
    
    // Extract title and content
    const content = articleData.content || '';
    let title = 'Unknown Title';
    
    // Extract title from content
    if (content) {
      const lines = content.split('\n');
      if (lines.length > 0 && lines[0].startsWith('# ')) {
        title = lines[0].substring(2).trim();
      }
    }
    
    // Find embedding if available
    let embedding = null;
    try {
      // Create a hash of the article content to look up potential embedding
      const contentHash = crypto.createHash('md5').update(content).digest('hex');
      const embeddingPaths = glob.sync(`${CACHE_DIR}/summary_${contentHash}*.json`);
      const localEmbeddingPaths = glob.sync(`${LOCAL_CACHE_DIR}/summary_${contentHash}*.json`);
      
      if (embeddingPaths.length > 0) {
        const embeddingData = JSON.parse(fs.readFileSync(embeddingPaths[0], 'utf-8'));
        embedding = embeddingData.embedding;
      } else if (localEmbeddingPaths.length > 0) {
        const embeddingData = JSON.parse(fs.readFileSync(localEmbeddingPaths[0], 'utf-8'));
        embedding = embeddingData.embedding;
      } else {
        // Try to find a matching summary by comparing title
        const summaryFiles = glob.sync(`${CACHE_DIR}/summary_*.json`);
        const localSummaryFiles = glob.sync(`${LOCAL_CACHE_DIR}/summary_*.json`);
        const allSummaryFiles = [...summaryFiles, ...localSummaryFiles];
        
        for (const summaryPath of allSummaryFiles) {
          try {
            const summaryData = JSON.parse(fs.readFileSync(summaryPath, 'utf-8'));
            const summaryTitle = summaryData.title || '';
            if ('embedding' in summaryData && summaryTitle.toLowerCase() === title.toLowerCase()) {
              embedding = summaryData.embedding;
              console.log(`Found embedding for article by title match: ${title}`);
              break;
            }
          } catch (error) {
            console.error(`Error reading summary file ${summaryPath}: ${error}`);
          }
        }
      }
    } catch (error) {
      console.error(`Error finding embedding: ${error}`);
    }
    
    // Build the response object
    const response = {
      status: "success",
      article: {
        id: articleId,
        title: title,
        content: content,
      }
    };
    
    // Add embedding if available
    if (embedding) {
      response.article.embedding = embedding;
    }
    
    return response;
  } catch (error) {
    console.error(`Error retrieving article ${articleId}: ${error}`);
    return {
      status: "error",
      message: `Error retrieving article: ${error.message}`
    };
  }
}

/**
 * Analyze interests endpoint implementation
 * @param {string} interests - Comma-separated user interests
 * @returns {Object} Response object with recommendations
 */
function analyze_interests_endpoint(interests) {
  try {
    if (!interests) {
      return {
        status: "error",
        message: "No interests provided"
      };
    }
    
    console.log(`Analyzing interests: ${interests}`);
    
    // Find any final articles in the cache
    const articleFiles = glob.sync(`${CACHE_DIR}/final_article_*.json`);
    const localArticleFiles = glob.sync(`${LOCAL_CACHE_DIR}/final_article_*.json`);
    const allArticleFiles = [...articleFiles, ...localArticleFiles];
    
    if (allArticleFiles.length === 0) {
      return {
        status: "error",
        message: "No articles found for analysis"
      };
    }
    
    console.log(`Found ${allArticleFiles.length} article files for analysis`);
    
    // Load articles and prepare for frontend analysis
    const articles = [];
    // Limit to 10 articles for performance
    const filesToProcess = allArticleFiles.slice(0, 10);
    
    for (const articlePath of filesToProcess) {
      try {
        const articleData = JSON.parse(fs.readFileSync(articlePath, 'utf-8'));
        
        const articleId = path.basename(articlePath).replace('final_article_', '').replace('.json', '');
        
        // Extract title from content
        let title = "Unknown Title";
        const content = articleData.content || '';
        const lines = content.split('\n');
        if (lines.length > 0 && lines[0].startsWith('# ')) {
          title = lines[0].substring(2).trim();
        }
        
        // Extract a summary from the content
        const summary = extractArticleSummary(content);
        
        // Find article embedding from summary files
        let embedding = null;
        try {
          // First try to find a summary file with matching title
          const summaryFiles = glob.sync(`${CACHE_DIR}/summary_*.json`);
          const localSummaryFiles = glob.sync(`${LOCAL_CACHE_DIR}/summary_*.json`);
          const allSummaryFiles = [...summaryFiles, ...localSummaryFiles];
          
          // Try to find a matching summary by comparing title
          for (const summaryPath of allSummaryFiles) {
            try {
              const summaryData = JSON.parse(fs.readFileSync(summaryPath, 'utf-8'));
              // If the summary contains an embedding and matches our title
              const summaryTitle = summaryData.title || '';
              if ('embedding' in summaryData && summaryTitle && (
                summaryTitle.toLowerCase() === title.toLowerCase() ||
                summaryTitle.toLowerCase().includes(title.toLowerCase()) ||
                title.toLowerCase().includes(summaryTitle.toLowerCase())
              )) {
                embedding = summaryData.embedding;
                console.log(`Found embedding for article: ${title}`);
                break;
              }
            } catch (error) {
              console.error(`Error reading summary file ${summaryPath}: ${error}`);
            }
          }
          
          // If still no embedding, try hash-based matching
          if (!embedding) {
            // Try matching by content hash
            const contentHash = crypto.createHash('md5').update(content).digest('hex');
            const embeddingPaths = glob.sync(`${CACHE_DIR}/summary_${contentHash}*.json`);
            const localEmbeddingPaths = glob.sync(`${LOCAL_CACHE_DIR}/summary_${contentHash}*.json`);
            
            if (embeddingPaths.length > 0) {
              const embeddingData = JSON.parse(fs.readFileSync(embeddingPaths[0], 'utf-8'));
              embedding = embeddingData.embedding;
            } else if (localEmbeddingPaths.length > 0) {
              const embeddingData = JSON.parse(fs.readFileSync(localEmbeddingPaths[0], 'utf-8'));
              embedding = embeddingData.embedding;
            }
          }
        } catch (error) {
          console.error(`Error finding embedding: ${error}`);
        }
        
        // Add to articles list
        const articleInfo = {
          id: articleId,
          title: title,
          content: content,
          summary: summary
        };
        
        // Add embedding if available
        if (embedding) {
          articleInfo.embedding = embedding;
        }
        
        articles.push(articleInfo);
      } catch (error) {
        console.error(`Error loading article ${articlePath}: ${error}`);
      }
    }
    
    // If we have no articles, return error
    if (articles.length === 0) {
      return {
        status: "error",
        message: "Failed to load any articles for analysis"
      };
    }
    
    // Return articles for frontend processing
    return {
      status: "success",
      message: "Retrieved articles for frontend analysis",
      articles: articles,
      calculation_steps: [] // Empty array for compatibility
    };
  } catch (error) {
    console.error(`Error in analyze_interests: ${error}`);
    return {
      status: "error",
      message: `An error occurred during analysis: ${error.message}`
    };
  }
}

// Export functions for use in other modules
module.exports = {
  process_articles_endpoint,
  get_article_endpoint,
  analyze_interests_endpoint
}; 