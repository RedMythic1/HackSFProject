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
    console.log(`Looking for articles in CACHE_DIR: ${CACHE_DIR}`);
    console.log(`Looking for articles in LOCAL_CACHE_DIR: ${LOCAL_CACHE_DIR}`);
    
    const finalArticles = glob.sync(`${CACHE_DIR}/final_article_*.json`);
    const localFinalArticles = glob.sync(`${LOCAL_CACHE_DIR}/final_article_*.json`);
    
    console.log(`Found ${finalArticles.length} articles in CACHE_DIR`);
    console.log(`Found ${localFinalArticles.length} articles in LOCAL_CACHE_DIR`);
    
    const allFinalArticles = [...finalArticles, ...localFinalArticles];
    
    if (allFinalArticles.length === 0) {
      console.warn("No articles found in either cache directory");
      return {
        status: "success",
        message: "No articles found",
        articles: []
      };
    }
    
    // Extract and load article data
    const articleData = [];
    const uniqueTitles = new Set();
    let errorCount = 0;
    
    for (const articlePath of allFinalArticles) {
      try {
        console.log(`Processing article: ${articlePath}`);
        const fileContent = fs.readFileSync(articlePath, 'utf-8');
        
        // Try parsing the article content
        let data;
        try {
          data = JSON.parse(fileContent);
        } catch (parseError) {
          console.error(`Error parsing article JSON ${articlePath}: ${parseError}`);
          errorCount++;
          continue;
        }
        
        // Extract filename
        const filename = path.basename(articlePath);
        
        // Extract timestamp and ID
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
        
        // Create a subject (keywords or summary extract)
        let subject = data.keywords || '';
        if (!subject && content) {
          // Extract a short snippet from content if no keywords
          const contentWithoutTitle = content.replace(lines[0], '').trim();
          const contentLines = contentWithoutTitle.split('\n');
          for (const line of contentLines) {
            if (line && !line.startsWith('#') && line.length > 10) {
              subject = line.length > 100 ? line.substring(0, 100) + '...' : line;
              break;
            }
          }
        }
        
        if (!uniqueTitles.has(title)) {
          uniqueTitles.add(title);
          articleData.push({
            id: articleId,
            title: title,
            subject: subject,
            score: 50, // Default score
            timestamp: data.timestamp || Date.now(),
            link: `https://news.ycombinator.com/item?id=${articleId}`
          });
        }
      } catch (error) {
        console.error(`Error loading article ${articlePath}: ${error}`);
        errorCount++;
      }
    }
    
    // Sort by timestamp (newest first)
    articleData.sort((a, b) => b.timestamp - a.timestamp);
    
    console.log(`Processed ${articleData.length} unique articles with ${errorCount} errors`);
    
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
    console.log(`Getting article details for ID: ${articleId}`);
    
    // Sanitize article_id to ensure it doesn't contain path traversal
    if (!articleId || !articleId.match(/^[a-zA-Z0-9_]+$/)) {
      console.warn(`Invalid article ID format: ${articleId}`);
      return {
        title: "Article Not Found",
        link: "#",
        summary: "The requested article could not be found. The ID format is invalid."
      };
    }
    
    // Lookup the article in the cache
    const articlePath = path.join(CACHE_DIR, `final_article_${articleId}.json`);
    const localArticlePath = path.join(LOCAL_CACHE_DIR, `final_article_${articleId}.json`);
    
    console.log(`Checking for article at: ${articlePath}`);
    console.log(`Checking for article at: ${localArticlePath}`);
    
    let articleData;
    let sourcePath;
    
    if (fs.existsSync(articlePath)) {
      console.log(`Article found in main cache: ${articlePath}`);
      sourcePath = articlePath;
      articleData = JSON.parse(fs.readFileSync(articlePath, 'utf-8'));
    } else if (fs.existsSync(localArticlePath)) {
      console.log(`Article found in local cache: ${localArticlePath}`);
      sourcePath = localArticlePath;
      articleData = JSON.parse(fs.readFileSync(localArticlePath, 'utf-8'));
    } else {
      console.warn(`Article not found for ID: ${articleId}`);
      
      // Try to find by partial ID match in filenames
      console.log("Attempting to find article by partial ID match...");
      const allArticles = [
        ...glob.sync(`${CACHE_DIR}/final_article_*${articleId}*.json`),
        ...glob.sync(`${LOCAL_CACHE_DIR}/final_article_*${articleId}*.json`)
      ];
      
      if (allArticles.length > 0) {
        console.log(`Found ${allArticles.length} potential matches by ID partial`);
        sourcePath = allArticles[0]; // Take the first match
        articleData = JSON.parse(fs.readFileSync(sourcePath, 'utf-8'));
      } else {
        return {
          title: "Article Not Found",
          link: "#",
          summary: "The requested article could not be found in our database."
        };
      }
    }
    
    // Extract title and content
    const content = articleData.content || '';
    let title = 'Unknown Title';
    
    // Extract title from content
    if (content) {
      const lines = content.split('\n');
      if (lines.length > 0) {
        if (lines[0].startsWith('# ')) {
          title = lines[0].substring(2).trim();
        } else {
          title = lines[0].trim();
        }
      }
    }
    
    // Create a summary from the content
    let summary = content;
    if (content && content.length > 200) {
      summary = extractArticleSummary(content);
    }
    
    // Create response object
    const response = {
      title: title,
      link: articleData.url || `https://news.ycombinator.com/item?id=${articleId}`,
      summary: summary
    };
    
    console.log(`Returning article: "${title}"`);
    return response;
  } catch (error) {
    console.error(`Error getting article by ID ${articleId}:`, error);
    return {
      title: "Error Loading Article",
      link: "#",
      summary: `An error occurred while loading the article: ${error.message}`
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