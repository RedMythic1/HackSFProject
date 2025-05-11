// JavaScript implementation of server.py for Vercel deployment
// No Python dependencies should be used

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { glob } = require('glob');
const axios = require('axios');
const { JSDOM } = require('jsdom');
const { put, list, head } = require('@vercel/blob');

// Verify that the imported functions exist
console.log(`Vercel Blob functions loaded:
  put: ${typeof put === 'function' ? 'Yes' : 'No'}
  list: ${typeof list === 'function' ? 'Yes' : 'No'}
  head: ${typeof head === 'function' ? 'Yes' : 'No'}`);

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
console.log(`Using storage:
  Using Vercel Blob Storage: Yes
  Current working directory: ${process.cwd()}`);

// In-memory cache for all environments as fallback
const MEMORY_CACHE = {
  articles: [],
  articleDetails: {},
  summaries: {}
};

// --- Helper Functions ---

// Safe file read function with blob storage
async function safeReadFile(filePath, defaultValue = null) {
  try {
    // Check if the file path is valid
    if (!filePath || typeof filePath !== 'string') {
      console.warn(`Invalid file path: ${filePath}`);
      return defaultValue;
    }
    
    // Convert file path to blob key
    const blobKey = getBlobKeyFromPath(filePath);
    
    console.log(`Attempting to read from blob storage: ${blobKey}`);
    try {
      // First check if the blob exists
      const blobMetadata = await head(blobKey);
      if (blobMetadata) {
        // If it exists, fetch its content
        const response = await fetch(blobMetadata.url);
        if (!response.ok) {
          throw new Error(`Failed to fetch blob content: ${response.status}`);
        }
        const content = await response.text();
        return JSON.parse(content);
      } else {
        console.warn(`Blob not found: ${blobKey}`);
      }
    } catch (blobError) {
      console.warn(`Error reading from blob storage: ${blobError.message}`);
    }
  } catch (error) {
    console.warn(`Error reading file ${filePath}: ${error.message}`);
  }
  return defaultValue;
}

// Safe file write function with blob storage
async function safeWriteFile(filePath, data) {
  try {
    // Convert file path to blob key
    const blobKey = getBlobKeyFromPath(filePath);
    
    console.log(`Writing to blob storage: ${blobKey}`);
    try {
      const { url } = await put(blobKey, JSON.stringify(data, null, 2), { 
        access: 'public',
        contentType: 'application/json'
      });
      console.log(`Successfully wrote to blob storage: ${url}`);
      return true;
    } catch (blobError) {
      console.warn(`Error writing to blob storage: ${blobError.message}`);
      return false;
    }
  } catch (error) {
    console.warn(`Error writing file ${filePath}: ${error.message}`);
  }
  return false;
}

// Helper function to convert file path to blob key
function getBlobKeyFromPath(filePath) {
  if (!filePath || typeof filePath !== 'string') {
    console.warn(`Invalid file path: ${filePath}`);
    return null;
  }

  const basename = path.basename(filePath);
  
  // Fix duplicated 'final_article_' prefix pattern
  if (basename.startsWith('final_article_final_article_')) {
    // Extract the actual ID after the duplicate prefix
    const id = basename.replace('final_article_final_article_', '');
    return BLOB_ARTICLE_PREFIX + id;
  } else if (basename.startsWith('final_article_')) {
    return BLOB_ARTICLE_PREFIX + basename.replace('final_article_', '');
  } else if (basename.startsWith('summary_')) {
    return BLOB_SUMMARY_PREFIX + basename.replace('summary_', '');
  } else if (basename.startsWith('search_')) {
    return BLOB_SEARCH_PREFIX + basename.replace('search_', '');
  } else {
    return BLOB_PREFIX + basename;
  }
}

// Helper function to get virtual path from blob key
function getVirtualPathFromBlobKey(blobKey) {
  if (!blobKey || typeof blobKey !== 'string') {
    console.warn(`Invalid blob key: ${blobKey}`);
    return null;
  }

  const basename = path.basename(blobKey);
  
  if (blobKey.startsWith(BLOB_ARTICLE_PREFIX)) {
    // Create a clean filename without duplicated prefixes
    const cleanName = basename.replace(/^final_article_/, '');
    return `/tmp/final_article_${cleanName}`;
  } else if (blobKey.startsWith(BLOB_SUMMARY_PREFIX)) {
    return `/tmp/summary_${basename}`;
  } else if (blobKey.startsWith(BLOB_SEARCH_PREFIX)) {
    return `/tmp/search_${basename}`;
  } else {
    return `/tmp/${basename}`;
  }
}

// List files from blob storage
async function listBlobFiles(pattern) {
  let prefix = BLOB_PREFIX;
  
  if (pattern.includes('final_article_')) {
    prefix = BLOB_ARTICLE_PREFIX;
  } else if (pattern.includes('summary_')) {
    prefix = BLOB_SUMMARY_PREFIX;
  } else if (pattern.includes('search_')) {
    prefix = BLOB_SEARCH_PREFIX;
  }
  
  try {
    console.log(`Listing blobs with prefix: ${prefix}`);
    
    // Check if token is available
    if (!process.env.BLOB_READ_WRITE_TOKEN) {
      console.error("Error listing blobs: Vercel Blob token not set in environment variables");
      return [];
    }
    
    const { blobs } = await list({ prefix });
    console.log(`Found ${blobs.length} blobs with prefix ${prefix}`);
    
    // Process each blob to handle duplicate prefixes
    const processedBlobs = blobs.map(blob => {
      // Check for duplicate prefixes in the pathname
      if (blob.pathname.includes('final_article_final_article_')) {
        console.log(`Found blob with duplicated prefix: ${blob.pathname}`);
        // Create a normalized pathname without the duplicate prefix
        const normalizedPathname = blob.pathname.replace('final_article_final_article_', 'final_article_');
        console.log(`Normalized pathname: ${normalizedPathname}`);
        return {
          ...blob,
          pathname: normalizedPathname
        };
      }
      return blob;
    });
    
    // Create virtualized paths to maintain compatibility with existing code
    return processedBlobs.map(blob => getVirtualPathFromBlobKey(blob.pathname));
  } catch (error) {
    console.error(`Error listing blobs: ${error.message}`);
    
    // Check for specific error messages
    if (error.message && error.message.includes("Access denied")) {
      console.error("Vercel Blob: Access denied error. Check that your token is valid and has proper permissions.");
    }
    
    return [];
  }
}

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

// Helper function to extract introduction from HTML content
function extractHtmlIntroduction(htmlContent) {
  try {
    // Look for the Introduction marker
    const introMarker = '#<h1>Introduction';
    const startIdx = htmlContent.indexOf(introMarker);
    if (startIdx === -1) return '';
    // Find the end of the introduction section (next #<h1> or end of string)
    const afterIntro = htmlContent.slice(startIdx + introMarker.length);
    const endIdx = afterIntro.indexOf('#<h1>');
    let introSection = endIdx !== -1 ? afterIntro.slice(0, endIdx) : afterIntro;
    // Remove HTML tags and decode entities
    introSection = introSection.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
    return introSection;
  } catch (e) {
    return '';
  }
}

// --- Demo Data Generator ---

/**
 * Create demo article data when no articles are available
 * @returns {Array} Array of demo article objects
 */
function generateDemoArticles() {
  console.log('Generating demo articles');
  
  // Sample demo articles
  return [
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
      title: 'Blockchain Technology Explained',
      subject: 'Understanding the fundamentals of blockchain technology and its potential beyond cryptocurrencies.',
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
}

/**
 * Get demo article detail by ID
 * @param {string} id - Article ID
 * @returns {Object} Article detail object
 */
function getDemoArticleDetail(id) {
  const demoDetails = {
    'demo1': {
      title: 'Introduction to Machine Learning',
      link: 'https://example.com/article/1',
      summary: 'Machine Learning is a rapidly growing field at the intersection of computer science and statistics. It focuses on developing algorithms that can learn from and make predictions on data. This article covers the fundamental concepts of Machine Learning, including supervised and unsupervised learning, regression, classification, and neural networks. We also explore real-world applications in fields like healthcare, finance, and transportation. Understanding these basics is essential for anyone looking to start a career in data science or AI development.'
    },
    'demo2': {
      title: 'The Future of Web Development',
      link: 'https://example.com/article/2',
      summary: 'Web development is constantly evolving with new technologies and approaches. This article examines the latest trends shaping the future of web development, including WebAssembly for high-performance code, Progressive Web Apps combining the best of web and mobile apps, and JAMstack architecture for faster and more secure websites. We also discuss the impact of AI-driven development tools, the growing importance of accessibility, and how edge computing is changing where web applications run. These developments are creating new opportunities for developers while addressing the increasing demands of modern web users.'
    },
    'demo3': {
      title: 'Blockchain Technology Explained',
      link: 'https://example.com/article/3',
      summary: 'Blockchain is a distributed ledger technology that enables secure, transparent, and immutable record-keeping without central authorities. This article breaks down how blockchain works, explaining concepts like consensus mechanisms, smart contracts, and cryptographic hashing. While most famous for powering cryptocurrencies like Bitcoin, blockchain has potential applications in supply chain management, voting systems, healthcare records, and digital identity verification. Understanding the fundamentals of blockchain helps separate the technology\'s genuine potential from market hype.'
    },
    'demo4': {
      title: 'Artificial Intelligence Ethics',
      link: 'https://example.com/article/4',
      summary: 'As AI systems become more powerful and widespread, ethical considerations become increasingly important. This article explores the key ethical challenges in AI development and deployment, including bias and fairness in algorithmic decision-making, privacy concerns with data collection, transparency and explainability of AI systems, and the potential impact on employment. We also discuss emerging frameworks for responsible AI development and the role of regulations in ensuring that AI benefits humanity. Addressing these ethical questions is essential for building AI systems that people can trust.'
    },
    'demo5': {
      title: 'Cloud Computing Fundamentals',
      link: 'https://example.com/article/5',
      summary: 'Cloud computing has transformed how businesses manage their IT resources and deliver digital services. This article provides an overview of cloud computing concepts, including the differences between IaaS, PaaS, and SaaS service models, and public, private, and hybrid deployment models. We cover key benefits like scalability, cost-efficiency, and global reach, along with important considerations regarding security, compliance, and vendor lock-in. For organizations considering moving to the cloud, we outline best practices for cloud migration and optimization.'
    },
    'demo6': {
      title: 'Data Science for Beginners',
      link: 'https://example.com/article/6',
      summary: 'Data science combines statistical analysis, computer science, and domain expertise to extract meaningful insights from data. This beginner-friendly guide introduces the essential components of data science, including data collection and cleaning, exploratory data analysis, statistical modeling, and communication of results. We review popular tools and languages like Python, R, SQL, and Jupyter Notebooks, and outline a learning path for aspiring data scientists. The article also discusses common challenges beginners face and strategies to overcome them while building practical skills through projects.'
    }
  };
  
  return demoDetails[id] || {
    title: 'Sample Article',
    link: '#',
    summary: 'This is a placeholder article summary. The requested article could not be found.'
  };
}

// --- API Endpoints Implementation ---

/**
 * Process articles endpoint
 * @param {Object} query - Query parameters
 * @returns {Array} Array of articles
 */
async function process_articles_endpoint(query) {
  try {
    console.log("Processing articles endpoint with query:", query);
    
    // First try to get articles from blob storage
    console.log("Looking for cached articles in blob storage");
    const finalArticleFiles = [
      ...(await listBlobFiles('final_article_*.json')),
      ...(await listBlobFiles('final_article_*.html'))
    ];
    console.log(`Found ${finalArticleFiles.length} final article files in blob storage`);
    
    if (finalArticleFiles.length > 0) {
      // We have cached articles, process them
      console.log(`Processing ${finalArticleFiles.length} articles from blob storage`);
      const articles = [];
      for (const filePath of finalArticleFiles.slice(0, 20)) { // Limit to 20 articles for performance
        try {
          // Load article data from blob storage
          const articleData = await safeReadFile(filePath);
          if (!articleData) {
            console.warn(`Skipping article ${filePath} - could not read file`);
            continue;
          }
          
          const filename = path.basename(filePath);
          const id = filename.replace('final_article_', '').replace('.json', '');
          
          // Extract title and summary from content
          let title = 'Untitled Article';
          let summary = '';
          let subject = '';
          
          if (filePath.endsWith('.html')) {
            // HTML article: extract introduction as summary
            summary = extractHtmlIntroduction(articleData.content || articleData.html || '');
            // Try to extract title from HTML content (fallback to filename)
            const match = (articleData.content || '').match(/<title>(.*?)<\/title>/i);
            if (match) {
              title = match[1];
            } else {
              title = filename.replace('final_article_', '').replace('.html', '');
            }
            subject = title.split(':')[0];
          } else {
            // JSON/Markdown: use summary or intro if available
            if (articleData.summary) {
              summary = articleData.summary;
            } else if (articleData.intro) {
              summary = articleData.intro;
            } else if (articleData.content) {
              // fallback to first paragraph or content parsing
              const paragraphs = articleData.content.split('\n\n').filter(p => p.trim() && !p.startsWith('#'));
              if (paragraphs.length > 0) {
                summary = paragraphs[0];
                if (paragraphs.length > 1) {
                  summary += '\n\n' + paragraphs[1];
                }
              }
            }
            // Extract title from content
            if (articleData.content) {
              const contentLines = articleData.content.split('\n');
              if (contentLines[0] && contentLines[0].startsWith('# ')) {
                title = contentLines[0].substring(2);
              }
            }
            // Try to extract subject from content
            if (articleData.content) {
              const subjectMatch = articleData.content.match(/##\s+(.+?)\n/);
              if (subjectMatch) {
                subject = subjectMatch[1].replace(/subject|topic|about|:/gi, '').trim();
              } else {
                subject = title.split(':')[0];
              }
            }
          }
          
          articles.push({
            id,
            title,
            link: `article/${id}`,
            summary: summary || 'No summary available',
            subject: subject || 'General',
            score: 0 // Will be scored by frontend
          });
        } catch (error) {
          console.error(`Error processing article ${filePath}:`, error);
        }
      }
      
      // Sort by ID (most recent first, assuming ID includes timestamp)
      articles.sort((a, b) => {
        const idA = a.id.split('_')[0];
        const idB = b.id.split('_')[0];
        return idB.localeCompare(idA);
      });
      
      console.log(`Returning ${articles.length} processed articles from blob storage`);
      return articles;
    }
    
    // If no blob storage articles, use demo data
    console.log("No articles in blob storage, returning demo articles");
    return generateDemoArticles();
  } catch (error) {
    console.error(`Error in process_articles_endpoint: ${error}`);
    return generateDemoArticles();
  }
}

/**
 * Get articles and process them for the homepage
 * @returns {Object} Processed articles or error
 */
async function get_homepage_articles_endpoint() {
  try {
    console.log('get_homepage_articles_endpoint: Retrieving articles from database');
    
    // Use listBlobFiles to get articles from Vercel Blob Storage
    const articleFiles = [
      ...(await listBlobFiles('final_article_*.json')),
      ...(await listBlobFiles('final_article_*.html'))
    ];
    console.log(`Found ${articleFiles.length} article files: homepage test`);
    
    if (articleFiles.length === 0) {
      // No articles found in blob storage, fall back to demo articles
      console.log('No articles found in storage, using demo articles');
      return {
        status: 'success',
        method: 'demo',
        articles: generateDemoArticles()
      };
    }

    // Process the articles
    const processedArticles = [];
    const skippedArticles = [];
    
    for (const articlePath of articleFiles) {
      try {
        const articleData = await safeReadFile(articlePath);
        if (articleData) {
          const articleId = path.basename(articlePath)
            .replace('final_article_', '')
            .replace('.json', '');
          
          const processedArticle = processArticleData(articleData, articleId);
          if (processedArticle) {
            processedArticles.push(processedArticle);
          }
        } else {
          console.warn(`Skipping article ${articlePath} - could not read file`);
          skippedArticles.push(articlePath);
        }
      } catch (articleError) {
        console.error(`Error processing article ${articlePath}: ${articleError.message}`);
        skippedArticles.push(articlePath);
      }
    }

    // Sort articles by score (descending)
    processedArticles.sort((a, b) => b.score - a.score);

    // Include skipped articles info in the response
    return {
      status: 'success',
      method: 'database',
      article_count: processedArticles.length,
      skipped_count: skippedArticles.length,
      articles: processedArticles
    };
  } catch (error) {
    console.error('Error retrieving articles:', error);
    return {
      status: 'error',
      message: `Failed to retrieve articles: ${error.message}`,
      method: 'demo',
      articles: generateDemoArticles()
    };
  }
}

/**
 * Get article endpoint
 * @param {string} articleId - Article ID
 * @returns {Object} Response with article data
 */
async function get_article_endpoint(articleId) {
  try {
    if (!articleId) {
      return { 
        status: 'error', 
        message: 'Article ID is required' 
      };
    }
    
    console.log(`Getting article details for: ${articleId}`);
    
    // Search for this article in blob storage
    const blobFiles = [
      ...(await listBlobFiles(`final_article_*${articleId}*.json`)),
      ...(await listBlobFiles(`final_article_*${articleId}*.html`))
    ];
    console.log(`Found ${blobFiles.length} matching blob files for article ID: ${articleId}`);
    
    if (blobFiles.length === 0) {
      // If not found directly, try listing all articles and finding by ID
      const allBlobFiles = [
        ...(await listBlobFiles('final_article_*.json')),
        ...(await listBlobFiles('final_article_*.html'))
      ];
      console.log(`Searching through ${allBlobFiles.length} total articles for ID: ${articleId}`);
      
      for (const filePath of allBlobFiles) {
        const filename = path.basename(filePath);
        // Check if the filename contains this ID
        if (filename.includes(articleId)) {
          blobFiles.push(filePath);
          console.log(`Found matching article in broader search: ${filename}`);
          break;
        }
      }
    }
    
    // If we found a matching file, read and process it
    if (blobFiles.length > 0) {
      // Use the first matching file
      const articlePath = blobFiles[0];
      console.log(`Reading article data from: ${articlePath}`);
      
      // Read blob data
      const articleData = await safeReadFile(articlePath);
      
      if (!articleData) {
        console.error(`Failed to read article data from: ${articlePath}`);
        return { 
          status: 'error', 
          message: 'Failed to read article data' 
        };
      }
      
      // Process the article data
      return processArticleData(articleData, articleId);
    }
    
    // If no article found in blob storage, return demo data
    console.log(`Article not found in storage, using demo data for: ${articleId}`);
    return getDemoArticleDetail(articleId);
  } catch (error) {
    console.error(`Error getting article ${articleId}:`, error);
    return { 
      status: 'error', 
      message: error.message || 'Internal server error' 
    };
  }
}

/**
 * Helper function to process article data and create a consistent response
 * @param {Object} articleData - Raw article data from file
 * @param {string} articleId - Article ID
 * @returns {Object} Formatted response with article data
 */
function processArticleData(articleData, articleId) {
  // Extract content and create title/summary
  const content = articleData.content || '';
  let title = 'Unknown Title';
  let summary = '';
  
  // Extract title from content
  if (content) {
    const lines = content.split('\n');
    if (lines.length > 0) {
      if (lines[0].startsWith('# ')) {
        title = lines[0].substring(2);
      } else {
        title = lines[0].trim();
      }
    }
    
    // Create a summary from the content
    summary = extractArticleSummary(content);
  }
  
  // Build response object
  return {
    status: "success",
    article: {
      id: articleId,
      title: title,
      link: articleData.url || `https://news.ycombinator.com/item?id=${articleId}`,
      summary: summary,
      content: content,
      timestamp: articleData.timestamp || Date.now()
    }
  };
}

/**
 * Analyze interests endpoint implementation
 * @param {string} interests - Comma-separated user interests
 * @returns {Object} Response object with recommendations
 */
async function analyze_interests_endpoint(interests) {
  try {
    console.log(`Analyzing interests: ${interests}`);
    
    if (!interests) {
      console.error("No interests provided for analysis");
      return { 
        status: 'error', 
        message: 'No interests provided' 
      };
    }
    
    // Get all final articles from blob storage
    console.log("Fetching articles from Vercel Blob Storage");
    const finalArticleFiles = [
      ...(await listBlobFiles('final_article_*.json')),
      ...(await listBlobFiles('final_article_*.html'))
    ];
    console.log(`Found ${finalArticleFiles.length} final article files in blob storage`);
    
    if (finalArticleFiles.length === 0) {
      return {
        status: 'success',
        articles: []
      };
    }
    
    // Process each article
    const articles = [];
    for (const filePath of finalArticleFiles) {
      try {
        // Load article data from blob storage
        const articleData = await safeReadFile(filePath);
        if (!articleData) {
          console.warn(`Skipping article ${filePath} - could not read file`);
          continue;
        }
        
        const filename = path.basename(filePath);
        const id = filename.replace('final_article_', '').replace('.json', '');
        
        // Extract title and content from article data
        let title = 'Untitled Article';
        let subject = '';
        
        if (articleData.content) {
          // Extract title from markdown content (the first line starting with # )
          const contentLines = articleData.content.split('\n');
          if (contentLines[0] && contentLines[0].startsWith('# ')) {
            title = contentLines[0].substring(2);
          }
          
          // Try to extract a subject by looking for certain patterns in the content
          // First, try to find a line starting with "## " that mentions "subject" or "topic"
          const subjectLinePattern = contentLines.find(line => 
            (line.startsWith('## ') && 
             (line.toLowerCase().includes('subject') || line.toLowerCase().includes('topic')))
          );
          
          if (subjectLinePattern) {
            subject = subjectLinePattern.replace(/^## /, '');
          } else {
            // If no explicit subject/topic heading, try to find a sentence mentioning "subject is" or "topic is"
            const subjectMentionPattern = /(?:subject|topic)\s+(?:is|was|about)?\s+["']?([^"'.]+)["']?/i;
            const fullContent = articleData.content;
            const subjectMatch = fullContent.match(subjectMentionPattern);
            
            if (subjectMatch && subjectMatch[1]) {
              subject = subjectMatch[1].trim();
            } else {
              // Fallback: just use the first part of the title
              subject = title.split(':')[0];
            }
          }
        }
        
        // Calculate a simple score based on text matching between interests and subject/title
        // This is a basic fallback since we no longer use Python for vectorization
        const interestTerms = interests.toLowerCase().split(/[,\s]+/).filter(term => term.length > 2);
        const articleText = `${title} ${subject}`.toLowerCase();
        
        let matchScore = 50; // Base score
        
        // Simple scoring based on term matching
        interestTerms.forEach(term => {
          if (articleText.includes(term)) {
            matchScore += 10;
          }
        });
        
        // Cap at 100
        matchScore = Math.min(100, matchScore);
        
        articles.push({
          id,
          title,
          subject,
          score: matchScore,
          filename
        });
      } catch (error) {
        console.error(`Error processing article ${filePath}:`, error);
      }
    }
    
    // Sort by score (highest first)
    articles.sort((a, b) => b.score - a.score);
    
    console.log(`Processed ${articles.length} articles, returning top scores`);
    
    return {
      status: 'success',
      articles
    };
  } catch (error) {
    console.error('Error analyzing interests:', error);
    return { 
      status: 'error', 
      message: error.message 
    };
  }
}

// Export functions for the API
module.exports = {
  get_homepage_articles_endpoint,
  get_article_endpoint,
  process_articles_endpoint,
  analyze_interests_endpoint
}; 