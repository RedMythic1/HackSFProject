// JavaScript implementation of server.py for Vercel deployment
// No Python dependencies should be used

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { glob } = require('glob');
const axios = require('axios');
const { JSDOM } = require('jsdom');
const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const { v4: uuidv4 } = require('uuid');
const WebSocket = require('ws');
const http = require('http');
const { 엄마APITESTER } = require('./test-exports.js');

// Import the Python command handler
let pythonCommand = 'python3'; // Default fallback
try {
  const serverHandler = require('./server_handler.js');
  if (serverHandler.pythonCommand) {
    pythonCommand = serverHandler.pythonCommand;
    console.log(`Using Python command: ${pythonCommand}`);
  }
} catch (error) {
  console.warn(`Could not load pythonCommand from server_handler.js: ${error.message}`);
  console.warn('Defaulting to python3');
}

// Define a log file path (same as in backtest.py for combined logging)
const LIVE_LOG_FILE_PATH = path.join(__dirname, 'backtest_live.log');

// Helper function for server-side logging
function logServerMessage(message, level = 'INFO') {
  const timestamp = new Date().toISOString();
  const logEntry = `[${timestamp}] [SERVER.JS] [${level.toUpperCase()}] ${message}`;
  console.log(logEntry); // Log to server console
  try {
    fs.appendFileSync(LIVE_LOG_FILE_PATH, logEntry + '\n'); // Append to live log file
  } catch (e) {
    console.error(`[SERVER.JS] [ERROR] Failed to append to live log file: ${e.message}`);
  }
}

// Initialize Express app
const app = express();
const port = process.env.PORT || 3000;

// Create HTTP server
const server = http.createServer(app);

// Create WebSocket server
const wss = new WebSocket.Server({ server });

// Middleware setup
app.use(cors());
app.use(express.json());

// Safe import for Vercel Blob Storage
let blobStorage = { put: null, list: null, head: null };
try {
  blobStorage = require('@vercel/blob');
  console.log(`Vercel Blob functions loaded:
    put: ${typeof blobStorage.put === 'function' ? 'Yes' : 'No'}
    list: ${typeof blobStorage.list === 'function' ? 'Yes' : 'No'}
    head: ${typeof blobStorage.head === 'function' ? 'Yes' : 'No'}`);
} catch (error) {
  console.warn(`Failed to load @vercel/blob: ${error.message}`);
  console.warn('Will fall back to in-memory storage only');
}

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

// Define local cache directory
const LOCAL_CACHE_DIR = path.join(__dirname, '../article_cache');
console.log(`Local cache directory configured at: ${LOCAL_CACHE_DIR}`);

// Log the actual paths being used
console.log(`Using storage:
  Using Vercel Blob Storage: ${isVercel ? 'Yes' : 'No'}
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
      const blobMetadata = await blobStorage.head(blobKey);
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
      const { url } = await blobStorage.put(blobKey, JSON.stringify(data, null, 2), { 
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
    
    const { blobs } = await blobStorage.list({ prefix });
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

// NEW consolidated listFiles function
async function listFiles(pattern) {
  if (isVercel && blobStorage.list && blobStorage.head) {
    let prefix = BLOB_PREFIX;
    if (pattern.includes('final_article_')) {
      prefix = BLOB_ARTICLE_PREFIX;
    } else if (pattern.includes('summary_')) {
      prefix = BLOB_SUMMARY_PREFIX;
    } else if (pattern.includes('search_')) {
      prefix = BLOB_SEARCH_PREFIX;
    }

    try {
      console.log(`Listing blobs with prefix: ${prefix} for pattern: ${pattern}`);
      if (!process.env.BLOB_READ_WRITE_TOKEN && prefix !== BLOB_PREFIX) { // Allow general prefix listing without token for broader discovery if needed, but specific prefixes might imply operations requiring tokens.
          //This logic might need refinement based on actual public/private nature of blobs.
          //console.warn("Vercel Blob token not set, listing might be restricted.");
      }
      const { blobs } = await blobStorage.list({ prefix });
      
      // Normalize blob pathnames and filter by the specific pattern
      // Convert glob pattern to regex for basename matching: 'final_article_*.json' -> /^final_article_.*\\.json$/
      const basePattern = pattern.substring(pattern.lastIndexOf('/') + 1); // e.g. final_article_*.json
      const regexPattern = `^${basePattern.replace(/\./g, '\\\\.').replace(/\*/g, '.*')}$`;
      const regex = new RegExp(regexPattern);

      const matchedBlobs = blobs
        .map(blob => { // Normalize pathname
          let currentPathname = blob.pathname;
          if (currentPathname.includes('final_article_final_article_')) {
            currentPathname = currentPathname.replace('final_article_final_article_', 'final_article_');
          }
          // Add other normalizations if necessary
          return { ...blob, pathname: currentPathname };
        })
        .filter(blob => regex.test(path.basename(blob.pathname)))
        .map(blob => blob.pathname); // Return normalized blob pathnames (keys)
      
      console.log(`Found ${matchedBlobs.length} blobs matching pattern ${pattern} (regex: ${regexPattern})`);
      return matchedBlobs;
    } catch (error) {
      console.error(`Error listing blobs with pattern ${pattern}: ${error.message}`);
      if (error.message && error.message.includes("Access denied")) {
        console.error("Vercel Blob: Access denied. Check token and permissions.");
      }
      return [];
    }
  } else {
    // Local file system fallback
    console.log(`Listing local files from ${LOCAL_CACHE_DIR} with pattern ${pattern}`);
    try {
      if (!fs.existsSync(LOCAL_CACHE_DIR)) {
        console.log(`Local cache directory ${LOCAL_CACHE_DIR} does not exist. Creating it.`);
        fs.mkdirSync(LOCAL_CACHE_DIR, { recursive: true });
        return []; // No files if just created
      }
      const filesInDir = await fs.promises.readdir(LOCAL_CACHE_DIR);
      const regexPattern = `^${pattern.replace(/\./g, '\\\\.').replace(/\*/g, '.*')}$`;
      const regex = new RegExp(regexPattern);
      
      const matchedFiles = filesInDir
        .filter(fileName => regex.test(fileName))
        .map(fileName => path.join(LOCAL_CACHE_DIR, fileName)); // Get absolute paths

      console.log(`Found ${matchedFiles.length} files locally in ${LOCAL_CACHE_DIR} matching pattern '${pattern}' (regex: ${regexPattern})`);
      return matchedFiles;
    } catch (error) {
      console.error(`Error listing local files from ${LOCAL_CACHE_DIR} with pattern ${pattern}: ${error.message}`);
      return [];
    }
  }
}

// NEW consolidated readFileContent function
async function readFileContent(filePathOrKey, defaultValue = null) {
  try {
    if (isVercel && blobStorage.head) { // Reading from Vercel Blob
      const blobKey = filePathOrKey; // filePathOrKey is a blob key from listFiles (Vercel path)
      console.log(`Attempting to read from blob storage: ${blobKey}`);
      const blobMetadata = await blobStorage.head(blobKey);
      if (blobMetadata) {
        const response = await fetch(blobMetadata.url); // fetch is global
        if (!response.ok) {
          throw new Error(`Failed to fetch blob content from ${blobMetadata.url}: ${response.status}`);
        }
        const content = await response.text();
        if (blobKey.endsWith('.html')) {
          return content; // Return raw string for HTML
        } else if (blobKey.endsWith('.json')) {
          return JSON.parse(content); // Parse JSON for .json files
        } else {
          console.warn(`Unsupported file type from blob for key ${blobKey}, returning raw text.`);
          return content; // Fallback for other types
        }
      } else {
        console.warn(`Blob not found in Vercel storage: ${blobKey}`);
      }
    } else if (!isVercel && typeof filePathOrKey === 'string' && fs.existsSync(filePathOrKey)) { // Reading from local file system
      const localFilePath = filePathOrKey; // filePathOrKey is an absolute local path
      console.log(`Attempting to read from local file: ${localFilePath}`);
      const content = await fs.promises.readFile(localFilePath, 'utf-8');
      if (localFilePath.endsWith('.html')) {
        return content; // Return raw string for HTML
      } else if (localFilePath.endsWith('.json')) {
        return JSON.parse(content); // Parse JSON for .json files
      } else {
        console.warn(`Unsupported local file type: ${localFilePath}, returning raw text.`);
        return content; // Fallback for other types
      }
    } else {
      if (!isVercel && typeof filePathOrKey === 'string' && !fs.existsSync(filePathOrKey)) {
        console.warn(`Local file not found: ${filePathOrKey}`);
      } else {
        console.warn(`File/Key not found or unsupported environment for: ${filePathOrKey}. isVercel: ${isVercel}`);
      }
    }
  } catch (error) {
    console.error(`Error reading file/key ${filePathOrKey}: ${error.message}`);
  }
  return defaultValue;
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
    
    console.log("Looking for articles...");
    const finalArticleFiles = [
      ...(await listFiles('final_article_*.json')),
      ...(await listFiles('final_article_*.html'))
    ];
    console.log(`Found ${finalArticleFiles.length} final article files.`);
    
    if (finalArticleFiles.length > 0) {
      console.log(`Processing ${finalArticleFiles.length} articles.`);
      const articles = [];
      for (const filePath of finalArticleFiles.slice(0, 20)) { // Limit to 20 articles for performance
        try {
          const articleData = await readFileContent(filePath);
          if (!articleData) {
            console.warn(`Skipping article ${filePath} - could not read or parse file content`);
            continue;
          }
          
          const id = path.basename(filePath).replace('final_article_', '').replace(/\.(json|html)$/, '');
          let title = 'Untitled Article';
          let summary = '';
          let subject = '';
          let articleContentSource = ''; // To hold the content string for parsing

          if (typeof articleData === 'string' && filePath.endsWith('.html')) { // HTML file read as raw string
            articleContentSource = articleData;
            summary = extractHtmlIntroduction(articleContentSource);
            const titleMatch = articleContentSource.match(/<title>(.*?)<\/title>/i);
            if (titleMatch && titleMatch[1]) title = titleMatch[1];
            else title = path.basename(filePath).replace('final_article_', '').replace('.html', '');
            subject = title.split(':')[0]; // Basic subject extraction
          } else if (typeof articleData === 'object' && articleData !== null && filePath.endsWith('.json')) { // JSON file parsed into object
            articleContentSource = articleData.content || ''; // Markdown content expected in .content
            if (articleData.summary) {
              summary = articleData.summary;
            } else if (articleData.intro) {
              summary = articleData.intro;
            } else if (articleContentSource) {
              const paragraphs = articleContentSource.split('\n\n').filter(p => p.trim() && !p.startsWith('#'));
              if (paragraphs.length > 0) summary = paragraphs[0];
            }
            
            if (articleData.title) {
                title = articleData.title;
            } else if (articleContentSource) {
              const contentLines = articleContentSource.split('\n');
              if (contentLines[0] && contentLines[0].startsWith('# ')) {
                title = contentLines[0].substring(2).trim();
              } else {
                title = 'Untitled JSON Article';
              }
            }
             if (articleData.subject){
                subject = articleData.subject;
            } else {
                subject = title.split(':')[0]; // Basic subject extraction
            }
          } else {
            console.warn(`Skipping article ${filePath} - unknown data format or missing .json/.html extension.`);
            continue;
          }
          
          articles.push({
            id,
            title: normalizeArticleTitle(title),
            link: `article/${id}`, // This might need adjustment if IDs are complex
            summary: summary || 'No summary available.',
            subject: subject || 'General',
            score: 0 // Will be scored by frontend or another process
          });
        } catch (error) {
          console.error(`Error processing article ${filePath}:`, error);
        }
      }
      
      articles.sort((a, b) => {
        const idA = (a.id.split('_')[0] || '0');
        const idB = (b.id.split('_')[0] || '0');
        return idB.localeCompare(idA, undefined, {numeric: true, sensitivity: 'base'});
      });
      
      console.log(`Returning ${articles.length} processed articles.`);
      return articles;
    }
    
    console.log("No articles found, returning demo articles");
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
    console.log('get_homepage_articles_endpoint: Retrieving articles');
    
    const articleFiles = [
      ...(await listFiles('final_article_*.json')),
      ...(await listFiles('final_article_*.html'))
    ];
    console.log(`Found ${articleFiles.length} article files for homepage.`);
    
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
        const articleData = await readFileContent(articlePath);
        if (articleData) {
          const articleId = path.basename(articlePath)
            .replace('final_article_', '')
            .replace(/\.(json|html)$/, '');
          
          const processedArticle = processArticleData(articleData, articleId, articlePath.endsWith('.html'));
          if (processedArticle && processedArticle.article) {
            processedArticles.push(processedArticle.article);
          } else {
             console.warn(`Skipping article ${articlePath} - processArticleData did not return a valid article object.`);
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
 * Load an article by ID using exact and partial match logic.
 * @param {string} articleId - Article ID
 * @returns {Object|null} Processed article data or null if not found
 */
async function loadArticleById(articleId) {
  // Try exact match
  const exactMatchPatterns = [`final_article_${articleId}.json`, `final_article_${articleId}.html`];
  let foundArticleData = null;
  let foundFilePath = null;

  for (const pattern of exactMatchPatterns) {
    const files = await listFiles(pattern);
    if (files.length > 0) {
      foundFilePath = files[0];
      foundArticleData = await readFileContent(foundFilePath);
      if (foundArticleData) break;
    }
  }

  // Try partial match if not found
  if (!foundArticleData) {
    const partialMatchPatterns = [`final_article_*${articleId}*.json`, `final_article_*${articleId}*.html`];
    for (const pattern of partialMatchPatterns) {
      const files = await listFiles(pattern);
      if (files.length > 0) {
        const preferredFile = files.find(f => f.endsWith('.json')) || files[0];
        foundFilePath = preferredFile;
        foundArticleData = await readFileContent(foundFilePath);
        if (foundArticleData) break;
      }
    }
  }

  // Return the processed article or null
  if (foundArticleData) {
    return processArticleData(foundArticleData, articleId, foundFilePath.endsWith('.html'));
  }
  return null;
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
    // Use the new reusable loader
    const result = await loadArticleById(articleId);
    if (result) {
      return result;
    }
    // Fallback to demo data if not found
    const demoResult = getDemoArticleDetail(articleId);
    return {
      status: "success",
      article: demoResult
    };
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
 * @param {boolean} isHtml - Flag to indicate if the source was an HTML file
 * @returns {Object} Formatted response with article data
 */
function processArticleData(articleData, articleId, isHtml = false) {
  let title = 'Unknown Title';
  let summary = '';
  let content = ''; // This will hold the primary textual content (Markdown or HTML)
  let sourceLink = `https://news.ycombinator.com/item?id=${articleId}`; // Default link

  if (isHtml && typeof articleData === 'string') { // HTML content as raw string
    content = articleData;
    const titleMatch = content.match(/<title>(.*?)<\/title>/i);
    if (titleMatch && titleMatch[1]) {
      title = titleMatch[1];
    } else {
      // Fallback title from ID if not in HTML
      title = articleId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()); // Basic pretty title
    }
    summary = extractHtmlIntroduction(content); // extractHtmlIntroduction expects HTML string

  } else if (!isHtml && typeof articleData === 'object' && articleData !== null) { // JSON object
    content = articleData.content || ''; // Markdown content
    title = articleData.title || (content.split('\n')[0]?.startsWith('# ') ? content.split('\n')[0].substring(2).trim() : 'Unknown Title');
    summary = articleData.summary || articleData.intro || extractArticleSummary(content); // extractArticleSummary expects markdown string
    sourceLink = articleData.url || sourceLink; // Use URL from JSON if available
  } else {
    console.warn(`processArticleData: articleData is not in expected format for ID ${articleId}. Type: ${typeof articleData}, IsHTML: ${isHtml}`);
     return { status: "error", message: "Invalid article data format" };
  }
  
  return {
    status: "success",
    article: {
      id: articleId,
      title: normalizeArticleTitle(title),
      link: sourceLink,
      summary: summary || "No summary available.",
      content: content, // Full content (Markdown or HTML)
      timestamp: (typeof articleData === 'object' ? articleData.timestamp : null) || Date.now()
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
    
    console.log("Fetching articles from Vercel Blob Storage or local .cache");
    const finalArticleFiles = [
      ...(await listFiles('final_article_*.json')),
      ...(await listFiles('final_article_*.html'))
    ];
    console.log(`Found ${finalArticleFiles.length} final article files for analysis.`);
    
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
        const articleData = await readFileContent(filePath);
        if (!articleData) {
          console.warn(`Skipping article ${filePath} for interest analysis - could not read or parse file content`);
          continue;
        }
        
        const filename = path.basename(filePath);
        const id = filename.replace('final_article_', '').replace('.json', '');
        
        // Extract title and content from article data
        let title = 'Untitled Article';
        let subject = '';
        let contentForAnalysis = '';

        if (typeof articleData === 'string' && filePath.endsWith('.html')) { // HTML file
            contentForAnalysis = articleData; // Use full HTML for keyword spotting, or strip tags first
            const titleMatch = contentForAnalysis.match(/<title>(.*?)<\/title>/i);
            if (titleMatch && titleMatch[1]) title = titleMatch[1];
            else title = path.basename(filePath).replace('final_article_', '').replace('.html', '');
            // For HTML, subject extraction might be simple (from title) or require DOM parsing (not done here)
            subject = title.split(':')[0];
        } else if (typeof articleData === 'object' && articleData !== null && filePath.endsWith('.json')) { // JSON file
            contentForAnalysis = articleData.content || '';
            title = articleData.title || (contentForAnalysis.split('\n')[0]?.startsWith('# ') ? contentForAnalysis.split('\n')[0].substring(2).trim() : 'Untitled JSON Article');
            subject = articleData.subject || title.split(':')[0];
        } else {
            console.warn(`Skipping article ${filePath} for interest analysis - unknown data format.`);
            continue;
        }
        
        const interestTerms = interests.toLowerCase().split(/[,\s]+/).filter(term => term.length > 2);
        // Use a combination of title, subject, and a snippet of content for matching
        const textForScoring = `${title} ${subject} ${contentForAnalysis.substring(0, 500)}`.toLowerCase();
        
        let matchScore = 50; // Base score
        
        // Simple scoring based on term matching
        interestTerms.forEach(term => {
          if (textForScoring.includes(term)) {
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

// Backtest API endpoint
app.post('/api/backtest', async (req, res) => {
  logServerMessage('Received request for /api/backtest');
  try {
    const { strategy } = req.body;
    
    if (!strategy) {
      logServerMessage('Strategy not provided in request body', 'WARN');
      return res.status(400).json({ 
        status: 'error', 
        error: 'Trading strategy is required'
      });
    }
    
    logServerMessage(`Strategy received: ${strategy.substring(0, 100)}${strategy.length > 100 ? '...' : ''}`);
    
    const backtestId = uuidv4().substring(0, 8);
    logServerMessage(`Generated backtest ID: ${backtestId}`);
    
    const scriptPath = path.join(__dirname, 'backtest.py');
    logServerMessage(`Python script path: ${scriptPath}`);
    
    process.env.BACKTEST_USE_TEST_DATASET = '1';
    logServerMessage(`Set BACKTEST_USE_TEST_DATASET=1`);

    // Set the PYTHONPATH to the local python_packages directory
    const pythonPackagesPath = path.join(__dirname, 'python_packages');
    const currentEnv = { ...process.env };
    currentEnv.PYTHONPATH = pythonPackagesPath;
    logServerMessage(`Set PYTHONPATH=${pythonPackagesPath} for Python process`);

    const pythonArgs = [scriptPath, '--json', strategy];
    logServerMessage(`Spawning ${pythonCommand} process with args: ${JSON.stringify(pythonArgs)}`, 'DEBUG');
    
    const pythonProcess = spawn(pythonCommand, pythonArgs, { env: currentEnv });
    
    let outputData = '';
    let errorData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      const dataStr = data.toString();
      outputData += dataStr;
      logServerMessage(`Python stdout (${backtestId}): ${dataStr.substring(0, 200)}${dataStr.length > 200 ? '...' : ''}`, 'DEBUG');
    });
    
    pythonProcess.stderr.on('data', (data) => {
      const dataStr = data.toString();
      errorData += dataStr;
      logServerMessage(`Python stderr (${backtestId}): ${dataStr.substring(0, 200)}${dataStr.length > 200 ? '...' : ''}`, 'ERROR');
    });
    
    pythonProcess.on('close', (code) => {
      logServerMessage(`Python process (${backtestId}) exited with code ${code}`);
      
      if (code !== 0) {
        logServerMessage(`Backtest failed with code ${code}. Error data: ${errorData}`, 'ERROR');
        return res.status(500).json({
          status: 'error',
          error: `Backtest failed with error: ${errorData || 'Unknown error'}`
        });
      }
      
      try {
        logServerMessage(`Attempting to parse JSON from Python output. Length: ${outputData.length}`, 'DEBUG');
        const jsonMatch = outputData.match(/\{.*\}/s);
        if (!jsonMatch) {
          logServerMessage(`No valid JSON found in Python output. Output preview: ${outputData.substring(0, 500)}${outputData.length > 500 ? '...' : ''}`, 'ERROR');
          return res.status(500).json({
            status: 'error',
            error: 'Could not parse backtest results from Python script'
          });
        }
        
        const resultJsonString = jsonMatch[0];
        logServerMessage(`Successfully extracted JSON string from Python output. Length: ${resultJsonString.length}`, 'DEBUG');
        const resultJson = JSON.parse(resultJsonString);
        logServerMessage('Successfully parsed JSON from Python output.', 'DEBUG');
        
        resultJson.status = resultJson.status || 'success';
        
        logServerMessage(`Sending response to client for backtest ${backtestId}: ${JSON.stringify(resultJson).substring(0,200)}...`, "DEBUG");
        res.json(resultJson);
      } catch (parseError) {
        logServerMessage(`Error parsing Python output JSON: ${parseError.message}. Raw output preview: ${outputData.substring(0, 500)}${outputData.length > 500 ? '...' : ''}`, 'ERROR');
        res.status(500).json({
          status: 'error',
          error: 'Failed to parse backtest results: ' + parseError.message
        });
      }
    });
    
    pythonProcess.on('error', (err) => {
      logServerMessage(`Failed to start Python process (${backtestId}): ${err.message}`, 'ERROR');
      res.status(500).json({
        status: 'error',
        error: `Failed to start backtesting process: ${err.message}`
      });
    });

  } catch (error) {
    logServerMessage(`Critical error in /api/backtest endpoint: ${error.message}\nStack: ${error.stack}`, 'ERROR');
    res.status(500).json({
      status: 'error',
      error: error.message || 'An unexpected error occurred in the backtest endpoint'
    });
  }
});

// New endpoint to get backtest logs
app.get('/api/backtest-logs', (req, res) => {
  logServerMessage('Received request for /api/backtest-logs');
  const logFilePath = path.join(__dirname, 'backtest_live.log');
  logServerMessage(`Attempting to read log file: ${logFilePath}`, 'DEBUG');
  fs.readFile(logFilePath, 'utf8', (err, data) => {
    if (err) {
      logServerMessage(`Error reading log file: ${err.message}`, 'ERROR');
      if (err.code === 'ENOENT') {
        logServerMessage('Log file not found (ENOENT). Sending appropriate message to client.', 'WARN');
        return res.status(200).send('Log file not yet created or backtest not run yet.');
      }
      return res.status(500).send('Error reading log file');
    }
    logServerMessage(`Successfully read log file. Length: ${data.length}. Sending to client.`, 'DEBUG');
    res.setHeader('Content-Type', 'text/plain');
    res.send(data);
  });
});

// Setup API routes
app.get('/api/articles', async (req, res) => {
  logServerMessage('Received request for /api/articles', 'INFO');
  try {
    const { interests } = req.query;
    const result = await process_articles_endpoint(req.query);
    res.json(result);
  } catch (error) {
    console.error('Error in /api/articles:', error);
    res.status(500).json({ error: 'Failed to process articles', details: error.message });
  }
});

app.get('/api/article/:id', async (req, res) => {
  const { id } = req.params;
  logServerMessage(`Received request for /api/article/${id}`, 'INFO');
  try {
    if (!id) {
      return res.status(400).json({ error: 'Article ID is required' });
    }
    
    const result = await get_article_endpoint(id);
    res.json(result);
  } catch (error) {
    console.error('Error in /api/article/:id:', error);
    res.status(500).json({ error: 'Failed to get article', details: error.message });
  }
});

app.post('/api/analyze-interests', async (req, res) => {
  logServerMessage('Received request for /api/analyze-interests', 'INFO');
  try {
    const { interests } = req.body;
    if (!interests) {
      return res.status(400).json({ error: 'Interests parameter is required' });
    }
    
    const result = await analyze_interests_endpoint(interests);
    res.json(result);
  } catch (error) {
    console.error('Error in /api/analyze-interests:', error);
    res.status(500).json({ error: 'Failed to analyze interests', details: error.message });
  }
});

app.post('/api/log', (req, res) => {
  // This endpoint is called by the frontend logger, avoid recursive logging here
  // or ensure it doesn't write to the same live log file if it causes issues.
  // For now, it just console.logs as per its original design.
  try {
    const { message, level = 'info', source = 'unknown' } = req.body;
    if (!message) {
      return res.status(400).json({ status: 'error', message: 'Missing log message' });
    }
    console.log(`[CLIENT LOG] [${source}] [${level.toUpperCase()}] ${message}`); // Distinguish client logs
    res.json({ status: 'success' });
  } catch (error) {
    console.error('[SERVER.JS] [ERROR] Error in /api/log endpoint:', error);
    res.status(500).json({ error: 'Failed to log message', details: error.message });
  }
});

// Serve static files from the 'src' directory
app.use(express.static(path.join(__dirname, '../src/')));

// Set up port configuration
const PORT = process.env.PORT || 3000;

// Start the server if this is not being imported
if (require.main === module) {
  server.listen(PORT, () => { // Make sure to use server.listen for WebSocket compatibility
    logServerMessage(`Server running on port ${PORT}. Main module.`);
    logServerMessage(`API is available at http://localhost:${PORT}/api`);
  });
} else {
  logServerMessage('Server.js loaded as a module, not starting listener independently.', 'DEBUG');
}

// Export for Vercel.js - ONLY THE APP
module.exports = app; 