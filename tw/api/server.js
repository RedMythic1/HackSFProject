// JavaScript implementation of server.py for Vercel deployment
// No Python dependencies should be used

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { glob } = require('glob');
const axios = require('axios');
const { JSDOM } = require('jsdom');

// Environment detection
const isVercel = process.env.VERCEL === '1';
console.log(`Running in ${isVercel ? 'Vercel' : 'local'} environment`);

// Constants - always use /tmp directory for Vercel
const CACHE_DIR = isVercel ? '/tmp/cache' : path.join(process.cwd(), '.vercel', 'cache');
const LOCAL_CACHE_DIR = isVercel ? '/tmp/local_cache' : path.join(process.cwd(), 'local_cache');

// Log the actual paths being used
console.log(`Using cache directories:
  CACHE_DIR: ${CACHE_DIR}
  LOCAL_CACHE_DIR: ${LOCAL_CACHE_DIR}
  Current working directory: ${process.cwd()}`);

// In-memory cache for all environments as fallback
const MEMORY_CACHE = {
  articles: [],
  articleDetails: {},
  summaries: {}
};

// Ensure cache directories exist, but handle failure gracefully
let cacheDirsAvailable = false;
try {
  if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
  }
  if (!fs.existsSync(LOCAL_CACHE_DIR)) {
    fs.mkdirSync(LOCAL_CACHE_DIR, { recursive: true });
  }
  cacheDirsAvailable = true;
  console.log(`Cache directories initialized: 
    CACHE_DIR: ${CACHE_DIR}
    LOCAL_CACHE_DIR: ${LOCAL_CACHE_DIR}`);
} catch (error) {
  console.warn(`Cannot create cache directories (this is normal in serverless environments): ${error.message}`);
  console.log('Using in-memory cache fallback');
}

// --- Helper Functions ---

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
 * Process articles endpoint implementation
 * @param {Object} query - Query parameters
 * @returns {Object} Response object with articles
 */
function process_articles_endpoint(query) {
  try {
    // Check if we have cached articles in memory
    if (MEMORY_CACHE.articles.length > 0) {
      console.log(`Returning ${MEMORY_CACHE.articles.length} articles from memory cache`);
      return MEMORY_CACHE.articles;
    }
    
    // Try to find articles in the filesystem
    console.log(`Looking for articles in CACHE_DIR and LOCAL_CACHE_DIR`);
    
    let finalArticles = [];
    let localFinalArticles = [];
    
    try {
      // Use glob pattern only if directory exists, otherwise use empty array
      if (cacheDirsAvailable && fs.existsSync(CACHE_DIR)) {
        finalArticles = glob.sync(`${CACHE_DIR}/final_article_*.json`);
        console.log(`Found ${finalArticles.length} articles in CACHE_DIR`);
      } 
      
      if (cacheDirsAvailable && fs.existsSync(LOCAL_CACHE_DIR)) {
        // Use safe directory reading that won't crash on Vercel
        try {
          const files = fs.readdirSync(LOCAL_CACHE_DIR);
          const articleFiles = files.filter(file => file.startsWith('final_article_') && file.endsWith('.json'));
          localFinalArticles = articleFiles.map(file => path.join(LOCAL_CACHE_DIR, file));
          console.log(`Found ${localFinalArticles.length} articles in LOCAL_CACHE_DIR`);
        } catch (dirError) {
          console.warn(`Error reading from LOCAL_CACHE_DIR: ${dirError.message}`);
        }
      }
    } catch (error) {
      console.warn(`Error searching for articles: ${error.message}`);
    }
    
    const allFinalArticles = [...finalArticles, ...localFinalArticles];
    console.log(`Total articles found: ${allFinalArticles.length}`);
    
    // If no articles found in filesystem, return demo data
    if (allFinalArticles.length === 0) {
      console.log("No articles found in either cache directory, using demo data");
      const demoArticles = generateDemoArticles();
      MEMORY_CACHE.articles = demoArticles; // Cache for future requests
      return demoArticles;
    }
    
    // Extract and load article data
    const articleData = [];
    const uniqueTitles = new Set();
    
    for (const articlePath of allFinalArticles) {
      try {
        // Use our safe file reading function
        const data = safeReadFile(articlePath, null);
        if (!data) {
          continue; // Skip if file couldn't be read
        }
        
        // Extract filename and article ID
        const filename = path.basename(articlePath);
        const articleId = filename.replace('final_article_', '').replace('.json', '');
        
        // Get the first line as the title
        const content = data.content || '';
        if (!content) {
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
            score: 0, // Scores will be calculated on the frontend
            timestamp: data.timestamp || Date.now(),
            link: `https://news.ycombinator.com/item?id=${articleId}`
          });
        }
      } catch (error) {
        console.error(`Error loading article ${articlePath}: ${error}`);
      }
    }
    
    // Sort by timestamp (newest first)
    articleData.sort((a, b) => b.timestamp - a.timestamp);
    
    console.log(`Processed ${articleData.length} unique articles`);
    
    // Cache in memory for future requests
    MEMORY_CACHE.articles = articleData;
    
    return articleData;
  } catch (error) {
    console.error(`Error getting articles: ${error}`);
    
    // Return demo data as fallback
    const demoArticles = generateDemoArticles();
    MEMORY_CACHE.articles = demoArticles;
    return demoArticles;
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
    
    // Check if it's a demo article ID
    if (articleId.startsWith('demo')) {
      const demoArticle = getDemoArticleDetail(articleId);
      console.log(`Returning demo article: "${demoArticle.title}"`);
      return {
        status: 'success',
        article: demoArticle
      };
    }
    
    // Check memory cache first
    if (MEMORY_CACHE.articleDetails[articleId]) {
      console.log(`Returning article from memory cache: "${MEMORY_CACHE.articleDetails[articleId].title}"`);
      return {
        status: 'success',
        article: MEMORY_CACHE.articleDetails[articleId]
      };
    }
    
    // Build possible file paths
    const cachePath = path.join(CACHE_DIR, `final_article_${articleId}.json`);
    const localCachePath = path.join(LOCAL_CACHE_DIR, `final_article_${articleId}.json`);
    
    // Try to read the article data using safe read function
    let articleData = safeReadFile(cachePath, null);
    
    // If not found in first location, try the second
    if (!articleData) {
      articleData = safeReadFile(localCachePath, null);
    }
    
    // Process the article data if found
    if (articleData) {
      const processedArticle = processArticleData(articleData, articleId);
      
      // Save to memory cache
      MEMORY_CACHE.articleDetails[articleId] = processedArticle;
      
      return {
        status: 'success',
        article: processedArticle
      };
    }
    
    // Article not found in any location
    console.log(`Article with ID ${articleId} not found`);
    return {
      status: 'error',
      message: 'Article not found'
    };
  } catch (error) {
    console.error(`Error getting article ${articleId}: ${error}`);
    return {
      status: 'error',
      message: `Error retrieving article: ${error.message}`
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
        title = lines[0].substring(2).trim();
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
function analyze_interests_endpoint(interests) {
  try {
    if (!interests) {
      return {
        status: "error",
        message: "No interests provided"
      };
    }
    
    console.log(`Analyzing interests: ${interests}`);
    
    // Generate demo recommendations based on interests
    const interestsList = interests.split(',').map(i => i.trim().toLowerCase());
    const demoArticles = generateDemoArticles();
    
    // Simple scoring based on keyword matching
    demoArticles.forEach(article => {
      let score = 50; // Base score
      interestsList.forEach(interest => {
        if (article.title.toLowerCase().includes(interest) || 
            article.subject.toLowerCase().includes(interest)) {
          score += 10;
        }
      });
      article.score = Math.min(score, 100); // Cap at 100
    });
    
    // Sort by score
    demoArticles.sort((a, b) => b.score - a.score);
    
    return demoArticles;
  } catch (error) {
    console.error(`Error in analyze_interests: ${error}`);
    return generateDemoArticles();
  }
}

// Export functions for use in other modules
module.exports = {
  process_articles_endpoint,
  get_article_endpoint,
  analyze_interests_endpoint
}; 