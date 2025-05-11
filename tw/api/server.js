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

// Constants - use tmp directory for Vercel
const CACHE_DIR = isVercel ? '/tmp/cache' : path.join(process.cwd(), '.vercel', 'cache');
const LOCAL_CACHE_DIR = isVercel ? '/tmp/local_cache' : path.join(process.cwd(), 'local_cache');

// In-memory cache for Vercel environment
const MEMORY_CACHE = {
  articles: [],
  articleDetails: {},
  summaries: {}
};

// Ensure cache directories exist, but don't crash if they can't be created
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
  console.warn(`Cannot create cache directories (this is normal in serverless environments): ${error.message}`);
  console.log('Using in-memory cache fallback');
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
      score: 85,
      timestamp: Date.now() - 3600000, // 1 hour ago
      link: 'https://example.com/article/1'
    },
    {
      id: 'demo2',
      title: 'The Future of Web Development',
      subject: 'Exploring emerging trends in web development including WebAssembly, Progressive Web Apps, and more.',
      score: 78,
      timestamp: Date.now() - 7200000, // 2 hours ago
      link: 'https://example.com/article/2'
    },
    {
      id: 'demo3',
      title: 'Blockchain Technology Explained',
      subject: 'Understanding the fundamentals of blockchain technology and its potential beyond cryptocurrencies.',
      score: 82,
      timestamp: Date.now() - 10800000, // 3 hours ago
      link: 'https://example.com/article/3'
    },
    {
      id: 'demo4',
      title: 'Artificial Intelligence Ethics',
      subject: 'Examining the ethical considerations in AI development and implementation in society.',
      score: 90,
      timestamp: Date.now() - 14400000, // 4 hours ago
      link: 'https://example.com/article/4'
    },
    {
      id: 'demo5',
      title: 'Cloud Computing Fundamentals',
      subject: 'An overview of cloud computing services, models, and best practices for businesses.',
      score: 75,
      timestamp: Date.now() - 18000000, // 5 hours ago
      link: 'https://example.com/article/5'
    },
    {
      id: 'demo6',
      title: 'Data Science for Beginners',
      subject: 'Getting started with data science: tools, techniques, and essential knowledge.',
      score: 88,
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
    // Check if we have cached articles in memory (for Vercel)
    if (isVercel && MEMORY_CACHE.articles.length > 0) {
      console.log(`Returning ${MEMORY_CACHE.articles.length} articles from memory cache`);
      return MEMORY_CACHE.articles;
    }
    
    // Try to find articles in the filesystem
    console.log(`Looking for articles in CACHE_DIR: ${CACHE_DIR}`);
    console.log(`Looking for articles in LOCAL_CACHE_DIR: ${LOCAL_CACHE_DIR}`);
    
    let finalArticles = [];
    let localFinalArticles = [];
    
    try {
      finalArticles = glob.sync(`${CACHE_DIR}/final_article_*.json`);
      localFinalArticles = glob.sync(`${LOCAL_CACHE_DIR}/final_article_*.json`);
    } catch (error) {
      console.warn(`Error searching for articles: ${error.message}`);
    }
    
    console.log(`Found ${finalArticles.length} articles in CACHE_DIR`);
    console.log(`Found ${localFinalArticles.length} articles in LOCAL_CACHE_DIR`);
    
    const allFinalArticles = [...finalArticles, ...localFinalArticles];
    
    // If no articles found in filesystem, return demo data
    if (allFinalArticles.length === 0) {
      console.warn("No articles found in either cache directory, using demo data");
      const demoArticles = generateDemoArticles();
      if (isVercel) {
        MEMORY_CACHE.articles = demoArticles; // Cache for future requests
      }
      return demoArticles;
    }
    
    // Extract and load article data
    const articleData = [];
    const uniqueTitles = new Set();
    let errorCount = 0;
    
    for (const articlePath of allFinalArticles) {
      try {
        console.log(`Processing article: ${articlePath}`);
        let fileContent;
        try {
          fileContent = fs.readFileSync(articlePath, 'utf-8');
        } catch (readError) {
          console.error(`Error reading article file ${articlePath}: ${readError.message}`);
          errorCount++;
          continue;
        }
        
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
    
    // Cache in memory for future requests if on Vercel
    if (isVercel) {
      MEMORY_CACHE.articles = articleData;
    }
    
    return articleData;
  } catch (error) {
    console.error(`Error getting articles: ${error}`);
    
    // Return demo data as fallback
    const demoArticles = generateDemoArticles();
    if (isVercel) {
      MEMORY_CACHE.articles = demoArticles;
    }
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
      return demoArticle;
    }
    
    // Check memory cache first (for Vercel)
    if (isVercel && MEMORY_CACHE.articleDetails[articleId]) {
      console.log(`Returning article from memory cache: "${MEMORY_CACHE.articleDetails[articleId].title}"`);
      return MEMORY_CACHE.articleDetails[articleId];
    }
    
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
    
    try {
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
        let allArticles = [];
        try {
          allArticles = [
            ...glob.sync(`${CACHE_DIR}/final_article_*${articleId}*.json`),
            ...glob.sync(`${LOCAL_CACHE_DIR}/final_article_*${articleId}*.json`)
          ];
        } catch (globError) {
          console.warn(`Error searching for partial matches: ${globError.message}`);
        }
        
        if (allArticles.length > 0) {
          console.log(`Found ${allArticles.length} potential matches by ID partial`);
          sourcePath = allArticles[0]; // Take the first match
          articleData = JSON.parse(fs.readFileSync(sourcePath, 'utf-8'));
        } else {
          // Return a demo article as fallback
          const demoArticle = getDemoArticleDetail('demo1');
          console.log(`No article found, returning fallback demo article`);
          return demoArticle;
        }
      }
    } catch (fsError) {
      console.error(`File system error: ${fsError.message}`);
      // Return a demo article as fallback
      const demoArticle = getDemoArticleDetail('demo1');
      console.log(`File system error, returning fallback demo article`);
      return demoArticle;
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
    
    // Cache in memory for future requests if on Vercel
    if (isVercel) {
      MEMORY_CACHE.articleDetails[articleId] = response;
    }
    
    console.log(`Returning article: "${title}"`);
    return response;
  } catch (error) {
    console.error(`Error getting article by ID ${articleId}:`, error);
    
    // Return a demo article as fallback
    const demoArticle = getDemoArticleDetail('demo1');
    console.log(`Error occurred, returning fallback demo article`);
    return demoArticle;
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