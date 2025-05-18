import axios from 'axios';
import { syncCache, getCachedArticle, getCachedSummary, getCachedSearch } from './utils/cacheSync';
import './style.css';

/**
 * SimilarityScorer class provides a fully offline similarity scoring algorithm 
 * for comparing strings on a scale of 0-100 with high accuracy.
 */
class SimilarityScorer {
  static preprocess(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, "")
      .split(/\s+/)
      .filter(Boolean);
  }

  static levenshteinSimilarity(a: string, b: string): number {
    const m = a.length, n = b.length;
    if (m === 0 || n === 0) return 0;

    const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;

    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        const cost = a[i - 1] === b[j - 1] ? 0 : 1;
        dp[i][j] = Math.min(
          dp[i - 1][j] + 1,
          dp[i][j - 1] + 1,
          dp[i - 1][j - 1] + cost
        );
      }
    }

    const distance = dp[m][n];
    const maxLen = Math.max(m, n);
    return 100 * (1 - distance / maxLen); // percentage
  }

  static jaccardSimilarity(aTokens: string[], bTokens: string[]): number {
    const setA = new Set(aTokens);
    const setB = new Set(bTokens);
    const intersection = Array.from(setA).filter(x => setB.has(x)).length;
    const union = Array.from(new Set([...aTokens, ...bTokens])).length;
    return union === 0 ? 0 : (intersection / union) * 100;
  }

  static tokenOverlap(aTokens: string[], bTokens: string[]): number {
    let match = 0;
    for (const word of aTokens) {
      if (bTokens.includes(word)) match++;
    }
    return aTokens.length === 0 ? 0 : (match / aTokens.length) * 100;
  }

  /**
   * Computes a similarity score between a short string and a longer string.
   * 
   * @param small - The small string or query
   * @param large - The larger content string
   * @returns A number from 0 to 100 indicating similarity
   */
  static compute(small: string, large: string): number {
    const aTokens = this.preprocess(small);
    const bTokens = this.preprocess(large);
    const smallClean = aTokens.join(" ");
    const largeClean = bTokens.join(" ");

    const lev = this.levenshteinSimilarity(smallClean, largeClean);
    const jac = this.jaccardSimilarity(aTokens, bTokens);
    const tok = this.tokenOverlap(aTokens, bTokens);

    const score = 0.4 * lev + 0.3 * jac + 0.3 * tok;
    return Math.round(score);
  }
}

// Define the type for calculation log steps
interface CalculationStep {
    step: string;
    value: any;
}

// Update the API_URL constant to use relative paths for Vercel compatibility
const API_URL = '';  // Empty string for relative paths

// Add a custom logger
const logToFile = async (message: string, level: 'info' | 'error' | 'debug' = 'info'): Promise<void> => {
    try {
        // Log to console
        const timestamp = new Date().toISOString();
        const formattedMessage = `[${timestamp}] [${level.toUpperCase()}] ${message}`;
        
        // Standard console logging
        if (level === 'error') {
            console.error(formattedMessage);
        } else if (level === 'debug') {
            console.debug(formattedMessage);
        } else {
            console.log(formattedMessage);
        }
        
        // Send log to backend to write to file
        await fetch(`${API_URL}/log`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: formattedMessage,
                level,
                source: 'frontend'
            })
        });
    } catch (error) {
        // If logging fails, at least show in console
        console.error(`Failed to log to file: ${error}`);
    }
};

interface Article {
    title: string;
    link: string;
    subject: string;
    score: number;
    summary?: string;
    id?: string;
    scoreComponents?: Record<string, string>;
}

interface ArticleDetail {
    title: string;
    link: string;
    summary: string;
    content?: string;
}

// Define types for the zero-shot classification results
interface ZeroShotClassificationOutput {
    sequence: string;
    labels: string[];
    scores: number[];
}

/**
 * Custom ranking system for articles - uses advanced similarity scoring
 * @param article Article to score
 * @param userInterests User's specified interests
 * @param articleCollection Full collection of articles for contextual ranking
 * @returns A score between 0-100
 */
const customRankingSystem = async (
    article: Article, 
    userInterests: string,
    articleCollection: Article[]
): Promise<number> => {
    try {
        console.log(`\n========== SCORING ARTICLE: "${article.title}" (ID: ${article.id || 'unknown'}) ==========`);
        console.log(`Interest Terms: "${userInterests}"`);
        
        // Base score (70-95 range)
        let baseScore = Math.floor(Math.random() * 25) + 70;
        console.log(`Base score (if no interests): ${baseScore}`);
        
        if (!userInterests.trim()) {
            console.log(`No interests provided - Using base score: ${baseScore}`);
            return baseScore; // Default score if no interests
        }
        
        // Process interest terms
        const interestTerms = userInterests
            .toLowerCase()
            .split(/[\s,]+/)
            .filter(term => term.length > 2)
            .map(term => term.trim());
        
        console.log(`Parsed interest terms (${interestTerms.length}):`, interestTerms);
        
        if (interestTerms.length === 0) {
            console.log(`No valid interest terms after filtering - Using base score: ${baseScore}`);
            return baseScore; // No valid interest terms
        }
        
        // Load summary from /data/article_cache/summary_{article.id}.json if possible
        let articleSummary = '';
        let summarySource = 'none';
        
        if (article.id) {
            try {
                const summaryPath = `/data/article_cache/summary_${article.id}.json`;
                console.log(`ATTEMPTING TO LOAD SUMMARY FILE: ${summaryPath}`);
                
                const response = await fetch(summaryPath);
                console.log(`Summary file response status:`, response.status, response.statusText);
                
                if (response.ok) {
                    const summaryData = await response.json();
                    console.log(`Summary file raw data:`, summaryData);
                    
                    if (summaryData.summary && typeof summaryData.summary === 'string') {
                        articleSummary = summaryData.summary;
                        summarySource = 'cache file';
                        console.log(`SUCCESS: Loaded summary from file (${articleSummary.length} chars)`);
                        console.log(`Summary preview: "${articleSummary.substring(0, 100)}${articleSummary.length > 100 ? '...' : ''}"`);
                    } else {
                        console.warn(`File loaded but no valid summary found in data`);
                    }
                } else {
                    console.warn(`Failed to load summary file: ${response.status} ${response.statusText}`);
                }
            } catch (err) {
                console.warn(`Error loading summary for article id ${article.id}:`, err);
            }
        } else {
            console.log(`No article ID available - Cannot load summary file`);
        }
        
        if (!articleSummary && article.summary) {
            articleSummary = article.summary;
            summarySource = 'article object';
            console.log(`FALLBACK: Using summary from article object (${articleSummary.length} chars)`);
            console.log(`Summary preview: "${articleSummary.substring(0, 100)}${articleSummary.length > 100 ? '...' : ''}"`);
        } else if (!articleSummary) {
            console.log(`WARNING: No summary available from any source!`);
        }
        
        console.log(`Final summary source: ${summarySource}`);
        console.log(`Final summary length: ${articleSummary.length} characters`);
        
        // Prepare article content for similarity scoring
        const titleText = article.title;
        const subjectText = article.subject || '';
        
        // Create full text representation for the article
        const articleFullText = `${titleText} ${subjectText} ${articleSummary}`;
        
        console.log(`\n----- SCORING COMPONENT 1: TEXT SIMILARITY (0-60 points) -----`);
        // ---- 1. TEXT SIMILARITY SCORING (0-60 points) ----
        let similarityScore = 0;
        
        if (articleFullText.length > 0) {
            console.log(`Computing text similarity using advanced methods...`);
            
            // Calculate similarity between user interests and article content
            const rawSimilarity = SimilarityScorer.compute(userInterests, articleFullText);
            console.log(`Raw similarity score: ${rawSimilarity} / 100`);
            
            // Weight the similarity score (max 60 points)
            similarityScore = (rawSimilarity / 100) * 60;
            console.log(`Weighted similarity score: ${similarityScore.toFixed(2)} / 60 points`);
            
            // Detailed similarity breakdown
            const interestTokens = SimilarityScorer.preprocess(userInterests);
            const articleTokens = SimilarityScorer.preprocess(articleFullText);
            
            // Calculate individual metrics for logging
            const levSim = SimilarityScorer.levenshteinSimilarity(
                interestTokens.join(" "), 
                articleTokens.slice(0, 200).join(" ")  // Limit for performance
            );
            const jacSim = SimilarityScorer.jaccardSimilarity(interestTokens, articleTokens);
            const tokSim = SimilarityScorer.tokenOverlap(interestTokens, articleTokens);
            
            console.log(`Similarity metrics breakdown:
  - Levenshtein similarity: ${levSim.toFixed(2)}%
  - Jaccard similarity:     ${jacSim.toFixed(2)}%
  - Token overlap:          ${tokSim.toFixed(2)}%`);
        } else {
            console.log(`Empty article content - Similarity score: 0 / 60 points`);
        }
        
        console.log(`\n----- SCORING COMPONENT 2: CONTEXTUAL RELEVANCE (0-30 points) -----`);
        // ---- 2. CONTEXTUAL RELEVANCE (0-30 points) ----
        let contextualScore = 0;
        
        // Title weight is higher if user interests appear in the title
        const titleImportance = userInterests.split(/\s+/).some(term => 
            article.title.toLowerCase().includes(term.toLowerCase())
        ) ? 2.0 : 1.0;
        
        if (titleImportance > 1.0) {
            console.log(`Interest terms found in title: applying ${titleImportance}x title importance multiplier`);
        }
        
        // Calculate semantic similarity based on key parts with different weights
        const titleSimilarity = SimilarityScorer.compute(userInterests, titleText) * titleImportance;
        console.log(`Title similarity: ${(titleSimilarity / titleImportance).toFixed(2)}% (raw) × ${titleImportance} = ${titleSimilarity.toFixed(2)}%`);
        
        const subjectSimilarity = subjectText ? 
            SimilarityScorer.compute(userInterests, subjectText) : 0;
        console.log(`Subject similarity: ${subjectSimilarity.toFixed(2)}%`);
        
        // Weighted average of title and subject similarities
        const weightedContextSimilarity = (titleSimilarity * 0.7) + (subjectSimilarity * 0.3);
        console.log(`Weighted context similarity: ${weightedContextSimilarity.toFixed(2)}%`);
        
        // Scale to the 30-point scoring component
        contextualScore = (weightedContextSimilarity / 100) * 30;
        console.log(`Contextual relevance score: ${contextualScore.toFixed(2)} / 30 points`);
        
        console.log(`\n----- SCORING COMPONENT 3: FRESHNESS & UNIQUENESS (0-10 points) -----`);
        // ---- 3. FRESHNESS FACTOR WITH UNIQUENESS (0-10 points) ----
        let freshnessScore = 0;
        
        // Base freshness score from article ID (consistent value)
        const baseFreshness = article.id ? 
            (parseInt(article.id.replace(/\D/g, '').slice(-2) || '0') % 8) : 
            Math.floor(Math.random() * 8);
        
        console.log(`Base freshness: ${baseFreshness} / 8`);
        
        // Uniqueness bonus: if article title contains uncommon words compared to other articles
        const allTitles = articleCollection.map(a => a.title.toLowerCase());
        const titleWords = titleText.toLowerCase().split(/\s+/).filter(w => w.length > 4); // Only consider substantial words
        
        let uncommonWordCount = 0;
        titleWords.forEach(word => {
            // Count how many other articles contain this word
            const occurrenceCount = allTitles.filter(t => t.includes(word)).length;
            
            // If word appears in less than 20% of articles, consider it uncommon
            if (occurrenceCount <= Math.max(1, Math.floor(articleCollection.length * 0.2))) {
                uncommonWordCount++;
                console.log(`  Uncommon word found: "${word}" (in ${occurrenceCount} articles)`);
            }
        });
        
        // Calculate uniqueness bonus (max +2 points)
        const uniquenessBonus = Math.min(2, uncommonWordCount * 0.5);
        console.log(`Uniqueness bonus: ${uniquenessBonus.toFixed(2)} points (${uncommonWordCount} uncommon words)`);
        
        // Combine base freshness with uniqueness bonus
        freshnessScore = baseFreshness + uniquenessBonus;
        console.log(`Freshness & uniqueness score: ${freshnessScore.toFixed(2)} / 10 points`);
        
        console.log(`\n----- FINAL SCORE CALCULATION -----`);
        // ---- COMBINE SCORES ----
        const finalScore = similarityScore + contextualScore + freshnessScore;
        console.log(`Final score breakdown:
  - Text Similarity:         ${similarityScore.toFixed(2)} / 60
  - Contextual Relevance:    ${contextualScore.toFixed(2)} / 30
  - Freshness & Uniqueness:  ${freshnessScore.toFixed(2)} / 10
  ----------------------
  TOTAL SCORE:               ${finalScore.toFixed(2)} / 100
`);
        
        // Log all scoring components
        const scoreBreakdown = {
            similarity: similarityScore.toFixed(2),
            contextual: contextualScore.toFixed(2),
            freshness: freshnessScore.toFixed(2),
            final: finalScore.toFixed(2)
        };
        
        console.log(`Score components stored:`, scoreBreakdown);
        
        // Store score components in the article object for display
        article.scoreComponents = scoreBreakdown;
        
        console.log(`\n----- SCORING COMPLETE -----`);
        
        // Ensure score is between 0-100
        return Math.min(100, Math.max(0, finalScore));
    } catch (error) {
        console.error('ERROR IN RANKING SYSTEM:', error);
        return Math.floor(Math.random() * 25) + 70; // Fallback score
    }
};

// API Service
class ApiService {
    private baseUrl: string = '';

    constructor() {
        // In development, use a different base URL
        this.baseUrl = window.location.hostname === 'localhost' 
            ? 'http://localhost:3000' 
            : '';
        
        console.log(`ApiService initialized with baseUrl: ${this.baseUrl}`);
        console.log(`Current hostname: ${window.location.hostname}`);
    }

    async getArticles(interests?: string): Promise<Article[]> {
        try {
            const queryParams = interests ? `?interests=${encodeURIComponent(interests)}` : '';
            const url = `${this.baseUrl}/api/articles${queryParams}`;
            console.log(`Fetching articles from: ${url}`);
            
            const response = await fetch(url);
            
            if (!response.ok) {
                console.error(`Error response from server: ${response.status}`);
                throw new Error(`Failed to fetch articles: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log(`Articles fetched successfully. Count: ${data.length}`);
            console.log(`Sample article data:`, data.length > 0 ? data[0] : 'No articles');
            
            // Ensure each article has the expected format and properties
            const formattedArticles = data.map((article: any) => ({
                id: article.id || '', // Preserve ID from API response
                title: article.title || 'Untitled Article',
                subject: article.subject || '',
                link: article.link || '',
                score: article.score || 0,
                summary: article.summary || ''
            }));
            
            return formattedArticles;
        } catch (error) {
            console.error('Error fetching articles:', error);
            return [];
        }
    }

    async getArticleDetails(articleId: string): Promise<ArticleDetail | null> {
        try {
            const url = `/data/article_cache/final_article_${articleId}.json`;
            console.log(`Fetching article details from: ${url}`);
            
            const response = await fetch(url);
            const data = await response.json();

            if (!response.ok) {
                console.error(`Error response from server: ${response.status}`);
                throw new Error(`Failed to fetch article details: ${response.statusText}`);
            }
            
            console.log('Article details response:', data);
            
            // If we get a proper response with article data
            if (data && (data.title || data.summary || data.content)) {
                console.log('Article details fetched successfully:', data.title);
                return {
                    title: data.title || '',
                    link: data.link || '',
                    summary: data.summary || '',
                    content: data.content || ''
                };
            } else {
                console.error('Invalid article data format:', data);
                return null;
            }
        } catch (error) {
            console.error('Error fetching article details:', error);
            return null;
        }
    }

    async analyzeInterests(interests: string): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/api/analyze-interests`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ interests })
            });
            
            if (!response.ok) {
                throw new Error('Failed to analyze interests');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error analyzing interests:', error);
            return null;
        }
    }
}

// UI Controller
class AppController {
    private apiService: ApiService;
    private articlesContainer: HTMLElement | null;
    private interestsInput: HTMLInputElement | null;
    private submitButton: HTMLElement | null;
    
    constructor() {
        this.apiService = new ApiService();
        this.articlesContainer = document.getElementById('articles-container');
        this.interestsInput = document.getElementById('interests-input') as HTMLInputElement;
        this.submitButton = document.getElementById('submit-interests');
        
        // Add navigation to backtesting page
        this.addNavigation();
        
        this.initEventListeners();
        this.loadArticles();
    }
    
    private addNavigation(): void {
        // Get the header element
        const header = document.querySelector('header');
        
        if (header) {
            // Check if navigation already exists
            let nav = header.querySelector('nav');
            
            if (!nav) {
                // Create a new navigation element
                nav = document.createElement('nav');
                nav.style.display = 'flex';
                nav.style.gap = '1rem';
                nav.style.justifyContent = 'center';
                nav.style.marginBottom = '1rem';
                
                // Create links
                const homeLink = document.createElement('a');
                homeLink.href = 'index.html';
                homeLink.className = 'nav-button active';
                homeLink.id = 'home-tab';
                homeLink.textContent = 'Home';
                
                const backtestLink = document.createElement('a');
                backtestLink.href = 'backtesting.html';
                backtestLink.className = 'nav-button';
                backtestLink.id = 'backtest-tab';
                backtestLink.textContent = 'Backtesting';
                
                // Add links to nav
                nav.appendChild(homeLink);
                nav.appendChild(backtestLink);
                
                // Add nav to header
                header.appendChild(nav);
            }
        }
    }
    
    private initEventListeners(): void {
        if (this.submitButton) {
            this.submitButton.addEventListener('click', () => this.handleInterestsSubmit());
        }
    }
    
    private async loadArticles(interests?: string): Promise<void> {
        if (!this.articlesContainer) return;

        try {
            console.log(`Loading articles with interests: ${interests || 'none'}`);
            this.articlesContainer.innerHTML = '<div class="loading">Loading articles...</div>';
            const articles = await this.apiService.getArticles(interests);
            if (articles.length === 0) {
                this.articlesContainer.innerHTML = '<div class="message">No articles found. Please try with different interests.</div>';
                return;
            }
            console.log(`Fetched ${articles.length} articles, now scoring based on interests: ${interests || 'none'}`);
            
            // Deduplicate articles by title
            const uniqueArticles = this.deduplicateArticles(articles);
            console.log(`Reduced to ${uniqueArticles.length} unique articles after deduplication`);
            
            // Score all articles using the custom ranking system
            const scoringPromises = uniqueArticles.map(article => 
                customRankingSystem(article, interests || '', uniqueArticles)
            );
            
            const scores = await Promise.all(scoringPromises);
            
            // Apply scores to articles
            const scoredArticles = uniqueArticles.map((article, index) => ({
                ...article,
                score: Math.round(scores[index]) // Round to nearest integer for display
            }));
            
            console.log(`Scored ${scoredArticles.length} articles using custom ranking system`);
            
            // Sort by score (highest first)
            scoredArticles.sort((a, b) => b.score - a.score);
            console.log("Articles sorted by score");
            
            this.renderArticles(scoredArticles);
        } catch (error) {
            console.error('Error loading articles:', error);
            this.articlesContainer.innerHTML = '<div class="error">Error loading articles. Please try again later.</div>';
        }
    }
    
    // Helper method to deduplicate articles based on title
    private deduplicateArticles(articles: Article[]): Article[] {
        const uniqueTitles = new Set<string>();
        return articles.filter(article => {
            // Normalize title for comparison
            const normalizedTitle = article.title.toLowerCase().trim();
            if (uniqueTitles.has(normalizedTitle)) {
                return false; // Skip duplicate
            }
            uniqueTitles.add(normalizedTitle);
            return true;
        });
    }
    
    private renderArticles(articles: Article[]): void {
        if (!this.articlesContainer) return;
        
        // Clear existing content
        this.articlesContainer.innerHTML = '';
        
        if (!articles || articles.length === 0) {
            this.articlesContainer.innerHTML = '<div class="no-results">No articles found. Try adjusting your interests.</div>';
            return;
        }
        
        // Display article count
        const articleCountEl = document.createElement('div');
        articleCountEl.className = 'article-count';
        articleCountEl.textContent = `Displaying ${articles.length} unique articles`;
        this.articlesContainer.appendChild(articleCountEl);
        
        // Render each article
        articles.forEach(article => {
            const card = document.createElement('div');
            card.className = 'article-card';
            card.dataset.id = article.id || '';
            
            // Add score badge
            const scoreClass = this.getScoreClass(article.score);
            const scoreBadge = document.createElement('div');
            scoreBadge.className = `score-badge ${scoreClass}`;
            scoreBadge.textContent = article.score.toString();
            card.appendChild(scoreBadge);
            
            // Add title
            const title = document.createElement('h3');
            title.className = 'article-title';
            title.textContent = article.title;
            card.appendChild(title);
            
            // Add subject if available
            if (article.subject) {
                const subject = document.createElement('div');
                subject.className = 'article-subject';
                subject.textContent = article.subject;
                card.appendChild(subject);
            }
            
            // Add summary if available
            if (article.summary) {
                const summary = document.createElement('div');
                summary.className = 'article-summary';
                summary.innerHTML = this.formatSummary(
                    this.truncateText(article.summary, 200)
                );
                card.appendChild(summary);
            }
            
            // Add score components if available
            if (article.scoreComponents) {
                const scoreDetails = document.createElement('div');
                scoreDetails.className = 'score-details';
                
                const detailsList = document.createElement('ul');
                
                // Fix the TS2531 error by adding a null check for scoreDetails
                const scoreDetailsButton = document.createElement('button');
                scoreDetailsButton.className = 'score-details-btn';
                scoreDetailsButton.textContent = 'Show score details';
                scoreDetailsButton.addEventListener('click', () => {
                    if (scoreDetails && scoreDetails.classList) {
                        scoreDetails.classList.toggle('visible');
                        scoreDetailsButton.textContent = 
                            scoreDetails.classList.contains('visible') ? 
                            'Hide score details' : 'Show score details';
                    }
                });
                
                // Add score breakdown
                Object.entries(article.scoreComponents).forEach(([key, value]) => {
                    const item = document.createElement('li');
                    const label = key.charAt(0).toUpperCase() + key.slice(1);
                    item.innerHTML = `<strong>${label}:</strong> ${value}`;
                    detailsList.appendChild(item);
                });
                
                scoreDetails.appendChild(detailsList);
                card.appendChild(scoreDetailsButton);
                card.appendChild(scoreDetails);
            }
            
            // Add view button
            const viewButton = document.createElement('button');
            viewButton.className = 'view-article-btn';
            viewButton.textContent = 'View Article';
            viewButton.addEventListener('click', () => {
                if (article.id) {
                    this.viewArticleDetails(article.id);
                }
            });
            card.appendChild(viewButton);
            
            // Add the card to the container
            this.articlesContainer.appendChild(card);
        });
        
        // Add click listeners to each article card for expanding
        const cards = document.querySelectorAll('.article-card');
        cards.forEach(card => {
            card.addEventListener('click', (e) => {
                // Fix the TS2531 error by adding a null check for card and e.target
                if (card && card.classList && e.target) {
                    const target = e.target as HTMLElement;
                    const isButton = target.tagName === 'BUTTON';
                    if (!isButton) {
                        card.classList.toggle('expanded');
                    }
                }
            });
        });
    }
    
    private async viewArticleDetails(articleId: string): Promise<void> {
        if (!this.articlesContainer) return;
        
        this.articlesContainer.innerHTML = '<div class="loading">Loading article details...</div>';
        
        try {
            const articleDetail = await this.apiService.getArticleDetails(articleId);
            
            if (!articleDetail) {
                this.articlesContainer.innerHTML = '<div class="error">Article not found</div>';
                // Add a back button even when article is not found
                const backButton = document.createElement('button');
                backButton.textContent = 'Back to Articles';
                backButton.className = 'back-button';
                backButton.addEventListener('click', () => {
                    const interests = this.interestsInput?.value || '';
                    this.loadArticles(interests);
                });
                this.articlesContainer.appendChild(backButton);
                return;
            }
            
            this.renderArticleDetail(articleDetail);
        } catch (error) {
            console.error('Error loading article details:', error);
            this.articlesContainer.innerHTML = '<div class="error">Error loading article details. Please try again later.</div>';
            // Add a back button even when there's an error
            const backButton = document.createElement('button');
            backButton.textContent = 'Back to Articles';
            backButton.className = 'back-button';
            backButton.addEventListener('click', () => {
                const interests = this.interestsInput?.value || '';
                this.loadArticles(interests);
            });
            this.articlesContainer.appendChild(backButton);
        }
    }
    
    private renderArticleDetail(article: ArticleDetail): void {
        if (!this.articlesContainer) return;
        
        this.articlesContainer.innerHTML = '';
        
        const articleDetail = document.createElement('div');
        articleDetail.className = 'article-detail';
        
        // Create article HTML content
        let articleContent = `
            <h2>${this.escapeHtml(article.title)}</h2>
        `;
        
        // Add the full article content if available, otherwise fall back to summary
        if (article.content) {
            // Check if content appears to be HTML (contains tags)
            if (article.content.includes('<') && article.content.includes('>')) {
                // It's HTML, insert directly with sanitization
                articleContent += `
                    <div class="article-content">
                        ${this.sanitizeHtml(article.content)}
                    </div>
                `;
            } else {
                // It's markdown or plain text, format with paragraphs
                articleContent += `
                    <div class="article-content">
                        ${this.formatSummary(article.content)}
                    </div>
                `;
            }
        } else {
            // No content, use summary
            articleContent += `
                <div class="article-summary">
                    ${this.formatSummary(article.summary)}
                </div>
            `;
        }
        
        // Add action button (centered, blue)
        articleContent += `
            <div class="article-actions" style="display: flex; justify-content: center; margin-top: 2.5rem;">
                <button id="back-to-articles" class="back-button blue">Back to Articles</button>
            </div>
        `;
        
        articleDetail.innerHTML = articleContent;
        
        // Add back button click event
        const backButton = articleDetail.querySelector('#back-to-articles');
        if (backButton) {
            backButton.addEventListener('click', () => {
                const interests = this.interestsInput?.value || '';
                this.loadArticles(interests);
            });
        }
        
        this.articlesContainer.appendChild(articleDetail);
    }
    
    private handleInterestsSubmit(): void {
        if (!this.interestsInput) return;
        
        const interests = this.interestsInput.value.trim();
        console.log(`Interests submitted: "${interests}"`);
        
        // Disable the button during loading
        if (this.submitButton) {
            this.submitButton.setAttribute('disabled', 'true');
            this.submitButton.textContent = 'Loading...';
        }
        
        // Show loading indicator in articles container
        if (this.articlesContainer) {
            this.articlesContainer.innerHTML = '<div class="loading">Scoring articles based on your interests...</div>';
        }
        
        // Load articles with interests
        this.loadArticles(interests)
            .finally(() => {
                // Re-enable the button when done
                if (this.submitButton) {
                    this.submitButton.removeAttribute('disabled');
                    this.submitButton.textContent = 'Update Interests';
                }
            });
    }
    
    private formatSummary(summary: string): string {
        // Replace newlines with paragraph breaks
        return summary
            .split('\n\n')
            .filter(paragraph => paragraph.trim().length > 0)
            .map(paragraph => `<p>${this.escapeHtml(paragraph)}</p>`)
            .join('');
    }
    
    private escapeHtml(unsafe: string): string {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    // Helper to truncate text with ellipsis
    private truncateText(text: string, maxLength: number): string {
        if (!text) return '';
        if (text.length <= maxLength) return this.escapeHtml(text);
        return this.escapeHtml(text.substring(0, maxLength)) + '...';
    }

    // Basic HTML sanitizer to avoid XSS while allowing formatted content
    private sanitizeHtml(html: string): string {
        // Create a temporary element
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;
        
        // Remove potentially dangerous elements/attributes
        const scripts = tempDiv.querySelectorAll('script');
        scripts.forEach(script => script.remove());
        
        const iframes = tempDiv.querySelectorAll('iframe');
        iframes.forEach(iframe => iframe.remove());
        
        // Remove on* attributes
        const allElements = tempDiv.querySelectorAll('*');
        allElements.forEach(el => {
            Array.from(el.attributes).forEach(attr => {
                if (attr.name.startsWith('on') || attr.name === 'href' && attr.value.startsWith('javascript:')) {
                    el.removeAttribute(attr.name);
                }
            });
        });
        
        return tempDiv.innerHTML;
    }

    private getScoreClass(score: number): string {
        if (score > 90) return 'high-score';
        if (score > 75) return 'medium-score';
        if (score > 60) return 'low-score';
        return 'no-score';
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AppController();
}); 