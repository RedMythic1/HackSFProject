import axios from 'axios';
import { pipeline } from '@xenova/transformers';
import { syncCache, getCachedArticle, getCachedSummary, getCachedSearch } from './utils/cacheSync';

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

// Initialize the embedding model
let embeddingModel: any = null;

// Initialize embedding extraction pipeline
const initEmbeddingModel = async (): Promise<void> => {
    try {
        console.log('Initializing embedding model...');
        embeddingModel = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        console.log('Embedding model initialized successfully');
    } catch (error) {
        console.error('Error initializing embedding model:', error);
    }
};

// Generate embeddings for text input
const generateEmbedding = async (text: string): Promise<number[]> => {
    if (!embeddingModel) {
        console.log('Embedding model not initialized, initializing now...');
        await initEmbeddingModel();
    }
    
    try {
        // Limit input text length to avoid issues with very long texts
        let inputText = text;
        if (text.length > 2048) {
            inputText = text.substring(0, 2048);
            console.log(`Text truncated from ${text.length} to 2048 chars for embedding generation`);
        }
        
        const result = await embeddingModel(inputText, {
            pooling: 'mean',
            normalize: true
        });
        
        // Convert tensor to regular array
        const embedding = Array.from(result.data).map(value => Number(value));
        console.log(`Generated embedding with ${embedding.length} dimensions`);
        return embedding;
    } catch (error) {
        console.error(`Error generating embedding: ${error}`);
        return [];
    }
};

// Calculate similarity between two vectors
const vectorSimilarity = (vecV: number[], vecW: number[]): number => {
    // Make sure we have valid inputs
    if (!vecV || !vecW || !Array.isArray(vecV) || !Array.isArray(vecW)) {
        console.error("Invalid vector inputs");
        return 25; // Return minimal score rather than 0
    }
    if (vecV.length !== vecW.length) {
        console.error(`Vector dimensions don't match: ${vecV.length} vs ${vecW.length}`);
        return 25; // Return minimal score rather than 0
    }
    if (vecV.length === 0 || vecW.length === 0) {
        console.error("Empty vectors");
        return 25; // Return minimal score rather than 0
    }
    const hasInvalidValues = (vec: number[]) => vec.some(v => isNaN(v) || !isFinite(v));
    if (hasInvalidValues(vecV) || hasInvalidValues(vecW)) {
        console.error("Vectors contain NaN or Infinity values");
        return 25; // Return minimal score rather than 0
    }
    // Calculate dot products and magnitudes
    let vDotW = 0, vDotV = 0, wDotW = 0;
    for (let i = 0; i < vecV.length; i++) {
        vDotW += vecV[i] * vecW[i];
        vDotV += vecV[i] * vecV[i];
        wDotW += vecW[i] * vecW[i];
    }
    if (vDotV <= 0 || wDotW <= 0) {
        console.error("One of the vectors has zero magnitude");
        return 25;
    }
    const magnitudeV = Math.sqrt(vDotV);
    const magnitudeW = Math.sqrt(wDotW);
    // Cosine similarity
    const cosineSimilarity = vDotW / (magnitudeV * magnitudeW);
    // Projection of w onto v: proj = (v . w) / (v . v) * v
    // v - proj(w on v):
    const scale = vDotW / vDotV;
    let diffSquaredSum = 0;
    for (let i = 0; i < vecV.length; i++) {
        const projComponent = scale * vecV[i];
        const diff = vecW[i] - projComponent;
        diffSquaredSum += diff * diff;
    }
    const absDiff = Math.sqrt(diffSquaredSum);
    // New denominator: (1 - abs(v - proj of w on v)^2)
    const denom = 1 - Math.pow(absDiff, 2);
    // Avoid division by zero or negative denominator
    let adjustedSimilarity = cosineSimilarity;
    if (denom > 0.00001) {
        adjustedSimilarity = cosineSimilarity / denom;
    } else {
        console.warn("Denominator for adjusted similarity is too small or negative, using cosineSimilarity only");
    }
    // Linear scaling from [-1, 1] to [0, 100]
    const normalizedScore = (adjustedSimilarity + 1) * 50;
    const finalScore = Math.max(5, Math.min(100, normalizedScore));
    return finalScore;
};

// Article scoring function using embeddings
const scoreArticle = async (article: Article, userInterests: string): Promise<number> => {
    try {
        if (!userInterests.trim()) {
            return Math.floor(Math.random() * 25) + 70; // Default score if no interests
        }

        // --- KEYWORD SCORE (unchanged) ---
        const interestTerms = userInterests.toLowerCase().split(/[\s,]+/).filter(term => term.length > 2);
        const articleTextLower = `${article.title} ${article.subject}`.toLowerCase();
        let keywordMatches = 0;
        interestTerms.forEach(term => {
            if (articleTextLower.includes(term)) {
                keywordMatches++;
            }
        });
        const keywordScore = interestTerms.length > 0 ? (keywordMatches / interestTerms.length) * 100 : 0;

        // --- VECTOR SCORE using summary embedding ---
        let vectorScore = 0;
        let articleEmbedding: number[] = [];
        try {
            if (article.id) {
                // Try to load the summary file for this article
                const summaryPath = `.cache/summary_${article.id}.json`;
                const response = await fetch(summaryPath);
                if (response.ok) {
                    const summaryData = await response.json();
                    if (summaryData.embedding && Array.isArray(summaryData.embedding)) {
                        articleEmbedding = summaryData.embedding;
                    }
                }
            }
        } catch (err) {
            console.warn(`Could not load summary embedding for article id ${article.id}:`, err);
        }

        // Generate embedding for user interests
        const interestsEmbedding = await generateEmbedding(userInterests);

        if (articleEmbedding.length && interestsEmbedding.length) {
            vectorScore = vectorSimilarity(articleEmbedding, interestsEmbedding);
        } else {
            // Fallback: use previous method (title+subject)
            const fallbackText = `${article.title}. ${article.subject}`;
            const fallbackEmbedding = await generateEmbedding(fallbackText);
            if (fallbackEmbedding.length && interestsEmbedding.length) {
                vectorScore = vectorSimilarity(fallbackEmbedding, interestsEmbedding);
            } else {
                vectorScore = Math.floor(Math.random() * 25) + 70;
            }
        }

        // Combine scores: 30% vector, 70% keyword (or whatever logic is current)
        const finalScore = 0.3 * vectorScore + 0.7 * keywordScore;
        console.log(`Combined score for "${article.title}": vector=${vectorScore}, keyword=${keywordScore}, final=${finalScore}`);
        return finalScore;
    } catch (error) {
        console.error('Error scoring article:', error);
        return Math.floor(Math.random() * 25) + 70;
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
            const url = `${this.baseUrl}/api/article/${articleId}`;
            console.log(`Fetching article details from: ${url}`);
            
            const response = await fetch(url);
            
            if (!response.ok) {
                console.error(`Error response from server: ${response.status}`);
                throw new Error(`Failed to fetch article details: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('Article details response:', data);
            
            // If we get a proper response with article data
            if (data.status === 'success' && data.article) {
                console.log('Article details fetched successfully:', data.article.title);
                return {
                    title: data.article.title,
                    link: data.article.link,
                    summary: data.article.summary,
                    content: data.article.content
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
        
        this.initEventListeners();
        this.loadArticles();
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
            let scoredArticles = [...articles];
            if (interests && interests.trim()) {
                // --- KEYWORD FIRST RANKING ---
                // Calculate keyword scores for all articles
                const interestTerms = interests.toLowerCase().split(/[\s,]+/).filter(term => term.length > 2);
                scoredArticles = articles.map(article => {
                    const articleTextLower = `${article.title} ${article.subject}`.toLowerCase();
                    let keywordMatches = 0;
                    interestTerms.forEach(term => {
                        if (articleTextLower.includes(term)) {
                            keywordMatches++;
                        }
                    });
                    const keywordScore = interestTerms.length > 0 ? (keywordMatches / interestTerms.length) * 100 : 0;
                    return { ...article, score: keywordScore };
                });
                // Sort by keyword score (descending)
                scoredArticles.sort((a, b) => b.score - a.score);
                // Take top 3 for vector reranking
                const topKeywordArticles = scoredArticles.slice(0, 3);
                const restArticles = scoredArticles.slice(3);
                // Score top 3 by vector similarity
                const vectorScores = await Promise.all(topKeywordArticles.map(article => scoreArticle(article, interests)));
                const top3WithVector = topKeywordArticles.map((article, idx) => ({ ...article, score: vectorScores[idx] }));
                // Sort top 3 by vector score (descending)
                top3WithVector.sort((a, b) => b.score - a.score);
                // Final list: top 3 (vector sorted) + rest (keyword sorted)
                scoredArticles = [...top3WithVector, ...restArticles];
            } else {
                // No interests provided, use default scoring
                console.log('No interests provided, using default scoring');
                scoredArticles = articles.map(article => ({
                    ...article,
                    score: article.score || Math.floor(Math.random() * 30) + 70
                }));
            }
            console.log(`Scored ${scoredArticles.length} articles`);
            // Sort by score (highest first)
            scoredArticles.sort((a, b) => b.score - a.score);
            console.log("Articles sorted by score");
            this.renderArticles(scoredArticles);
        } catch (error) {
            console.error('Error loading articles:', error);
            this.articlesContainer.innerHTML = '<div class="error">Error loading articles. Please try again later.</div>';
        }
    }
    
    private renderArticles(articles: Article[]): void {
        if (!this.articlesContainer) return;
        
        console.log(`Rendering ${articles.length} articles`);
        this.articlesContainer.innerHTML = '';
        
        articles.forEach(article => {
            // Extract article ID from link - Use article.id if available, otherwise extract from link
            let articleId;
            
            // Check if the article already has an id property
            if ('id' in article && article.id) {
                articleId = article.id;
            } else {
                // Extract from link - making this more robust
                const linkParts = article.link.split('/');
                articleId = linkParts[linkParts.length - 1]; // Get the last part of the link
                
                // If the link doesn't contain a valid ID, generate one from the title
                if (!articleId || articleId === '#') {
                    // Make a slug from title + random numbers to ensure uniqueness
                    const titleSlug = article.title
                        .toLowerCase()
                        .replace(/[^a-z0-9]+/g, '-')
                        .replace(/(^-|-$)/g, '');
                    articleId = `${titleSlug}-${Date.now().toString().slice(-6)}`;
                }
            }
            
            console.log(`Rendering article: "${article.title}" (ID: ${articleId})`);
            
            const articleCard = document.createElement('div');
            articleCard.className = 'article-card';
            articleCard.innerHTML = `
                <div class="article-header">
                    <h3 class="article-title">${this.escapeHtml(article.title)}</h3>
                    <span class="article-score">Score: ${article.score}</span>
                </div>
                <div class="article-body">
                    <div class="article-subject">${this.escapeHtml(article.subject)}</div>
                    <div class="article-introduction">
                        <h4>Introduction</h4>
                        <p>${this.truncateText(article.summary || 'No introduction available', 100)}</p>
                    </div>
                    <a href="#" class="article-link" data-article-id="${articleId}">Read more</a>
                </div>
            `;
            
            // Add click event to view article details - Log the actual ID being used
            const articleLink = articleCard.querySelector('.article-link') as HTMLAnchorElement;
            if (articleLink) {
                articleLink.addEventListener('click', (e) => {
                    e.preventDefault();
                    // Get the ID from the data attribute to ensure consistency
                    const clickedId = articleLink.getAttribute('data-article-id');
                    console.log(`Article link clicked, ID from attribute: ${clickedId}, original ID: ${articleId}`);
                    if (clickedId) {
                        this.viewArticleDetails(clickedId);
                    } else {
                        console.error('Missing article ID in data attribute');
                    }
                });
            }
            
            this.articlesContainer?.appendChild(articleCard);
        });
        
        // Add a debug counter
        const debugInfo = document.createElement('div');
        debugInfo.className = 'debug-info';
        debugInfo.textContent = `Displaying ${articles.length} articles`;
        debugInfo.style.marginTop = '20px';
        debugInfo.style.fontSize = '12px';
        this.articlesContainer.appendChild(debugInfo);
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
        
        // Add action buttons
        articleContent += `
            <div class="article-actions">
                <a href="${article.link}" target="_blank" class="article-link">View Original</a>
                <button id="back-to-articles" class="back-button">Back to Articles</button>
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
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize the embedding model when the app starts
    initEmbeddingModel().then(() => {
        console.log('Embedding model initialized on startup');
    });
    
    new AppController();
}); 