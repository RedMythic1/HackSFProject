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
}

interface ArticleDetail {
    title: string;
    link: string;
    summary: string;
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
    
    // Make sure vectors have non-zero length
    if (vecV.length === 0 || vecW.length === 0) {
        console.error("Empty vectors");
        return 25; // Return minimal score rather than 0
    }
    
    // Check for NaN or Infinity values
    const hasInvalidValues = (vec: number[]) => vec.some(v => isNaN(v) || !isFinite(v));
    if (hasInvalidValues(vecV) || hasInvalidValues(vecW)) {
        console.error("Vectors contain NaN or Infinity values");
        return 25; // Return minimal score rather than 0
    }
    
    // Calculate dot product of v and w
    let vDotW = 0;
    // Calculate dot product of v with itself
    let vDotV = 0;
    // Calculate dot product of w with itself
    let wDotW = 0;
    
    for (let i = 0; i < vecV.length; i++) {
        vDotW += vecV[i] * vecW[i];
        vDotV += vecV[i] * vecV[i];
        wDotW += vecW[i] * vecW[i];
    }
    
    // Guard against division by zero
    if (vDotV <= 0 || wDotW <= 0) {
        console.error("One of the vectors has zero magnitude");
        return 25; // Return minimal score rather than 0
    }
    
    // Calculate magnitudes
    const magnitudeV = Math.sqrt(vDotV);
    const magnitudeW = Math.sqrt(wDotW);
    
    // Calculate cosine similarity
    const cosineSimilarity = vDotW / (magnitudeV * magnitudeW);
    
    // Direct linear scaling from cosine similarity (-1 to 1) to score (0 to 100)
    const normalizedScore = (cosineSimilarity + 1) * 50;
    
    // Ensure the score is in the range [5, 100]
    const finalScore = Math.max(5, Math.min(100, normalizedScore));
    
    return finalScore;
};

// Article scoring function using embeddings
const scoreArticle = async (article: Article, userInterests: string): Promise<number> => {
    try {
        // Get user interests as text
        if (!userInterests.trim()) {
            return Math.floor(Math.random() * 25) + 70; // Default score if no interests
        }
        
        // Generate embeddings for the article and user interests
        const articleText = `${article.title}. ${article.subject}`;
        const articleEmbedding = await generateEmbedding(articleText);
        const interestsEmbedding = await generateEmbedding(userInterests);
        
        // Check if we have valid embeddings
        if (!articleEmbedding.length || !interestsEmbedding.length) {
            console.warn("Could not generate embeddings, using fallback scoring");
            return Math.floor(Math.random() * 25) + 70;
        }
        
        // Calculate similarity score
        const score = vectorSimilarity(articleEmbedding, interestsEmbedding);
        console.log(`Vector similarity score for "${article.title}": ${score}`);
        
        return score;
    } catch (error) {
        console.error('Error scoring article:', error);
        // Fallback to random score between 70-95 if scoring fails
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
            return data;
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
            console.log('Article details fetched successfully');
            return data;
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
        
        this.articlesContainer.innerHTML = '<div class="loading">Loading articles...</div>';
        
        try {
            const articles = await this.apiService.getArticles(interests);
            
            if (articles.length === 0) {
                this.articlesContainer.innerHTML = '<div class="error">No articles found</div>';
                return;
            }
            
            this.renderArticles(articles);
        } catch (error) {
            console.error('Error loading articles:', error);
            this.articlesContainer.innerHTML = '<div class="error">Error loading articles. Please try again later.</div>';
        }
    }
    
    private renderArticles(articles: Article[]): void {
        if (!this.articlesContainer) return;
        
        this.articlesContainer.innerHTML = '';
        
        articles.forEach(article => {
            // Extract article ID from link
            const articleId = article.link.split('id=')[1] || '';
            
            const articleCard = document.createElement('div');
            articleCard.className = 'article-card';
            articleCard.innerHTML = `
                <div class="article-header">
                    <h3 class="article-title">${this.escapeHtml(article.title)}</h3>
                    <span class="article-score">Score: ${article.score}</span>
                </div>
                <div class="article-body">
                    <div class="article-subject">${this.escapeHtml(article.subject)}</div>
                    <a href="/article/${articleId}" class="article-link" data-article-id="${articleId}">Read more</a>
                </div>
            `;
            
            // Add click event to view article details
            const articleLink = articleCard.querySelector('.article-link') as HTMLAnchorElement;
            if (articleLink) {
                articleLink.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.viewArticleDetails(articleId);
                });
            }
            
            this.articlesContainer?.appendChild(articleCard);
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
        articleDetail.innerHTML = `
            <h2>${this.escapeHtml(article.title)}</h2>
            <div class="article-summary">
                ${this.formatSummary(article.summary)}
            </div>
            <div class="article-actions">
                <a href="${article.link}" target="_blank" class="article-link">View Original</a>
                <button id="back-to-articles" class="back-button">Back to Articles</button>
            </div>
        `;
        
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
        const interests = this.interestsInput?.value || '';
        this.loadArticles(interests);
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
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize the embedding model when the app starts
    initEmbeddingModel().then(() => {
        console.log('Embedding model initialized on startup');
    });
    
    new AppController();
}); 