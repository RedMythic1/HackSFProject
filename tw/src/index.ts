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

/**
 * Custom ranking system for articles - combines multiple signals for more accurate scoring
 * @param article Article to score
 * @param userInterests User's specified interests
 * @param articleCollection Full collection of articles (for contextual ranking)
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
        
        // Split interest terms and normalize
        const interestTerms = userInterests.toLowerCase().split(/[\s,]+/).filter(term => term.length > 2);
        console.log(`Parsed interest terms (${interestTerms.length}):`, interestTerms);
        
        if (interestTerms.length === 0) {
            console.log(`No valid interest terms after filtering - Using base score: ${baseScore}`);
            return baseScore; // No valid interest terms
        }
        
        // Load summary from .cache/summary_{article.id}.json if possible
        let articleSummary = '';
        let summarySource = 'none';
        
        if (article.id) {
            try {
                const summaryPath = `.cache/summary_${article.id}.json`;
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
        
        const articleText = `${article.title} ${article.subject || ''} ${articleSummary}`.toLowerCase();
        const articleSummaryLower = articleSummary.toLowerCase();
        
        console.log(`\n----- SCORING COMPONENT 1: KEYWORD FREQUENCY IN SUMMARY (0-30 points) -----`);
        // ---- 1. KEYWORD FREQUENCY IN SUMMARY (0-30 points) ----
        let keywordFrequencyScore = 0;
        if (articleSummaryLower.length > 0) {
            console.log(`Analyzing summary text (${articleSummaryLower.length} chars)`);
            let totalKeywordCount = 0;
            let termCounts = new Map<string, number>();
            
            // Count each keyword occurrence in the summary
            interestTerms.forEach(term => {
                let count = 0;
                let position = articleSummaryLower.indexOf(term);
                
                while (position !== -1) {
                    count++;
                    position = articleSummaryLower.indexOf(term, position + 1);
                }
                
                termCounts.set(term, count);
                totalKeywordCount += count;
                console.log(`  Term "${term}": ${count} occurrences`);
            });
            
            // Calculate percentage of keywords in the summary (words)
            const summaryWordCount = articleSummaryLower.split(/\s+/).length;
            console.log(`Summary word count: ${summaryWordCount} words`);
            console.log(`Total keyword occurrences: ${totalKeywordCount}`);
            
            const keywordPercentage = summaryWordCount > 0 ? 
                (totalKeywordCount / summaryWordCount) * 100 : 0;
            
            console.log(`Keyword percentage: ${keywordPercentage.toFixed(2)}%`);
            
            // Convert to score (max 30 points)
            // We cap at 10% to avoid overweighting articles that just repeat keywords
            keywordFrequencyScore = Math.min(30, keywordPercentage * 3);
            console.log(`Keyword frequency score: ${keywordFrequencyScore.toFixed(2)} / 30 points (${keywordPercentage.toFixed(2)}% × 3, capped at 30)`);
        } else {
            console.log(`Empty summary - Keyword frequency score: 0 / 30 points`);
        }
        
        console.log(`\n----- SCORING COMPONENT 2: EXACT MATCH SCORE (0-20 points) -----`);
        // ---- 2. EXACT MATCH SCORE (0-20 points) ----
        let exactMatchScore = 0;
        let exactMatchCount = 0;
        
        // Count exact matches
        interestTerms.forEach(term => {
            if (articleText.includes(term)) {
                exactMatchCount++;
                console.log(`  Exact match found for "${term}"`);
            } else {
                console.log(`  No exact match for "${term}"`);
            }
        });
        
        // Calculate normalized exact match score
        console.log(`Total exact matches: ${exactMatchCount} / ${interestTerms.length} terms`);
        const normalizedExactMatches = exactMatchCount / interestTerms.length;
        console.log(`Normalized exact match score: ${normalizedExactMatches.toFixed(2)}`);
        exactMatchScore = normalizedExactMatches * 20;
        console.log(`Exact match score: ${exactMatchScore.toFixed(2)} / 20 points (${normalizedExactMatches.toFixed(2)} × 20)`);
        
        console.log(`\n----- SCORING COMPONENT 3: VECTOR SIMILARITY METRICS (0-40 points) -----`);
        // ---- 3. VECTOR SIMILARITY METRICS (0-40 points) ----
        let vectorScore = 0;
        let vectorDistanceScore = 0;
        let cosineSimilarityScore = 0;
        
        // Get article embedding (either from summary or title+subject)
        let articleEmbedding: number[] = [];
        let embeddingSource = 'none';
        
        try {
            if (article.id) {
                // Try to load the summary file for this article
                const summaryPath = `.cache/summary_${article.id}.json`;
                console.log(`Attempting to load embedding from: ${summaryPath}`);
                const response = await fetch(summaryPath);
                if (response.ok) {
                    const summaryData = await response.json();
                    if (summaryData.embedding && Array.isArray(summaryData.embedding)) {
                        articleEmbedding = summaryData.embedding;
                        embeddingSource = 'cache file';
                        console.log(`Successfully loaded embedding from file (${articleEmbedding.length} dimensions)`);
                    } else {
                        console.log(`No valid embedding found in summary file`);
                    }
                } else {
                    console.log(`Failed to load embedding from file: ${response.status}`);
                }
            }
            
            if (!articleEmbedding.length) {
                // Fallback to generating embedding from title+subject
                console.log(`Fallback: Generating new embedding from title+subject`);
                const fallbackText = `${article.title}. ${article.subject || ''}`;
                console.log(`Generating embedding for text: "${fallbackText}"`);
                articleEmbedding = await generateEmbedding(fallbackText);
                if (articleEmbedding.length) {
                    embeddingSource = 'generated';
                    console.log(`Successfully generated embedding (${articleEmbedding.length} dimensions)`);
                } else {
                    console.log(`Failed to generate embedding`);
                }
            }
            
            // Get interests embedding
            console.log(`Generating embedding for user interests: "${userInterests}"`);
            const interestsEmbedding = await generateEmbedding(userInterests);
            console.log(`User interests embedding generated (${interestsEmbedding.length} dimensions)`);
            
            if (articleEmbedding.length && interestsEmbedding.length) {
                console.log(`Computing vector similarity between article (${embeddingSource}) and interests...`);
                
                // Calculate cosine similarity (0-25 points)
                const similarity = vectorSimilarity(articleEmbedding, interestsEmbedding);
                console.log(`Cosine similarity: ${similarity.toFixed(2)} / 100`);
                cosineSimilarityScore = (similarity / 100) * 25;
                console.log(`Cosine similarity score: ${cosineSimilarityScore.toFixed(2)} / 25 points (${similarity.toFixed(2)}% × 0.25)`);
                
                // Calculate vector distance (0-15 points)
                // Lower distance = better match = higher score
                if (articleEmbedding.length === interestsEmbedding.length) {
                    // Calculate Euclidean distance between vectors
                    let sumSquaredDiff = 0;
                    for (let i = 0; i < articleEmbedding.length; i++) {
                        const diff = articleEmbedding[i] - interestsEmbedding[i];
                        sumSquaredDiff += diff * diff;
                    }
                    const distance = Math.sqrt(sumSquaredDiff);
                    console.log(`Euclidean distance: ${distance.toFixed(4)}`);
                    
                    // Normalize distance to score (inversely related)
                    // We expect most distances to be between 0-2 for normalized vectors
                    const normalizedDistance = Math.min(2, distance);
                    console.log(`Normalized distance: ${normalizedDistance.toFixed(4)} (capped at 2.0)`);
                    vectorDistanceScore = 15 * (1 - (normalizedDistance / 2));
                    console.log(`Vector distance score: ${vectorDistanceScore.toFixed(2)} / 15 points (15 × (1 - ${normalizedDistance.toFixed(2)}/2))`);
                } else {
                    console.log(`Vector dimensions don't match: ${articleEmbedding.length} vs ${interestsEmbedding.length} - Cannot calculate distance`);
                }
                
                // Combine vector scores
                vectorScore = cosineSimilarityScore + vectorDistanceScore;
                console.log(`Combined vector score: ${vectorScore.toFixed(2)} / 40 points (${cosineSimilarityScore.toFixed(2)} + ${vectorDistanceScore.toFixed(2)})`);
            } else {
                console.log(`Missing embeddings - Cannot calculate vector similarity`);
            }
        } catch (err) {
            console.warn(`Error calculating vector scores:`, err);
            // Default vector score if we can't calculate it
            vectorScore = 20;
            console.log(`Using default vector score due to error: ${vectorScore} / 40 points`);
        }
        
        console.log(`\n----- SCORING COMPONENT 4: FRESHNESS FACTOR (0-10 points) -----`);
        // ---- 4. FRESHNESS FACTOR (0-10 points) ----
        // A consistent value based on article ID to ensure the same article always gets the same freshness score
        const freshnessScore = article.id ? 
            (parseInt(article.id.replace(/\D/g, '').slice(-2) || '0') % 10) : 
            Math.floor(Math.random() * 10);
        console.log(`Freshness score calculation: article.id=${article.id}, extracted digits=${article.id ? article.id.replace(/\D/g, '').slice(-2) : 'none'}`);
        console.log(`Freshness score: ${freshnessScore} / 10 points`);
        
        console.log(`\n----- FINAL SCORE CALCULATION -----`);
        // ---- COMBINE SCORES ----
        const finalScore = keywordFrequencyScore + exactMatchScore + vectorScore + freshnessScore;
        console.log(`Final score breakdown:
  - Keyword Frequency: ${keywordFrequencyScore.toFixed(2)} / 30
  - Exact Match:      ${exactMatchScore.toFixed(2)} / 20
  - Vector Similarity: ${vectorScore.toFixed(2)} / 40
    - Cosine Similarity: ${cosineSimilarityScore.toFixed(2)} / 25
    - Vector Distance:   ${vectorDistanceScore.toFixed(2)} / 15
  - Freshness:        ${freshnessScore} / 10
  ----------------------
  TOTAL SCORE:        ${finalScore.toFixed(2)} / 100
`);
        
        // Log all scoring components
        const scoreBreakdown = {
            keywordFrequency: keywordFrequencyScore.toFixed(2),
            exactMatch: exactMatchScore.toFixed(2),
            vectorTotal: vectorScore.toFixed(2),
            cosineSimilarity: cosineSimilarityScore.toFixed(2),
            vectorDistance: vectorDistanceScore.toFixed(2),
            freshness: freshnessScore.toString(),
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
            
            // Score all articles using the custom ranking system
            const scoringPromises = articles.map(article => 
                customRankingSystem(article, interests || '', articles)
            );
            
            const scores = await Promise.all(scoringPromises);
            
            // Apply scores to articles
            const scoredArticles = articles.map((article, index) => ({
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
                    <div class="score-container">
                        <span class="article-score">Score: ${article.score}</span>
                        <div class="score-breakdown">
                            <div class="score-meter">
                                <div class="relevance" style="width: ${Math.min(75, article.score)}%"></div>
                            </div>
                            <div class="score-tags">
                                ${article.score > 90 ? '<span class="tag excellent">Excellent Match</span>' : 
                                  article.score > 75 ? '<span class="tag good">Good Match</span>' : 
                                  article.score > 60 ? '<span class="tag moderate">Moderate Match</span>' :
                                  '<span class="tag low">Low Match</span>'}
                            </div>
                            ${article.scoreComponents ? `
                            <div class="score-details">
                                <div class="score-component">
                                    <span class="component-label">Keyword Frequency:</span>
                                    <span class="component-value">${article.scoreComponents.keywordFrequency}</span>
                                </div>
                                <div class="score-component">
                                    <span class="component-label">Exact Match:</span>
                                    <span class="component-value">${article.scoreComponents.exactMatch}</span>
                                </div>
                                <div class="score-component">
                                    <span class="component-label">Vector Similarity:</span>
                                    <span class="component-value">${article.scoreComponents.vectorTotal}</span>
                                </div>
                            </div>` : ''}
                        </div>
                    </div>
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