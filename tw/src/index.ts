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

class App {
    private emailForm: HTMLFormElement | null = null;
    private emailInput: HTMLInputElement | null = null;
    private interestsForm: HTMLFormElement | null = null;
    private interestsInput: HTMLInputElement | null = null;
    private commandForm: HTMLFormElement | null = null;
    private commandInput: HTMLInputElement | null = null;
    private commandOutput: HTMLDivElement | null = null;
    private loadingSection: HTMLDivElement | null = null;
    private loadingMessage: HTMLParagraphElement | null = null;
    private emailSection: HTMLDivElement | null = null;
    private interestsSection: HTMLDivElement | null = null;
    private adminToggleButton: HTMLButtonElement | null = null;
    private embeddingExtractor: any = null;
    private calculationLogs: CalculationStep[] = [];
    private articleEmbeddingCache: Map<string, number[]> = new Map();
    private cacheInitialized: boolean = false;

    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.copyArticleFiles();
        this.syncAndCheckCache();
        this.initEmbeddingModel().then(() => {
            this.preloadArticleEmbeddings();
        });
    }

    private initializeElements(): void {
        // Get form elements
        this.emailForm = document.getElementById('email-form') as HTMLFormElement;
        this.emailInput = document.getElementById('email-input') as HTMLInputElement;
        this.interestsForm = document.getElementById('interests-form') as HTMLFormElement;
        this.interestsInput = document.getElementById('interests-input') as HTMLInputElement;
        this.commandForm = document.getElementById('command-form') as HTMLFormElement;
        this.commandInput = document.getElementById('command-input') as HTMLInputElement;
        
        // Get output element
        this.commandOutput = document.getElementById('command-output') as HTMLDivElement;
        
        // Get section elements
        this.loadingSection = document.getElementById('loading-section') as HTMLDivElement;
        this.loadingMessage = document.getElementById('loading-message') as HTMLParagraphElement;
        this.emailSection = document.getElementById('email-section') as HTMLDivElement;
        this.interestsSection = document.getElementById('interests-section') as HTMLDivElement;
    }

    private setupEventListeners(): void {
        // Add form submit event listeners
        if (this.emailForm) {
            this.emailForm.addEventListener('submit', this.handleEmailSubmit.bind(this));
        } else {
            console.error('Email form not found');
        }
        
        if (this.interestsForm) {
            this.interestsForm.addEventListener('submit', this.handleInterestsSubmit.bind(this));
        } else {
            console.error('Interests form not found');
        }
        
        if (this.commandForm) {
            this.commandForm.addEventListener('submit', this.handleCommandSubmit.bind(this));
        } else {
            console.error('Command form not found');
        }
        
        document.addEventListener('keydown', (event) => {
            // Handle keyboard shortcuts (like Escape to close modals)
            if (event.key === 'Escape') {
                const modals = document.querySelectorAll('.article-modal');
                if (modals.length > 0) {
                    modals.forEach(modal => {
                        document.body.removeChild(modal);
                    });
                }
            }
        });
    }

    private async syncAndCheckCache(): Promise<void> {
        try {
            // Synchronize the local cache with the main cache
            await logToFile('Synchronizing local cache with main cache...', 'info');
            this.updateLoadingMessage('Synchronizing local cache with main cache...');
            
            // Perform the sync asynchronously
            const syncPromise = new Promise<void>(async (resolve) => {
                try {
                    const stats = await syncCache();
                    logToFile(`Cache sync complete: ${stats.added} added, ${stats.updated} updated, ${stats.skipped} skipped`, 'info');
                    this.cacheInitialized = true;
                    resolve();
                } catch (error) {
                    logToFile(`Error syncing cache: ${error}`, 'error');
                    // Continue even if sync fails
                    this.cacheInitialized = false;
                    resolve();
                }
            });
            
            // Wait for the sync to complete
            await syncPromise;
            
            // Now proceed with the normal cache check
            await this.checkCache();
        } catch (error) {
            await logToFile(`Error in syncAndCheckCache: ${error}`, 'error');
            // Continue to the cache check anyway
            await this.checkCache();
        }
    }

    private async checkCache(): Promise<void> {
        try {
            await logToFile('Checking article cache...', 'info');
            this.updateLoadingMessage('Checking article cache...');
            
            // First check if our local cache is initialized
            if (this.cacheInitialized) {
                this.updateLoadingMessage('Using local cache...');
                
                // Try to fetch cache status from backend
                const response = await fetch(`${API_URL}/check-cache`, {
                    credentials: 'include'
                });
                const data = await response.json();
                
                if (data.cached) {
                    const validCount = data.valid_article_count || 0;
                    const uniqueCount = data.final_article_count || 0;
                    
                    await logToFile(`Found ${data.article_count} cached articles and ${uniqueCount} final articles`, 'info');
                    this.updateLoadingMessage(`Found ${data.article_count} cached articles and ${uniqueCount} final articles. Preparing interface...`);
                    
                    // Add a status element for live cache updates - INSERT AT TOP
                    const cacheStatusDiv = document.createElement('div');
                    cacheStatusDiv.classList.add('cache-info');
                    cacheStatusDiv.textContent = `Cached articles: ${data.article_count}, Final articles: ${uniqueCount}`;
                    // Insert at the top instead of appending
                    this.commandOutput?.insertBefore(cacheStatusDiv, this.commandOutput.firstChild);
                    
                    // Don't show any final articles automatically - hide until user requests them
                    if (uniqueCount > 0) {
                        setTimeout(() => {
                            // Use the addMessageToOutput method which already inserts at the top
                            this.addMessageToOutput(`${uniqueCount} final articles with questions & answers are available.`, 'info');
                            
                            // Add a button to view articles if we have any
                            const viewArticlesButton = document.createElement('button');
                            viewArticlesButton.textContent = 'View Final Articles';
                            viewArticlesButton.classList.add('view-articles-btn');
                            viewArticlesButton.addEventListener('click', () => {
                                this.handleViewArticles(true);
                            });
                            
                            // Create a container for the button
                            const buttonContainer = document.createElement('div');
                            buttonContainer.classList.add('button-container');
                            buttonContainer.appendChild(viewArticlesButton);
                            
                            // Insert at the top instead of appending
                            this.commandOutput?.insertBefore(buttonContainer, this.commandOutput.firstChild);
                        }, 3000);
                    }
                } else {
                    this.updateLoadingMessage('No cached articles found. Use shell commands to process articles first.');
                    
                    // Add an empty status element that will be updated by polling - INSERT AT TOP
                    const cacheStatusDiv = document.createElement('div');
                    cacheStatusDiv.classList.add('cache-info');
                    cacheStatusDiv.textContent = 'Cached articles: 0, Final articles: 0';
                    // Insert at the top instead of appending
                    this.commandOutput?.insertBefore(cacheStatusDiv, this.commandOutput.firstChild);
                }
            } else {
                // Continue with the original API check if local cache initialization failed
                // Call the backend endpoint
                const response = await fetch(`${API_URL}/check-cache`, {
                    credentials: 'include'
                });
                const data = await response.json();
                
                // Process the response as before...
                // This is the existing implementation
                if (data.cached) {
                    const validCount = data.valid_article_count || 0;
                    const uniqueCount = data.final_article_count || 0;
                    
                    await logToFile(`Found ${data.article_count} cached articles and ${uniqueCount} final articles`, 'info');
                    this.updateLoadingMessage(`Found ${data.article_count} cached articles and ${uniqueCount} final articles. Preparing interface...`);
                    
                    // Add a status element for live cache updates - INSERT AT TOP
                    const cacheStatusDiv = document.createElement('div');
                    cacheStatusDiv.classList.add('cache-info');
                    cacheStatusDiv.textContent = `Cached articles: ${data.article_count}, Final articles: ${uniqueCount}`;
                    // Insert at the top instead of appending
                    this.commandOutput?.insertBefore(cacheStatusDiv, this.commandOutput.firstChild);
                    
                    // Same code as above for showing the button...
                    if (uniqueCount > 0) {
                        setTimeout(() => {
                            // Use the addMessageToOutput method which already inserts at the top
                            this.addMessageToOutput(`${uniqueCount} final articles with questions & answers are available.`, 'info');
                            
                            const viewArticlesButton = document.createElement('button');
                            viewArticlesButton.textContent = 'View Final Articles';
                            viewArticlesButton.classList.add('view-articles-btn');
                            viewArticlesButton.addEventListener('click', () => {
                                this.handleViewArticles(true);
                            });
                            
                            const buttonContainer = document.createElement('div');
                            buttonContainer.classList.add('button-container');
                            buttonContainer.appendChild(viewArticlesButton);
                            
                            // Insert at the top instead of appending
                            this.commandOutput?.insertBefore(buttonContainer, this.commandOutput.firstChild);
                        }, 3000);
                    }
                } else {
                    this.updateLoadingMessage('No cached articles found. Use shell commands to process articles first.');
                    
                    const cacheStatusDiv = document.createElement('div');
                    cacheStatusDiv.classList.add('cache-info');
                    cacheStatusDiv.textContent = 'Cached articles: 0, Final articles: 0';
                    // Insert at the top instead of appending
                    this.commandOutput?.insertBefore(cacheStatusDiv, this.commandOutput.firstChild);
                }
            }
            
            // Artificial delay to ensure everything is ready
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            // Hide loading and show email form
            this.toggleSections('loading-section', 'email-section');
            
        } catch (error) {
            await logToFile(`Error checking cache: ${error}`, 'error');
            this.addMessageToOutput(`Error checking cache: ${error}`, 'error');
            
            // Continue to the app interface even on error
            setTimeout(() => {
                this.toggleSections('loading-section', 'email-section');
            }, 2000);
        }
    }
    
    private updateLoadingMessage(message: string): void {
        if (this.loadingMessage) {
            this.loadingMessage.textContent = message;
        }
    }

    private async handleEmailSubmit(event: Event): Promise<void> {
        event.preventDefault();
        
        if (!this.emailInput || !this.commandOutput) {
            return;
        }

        const email = this.emailInput.value.trim();
        
        if (!email) {
            this.addMessageToOutput('Please enter a valid email address', 'error');
            return;
        }
        
        // Basic email validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            this.addMessageToOutput('Please enter a valid email address', 'error');
            return;
        }

        try {
            // Call the backend endpoint
            const response = await fetch(`${API_URL}/verify-email`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                credentials: 'include', // Include cookies in the request
                body: JSON.stringify({ email })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Display success message
                this.addMessageToOutput(`Email registered: ${email}`, 'success');
                
                // Hide email section and show interests section
                this.toggleSections('email-section', 'interests-section');
            } else {
                this.addMessageToOutput(data.message || 'Error processing email', 'error');
            }
        } catch (error) {
            await logToFile(`Error verifying email: ${error}`, 'error');
            console.error('Error verifying email:', error);
            this.addMessageToOutput(`Error verifying email: ${error}`, 'error');
        }
    }

    private async handleInterestsSubmit(event: Event): Promise<void> {
        event.preventDefault();
        
        if (!this.interestsInput || !this.commandOutput) {
            return;
        }

        const interests = this.interestsInput.value.trim();
        
        if (!interests) {
            this.addMessageToOutput('Please enter your interests', 'error');
            return;
        }

        await logToFile(`User submitted interests: ${interests}`, 'info');

        // Clear any previous articles from the view
        const previousArticles = document.querySelectorAll('.articles-container, .article-recommendation, .article-content');
        previousArticles.forEach(element => element.remove());

        // Show loading message
        this.addMessageToOutput(`Analyzing articles for interests: ${interests}...`, 'info');
        
        try {
            // Generate embedding for the user's interests
            await logToFile('Generating embedding for user interests', 'info');
            const interestsEmbedding = await this.generateEmbedding(interests);
            
            if (!interestsEmbedding || interestsEmbedding.length === 0) {
                await logToFile('Could not generate embeddings for interests', 'error');
                this.addMessageToOutput("Could not generate embeddings. Using keyword matching fallback.", "error");
            } else {
                await logToFile(`Successfully generated embedding with ${interestsEmbedding.length} dimensions`, 'info');
                this.addMessageToOutput(`Generated vector embedding with ${interestsEmbedding.length} dimensions`, "info");
            }
            
            // First try to use the direct /analyze-interests endpoint
            await logToFile('Calling backend analyze-interests endpoint', 'info');
            const response = await fetch(`${API_URL}/analyze-interests`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                credentials: 'include', // Include cookies in the request
                body: JSON.stringify({ interests })
            });
            
            const data = await response.json();
            await logToFile(`Received response from analyze-interests: ${data.status}`, 'info');
            
            if (data.status === 'success' && data.articles) {
                await logToFile(`Got ${data.articles.length} articles from backend, processing with vector similarity`, 'info');
                // Process the articles with vector similarity
                const scoredArticles = await this.matchArticlesByInterest(interests, data.articles, interestsEmbedding);
                
                if (scoredArticles && scoredArticles.length > 0) {
                    // Find the top article (with highest score)
                    const bestArticle = scoredArticles[0];
                    await logToFile(`Best matching article: "${bestArticle.title}" with score ${bestArticle.match_score}/100`, 'info');
                    
                    // Display best match with calculation table
                    this.displayBestMatch(bestArticle, interests);
                    
                    // Add a "View Other Articles" button
                    const viewAllBtn = document.createElement('button');
                    viewAllBtn.textContent = 'View Other Articles';
                    viewAllBtn.className = 'view-all-btn';
                    viewAllBtn.addEventListener('click', () => {
                        // Display all articles
                        const articlesContainer = document.createElement('div');
                        articlesContainer.className = 'all-articles-container';
                        
                        // Add heading
                        const heading = document.createElement('h3');
                        heading.textContent = 'All Matching Articles';
                        articlesContainer.appendChild(heading);
                        
                        // Add each article with scores
                        scoredArticles.forEach((article, index) => {
                            if (index === 0) return; // Skip best match as it's already displayed
                            
                            const articleDiv = document.createElement('div');
                            articleDiv.className = 'article-item';
                            
                            const title = document.createElement('h4');
                            title.textContent = article.title;
                            
                            const score = document.createElement('div');
                            score.className = 'item-score';
                            score.textContent = `Match: ${article.match_score.toFixed(1)}/100`;
                            
                            const viewBtn = document.createElement('button');
                            viewBtn.textContent = 'View';
                            viewBtn.className = 'view-item-btn';
                            viewBtn.dataset.articleId = article.id;
                            viewBtn.addEventListener('click', (e) => {
                                e.preventDefault();
                                this.handleViewArticle(e);
                            });
                            
                            articleDiv.appendChild(title);
                            articleDiv.appendChild(score);
                            articleDiv.appendChild(viewBtn);
                            
                            articlesContainer.appendChild(articleDiv);
                        });
                        
                        // Insert articles container at the top instead of appending
                        this.commandOutput?.insertBefore(articlesContainer, this.commandOutput.firstChild);
                    });
                    
                    // Insert view all button at the top instead of appending
                    this.commandOutput.insertBefore(viewAllBtn, this.commandOutput.firstChild);
                    
                    // Add modern styling
                    const style = document.createElement('style');
                    style.textContent = `
                        .view-all-btn {
                            background-color: #3f51b5;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            padding: 8px 16px;
                            font-size: 14px;
                            cursor: pointer;
                            margin: 10px 0;
                            transition: background-color 0.3s ease;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                        }
                        
                        .view-all-btn:hover {
                            background-color: #303f9f;
                        }
                        
                        .article-item {
                            background-color: #fff;
                            margin: 8px 0;
                            padding: 16px;
                            border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            transition: transform 0.2s ease, box-shadow 0.2s ease;
                        }
                        
                        .article-item:hover {
                            transform: translateY(-2px);
                            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                        }
                        
                        .all-articles-container h3 {
                            font-size: 18px;
                            color: #333;
                            margin-bottom: 16px;
                            padding-bottom: 8px;
                            border-bottom: 2px solid #e0e0e0;
                        }
                        
                        .view-item-btn {
                            background-color: #2196F3;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            padding: 6px 12px;
                            cursor: pointer;
                            transition: background-color 0.2s ease;
                        }
                        
                        .view-item-btn:hover {
                            background-color: #1976D2;
                        }
                        
                        .item-score {
                            display: inline-block;
                            padding: 4px 8px;
                            margin: 8px 0;
                            background-color: #f5f5f5;
                            border-radius: 16px;
                            font-size: 13px;
                            color: #555;
                        }
                    `;
                    document.head.appendChild(style);
                    
                } else {
                    await logToFile('No matching articles found after scoring', 'error');
                    this.addMessageToOutput('No matching articles found for your interests.', 'error');
                }
            } else {
                await logToFile(`Failed to analyze interests: ${data.message || 'Unknown error'}`, 'error');
                this.addMessageToOutput(data.message || 'Failed to analyze interests', 'error');
            }
        } catch (error) {
            await logToFile(`Error in handleInterestsSubmit: ${error}`, 'error');
            console.error('Error analyzing interests:', error);
            this.addMessageToOutput(`Error analyzing interests: ${error}`, 'error');
        }
    }
    
    // Initialize the embedding model
    private async initEmbeddingModel(): Promise<void> {
        try {
            await logToFile('Initializing embedding model...', 'info');
            this.updateLoadingMessage('Initializing embedding model...');
            this.embeddingExtractor = await pipeline(
                'feature-extraction',
                'Xenova/all-MiniLM-L6-v2'
            );
            await logToFile('Embedding model initialized successfully', 'info');
        } catch (error) {
            await logToFile(`Error initializing embedding model: ${error}`, 'error');
        }
    }
    
    // Generate embeddings for text input
    private async generateEmbedding(text: string): Promise<number[]> {
        if (!this.embeddingExtractor) {
            await logToFile('Embedding extractor not initialized, initializing now...', 'info');
            await this.initEmbeddingModel();
        }
        
        try {
            await logToFile(`Generating embedding for text: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`, 'info');
            
            // Limit input text length to avoid issues with very long texts
            // Most embedding models work best with 512-1024 tokens
            let inputText = text;
            if (text.length > 2048) {
                inputText = text.substring(0, 2048);
                await logToFile(`Text truncated from ${text.length} to 2048 chars for embedding generation`, 'info');
            }
            
            const result = await this.embeddingExtractor(inputText, {
                pooling: 'mean',
                normalize: true
            });
            
            // Convert tensor to regular array with proper typing
            // Map each value explicitly to number to fix the TypeScript error
            const embedding = Array.from(result.data).map(value => Number(value));
            await logToFile(`Generated embedding with ${embedding.length} dimensions`, 'info');
            return embedding;
        } catch (error) {
            await logToFile(`Error generating embedding: ${error}`, 'error');
            return [];
        }
    }
    
    // Get article embedding - either from backend, cache, or generate new one
    private async getArticleEmbedding(article: any, summary: string, title: string): Promise<number[]> {
        const articleId = article.id;
        
        // First check if we already have a cached embedding for this article
        if (this.articleEmbeddingCache.has(articleId)) {
            const cachedEmbedding = this.articleEmbeddingCache.get(articleId);
            await logToFile(`Using cached embedding for article "${title}" (ID: ${articleId})`, 'info');
            return cachedEmbedding!;
        }
        
        // If article has an embedding from backend, use it
        if (article.embedding && Array.isArray(article.embedding) && article.embedding.length > 0) {
            await logToFile(`Using backend embedding for article "${title}" (ID: ${articleId})`, 'info');
            // Cache the embedding for future use
            this.articleEmbeddingCache.set(articleId, article.embedding);
            return article.embedding;
        }
        
        // No embedding available, generate a new one
        await logToFile(`No embedding found for article "${title}" (ID: ${articleId}), generating one`, 'info');
        
        // Use summary for embedding if available, otherwise use title
        const textForEmbedding = summary || title;
        const newEmbedding = await this.generateEmbedding(textForEmbedding);
        
        if (newEmbedding && newEmbedding.length > 0) {
            await logToFile(`Successfully generated article embedding with ${newEmbedding.length} dimensions for article ${articleId}`, 'info');
            // Cache the embedding for future use
            this.articleEmbeddingCache.set(articleId, newEmbedding);
            return newEmbedding;
        } else {
            await logToFile(`Failed to generate article embedding for article ${articleId}`, 'error');
            return [];
        }
    }
    
    // Calculate custom similarity between two vectors
    private async vectorSimilarity(vecV: number[], vecW: number[]): Promise<number> {
        // Reset calculation logs for new calculation
        this.calculationLogs = [];
        
        const logStep = async (step: string, value: any): Promise<void> => {
            this.calculationLogs.push({ step, value });
            const logMessage = `${step}: ${typeof value === 'number' ? value.toFixed(6) : value}`;
            await logToFile(logMessage, 'debug');
        };
        
        // Make sure we have valid inputs
        if (!vecV || !vecW || !Array.isArray(vecV) || !Array.isArray(vecW)) {
            await logStep("Error", "Invalid vector inputs");
            return 25; // Return minimal score rather than 0
        }
        
        if (vecV.length !== vecW.length) {
            await logStep("Error", `Vector dimensions don't match: ${vecV.length} vs ${vecW.length}`);
            return 25; // Return minimal score rather than 0
        }
        
        // Make sure vectors have non-zero length
        if (vecV.length === 0 || vecW.length === 0) {
            await logStep("Error", "Empty vectors");
            return 25; // Return minimal score rather than 0
        }
        
        // Check for NaN or Infinity values
        const hasInvalidValues = (vec: number[]) => vec.some(v => isNaN(v) || !isFinite(v));
        if (hasInvalidValues(vecV) || hasInvalidValues(vecW)) {
            await logStep("Error", "Vectors contain NaN or Infinity values");
            return 25; // Return minimal score rather than 0
        }
        
        // Log the input vectors (just first 5 elements for readability)
        await logStep("Vector V (first 5)", vecV.slice(0, 5).map(v => v.toFixed(4)).join(", "));
        await logStep("Vector W (first 5)", vecW.slice(0, 5).map(w => w.toFixed(4)).join(", "));
        
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
        
        await logStep("v·w (dot product)", vDotW);
        await logStep("v·v (dot product)", vDotV);
        await logStep("w·w (dot product)", wDotW);
        
        // Guard against division by zero
        if (vDotV <= 0 || wDotW <= 0) {
            await logStep("Error", "One of the vectors has zero magnitude");
            return 25; // Return minimal score rather than 0
        }
        
        // Calculate magnitudes
        const magnitudeV = Math.sqrt(vDotV);
        const magnitudeW = Math.sqrt(wDotW);
        
        await logStep("||v|| (magnitude of v)", magnitudeV);
        await logStep("||w|| (magnitude of w)", magnitudeW);
        
        // Calculate cosine similarity
        const cosineSimilarity = vDotW / (magnitudeV * magnitudeW);
        await logStep("Cosine similarity", cosineSimilarity);
        
        // Direct linear scaling from cosine similarity (-1 to 1) to score (0 to 100)
        const normalizedScore = (cosineSimilarity + 1) * 50;
        await logStep("Normalized score (0-100)", normalizedScore);
        
        // Ensure the score is in the range [5, 100]
        const finalScore = Math.max(5, Math.min(100, normalizedScore));
        await logStep("Final score", finalScore);
        
        return finalScore;
    }
    
    // Updated method to use only vector similarity, no keyword matching
    private async matchArticlesByInterest(interests: string, articles: any[], interestsEmbedding?: number[]): Promise<any[]> {
        // Parse interests into keywords (for logging only)
        const interestKeywords = interests.split(',')
            .map(interest => interest.trim().toLowerCase())
            .filter(interest => interest.length > 0);
            
        if (interestKeywords.length === 0) return [];
        
        await logToFile(`Starting to match ${articles.length} articles with interests: ${interests}`, 'info');
        await logToFile(`Interest embedding available: ${interestsEmbedding ? 'Yes' : 'No'}`, 'info');
        
        // Score each article based purely on vector similarity
        const scoredArticles = await Promise.all(articles.map(async (article) => {
            try {
                await logToFile(`Processing article: "${article.title}" (ID: ${article.id})`, 'info');
                
                // First, try to get the article content
                const articleResponse = await fetch(`${API_URL}/get-final-article/${article.id}`);
                const articleData = await articleResponse.json();
                
                if (articleData.status === 'success') {
                    const title = articleData.article.title.toLowerCase();
                    const content = articleData.article.content.toLowerCase();
                    
                    // Extract or use summary if available
                    let summary = '';
                    if (articleData.article.summary) {
                        summary = articleData.article.summary.toLowerCase();
                        await logToFile(`Using provided summary for article "${article.title}"`, 'info');
                    } else {
                        // Try to extract summary from content if not provided
                        const summaryMatch = content.match(/## summary\s+([\s\S]+?)(?=##|$)/i);
                        if (summaryMatch && summaryMatch[1]) {
                            summary = summaryMatch[1].trim().toLowerCase();
                            await logToFile(`Extracted summary from content for article "${article.title}"`, 'info');
                        } else if (content.length > 0) {
                            // Use first paragraph as fallback summary
                            const paragraphs = content.split('\n\n');
                            // Skip title paragraph if it exists
                            const startIdx = paragraphs[0].startsWith('#') ? 1 : 0;
                            if (paragraphs.length > startIdx) {
                                summary = paragraphs[startIdx].trim().toLowerCase();
                                await logToFile(`Using first paragraph as summary for article "${article.title}"`, 'info');
                            }
                        }
                    }
                    
                    // Log summary length
                    if (summary) {
                        await logToFile(`Summary length: ${summary.length} characters`, 'debug');
                    } else {
                        await logToFile(`No summary available, using full content`, 'debug');
                        summary = content; // Fallback to full content if no summary
                    }
                    
                    let score = 0;
                    let explanation = '';
                    
                    // Debug log
                    await logToFile(`Article has embedding: ${articleData.article.embedding ? 'Yes' : 'No'}`, 'debug');
                    
                    // Check if we have embeddings for the interests
                    if (interestsEmbedding && interestsEmbedding.length > 0) {
                        // Get article embedding - either from backend, cache, or generate new one
                        const articleEmbedding = await this.getArticleEmbedding(articleData.article, summary, title);
                        
                        // Make sure we have valid embeddings
                        if (!Array.isArray(articleEmbedding) || articleEmbedding.length === 0) {
                            await logToFile("Article embedding is invalid or empty", 'error');
                            score = 25; // Default score when embedding is invalid
                            explanation = "Using fallback scoring due to invalid article embedding.";
                        } else {
                            await logToFile(`Article embedding dimensions: ${articleEmbedding.length}`, 'debug');
                            await logToFile(`Interest embedding dimensions: ${interestsEmbedding.length}`, 'debug');
                            
                            // If the dimensions don't match, we need to handle this case
                            if (articleEmbedding.length !== interestsEmbedding.length) {
                                await logToFile(`Embedding dimension mismatch: article ${articleEmbedding.length}, interests ${interestsEmbedding.length}`, 'error');
                                score = 30; // Use a slightly higher fallback score
                                explanation = "Using estimated scoring due to embedding dimension mismatch.";
                            } else {
                                // Calculate similarity using our custom metric - pure vector similarity
                                score = await this.vectorSimilarity(interestsEmbedding, articleEmbedding);
                                await logToFile(`Vector similarity score: ${score}`, 'info');
                                
                                // Create explanation based on similarity score
                                const strength = score >= 85 ? "excellent" : 
                                                score >= 70 ? "strong" : 
                                                score >= 50 ? "good" : 
                                                score >= 25 ? "moderate" : "minimal";
                                                
                                explanation = `This article has a ${strength} semantic match (${score.toFixed(1)}/100) with your interests based on pure vector similarity.`;
                            }
                        }
                    } else {
                        // Fallback when no embeddings are available
                        score = 25; // Default score
                        explanation = "No embedding data available for proper analysis.";
                        await logToFile(`No embeddings available for article "${article.title}", using default score`, 'info');
                    }
                    
                    // Make sure we have a non-zero score (at least some minimal match)
                    if (score <= 0) {
                        score = 5; // Minimum score to avoid complete zeros
                    }
                    
                    await logToFile(`Final score for "${article.title}": ${score}/100`, 'info');
                    
                    // Return the scored article with content and explanation
                    return {
                        ...article,
                        content: articleData.article.content,
                        summary: articleData.article.summary || summary, // Include summary
                        match_score: score,
                        explanation: explanation
                    };
                }
                
                await logToFile(`Could not load article content for "${article.title}"`, 'error');
                // Fallback if article content can't be loaded
                return {
                    ...article,
                    match_score: 5, // Minimum non-zero score
                    explanation: "Could not analyze article content"
                };
            } catch (error) {
                await logToFile(`Error analyzing article ${article.id}: ${error}`, 'error');
                return {
                    ...article,
                    match_score: 5, // Minimum non-zero score
                    explanation: "Error analyzing article"
                };
            }
        }));
        
        // Sort by score in descending order
        const sortedArticles = scoredArticles.sort((a, b) => b.match_score - a.match_score);
        await logToFile(`Sorted ${sortedArticles.length} articles by score`, 'info');
        
        return sortedArticles;
    }
    
    // Display the best matching article
    private displayBestMatch(article: any, interests: string): void {
        if (!this.commandOutput) return;
        
        // Ensure we have a valid score
        if (typeof article.match_score !== 'number') {
            article.match_score = 5; // Minimum default score
        }
        
        // Create the best match container
        const matchContainer = document.createElement('div');
        matchContainer.className = 'article-recommendation best-article-match';
        
        // Create header with match info
        const header = document.createElement('div');
        header.className = 'recommendation-header';
        
        const title = document.createElement('h3');
        title.textContent = 'Best Article Match';
        header.appendChild(title);
        
        const score = document.createElement('div');
        score.className = 'match-score';
        
        // Format score with proper decimal places
        const formattedScore = Math.round(article.match_score * 10) / 10; // Round to 1 decimal place
        score.textContent = `${formattedScore}/100`;
        
        // Apply CSS class based on score
        if (article.match_score >= 70) {
            score.classList.add('high-score');
        } else if (article.match_score >= 40) {
            score.classList.add('medium-score');
        } else {
            score.classList.add('low-score');
        }
        
        header.appendChild(score);
        
        // Add explanation
        const explanation = document.createElement('p');
        explanation.className = 'match-explanation';
        explanation.textContent = article.explanation || 'No explanation available';
        
        // Article title and preview
        const articleTitle = document.createElement('h4');
        articleTitle.textContent = article.title;
        
        // Create article summary section
        const summarySection = document.createElement('div');
        summarySection.className = 'article-summary-section';
        
        // Create summary title
        const summaryTitle = document.createElement('h5');
        summaryTitle.textContent = 'Article Summary';
        summaryTitle.className = 'summary-title';
        summarySection.appendChild(summaryTitle);
        
        // Create article summary from the available summary or extract from content
        const articleSummary = document.createElement('p');
        articleSummary.className = 'article-summary';
        
        // Use summary if available, otherwise create from content
        let summaryText = '';
        if (article.summary) {
            // Use provided summary
            summaryText = article.summary;
            if (summaryText.length > 300) summaryText = summaryText.substring(0, 300) + '...';
        } else if (article.content) {
            // Extract from content - this is a fallback
            const plainContent = article.content.replace(/#{1,6}\s+/g, '');
            const lines = plainContent.split('\n').filter((line: string) => line.trim());
            if (lines.length > 1) {
                summaryText = lines.slice(1, 3).join(' ').substring(0, 300);
                if (summaryText.length === 300) summaryText += '...';
            }
        }
        
        articleSummary.textContent = summaryText || 'No summary available';
        summarySection.appendChild(articleSummary);
        
        // View button
        const viewButton = document.createElement('button');
        viewButton.className = 'view-article-btn';
        viewButton.textContent = 'Read Full Article';
        viewButton.dataset.articleId = article.id;
        viewButton.addEventListener('click', (e) => {
            e.preventDefault();
            this.handleViewArticle(e);
        });
        
        // Assemble the recommendation
        matchContainer.appendChild(header);
        matchContainer.appendChild(explanation);
        matchContainer.appendChild(articleTitle);
        matchContainer.appendChild(summarySection);
        matchContainer.appendChild(viewButton);
        
        // Add to output - insert at the top instead of appending
        this.commandOutput.insertBefore(matchContainer, this.commandOutput.firstChild);
        
        // Inject CSS for score styling
        const style = document.createElement('style');
        style.textContent = `
            .best-article-match {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.15);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            
            .best-article-match:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
            
            .recommendation-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                border-bottom: 2px solid #f0f0f0;
                padding-bottom: 10px;
            }
            
            .recommendation-header h3 {
                font-size: 18px;
                margin: 0;
                color: #333;
            }
            
            .match-score {
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 20px;
                color: white;
                display: inline-block;
                font-size: 14px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            
            .high-score {
                background-color: #4CAF50;
            }
            
            .medium-score {
                background-color: #FF9800;
            }
            
            .low-score {
                background-color: #F44336;
            }
            
            .match-explanation {
                font-size: 14px;
                color: #555;
                margin-bottom: 15px;
                line-height: 1.5;
            }
            
            .best-article-match h4 {
                font-size: 20px;
                color: #2c3e50;
                margin: 15px 0 10px;
            }
            
            .article-summary-section {
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
                border-left: 4px solid #3f51b5;
            }
            
            .summary-title {
                color: #3f51b5;
                margin-top: 0;
                margin-bottom: 8px;
                font-size: 14px;
                font-weight: 600;
            }
            
            .article-summary {
                font-size: 14px;
                line-height: 1.6;
                color: #444;
                margin: 0;
            }
            
            .view-article-btn {
                background-color: #3f51b5;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
                cursor: pointer;
                margin-top: 15px;
                transition: background-color 0.3s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            
            .view-article-btn:hover {
                background-color: #303f9f;
            }
        `;
        document.head.appendChild(style);
    }

    private async handleCommandSubmit(event: Event): Promise<void> {
        event.preventDefault();
        
        if (!this.commandInput || !this.commandOutput) {
            return;
        }

        const command = this.commandInput.value.trim();
        
        if (!command) {
            return;
        }

        try {
            // Display sending status
            this.addMessageToOutput('Sending command...', 'info');
            
            // Send command to Python backend
            const response = await axios.post('http://localhost:5001/command', {
                command: command
            });
            
            // Display success message
            this.addMessageToOutput(`Success: ${response.data.message}`, 'success');
            
            // Clear input
            this.commandInput.value = '';
            
        } catch (error) {
            // Display error message
            this.addMessageToOutput('Error sending command. Is the server running?', 'error');
            console.error('Error:', error);
        }
    }

    private async handleViewArticles(showLoadingMessage: boolean = true): Promise<void> {
        try {
            if (showLoadingMessage) {
                this.addMessageToOutput('Loading articles...', 'info');
            }

            // First try to get from our local cache
            let articles: any[] = [];
            
            if (this.cacheInitialized) {
                // Get a list of final article files from the API
                const response = await fetch(`${API_URL}/final-articles`, {
                    credentials: 'include'
                });
                const data = await response.json();
                
                // Use our local cache to load each article
                if (data.articles && Array.isArray(data.articles)) {
                    articles = await Promise.all(data.articles.map(async (article: any) => {
                        // Extract subject from the filename
                        const match = article.filename.match(/final_article_\d+_(.+)\.json$/);
                        if (match) {
                            const subject = match[1].replace(/_/g, ' ');
                            // Try to get from our local cache
                            const cachedArticle = getCachedArticle(subject);
                            if (cachedArticle) {
                                return {
                                    ...cachedArticle,
                                    subject,
                                    from_local_cache: true
                                };
                            }
                        }
                        
                        // Fall back to API if not in local cache
                        const articleResponse = await fetch(`${API_URL}/final-article/${article.filename}`, {
                            credentials: 'include'
                        });
                        return await articleResponse.json();
                    }));
                }
            } else {
                // Fall back to the original implementation using the API
                const response = await fetch(`${API_URL}/final-articles`, {
                    credentials: 'include'
                });
                const data = await response.json();
                
                if (data.articles && Array.isArray(data.articles)) {
                    articles = await Promise.all(data.articles.map(async (article: any) => {
                        const articleResponse = await fetch(`${API_URL}/final-article/${article.filename}`, {
                            credentials: 'include'
                        });
                        return await articleResponse.json();
                    }));
                }
            }

            // Display the articles - same as original implementation
            if (articles.length > 0) {
                const articleContainer = document.createElement('div');
                articleContainer.classList.add('article-list');
                
                // Sort articles by timestamp if available (newest first)
                articles.sort((a, b) => {
                    const timestampA = a.timestamp || 0;
                    const timestampB = b.timestamp || 0;
                    return timestampB - timestampA;
                });
                
                articles.forEach(article => {
                    const articleElement = document.createElement('div');
                    articleElement.classList.add('article-item');
                    
                    // Extract title from content
                    let title = 'Untitled Article';
                    const contentLines = article.content.split('\n');
                    if (contentLines.length > 0 && contentLines[0].startsWith('# ')) {
                        title = contentLines[0].substring(2).trim();
                    }
                    
                    articleElement.innerHTML = `
                        <h3>${title}</h3>
                        <div class="article-meta">
                            ${article.from_local_cache ? '<span class="local-cache-badge">Local Cache</span>' : ''}
                            <span class="article-date">${new Date(article.timestamp * 1000).toLocaleDateString()}</span>
                        </div>
                        <button class="view-article-btn">Read Article</button>
                    `;
                    
                    // Add click event to view the article
                    const viewButton = articleElement.querySelector('.view-article-btn');
                    if (viewButton) {
                        viewButton.addEventListener('click', () => {
                            this.displayArticleModal(article);
                        });
                    }
                    
                    articleContainer.appendChild(articleElement);
                });
                
                // Clear previous article list if any
                const existingList = document.querySelector('.article-list');
                if (existingList) {
                    existingList.remove();
                }
                
                // Insert at the top instead of appending
                this.commandOutput?.insertBefore(articleContainer, this.commandOutput.firstChild);
                
                if (showLoadingMessage) {
                    this.addMessageToOutput(`Displaying ${articles.length} articles.`, 'success');
                }
            } else {
                if (showLoadingMessage) {
                    this.addMessageToOutput('No articles found.', 'info');
                }
            }
        } catch (error) {
            await logToFile(`Error loading articles: ${error}`, 'error');
            this.addMessageToOutput(`Error loading articles: ${error}`, 'error');
        }
    }

    private handleViewArticle(event: Event): void {
        const target = event.currentTarget as HTMLElement;
        const articleId = target.getAttribute('data-article-id');
        
        if (!articleId) {
            this.addMessageToOutput('Article ID not found', 'error');
            return;
        }
        
        this.addMessageToOutput(`Loading article ${articleId}...`, 'info');
        
        fetch(`${API_URL}/get-final-article/${articleId}`, {
            credentials: 'include' // Include cookies in the request
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    const article = data.article;
                    
                    // Create a modal for the article
                    const modal = document.createElement('div');
                    modal.className = 'article-modal';
                    
                    // Add close button
                    const closeButton = document.createElement('button');
                    closeButton.className = 'close-button';
                    closeButton.innerHTML = '&times;';
                    closeButton.addEventListener('click', () => {
                        document.body.removeChild(modal);
                    });
                    
                    // Create container for content
                    const contentContainer = document.createElement('div');
                    contentContainer.className = 'article-content-container';
                    
                    // Add title
                    const title = document.createElement('h2');
                    title.textContent = article.title;
                    
                    // Convert markdown to HTML
                    const htmlContent = this.markdownToHtml(article.content);
                    
                    // Add content
                    const content = document.createElement('div');
                    content.className = 'article-content';
                    content.innerHTML = htmlContent;
                    
                    // Create button container
                    const buttonContainer = document.createElement('div');
                    buttonContainer.className = 'article-actions';
                    
                    // Add HTML view button
                    const viewHtmlButton = document.createElement('button');
                    viewHtmlButton.textContent = 'View HTML Version';
                    viewHtmlButton.addEventListener('click', () => {
                        // Open HTML version in new tab
                        window.open(`${API_URL}/articles/tech_deep_dive_${article.id}.html`, '_blank');
                    });
                    
                    // Add buttons to container
                    buttonContainer.appendChild(viewHtmlButton);
                    
                    // Add HTML view section
                    const htmlViewSection = document.createElement('div');
                    htmlViewSection.className = 'html-view-section';
                    
                    // Try to fetch the HTML version
                    fetch(`${API_URL}/articles/tech_deep_dive_${article.id}.html`, {
                        credentials: 'include' // Include cookies in the request
                    })
                        .then(response => {
                            if (response.ok) {
                                // HTML exists - no need to do anything, button will work
                            } else {
                                // HTML doesn't exist
                                const noHtml = document.createElement('p');
                                noHtml.className = 'warning-message';
                                noHtml.textContent = 'HTML version not available.';
                                htmlViewSection.appendChild(noHtml);
                            }
                        })
                        .catch(error => {
                            console.error('Error checking HTML version:', error);
                            const noHtml = document.createElement('p');
                            noHtml.className = 'error-message';
                            noHtml.textContent = 'Could not check HTML version.';
                            htmlViewSection.appendChild(noHtml);
                        });
                    
                    // Assemble modal content
                    contentContainer.appendChild(title);
                    contentContainer.appendChild(content);
                    contentContainer.appendChild(buttonContainer);
                    contentContainer.appendChild(htmlViewSection);
                    
                    modal.appendChild(closeButton);
                    modal.appendChild(contentContainer);
                    
                    // Add modal to page
                    document.body.appendChild(modal);
                    
                    this.addMessageToOutput(`Displaying article: ${article.title}`, 'success');
                } else {
                    this.addMessageToOutput(`Error retrieving article: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                this.addMessageToOutput(`Error: ${error.message}`, 'error');
            });
    }

    private markdownToHtml(markdown: string): string {
        // Simple markdown parsing
        return markdown
            .replace(/^# (.*?)$/gm, '<h1>$1</h1>')
            .replace(/^## (.*?)$/gm, '<h2>$1</h2>')
            .replace(/^### (.*?)$/gm, '<h3>$1</h3>')
            .replace(/^#### (.*?)$/gm, '<h4>$1</h4>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');
    }

    private addMessageToOutput(message: string, type: 'info' | 'success' | 'error'): void {
        if (!this.commandOutput) {
            return;
        }

        const messageElement = document.createElement('div');
        messageElement.classList.add('message', type);
        messageElement.textContent = message;
        
        // Add new message at the top
        this.commandOutput.insertBefore(messageElement, this.commandOutput.firstChild);
    }
    
    private addRawTextToOutput(text: string): void {
        if (!this.commandOutput) {
            return;
        }

        const preElement = document.createElement('pre');
        preElement.classList.add('raw-output');
        preElement.textContent = text;
        
        // Add raw text at the top
        this.commandOutput.insertBefore(preElement, this.commandOutput.firstChild);
    }
    
    private toggleSections(hideId: string, showId: string): void {
        const hideElement = document.getElementById(hideId);
        const showElement = document.getElementById(showId);
        
        if (hideElement) {
            hideElement.style.display = 'none';
        }
        
        if (showElement) {
            showElement.style.display = 'block';
        }
    }

    private setupPlatformSpecificFeatures(): void {
        // Detect platform
        const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
        
        if (isMac) {
            // Add Mac-specific features if needed
        }
    }

    private async copyArticleFiles(): Promise<void> {
        try {
            // Don't show any UI for this operation
            console.log('Checking for article files to copy...');
            
            // Call the backend endpoint to copy article files
            const response = await fetch(`${API_URL}/copy-article-files`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({}),  // No source directory specified, use defaults
                credentials: 'include' // Include cookies in the request
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                if (data.copied_count > 0) {
                    console.log(`Successfully copied ${data.copied_count} article files from ${data.source_dir}`);
                    // We'll add a message to the UI after checkCache completes
                    setTimeout(() => {
                        this.addMessageToOutput(`Copied ${data.copied_count} additional article files from external cache.`, 'info');
                    }, 5000);
                } else {
                    console.log('No new article files found to copy');
                }
            } else {
                console.error(`Error copying article files: ${data.message}`);
            }
        } catch (error) {
            console.error('Error copying article files:', error);
            // Don't show an error to the user, just log it
        }
    }

    private async preloadArticleEmbeddings(): Promise<void> {
        try {
            await logToFile('Preloading article embeddings...', 'info');
            this.updateLoadingMessage('Preloading article embeddings...');
            
            // Fetch list of all articles
            const response = await fetch(`${API_URL}/list-articles`);
            const data = await response.json();
            
            if (data.status === 'success' && data.articles) {
                await logToFile(`Found ${data.articles.length} articles to preload embeddings for`, 'info');
                
                // For better UX, don't wait for all embeddings to be generated
                // Just start the process in the background
                Promise.all(data.articles.map(async (article: any) => {
                    try {
                        // Get full article data
                        const articleResponse = await fetch(`${API_URL}/get-final-article/${article.id}`);
                        const articleData = await articleResponse.json();
                        
                        if (articleData.status === 'success') {
                            const title = articleData.article.title.toLowerCase();
                            const content = articleData.article.content.toLowerCase();
                            
                            // Extract or use summary if available
                            let summary = '';
                            if (articleData.article.summary) {
                                summary = articleData.article.summary.toLowerCase();
                            } else {
                                // Extract summary from content
                                const summaryMatch = content.match(/## summary\s+([\s\S]+?)(?=##|$)/i);
                                if (summaryMatch && summaryMatch[1]) {
                                    summary = summaryMatch[1].trim().toLowerCase();
                                } else if (content.length > 0) {
                                    // Use first paragraph as fallback summary
                                    const paragraphs = content.split('\n\n');
                                    const startIdx = paragraphs[0].startsWith('#') ? 1 : 0;
                                    if (paragraphs.length > startIdx) {
                                        summary = paragraphs[startIdx].trim().toLowerCase();
                                    }
                                }
                            }
                            
                            // Generate and cache embedding if not already available
                            await this.getArticleEmbedding(articleData.article, summary, title);
                        }
                    } catch (error) {
                        await logToFile(`Error preloading embedding for article ${article.id}: ${error}`, 'error');
                    }
                })).then(() => {
                    logToFile('Completed preloading all article embeddings', 'info');
                }).catch(error => {
                    logToFile(`Error in preloading embeddings: ${error}`, 'error');
                });
                
                // Don't wait for all embeddings to complete before continuing
                await logToFile('Started preloading embeddings in background', 'info');
            } else {
                await logToFile('Failed to fetch articles for preloading embeddings', 'error');
            }
        } catch (error) {
            await logToFile(`Error preloading article embeddings: ${error}`, 'error');
        }
    }

    /**
     * Display an article in a modal dialog
     */
    private displayArticleModal(article: any): void {
        // Create the modal container
        const modal = document.createElement('div');
        modal.classList.add('article-modal');
        
        // Extract title from content
        let title = 'Untitled Article';
        const contentLines = article.content.split('\n');
        if (contentLines.length > 0 && contentLines[0].startsWith('# ')) {
            title = contentLines[0].substring(2).trim();
        }
        
        // Create the modal content
        modal.innerHTML = `
            <div class="article-modal-content">
                <div class="article-modal-header">
                    <h2>${title}</h2>
                    <button class="article-modal-close">&times;</button>
                </div>
                <div class="article-modal-body">
                    ${this.markdownToHtml(article.content)}
                </div>
                <div class="article-modal-footer">
                    <div class="article-meta">
                        ${article.from_local_cache ? '<span class="local-cache-badge">Local Cache</span>' : ''}
                        <span class="article-date">${new Date(article.timestamp * 1000).toLocaleDateString()}</span>
                    </div>
                </div>
            </div>
        `;
        
        // Add close button event
        const closeButton = modal.querySelector('.article-modal-close');
        if (closeButton) {
            closeButton.addEventListener('click', () => {
                document.body.removeChild(modal);
            });
        }
        
        // Close modal when clicking outside content
        modal.addEventListener('click', (event) => {
            if (event.target === modal) {
                document.body.removeChild(modal);
            }
        });
        
        // Add to body
        document.body.appendChild(modal);
        
        // Add ESC key handler for the modal
        const escHandler = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                document.body.removeChild(modal);
                document.removeEventListener('keydown', escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new App();
}); 