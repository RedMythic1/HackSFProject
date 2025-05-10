import axios from 'axios';

// Update the API_URL constant to point to the new Node.js backend
const API_URL = 'http://localhost:5001';

class CommandApp {
    private inputElement: HTMLInputElement | null = null;
    private formElement: HTMLFormElement | null = null;
    private interestsInputElement: HTMLInputElement | null = null;
    private interestsFormElement: HTMLFormElement | null = null;
    private emailInputElement: HTMLInputElement | null = null;
    private emailFormElement: HTMLFormElement | null = null;
    private outputElement: HTMLDivElement | null = null;
    private loadingMessageElement: HTMLElement | null = null;
    private viewArticlesButton: HTMLButtonElement | null = null;
    private adminToggleButton: HTMLButtonElement | null = null;
    private cacheCheckInterval: number | null = null;
    
    // Store the user email
    private userEmail: string = '';

    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.checkCache();
        this.checkEmailVerification();
        
        // Log platform information to help with debugging
        const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
        console.log(`Platform detected: ${navigator.platform} - Is Mac: ${isMac}`);
    }

    private initializeElements(): void {
        // Get loading elements
        this.loadingMessageElement = document.getElementById('loading-message');
        
        // Get email form elements
        this.emailFormElement = document.getElementById('email-form') as HTMLFormElement;
        this.emailInputElement = document.getElementById('email-input') as HTMLInputElement;
        
        // Get command form elements
        this.formElement = document.getElementById('command-form') as HTMLFormElement;
        this.inputElement = document.getElementById('command-input') as HTMLInputElement;
        
        // Get interests form elements
        this.interestsFormElement = document.getElementById('interests-form') as HTMLFormElement;
        this.interestsInputElement = document.getElementById('interests-input') as HTMLInputElement;
        
        // Get admin elements
        this.viewArticlesButton = document.getElementById('view-articles-btn') as HTMLButtonElement;
        this.adminToggleButton = document.getElementById('admin-toggle-btn') as HTMLButtonElement;
        
        // Get output element
        this.outputElement = document.getElementById('command-output') as HTMLDivElement;
    }

    private setupEventListeners(): void {
        // Add email form submit event listener
        if (this.emailFormElement) {
            this.emailFormElement.addEventListener('submit', this.handleEmailSubmit.bind(this));
        } else {
            console.error('Email form not found');
        }
        
        // Add command form submit event listener
        if (this.formElement) {
            this.formElement.addEventListener('submit', this.handleFormSubmit.bind(this));
        } else {
            console.error('Command form not found');
        }
        
        // Add interests form submit event listener
        if (this.interestsFormElement) {
            this.interestsFormElement.addEventListener('submit', this.handleInterestsSubmit.bind(this));
        } else {
            console.error('Interests form not found');
        }
        
        // Add view articles button event listener
        if (this.viewArticlesButton) {
            this.viewArticlesButton.addEventListener('click', (event) => {
                this.handleViewArticles(true);
            });
        } else {
            console.error('View articles button not found');
        }
        
        // Add admin toggle button event listener
        if (this.adminToggleButton) {
            this.adminToggleButton.addEventListener('click', this.toggleAdminPanel.bind(this));
        } else {
            console.error('Admin toggle button not found');
        }
        
        // Add keyboard shortcut for admin panel
        document.addEventListener('keydown', this.handleKeyDown.bind(this));
        
        // Start cache polling
        this.startCachePolling();
        
        // Stop polling when the window is closed
        window.addEventListener('beforeunload', () => {
            this.stopCachePolling();
        });
    }
    
    private handleKeyDown(event: KeyboardEvent): void {
        // For Mac compatibility, check for both Alt (Option) key and Meta (Command) key
        // Also accept both uppercase and lowercase 'a'
        if (
            (event.shiftKey && (event.altKey || event.metaKey) && (event.key === 'A' || event.key === 'a')) ||
            (event.shiftKey && event.key === 'a') // Simpler alternative if the above doesn't work
        ) {
            console.log('Admin panel shortcut triggered');
            this.toggleAdminPanel();
        }
    }
    
    private toggleAdminPanel(): void {
        const adminSection = document.getElementById('admin-section');
        if (adminSection) {
            // Toggle visibility (ensure we handle both empty string and 'block' cases)
            const isVisible = adminSection.style.display !== 'none';
            adminSection.style.display = isVisible ? 'none' : 'block';
            
            // If showing admin panel, add a message to the output
            if (adminSection.style.display === 'block') {
                this.addMessageToOutput('Admin panel activated', 'info');
                this.addMessageToOutput('Article processing is now handled via shell commands', 'info');
            }
        } else {
            console.error('Admin section not found in DOM');
        }
    }
    
    private async checkCache(): Promise<void> {
        try {
            this.updateLoadingMessage('Checking article cache...');
            
            // Call the backend endpoint
            const response = await fetch(`${API_URL}/check-cache`);
            const data = await response.json();
            
            if (data.cached) {
                const validCount = data.valid_article_count || 0;
                const uniqueCount = data.final_article_count || 0;
                
                this.updateLoadingMessage(`Found ${data.article_count} cached articles and ${uniqueCount} final articles. Preparing interface...`);
                
                // Add a status element for live cache updates
                const cacheStatusDiv = document.createElement('div');
                cacheStatusDiv.classList.add('cache-info');
                cacheStatusDiv.textContent = `Cached articles: ${data.article_count}, Final articles: ${uniqueCount}`;
                this.outputElement?.appendChild(cacheStatusDiv);
                
                // Don't show any final articles automatically - hide until user requests them
                if (uniqueCount > 0) {
                    setTimeout(() => {
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
                        
                        // Add to output
                        this.outputElement?.appendChild(buttonContainer);
                    }, 3000);
                }
            } else {
                this.updateLoadingMessage('No cached articles found. Use shell commands to process articles first.');
                
                // Add an empty status element that will be updated by polling
                const cacheStatusDiv = document.createElement('div');
                cacheStatusDiv.classList.add('cache-info');
                cacheStatusDiv.textContent = 'Cached articles: 0, Final articles: 0';
                this.outputElement?.appendChild(cacheStatusDiv);
            }
            
            // Artificial delay to ensure everything is ready
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            // Hide loading and show email form
            this.toggleSections('loading-section', 'email-section');
            
        } catch (error) {
            console.error('Error checking cache:', error);
            this.addMessageToOutput(`Error checking cache: ${error}`, 'error');
            
            // Continue to the app interface even on error
            setTimeout(() => {
                this.toggleSections('loading-section', 'email-section');
            }, 2000);
        }
    }
    
    private updateLoadingMessage(message: string): void {
        if (this.loadingMessageElement) {
            this.loadingMessageElement.textContent = message;
        }
    }
    
    private async checkEmailVerification(): Promise<void> {
        try {
            // Call the new Node.js backend endpoint
            const response = await fetch(`${API_URL}/check-email-verification`);
            const data = await response.json();
            
            if (data.status === 'success' && data.verified) {
                // Email is already verified, store it and show interests section
                this.userEmail = data.email;
                this.toggleSections('email-section', 'interests-section');
                this.addMessageToOutput(`Welcome back! Using email: ${data.email}`, 'success');
            }
        } catch (error) {
            console.error('Error checking email verification:', error);
        }
    }

    private async handleEmailSubmit(event: Event): Promise<void> {
        event.preventDefault();
        
        if (!this.emailInputElement || !this.outputElement) {
            return;
        }

        const email = this.emailInputElement.value.trim();
        
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
                body: JSON.stringify({ email })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Store the email
                this.userEmail = email;
                
                // Display success message
                this.addMessageToOutput(`Email registered: ${email}`, 'success');
                
                // Hide email section and show interests section
                this.toggleSections('email-section', 'interests-section');
            } else {
                this.addMessageToOutput(data.message || 'Error processing email', 'error');
            }
        } catch (error) {
            console.error('Error verifying email:', error);
            this.addMessageToOutput(`Error verifying email: ${error}`, 'error');
        }
    }

    private async handleFormSubmit(event: Event): Promise<void> {
        event.preventDefault();
        
        if (!this.inputElement || !this.outputElement) {
            return;
        }

        const command = this.inputElement.value.trim();
        
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
            this.inputElement.value = '';
            
        } catch (error) {
            // Display error message
            this.addMessageToOutput('Error sending command. Is the server running?', 'error');
            console.error('Error:', error);
        }
    }
    
    private async handleInterestsSubmit(event: Event): Promise<void> {
        event.preventDefault();
        
        if (!this.interestsInputElement || !this.outputElement) {
            return;
        }

        const interests = this.interestsInputElement.value.trim();
        
        if (!interests) {
            this.addMessageToOutput('Please enter your interests', 'error');
            return;
        }

        // Clear any previous articles from the view
        const previousArticles = document.querySelectorAll('.articles-container, .article-recommendation, .article-content');
        previousArticles.forEach(element => element.remove());

        // Display a helpful message about shell commands
        this.addMessageToOutput(`You entered interests: ${interests}`, 'info');
        this.addMessageToOutput('Article processing is now handled via shell commands. Please use the terminal.', 'info');
        this.addMessageToOutput('Command example:', 'info');
        this.addMessageToOutput(`cd /path/to/project && ./fetch_articles.sh process "${interests}"`, 'info');
        
        // Suggest viewing articles
        this.addMessageToOutput('After processing articles via shell commands, you can view them using the "View Articles" button.', 'info');
    }
    
    private async pollTaskProgress(taskId: string, loadingElement: HTMLElement): Promise<void> {
        const maxAttempts = 60; // 2 minutes maximum (2s intervals)
        let attempts = 0;
        let progressBarDiv: HTMLDivElement | null = null;
        
        // Create progress bar
        progressBarDiv = document.createElement('div');
        progressBarDiv.className = 'progress-container';
        progressBarDiv.innerHTML = `
            <div class="progress-bar">
                <div class="progress-fill" style="width: 0%"></div>
            </div>
            <p class="progress-text">0%</p>
        `;
        loadingElement.appendChild(progressBarDiv);
        
        const pollInterval = setInterval(async () => {
            try {
                attempts++;
                // Check task progress
                const response = await axios.get(`http://localhost:5001/get-match-progress/${taskId}`);
                
                // Update progress bar if available
                if (progressBarDiv && response.data.percentage) {
                    const progressFill = progressBarDiv.querySelector('.progress-fill');
                    const progressText = progressBarDiv.querySelector('.progress-text');
                    if (progressFill && progressText) {
                        progressFill.setAttribute('style', `width: ${response.data.percentage}%`);
                        progressText.textContent = `${response.data.percentage}% - ${response.data.status || 'Processing...'}`;
                    }
                }
                
                // Task is complete
                if (response.data.completed) {
                    clearInterval(pollInterval);
                    loadingElement.remove();
                    
                    if (response.data.final_result?.status === 'success') {
                        const article = response.data.final_result.article;
                        
                        // Create a message with the article info
                        const matchMessage = `Found the best article match for your interests (${article.match_score}/100):`;
                        this.addMessageToOutput(matchMessage, 'success');
                        
                        // Create a div for the article recommendation
                        const articleDiv = document.createElement('div');
                        articleDiv.className = 'article-recommendation best-article-match';
                        
                        // Create a header container with score indicator
                        const headerContainer = document.createElement('div');
                        headerContainer.className = 'best-match-header';
                        
                        // Add a "Best Match" badge
                        const matchBadge = document.createElement('div');
                        matchBadge.className = 'match-badge';
                        matchBadge.innerHTML = `<span class="match-score">${article.match_score}</span><span class="match-label">MATCH SCORE</span>`;
                        headerContainer.appendChild(matchBadge);
                        
                        // Create a header for the article
                        const articleTitle = document.createElement('h3');
                        articleTitle.textContent = article.title;
                        headerContainer.appendChild(articleTitle);
                        
                        // Add the header container to the article div
                        articleDiv.appendChild(headerContainer);
                        
                        // Add article summary
                        const summary = this.extractArticleSummary(article.content);
                        const summaryContainer = document.createElement('div');
                        summaryContainer.className = 'article-summary';
                        summaryContainer.textContent = summary;
                        articleDiv.appendChild(summaryContainer);
                        
                        // Add match explanation if available
                        if (article.match_explanation) {
                            const matchExplanation = document.createElement('div');
                            matchExplanation.className = 'match-explanation';
                            matchExplanation.textContent = article.match_explanation;
                            articleDiv.appendChild(matchExplanation);
                        } else {
                            // Add a generic explanation
                            const matchExplanation = document.createElement('div');
                            matchExplanation.className = 'match-explanation';
                            matchExplanation.textContent = `This article scored ${article.match_score}/100 based on your stated interests. Articles are scored based on relevant content, topic overlap, and depth of coverage.`;
                            articleDiv.appendChild(matchExplanation);
                        }
                        
                        // If we have an HTML version, add a link to it
                        if (article.has_html) {
                            // Extract just the filename from the path
                            const pathParts = article.html_path.split('/');
                            const htmlFilename = pathParts[pathParts.length - 1];
                            
                            // Create a button container for actions
                            const buttonContainer = document.createElement('div');
                            buttonContainer.className = 'article-actions';
                            
                            // Create a button to view the full article
                            const viewButton = document.createElement('button');
                            viewButton.className = 'view-article-btn';
                            viewButton.textContent = 'View Full Article';
                            viewButton.addEventListener('click', () => {
                                // Instead of opening in new tab, display content directly in UI
                                this.handleViewArticleContent(article.id);
                            });
                            
                            buttonContainer.appendChild(viewButton);
                            articleDiv.appendChild(buttonContainer);
                        } else {
                            // No HTML version available
                            const noHtml = document.createElement('p');
                            noHtml.textContent = 'HTML version not available. You can view the article in the admin panel.';
                            articleDiv.appendChild(noHtml);
                        }
                        
                        // Create a simple message that explains what the user can do
                        const explainText = document.createElement('p');
                        explainText.className = 'article-explanation';
                        explainText.textContent = 'This article closely matches your interests. Click the button above to view the full article with deep dive questions and further exploration sections.';
                        articleDiv.appendChild(explainText);
                        
                        // Add the article div to the output
                        this.outputElement?.appendChild(articleDiv);
                    } else {
                        // Error in the final result
                        this.addMessageToOutput(`Error: ${response.data.final_result?.message || 'Could not find matching article'}`, 'error');
                    }
                } else if (attempts >= maxAttempts) {
                    // Timeout after max attempts
                    clearInterval(pollInterval);
                    loadingElement.remove();
                    this.addMessageToOutput('Request timed out. The article matching is taking longer than expected.', 'error');
                }
            } catch (error) {
                console.error('Error polling task progress:', error);
                attempts++;
                
                if (attempts >= maxAttempts) {
                    clearInterval(pollInterval);
                    loadingElement.remove();
                    this.addMessageToOutput('Error tracking progress of article matching. Please try again.', 'error');
                }
            }
        }, 2000); // Poll every 2 seconds
    }

    private addMessageToOutput(message: string, type: 'info' | 'success' | 'error'): void {
        if (!this.outputElement) {
            return;
        }

        const messageElement = document.createElement('div');
        messageElement.classList.add('message', type);
        messageElement.textContent = message;
        
        // Add new message at the top
        this.outputElement.insertBefore(messageElement, this.outputElement.firstChild);
    }
    
    private addRawTextToOutput(text: string): void {
        if (!this.outputElement) {
            return;
        }

        const preElement = document.createElement('pre');
        preElement.classList.add('raw-output');
        preElement.textContent = text;
        
        // Add raw text at the top
        this.outputElement.insertBefore(preElement, this.outputElement.firstChild);
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

    private async handleViewArticles(showLoadingMessage: boolean = true): Promise<void> {
        try {
            if (!this.outputElement) {
                console.error('Output element not initialized');
                return;
            }
            
            if (showLoadingMessage) {
                this.updateLoadingMessage('Loading articles...');
            }
            
            // Call the backend endpoint
            const response = await fetch(`${API_URL}/get-final-articles`);
            const data = await response.json();
            
            if (data.status === 'success' && data.articles && data.articles.length > 0) {
                // Create a container for the articles
                const articlesContainer = document.createElement('div');
                articlesContainer.className = 'articles-container';
                
                // Add a heading
                const heading = document.createElement('h3');
                heading.textContent = `Available Articles (${data.articles.length})`;
                articlesContainer.appendChild(heading);
                
                // Create the list
                const articleList = document.createElement('ul');
                articleList.className = 'article-list';
                
                // Add each article
                data.articles.forEach((article: any) => {
                    const listItem = document.createElement('li');
                    
                    // Create link
                    const articleLink = document.createElement('a');
                    articleLink.href = '#';
                    articleLink.textContent = article.title;
                    articleLink.dataset.articleId = article.id;
                    articleLink.addEventListener('click', (e) => {
                        e.preventDefault();
                        this.handleViewArticleContent(article.id);
                    });
                    
                    // Add the link to the list item
                    listItem.appendChild(articleLink);
                    
                    // Add the list item to the list
                    articleList.appendChild(listItem);
                });
                
                // Add the list to the container
                articlesContainer.appendChild(articleList);
                
                // Add the container to the output
                this.outputElement.appendChild(articlesContainer);
                
                // Scroll to the articles container
                articlesContainer.scrollIntoView({ behavior: 'smooth' });
                
                // Add a note about the count
                if (data.invalid_count > 0) {
                    this.addMessageToOutput(`Note: ${data.invalid_count} invalid article files were cleaned up during this process.`, 'info');
                }
            } else {
                if (showLoadingMessage) {
                    this.addMessageToOutput('No final articles found. Please process articles using shell commands first.', 'info');
                    this.addMessageToOutput('Example: ./fetch_articles.sh process "technology, science"', 'info');
                }
            }
        } catch (error) {
            console.error('Error loading articles:', error);
            this.addMessageToOutput(`Error loading articles: ${error}`, 'error');
        }
    }
    
    private async handleViewArticleContent(articleId: string): Promise<void> {
        try {
            if (!this.outputElement) {
                console.error('Output element not initialized');
                return;
            }
            
            this.updateLoadingMessage('Loading article content...');
            
            // Call the backend endpoint
            const response = await fetch(`${API_URL}/get-final-article/${articleId}`);
            const data = await response.json();
            
            if (data.status === 'success' && data.article) {
                const article = data.article;
                
                // Create a container for the article content
                const contentContainer = document.createElement('div');
                contentContainer.className = 'article-content';
                
                // Add article title
                const title = document.createElement('h2');
                title.textContent = article.title;
                contentContainer.appendChild(title);
                
                // Add a close button
                const closeButton = document.createElement('button');
                closeButton.textContent = 'Close Article';
                closeButton.className = 'close-article-btn';
                closeButton.addEventListener('click', () => {
                    contentContainer.remove();
                });
                contentContainer.appendChild(closeButton);
                
                // Add article summary
                const summary = this.extractArticleSummary(article.content);
                const summaryContainer = document.createElement('div');
                summaryContainer.className = 'article-summary';
                
                const summaryTitle = document.createElement('h3');
                summaryTitle.textContent = 'Summary';
                summaryContainer.appendChild(summaryTitle);
                
                const summaryText = document.createElement('p');
                summaryText.textContent = summary;
                summaryContainer.appendChild(summaryText);
                contentContainer.appendChild(summaryContainer);
                
                // Add the markdown content
                const markdownContent = document.createElement('div');
                markdownContent.className = 'markdown-content';
                
                // Simple markdown parsing (very basic)
                const formattedContent = article.content
                    .replace(/^# (.*?)$/gm, '<h1>$1</h1>')
                    .replace(/^## (.*?)$/gm, '<h2>$1</h2>')
                    .replace(/^### (.*?)$/gm, '<h3>$1</h3>')
                    .replace(/^#### (.*?)$/gm, '<h4>$1</h4>')
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/\n\n/g, '</p><p>')
                    .replace(/\n/g, '<br>');
                    
                markdownContent.innerHTML = `<p>${formattedContent}</p>`;
                contentContainer.appendChild(markdownContent);
                
                // Add the container to the output
                this.outputElement.appendChild(contentContainer);
                
                // Scroll to the content
                contentContainer.scrollIntoView({ behavior: 'smooth' });
            } else {
                this.addMessageToOutput('Error: Could not load article content.', 'error');
            }
        } catch (error) {
            console.error('Error loading article content:', error);
            this.addMessageToOutput(`Error loading article content: ${error}`, 'error');
        }
    }

    private extractArticleSummary(content: string): string {
        // Logic to extract a summary from the article content
        // This is a simple implementation - first look for a section called "Summary"
        const summaryRegex = /## Summary\s+([\s\S]+?)(?=##|$)/;
        const match = content.match(summaryRegex);
        
        if (match && match[1]) {
            return match[1].trim();
        }
        
        // If no Summary section, generate one from the first few paragraphs
        const paragraphs = content.split(/\n\n+/);
        let summary = '';
        
        // Skip the title (first paragraph) and use the next 2-3 paragraphs
        for (let i = 1; i < Math.min(4, paragraphs.length); i++) {
            // Clean up markdown formatting for the summary
            const cleaned = paragraphs[i]
                .replace(/^#+ /gm, '') // Remove headings
                .replace(/\*\*/g, '') // Remove bold
                .replace(/\*/g, '') // Remove italic
                .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Replace links with just the text
                .trim();
                
            if (cleaned) {
                summary += cleaned + ' ';
            }
            
            // Limit summary length
            if (summary.length > 300) {
                summary = summary.substring(0, 300) + '...';
                break;
            }
        }
        
        return summary.trim() || 'No summary available for this article.';
    }

    private startCachePolling(): void {
        // Poll every 5 seconds
        this.cacheCheckInterval = window.setInterval(async () => {
            try {
                const response = await axios.get('http://localhost:5001/check-cache');
                
                // Update the UI with the new cache info
                const cachedCount = response.data.article_count;
                const finalCount = response.data.final_article_count || 0;
                const previousFinalCount = document.querySelectorAll('.cache-info').length > 0 ? 
                    parseInt(document.querySelector('.cache-info')?.textContent?.split(':')[2]?.trim() || '0') : 0;
                
                // Check if the count changed
                const countChanged = finalCount !== previousFinalCount;
                
                // Update any UI elements that show cache counts
                const cacheInfoElements = document.querySelectorAll('.cache-info');
                cacheInfoElements.forEach(element => {
                    element.textContent = `Cached articles: ${cachedCount}, Final articles: ${finalCount}`;
                });
                
                // If there are now articles available but weren't before, add the button
                if (finalCount > 0 && !document.querySelector('.view-articles-btn')) {
                    this.addMessageToOutput(`${finalCount} final articles with questions & answers are available.`, 'info');
                    
                    // Add a button to view articles
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
                    
                    // Add to output
                    this.outputElement?.appendChild(buttonContainer);
                }
                
                // If the count changed and we have an articles container displayed, refresh it
                if (countChanged && document.querySelector('.articles-container')) {
                    this.handleViewArticles(false); // Refresh without showing loading message
                }
            } catch (error) {
                console.error('Error checking cache during polling:', error);
                // Don't display errors during automatic polling
            }
        }, 5000);
    }

    private stopCachePolling(): void {
        if (this.cacheCheckInterval !== null) {
            window.clearInterval(this.cacheCheckInterval);
            this.cacheCheckInterval = null;
        }
    }

    public init(): void {
        console.log('Application initialized');
        
        // Check if on Mac and provide hint about admin panel
        const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
        if (isMac) {
            setTimeout(() => {
                this.addMessageToOutput('Mac detected. Access admin panel using the "Admin" button or press Shift+Option+A', 'info');
            }, 3000);
        }
    }
}

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new CommandApp();
    app.init();
}); 