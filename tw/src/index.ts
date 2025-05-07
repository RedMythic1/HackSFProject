import axios from 'axios';

class CommandApp {
    private inputElement: HTMLInputElement | null = null;
    private formElement: HTMLFormElement | null = null;
    private interestsInputElement: HTMLInputElement | null = null;
    private interestsFormElement: HTMLFormElement | null = null;
    private emailInputElement: HTMLInputElement | null = null;
    private emailFormElement: HTMLFormElement | null = null;
    private outputElement: HTMLDivElement | null = null;
    private loadingMessageElement: HTMLElement | null = null;
    private cacheArticlesButton: HTMLButtonElement | null = null;
    private generateQuestionsButton: HTMLButtonElement | null = null;
    private adminToggleButton: HTMLButtonElement | null = null;
    
    // Store the user email
    private userEmail: string = '';

    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.checkCache();
        
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
        this.cacheArticlesButton = document.getElementById('cache-articles-btn') as HTMLButtonElement;
        this.generateQuestionsButton = document.getElementById('generate-questions-btn') as HTMLButtonElement;
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
        
        // Add cache articles button event listener
        if (this.cacheArticlesButton) {
            this.cacheArticlesButton.addEventListener('click', this.handleCacheArticles.bind(this));
        } else {
            console.error('Cache articles button not found');
        }
        
        // Add generate questions button event listener
        if (this.generateQuestionsButton) {
            this.generateQuestionsButton.addEventListener('click', this.handleGenerateQuestions.bind(this));
        } else {
            console.error('Generate questions button not found');
        }
        
        // Add admin toggle button event listener
        if (this.adminToggleButton) {
            this.adminToggleButton.addEventListener('click', this.toggleAdminPanel.bind(this));
        } else {
            console.error('Admin toggle button not found');
        }
        
        // Add keyboard shortcut for admin panel
        document.addEventListener('keydown', this.handleKeyDown.bind(this));
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
            }
        } else {
            console.error('Admin section not found in DOM');
        }
    }
    
    private async handleCacheArticles(): Promise<void> {
        if (!this.cacheArticlesButton || !this.outputElement) {
            return;
        }
        
        try {
            // Disable the button while processing
            this.cacheArticlesButton.disabled = true;
            this.cacheArticlesButton.textContent = 'Processing...';
            
            // Display processing status
            this.addMessageToOutput('Starting article caching process...', 'info');
            
            // Call the cache-articles endpoint
            const response = await axios.get('http://localhost:5001/cache-articles');
            
            // Display success message
            this.addMessageToOutput(`${response.data.message}`, 'success');
            
            // Add a note about the terminal window if on Mac
            const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
            if (isMac) {
                this.addMessageToOutput('A new Terminal window has been opened to show the caching process.', 'info');
                this.addMessageToOutput('You can monitor the progress in that window.', 'info');
            }
            
        } catch (error: any) {
            // Display error message
            const errorMessage = error.response?.data?.message || 'Error caching articles. Is the server running?';
            this.addMessageToOutput(errorMessage, 'error');
            console.error('Error:', error);
        } finally {
            // Re-enable the button
            this.cacheArticlesButton.disabled = false;
            this.cacheArticlesButton.textContent = 'Cache Articles Only';
        }
    }
    
    private async checkCache(): Promise<void> {
        try {
            this.updateLoadingMessage('Checking article cache...');
            
            // Check if articles are cached
            const response = await axios.get('http://localhost:5001/check-cache');
            
            if (response.data.cached) {
                this.updateLoadingMessage(`Found ${response.data.article_count} cached articles and ${response.data.final_article_count || 0} final articles. Preparing interface...`);
                
                // Don't show any final articles automatically - hide until user requests them
                if (response.data.final_article_count > 0) {
                    setTimeout(() => {
                        this.addMessageToOutput(`${response.data.final_article_count} final articles with questions & answers are available.`, 'info');
                        
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
                this.updateLoadingMessage('No cached articles found. You may want to cache articles first.');
            }
            
            // Artificial delay to ensure everything is ready
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            // Hide loading and show email form
            this.toggleSections('loading-section', 'email-section');
            
        } catch (error) {
            console.error('Error checking cache:', error);
            this.updateLoadingMessage('Error checking cache. Proceeding anyway...');
            
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
            // Store the email
            this.userEmail = email;
            
            // Display success message
            this.addMessageToOutput(`Email registered: ${email}`, 'success');
            
            // Hide email section and show interests section
            this.toggleSections('email-section', 'interests-section');
            
        } catch (error) {
            // Display error message
            this.addMessageToOutput('Error processing email', 'error');
            console.error('Error:', error);
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

        // Display a nice message about what we're doing
        this.addMessageToOutput(`Analyzing your interests: ${interests}`, 'info');
        
        try {
            // Send the interests to the server along with the email
            const response = await axios.post('http://localhost:5001/get-best-article-match', {
                interests: interests
            });
            
            // Handle the response
            if (response.data.status === 'success') {
                const article = response.data.article;
                
                // Create a message with the article info
                const matchMessage = `Found the best article match for your interests (${article.match_score}/100):`;
                this.addMessageToOutput(matchMessage, 'success');
                
                // Create a div for the article recommendation
                const articleDiv = document.createElement('div');
                articleDiv.className = 'article-recommendation';
                
                // Create a header for the article
                const articleTitle = document.createElement('h3');
                articleTitle.textContent = article.title;
                articleDiv.appendChild(articleTitle);
                
                // If we have an HTML version, add a link to it
                if (article.has_html) {
                    // Extract just the filename from the path
                    const pathParts = article.html_path.split('/');
                    const htmlFilename = pathParts[pathParts.length - 1];
                    
                    // Create a button to view the full article
                    const viewButton = document.createElement('button');
                    viewButton.className = 'view-article-btn';
                    viewButton.textContent = 'View Full Article';
                    viewButton.addEventListener('click', () => {
                        // Open the HTML article in a new tab/window
                        window.open(`/final_articles/html/${htmlFilename}`, '_blank');
                    });
                    
                    articleDiv.appendChild(viewButton);
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
                this.outputElement.appendChild(articleDiv);
                
                // After 5 seconds, get all available articles
                // setTimeout(() => {
                //     this.addMessageToOutput('Other articles you might be interested in:', 'info');
                //     this.handleViewArticles(false); // Don't show loading message
                // }, 5000);
                
                // Suggest other interests to try if they want to see more articles
                // this.addMessageToOutput('Want to explore more? Try entering different interests!', 'info');
                
            } else {
                // Error message from the server
                this.addMessageToOutput(`Error: ${response.data.message}`, 'error');
            }
        } catch (error: any) {
            console.error('Error:', error);
            const errorMessage = error.response?.data?.message || 'Error processing your interests. Is the server running?';
            this.addMessageToOutput(errorMessage, 'error');
        }
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

    private async handleGenerateQuestions(): Promise<void> {
        if (!this.generateQuestionsButton || !this.outputElement) {
            return;
        }
        
        try {
            // Disable the button while processing
            this.generateQuestionsButton.disabled = true;
            this.generateQuestionsButton.textContent = 'Processing...';
            
            // Display processing status
            this.addMessageToOutput('Starting full question and answer generation...', 'info');
            
            // Call the generate-questions endpoint
            const response = await axios.get('http://localhost:5001/generate-questions');
            
            // Display success message
            this.addMessageToOutput(`${response.data.message}`, 'success');
            
            // Add a note about the terminal window if on Mac
            const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
            if (isMac) {
                this.addMessageToOutput('A new Terminal window has been opened to show the generation process.', 'info');
                this.addMessageToOutput('This process will take several minutes. You can monitor the progress in that window.', 'info');
            }
            
        } catch (error: any) {
            // Display error message
            const errorMessage = error.response?.data?.message || 'Error generating questions. Is the server running?';
            this.addMessageToOutput(errorMessage, 'error');
            console.error('Error:', error);
        } finally {
            // Re-enable the button
            this.generateQuestionsButton.disabled = false;
            this.generateQuestionsButton.textContent = 'Generate Questions & Answers';
        }
    }

    private async handleViewArticles(showLoadingMessage: boolean = true): Promise<void> {
        if (!this.outputElement) {
            return;
        }
        
        try {
            // Display loading status if requested
            if (showLoadingMessage) {
                this.addMessageToOutput('Loading final articles...', 'info');
            }
            
            // Get the list of final articles
            const response = await axios.get('http://localhost:5001/get-final-articles');
            
            if (response.data.articles && response.data.articles.length > 0) {
                // Display articles
                if (showLoadingMessage) {
                    this.addMessageToOutput(`Found ${response.data.articles.length} final articles:`, 'success');
                }
                
                // Create a container for the articles
                const articlesContainer = document.createElement('div');
                articlesContainer.classList.add('articles-container');
                
                // Add each article
                response.data.articles.forEach((article: any) => {
                    const articleElement = document.createElement('div');
                    articleElement.classList.add('article-item');
                    
                    // Format date
                    const date = new Date(article.timestamp * 1000);
                    const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
                    
                    // Add article title and timestamp
                    articleElement.innerHTML = `
                        <h3>${article.title}</h3>
                        <p>Generated: ${dateStr}</p>
                        <button class="view-article-btn" data-id="${article.id}">View Article</button>
                    `;
                    
                    // Add to container
                    articlesContainer.appendChild(articleElement);
                });
                
                // Add to output
                this.outputElement.appendChild(articlesContainer);
                
                // Add event listeners to view buttons
                const viewButtons = document.querySelectorAll('.view-article-btn');
                viewButtons.forEach(button => {
                    button.addEventListener('click', (event) => {
                        const id = (event.target as HTMLElement).getAttribute('data-id');
                        if (id) {
                            this.handleViewArticleContent(id);
                        }
                    });
                });
            } else {
                if (showLoadingMessage) {
                    this.addMessageToOutput('No final articles found.', 'info');
                }
            }
        } catch (error: any) {
            // Display error message
            const errorMessage = error.response?.data?.message || 'Error loading articles. Is the server running?';
            this.addMessageToOutput(errorMessage, 'error');
            console.error('Error:', error);
        }
    }
    
    private async handleViewArticleContent(articleId: string): Promise<void> {
        if (!this.outputElement) {
            return;
        }
        
        try {
            // Display loading status
            this.addMessageToOutput(`Loading article content...`, 'info');
            
            // Get the article content
            const response = await axios.get(`http://localhost:5001/get-final-article/${articleId}`);
            
            if (response.data.article) {
                const article = response.data.article;
                
                // Create a container for the article content
                const contentContainer = document.createElement('div');
                contentContainer.classList.add('article-content');
                
                // Add article content
                const contentElement = document.createElement('div');
                contentElement.classList.add('markdown-content');
                
                // Convert Markdown content to HTML
                const markdownContent = article.content;
                
                // Simple Markdown to HTML conversion for headers and paragraphs
                let htmlContent = markdownContent
                    .replace(/^# (.*$)/gm, '<h1>$1</h1>')
                    .replace(/^## (.*$)/gm, '<h2>$1</h2>')
                    .replace(/^### (.*$)/gm, '<h3>$1</h3>')
                    .replace(/^#### (.*$)/gm, '<h4>$1</h4>')
                    .replace(/^##### (.*$)/gm, '<h5>$1</h5>')
                    .replace(/\n\n/gm, '</p><p>')
                    .replace(/\*\*(.*?)\*\*/gm, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/gm, '<em>$1</em>');
                
                // Wrap in paragraphs
                htmlContent = '<p>' + htmlContent + '</p>';
                
                contentElement.innerHTML = htmlContent;
                contentContainer.appendChild(contentElement);
                
                // Add a close button
                const closeButton = document.createElement('button');
                closeButton.textContent = 'Close Article';
                closeButton.classList.add('close-article-btn');
                closeButton.addEventListener('click', () => {
                    // Remove the content container
                    contentContainer.remove();
                });
                contentContainer.appendChild(closeButton);
                
                // Add to output
                this.outputElement.appendChild(contentContainer);
                
                // Scroll to the content
                contentContainer.scrollIntoView({ behavior: 'smooth' });
            } else {
                this.addMessageToOutput('Article content not found.', 'error');
            }
        } catch (error: any) {
            // Display error message
            const errorMessage = error.response?.data?.message || 'Error loading article content. Is the server running?';
            this.addMessageToOutput(errorMessage, 'error');
            console.error('Error:', error);
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