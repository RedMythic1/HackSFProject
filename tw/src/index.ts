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
    private cacheCheckInterval: number | null = null;
    
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
                const validCount = response.data.valid_article_count || 0;
                const uniqueCount = response.data.final_article_count || 0;
                
                this.updateLoadingMessage(`Found ${response.data.article_count} cached articles and ${uniqueCount} final articles. Preparing interface...`);
                
                // Add a status element for live cache updates
                const cacheStatusDiv = document.createElement('div');
                cacheStatusDiv.classList.add('cache-info');
                cacheStatusDiv.textContent = `Cached articles: ${response.data.article_count}, Final articles: ${uniqueCount}`;
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
                this.updateLoadingMessage('No cached articles found. You may want to cache articles first.');
                
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
            
            // Check URL hash to see if we should use a specific article task
            if (window.location.hash) {
                const hashParams = new URLSearchParams(window.location.hash.substring(1));
                const taskId = hashParams.get('task');
                if (taskId) {
                    // If we have a task_id in the URL, poll for its status
                    this.addMessageToOutput('Loading article from link...', 'info');
                    this.pollArticleMatchTask(taskId);
                    return;
                }
            }
            
            // Handle the response - check for 202 status (Accepted) which means task was started
            if (response.status === 202 && response.data.task_id) {
                // Store task_id in URL hash
                window.location.hash = `task=${response.data.task_id}`;
                
                this.addMessageToOutput('Article matching process initiated...', 'info');
                
                // Start polling for the result
                this.pollArticleMatchTask(response.data.task_id);
                
            } else if (response.data.status === 'success') {
                // Direct success response (unlikely with the new task-based approach)
                const article = response.data.article;
                this.displayArticleMatch(article);
            } else {
                // Error message from the server
                this.addMessageToOutput(`Processing error: ${response.data.message}`, 'error');
            }
        } catch (error: any) {
            console.error('Error:', error);
            const errorMessage = error.response?.data?.message || 'Error processing your interests. Is the server running?';
            this.addMessageToOutput(errorMessage, 'error');
        }
    }
    
    private pollArticleMatchTask(taskId: string, attempts: number = 0): void {
        const maxAttempts = 60; // Poll for up to 2 minutes (5 * 30 seconds)
        const pollInterval = 2000; // 2 seconds
        
        if (attempts >= maxAttempts) {
            this.addMessageToOutput('Article matching process timed out. Please try again.', 'error');
            return;
        }
        
        setTimeout(async () => {
            try {
                const response = await axios.get(`http://localhost:5001/get-match-progress/${taskId}`);
                
                // Update progress display
                const progress = response.data.percentage || 0;
                const status = response.data.status || 'Processing...';
                
                // Update or create a progress bar
                this.updateProgressBar(progress, status);
                
                // Check if task is completed
                if (response.data.completed) {
                    // Remove progress bar
                    const progressBar = document.querySelector('.task-progress-container');
                    if (progressBar) {
                        progressBar.remove();
                    }
                    
                    if (response.data.final_result && response.data.final_result.status === 'success') {
                        // Display the article match
                        const article = response.data.final_result.article;
                        this.displayArticleMatch(article);
                        
                        // Add direct article link
                        if (article.article_url) {
                            window.open(article.article_url, '_blank');
                        }
                    } else {
                        // Error in task result
                        const errorMsg = response.data.final_result?.message || 'Error finding article match';
                        this.addMessageToOutput(`Processing error: ${errorMsg}`, 'error');
                    }
                } else {
                    // Continue polling
                    this.pollArticleMatchTask(taskId, attempts + 1);
                }
            } catch (error) {
                console.error('Error polling for task status:', error);
                this.addMessageToOutput('Error checking article match status. Will retry...', 'info');
                // Continue polling even on error
                this.pollArticleMatchTask(taskId, attempts + 1);
            }
        }, pollInterval);
    }
    
    private updateProgressBar(progress: number, status: string): void {
        // Find or create progress bar container
        let progressContainer = document.querySelector('.task-progress-container') as HTMLDivElement;
        
        if (!progressContainer) {
            progressContainer = document.createElement('div');
            progressContainer.className = 'task-progress-container';
            
            const progressBarOuter = document.createElement('div');
            progressBarOuter.className = 'progress-bar-outer';
            
            const progressBarInner = document.createElement('div');
            progressBarInner.className = 'progress-bar-inner';
            
            const statusText = document.createElement('p');
            statusText.className = 'progress-status';
            
            progressBarOuter.appendChild(progressBarInner);
            progressContainer.appendChild(progressBarOuter);
            progressContainer.appendChild(statusText);
            
            // Add to the beginning of the output
            if (this.outputElement) {
                this.outputElement.insertBefore(progressContainer, this.outputElement.firstChild);
            }
        }
        
        // Update progress bar
        const progressBar = progressContainer.querySelector('.progress-bar-inner') as HTMLDivElement;
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
        
        // Update status text
        const statusElement = progressContainer.querySelector('.progress-status') as HTMLParagraphElement;
        if (statusElement) {
            statusElement.textContent = `${status} (${progress}%)`;
        }
    }
    
    private displayArticleMatch(article: any): void {
        if (!this.outputElement) return;
        
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
        
        // Create a button to view the article
        const viewButton = document.createElement('button');
        viewButton.className = 'view-article-btn';
        viewButton.textContent = 'View Full Article';
        
        // Set up the button click event
        viewButton.addEventListener('click', () => {
            // Check for direct article URL from the server
            if (article.article_url) {
                window.open(article.article_url, '_blank');
            } else if (article.id) {
                // Fallback to local URL construction
                window.open(`http://localhost:5001/article-html/${article.id}`, '_blank');
            }
        });
        
        articleDiv.appendChild(viewButton);
        
        // Create a simple message that explains what the user can do
        const explainText = document.createElement('p');
        explainText.className = 'article-explanation';
        explainText.textContent = 'This article closely matches your interests. Click the button above to view the full article.';
        articleDiv.appendChild(explainText);
        
        // Add the article div to the output
        this.outputElement.appendChild(articleDiv);
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
            if (showLoadingMessage) {
                this.addMessageToOutput('Loading article list...', 'info');
            }
            
            // Remove any existing article containers
            const existingContainers = document.querySelectorAll('.articles-container');
            existingContainers.forEach(container => container.remove());

            // Get the list of final articles
            const response = await axios.get('http://localhost:5001/get-final-articles');
            
            if (response.data.status === 'success' && response.data.articles && response.data.articles.length > 0) {
                // Create a container for the articles
                const articlesContainer = document.createElement('div');
                articlesContainer.className = 'articles-container';
                
                // Add a heading
                const heading = document.createElement('h3');
                heading.textContent = `Available Articles (${response.data.articles.length})`;
                articlesContainer.appendChild(heading);
                
                // Create the list
                const articleList = document.createElement('ul');
                articleList.className = 'article-list';
                
                // Add each article
                response.data.articles.forEach((article: any) => {
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
                if (response.data.invalid_count > 0) {
                    this.addMessageToOutput(`Note: ${response.data.invalid_count} invalid article files were cleaned up during this process.`, 'info');
                }
            } else {
                if (showLoadingMessage) {
                    this.addMessageToOutput('No final articles found. You may need to generate them first.', 'error');
                }
            }
        } catch (error) {
            console.error('Error viewing articles:', error);
            if (showLoadingMessage) {
                this.addMessageToOutput('Error retrieving articles. Please try again.', 'error');
            }
        }
    }
    
    private async handleViewArticleContent(articleId: string): Promise<void> {
        if (!this.outputElement) {
            return;
        }

        try {
            this.addMessageToOutput('Loading article content...', 'info');
            
            // Remove any existing article content display
            const existingContent = document.querySelectorAll('.article-content');
            existingContent.forEach(content => content.remove());

            // Get the article content
            const response = await axios.get(`http://localhost:5001/get-final-article/${articleId}`);
            
            if (response.data.status === 'success' && response.data.article) {
                const article = response.data.article;
                
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
            this.addMessageToOutput('Error loading article content. Please try again.', 'error');
        }
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