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
    
    // Store the user email
    private userEmail: string = '';

    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.checkCache();
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
        
        // Add keyboard shortcut for admin panel
        document.addEventListener('keydown', this.handleKeyDown.bind(this));
    }
    
    private handleKeyDown(event: KeyboardEvent): void {
        // Show admin panel when Shift+Alt+A is pressed
        if (event.shiftKey && event.altKey && event.key === 'A') {
            const adminSection = document.getElementById('admin-section');
            if (adminSection) {
                adminSection.style.display = adminSection.style.display === 'none' ? 'block' : 'none';
                
                // If showing admin panel, add a message to the output
                if (adminSection.style.display === 'block') {
                    this.addMessageToOutput('Admin panel activated', 'info');
                }
            }
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
            this.addMessageToOutput('Starting article caching and question generation...', 'info');
            
            // Call the cache-articles endpoint
            const response = await axios.get('http://localhost:5001/cache-articles');
            
            // Display success message
            this.addMessageToOutput(`${response.data.message}`, 'success');
            
        } catch (error: any) {
            // Display error message
            const errorMessage = error.response?.data?.message || 'Error caching articles. Is the server running?';
            this.addMessageToOutput(errorMessage, 'error');
            console.error('Error:', error);
        } finally {
            // Re-enable the button
            this.cacheArticlesButton.disabled = false;
            this.cacheArticlesButton.textContent = 'Cache Articles & Generate Questions';
        }
    }
    
    private async checkCache(): Promise<void> {
        try {
            this.updateLoadingMessage('Checking article cache...');
            
            // Check if articles are cached
            const response = await axios.get('http://localhost:5001/check-cache');
            
            if (response.data.cached) {
                this.updateLoadingMessage(`Found ${response.data.article_count} cached articles. Preparing interface...`);
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
            this.addMessageToOutput('Please enter at least one interest', 'error');
            return;
        }

        try {
            // Display processing status
            this.addMessageToOutput('Processing interests... This may take a while.', 'info');
            
            // Disable the button and input while processing
            const submitButton = this.interestsFormElement?.querySelector('button');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.textContent = 'Processing...';
            }
            if (this.interestsInputElement) {
                this.interestsInputElement.disabled = true;
            }
            
            // Send interests and email to Python backend
            const response = await axios.post('http://localhost:5001/run-ansys', {
                email: this.userEmail,
                interests: interests
            });
            
            // Display success message
            this.addMessageToOutput(`Analysis complete!`, 'success');
            
            // Display terminal message
            this.addMessageToOutput(`A new terminal window has been opened to show real-time output. Please check your desktop.`, 'info');
            
            // Display the output from ansys.py
            if (response.data.output) {
                this.addMessageToOutput(`Web output summary:`, 'info');
                this.addRawTextToOutput(response.data.output);
            }
            
            // Clear input
            this.interestsInputElement.value = '';
            
        } catch (error: any) {
            // Display error message
            const errorMessage = error.response?.data?.message || 'Error running analysis. Is the server running?';
            this.addMessageToOutput(errorMessage, 'error');
            console.error('Error:', error);
        } finally {
            // Re-enable the button and input
            const submitButton = this.interestsFormElement?.querySelector('button');
            if (submitButton) {
                submitButton.disabled = false;
                submitButton.textContent = 'Analyze';
            }
            if (this.interestsInputElement) {
                this.interestsInputElement.disabled = false;
            }
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

    public init(): void {
        console.log('Application initialized');
    }
}

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new CommandApp();
    app.init();
}); 