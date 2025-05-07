# HackSF Project - Personalized Tech Newsletter

A web application that analyzes Hacker News articles, identifies content matching user interests, and generates personalized tech deep dives based on user preferences.

## Project Overview

This project fetches articles from Hacker News, caches them, analyzes their content, and ranks them based on user interests. It then generates detailed summaries and research about the most relevant articles for the user.

## File Structure and Purpose

### Core Components

- **ansys.py**: Main analysis script that handles article fetching, caching, processing, and ranking based on user interests. Generates questions and summaries for articles.
- **hackernews_summarizer.py**: Specialized module for summarizing Hacker News content, extracting relevant information, and cleaning text.
- **emailsender.py**: Utility for sending email notifications with personalized tech deep dives.

### Web Interface

- **typescript-webpage/**: Directory containing the web interface
  - **server.py**: Flask backend server that provides API endpoints for the frontend.
  - **src/index.ts**: Main TypeScript file controlling the frontend behavior.
  - **public/index.html**: HTML structure for the web interface.
  - **public/styles.css**: CSS styling for the web interface.

### Scripts and Utilities

- **typescript-webpage/init.sh**: Script to start both the frontend and backend servers.
- **typescript-webpage/setup-ansys.sh**: Script to set up ansys.py in the correct location.
- **typescript-webpage/test-cache.sh**: Script to test the article caching functionality.

### Configuration Files

- **typescript-webpage/package.json**: Frontend dependencies and scripts.
- **typescript-webpage/tsconfig.json**: TypeScript configuration.
- **typescript-webpage/webpack.config.js**: Webpack build configuration.
- **typescript-webpage/requirements.txt**: Python dependencies for the backend.

### Data Storage

- **.cache/**: Directory for storing cached articles and generated content.
- **typescript-webpage/user_data/**: Directory for storing user preference data.

## Workflow

### Setup Process

1. **Initial Setup**:
   ```bash
   cd typescript-webpage
   npm install
   pip install -r requirements.txt
   ./setup-ansys.sh
   ```

2. **Start the Application**:
   ```bash
   ./init.sh
   ```

### User Flow

1. **Article Caching**:
   - Access the admin panel by pressing `Shift+Alt+A`
   - Click "Cache Articles & Generate Questions"
   - Wait for the process to complete

2. **User Input**:
   - Enter email address
   - Specify interests (comma-separated)
   - Submit for analysis

3. **Backend Processing**:
   - Server receives user interests
   - ansys.py ranks cached articles based on interests
   - Generates questions about top articles
   - Researches answers from the web
   - Creates a detailed summary/deep dive

4. **Result Delivery**:
   - Results are displayed on the webpage
   - Optionally sent to the user's email

### Development Workflow

1. **Frontend Changes**:
   - Modify files in typescript-webpage/src/ or typescript-webpage/public/
   - Webpack automatically rebuilds when using init.sh

2. **Backend Changes**:
   - Modify server.py or ansys.py
   - Restart the server to apply changes

3. **Testing**:
   - Use test-cache.sh to verify article caching
   - Check server.log and frontend.log for errors

## Key Features

- Semantic article analysis and ranking
- Web content extraction and summarization
- Question generation for deeper research
- Responsive web interface
- Admin controls for article caching
- Email delivery system

## Requirements

- Node.js v14+
- Python 3.8+
- Flask and related dependencies
- Modern web browser 