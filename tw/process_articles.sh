#!/bin/bash
# process_articles.sh - Command-line interface for article processing
# This script provides all article functions that were previously in the web interface

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Ensure we're using the tw/local_cache directory for local operations
CACHE_DIR="$SCRIPT_DIR/local_cache"
# Path to local ansys.py file
ANSYS_LOCAL_PATH="$SCRIPT_DIR/ansys_local.py"

# Load Edge Config environment
source "$SCRIPT_DIR/setup-edge-config.sh" >/dev/null 2>&1 || true

# Edge Config variables
EDGE_CONFIG_URL="https://edge-config.vercel.com/ecfg_xsczamr0q3eodjuagxzwjiznqxxs?token=854495e2-1208-47c1-84a6-213468e23510"

# Make sure cache directory exists (for local fallback)
mkdir -p "$CACHE_DIR"

# Function to print a styled header
print_header() {
    echo -e "\n${BLUE}========== $1 ==========${NC}\n"
}

# Function to print a success message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print an error message
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print a warning/info message
print_info() {
    echo -e "${YELLOW}! $1${NC}"
}

# Function to print a warning message
print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

# Function to check if ansys_local.py exists
check_ansys() {
    if [ ! -f "$ANSYS_LOCAL_PATH" ]; then
        print_error "ansys_local.py not found at $ANSYS_LOCAL_PATH"
        echo "Please make sure ansys_local.py is located in the current directory."
        exit 1
    fi
    print_success "Found ansys_local.py at $ANSYS_LOCAL_PATH"
    echo ""
    return 0
}

# Function to check if Edge Config environment variables are set
check_edge_config() {
    if [ -z "$EDGE_CONFIG" ]; then
        print_warning "EDGE_CONFIG not set. Falling back to local cache."
        return 1
    fi
    
    # Check if @vercel/edge-config is installed
    if ! node -e "try { require('@vercel/edge-config'); } catch(e) { process.exit(1); }" &> /dev/null; then
        print_warning "@vercel/edge-config package not found. Falling back to local cache."
        return 1
    }
    
    print_success "Using Vercel Edge Config for storage"
    return 0
}

# Function to handle caching of articles (only subjectizes and caches subject lines)
cache_articles() {
    print_header "CACHING ARTICLES"
    print_info "This only subjectizes articles and caches the subject lines"
    
    # Check for ansys_local.py
    check_ansys
    
    # Get the already processed article IDs
    PROCESSED_IDS_FILE=$(mktemp)
    
    # Try to get processed_articles.json from Edge Config first
    if check_edge_config; then
        print_info "Attempting to get processed articles from Edge Config"
        node -e "
        const { createClient } = require('@vercel/edge-config');
        
        async function getProcessedArticles() {
            try {
                const edgeConfig = createClient('$EDGE_CONFIG');
                const data = await edgeConfig.get('articles/processed_articles.json');
                if (data) {
                    console.log(JSON.stringify(data));
                    return true;
                }
                return false;
            } catch (error) {
                console.error('Error getting processed articles:', error.message);
                return false;
            }
        }
        getProcessedArticles();" > "$PROCESSED_IDS_FILE" 2>/dev/null || true
        
        if [ -s "$PROCESSED_IDS_FILE" ]; then
            print_success "Retrieved processed articles from Edge Config"
        else
            echo '{"processed_ids":[]}' > "$PROCESSED_IDS_FILE"
            print_info "Starting fresh with no processed articles"
        fi
    elif [ -f "$CACHE_DIR/processed_articles.json" ]; then
        cp "$CACHE_DIR/processed_articles.json" "$PROCESSED_IDS_FILE"
        print_info "Using existing processed articles list from local cache"
    else
        echo '{"processed_ids":[]}' > "$PROCESSED_IDS_FILE"
        print_info "Starting fresh with no processed articles"
    fi
    
    # Set environment variables for ansys_local.py
    export ANSYS_PROCESSED_IDS_FILE="$PROCESSED_IDS_FILE"
    export ANSYS_NO_SCORE=1
    # Force using tw/local_cache directory for local files
    export CACHE_DIR="$CACHE_DIR"
    # Set Edge Config environment variables
    export EDGE_CONFIG="$EDGE_CONFIG_URL"
    export USE_EDGE_CONFIG=1
    
    print_info "Starting article caching process (subjectizing only)..."
    
    # Use default interests (doesn't matter for caching)
    echo "technology, programming, science" | python3 "$ANSYS_LOCAL_PATH" --cache-only
    
    # Check if caching was successful
    if [ $? -eq 0 ]; then
        print_success "Articles cached and subjectized successfully"
        
        # Upload the processed_ids file to Edge Config if available
        if check_edge_config; then
            print_info "Uploading processed articles file to Edge Config"
            node -e "
            const fs = require('fs');
            const { createClient } = require('@vercel/edge-config');
            
            async function uploadProcessedArticles() {
                try {
                    const edgeConfig = createClient('$EDGE_CONFIG');
                    const data = JSON.parse(fs.readFileSync('$PROCESSED_IDS_FILE', 'utf8'));
                    
                    // Get existing items to create patch
                    const allItems = await edgeConfig.getAll();
                    const patch = {
                        ...allItems,
                        'articles/processed_articles.json': data
                    };
                    
                    await edgeConfig.patch(patch);
                    console.log('Uploaded processed articles to Edge Config');
                    return true;
                } catch (error) {
                    console.error('Error uploading processed articles:', error.message);
                    return false;
                }
            }
            uploadProcessedArticles();" >/dev/null 2>&1 || true
            
            # Also run the upload cache script to ensure all files are in Edge Config
            print_info "Ensuring all cache files are in Edge Config..."
            node "$SCRIPT_DIR/tools/edge-config-utils.js" upload-dir "$CACHE_DIR" "articles/" >/dev/null 2>&1 || true
        else
            # Fallback to local file
            cp "$PROCESSED_IDS_FILE" "$CACHE_DIR/processed_articles.json"
        fi
    else
        print_error "Error caching articles"
    fi
    
    # Clean up temporary file
    rm -f "$PROCESSED_IDS_FILE"
}

# Function to process articles with user interests
process_articles() {
    INTERESTS="$1"
    
    if [ -z "$INTERESTS" ]; then
        print_error "No interests provided. Please specify interests."
        echo "Usage: $0 process \"technology, AI, science\""
        return 1
    fi
    
    print_header "PROCESSING ARTICLES WITH INTERESTS: $INTERESTS"
    
    # Check for ansys_local.py
    check_ansys
    
    # Get the already processed article IDs
    PROCESSED_IDS_FILE=$(mktemp)
    
    # Try to get processed_articles.json from Edge Config first
    if check_edge_config; then
        print_info "Attempting to get processed articles from Edge Config"
        node -e "
        const { createClient } = require('@vercel/edge-config');
        
        async function getProcessedArticles() {
            try {
                const edgeConfig = createClient('$EDGE_CONFIG');
                const data = await edgeConfig.get('articles/processed_articles.json');
                if (data) {
                    console.log(JSON.stringify(data));
                    return true;
                }
                return false;
            } catch (error) {
                console.error('Error getting processed articles:', error.message);
                return false;
            }
        }
        getProcessedArticles();" > "$PROCESSED_IDS_FILE" 2>/dev/null || true
        
        if [ -s "$PROCESSED_IDS_FILE" ]; then
            print_success "Retrieved processed articles from Edge Config"
        else
            echo '{"processed_ids":[]}' > "$PROCESSED_IDS_FILE"
            print_info "Starting fresh with no processed articles"
        fi
    elif [ -f "$CACHE_DIR/processed_articles.json" ]; then
        cp "$CACHE_DIR/processed_articles.json" "$PROCESSED_IDS_FILE"
        print_info "Using existing processed articles list from local cache"
    else
        echo '{"processed_ids":[]}' > "$PROCESSED_IDS_FILE"
        print_info "Starting fresh with no processed articles"
    fi
    
    # Set environment variables for ansys_local.py
    export ANSYS_PROCESSED_IDS_FILE="$PROCESSED_IDS_FILE"
    # Force using tw/local_cache directory for local fallback
    export CACHE_DIR="$CACHE_DIR"
    # Set Edge Config environment variables
    export EDGE_CONFIG="$EDGE_CONFIG_URL"
    export USE_EDGE_CONFIG=1
    
    print_info "Starting article processing with interests: $INTERESTS"
    
    # Run ansys_local.py with the provided interests
    echo "$INTERESTS" | python3 "$ANSYS_LOCAL_PATH"
    
    # Check if processing was successful
    if [ $? -eq 0 ]; then
        print_success "Articles processed successfully"
        
        # Upload the processed_ids file to Edge Config if available
        if check_edge_config; then
            print_info "Uploading processed articles file to Edge Config"
            node -e "
            const fs = require('fs');
            const { createClient } = require('@vercel/edge-config');
            
            async function uploadProcessedArticles() {
                try {
                    const edgeConfig = createClient('$EDGE_CONFIG');
                    const data = JSON.parse(fs.readFileSync('$PROCESSED_IDS_FILE', 'utf8'));
                    
                    // Get existing items to create patch
                    const allItems = await edgeConfig.getAll();
                    const patch = {
                        ...allItems,
                        'articles/processed_articles.json': data
                    };
                    
                    await edgeConfig.patch(patch);
                    console.log('Uploaded processed articles to Edge Config');
                    return true;
                } catch (error) {
                    console.error('Error uploading processed articles:', error.message);
                    return false;
                }
            }
            uploadProcessedArticles();" >/dev/null 2>&1 || true
            
            # Also run the upload cache script to ensure all files are in Edge Config
            print_info "Ensuring all cache files are in Edge Config..."
            node "$SCRIPT_DIR/tools/edge-config-utils.js" upload-dir "$CACHE_DIR" "articles/" >/dev/null 2>&1 || true
        else
            # Fallback to local file
            cp "$PROCESSED_IDS_FILE" "$CACHE_DIR/processed_articles.json"
        fi
        
        # Check if we need to copy HTML files to public directory
        HTML_DIR="$SCRIPT_DIR/public/articles"
        
        # Look for HTML files in current directory
        HTML_FILES=$(ls tech_deep_dive_*.html 2>/dev/null)
        
        if [ -n "$HTML_FILES" ]; then
            mkdir -p "$HTML_DIR"
            print_info "Copying HTML files to public directory..."
            cp -n $HTML_FILES "$HTML_DIR/" 2>/dev/null || true
            print_success "HTML files copied successfully"
        fi
    else
        print_error "Error processing articles"
    fi
    
    # Clean up temporary file
    rm -f "$PROCESSED_IDS_FILE"
}

# Function to list all cached articles
list_articles() {
    print_header "CACHED ARTICLES"
    
    if check_edge_config; then
        print_info "Listing articles from Edge Config"
        node -e "
        const { createClient } = require('@vercel/edge-config');
        
        async function listArticles() {
            try {
                const edgeConfig = createClient('$EDGE_CONFIG');
                const allItems = await edgeConfig.getAll();
                
                // Filter keys that start with the article prefix
                const articleKeys = Object.keys(allItems).filter(key => 
                    key.startsWith('articles/final_article_')
                );
                
                console.log(`Found ${articleKeys.length} articles in Edge Config`);
                
                if (articleKeys.length === 0) {
                    console.log('No articles found. You may need to process articles first.');
                    return;
                }
                
                console.log('\nFinal Articles:');
                console.log('----------------');
                
                // Sort keys (newest first based on timestamp in the path)
                articleKeys.sort((a, b) => b.localeCompare(a));
                
                // Display articles
                for (let i = 0; i < articleKeys.length; i++) {
                    const key = articleKeys[i];
                    const id = key.replace('articles/final_article_', '').replace('.json', '');
                    const data = allItems[key];
                    
                    try {
                        // Extract title from content
                        let title = 'Unknown Title';
                        if (data.content) {
                            const lines = data.content.split('\\n');
                            if (lines[0]) {
                                title = lines[0].startsWith('# ') ? lines[0].substring(2) : lines[0];
                            }
                        }
                        
                        console.log(`${i+1}. ${title} (${id})`);
                    } catch (error) {
                        console.log(`${i+1}. Error parsing article ${id}: ${error.message}`);
                    }
                }
            } catch (error) {
                console.error('Error listing articles:', error.message);
            }
        }
        listArticles();" || true
    else
        # Fallback to local listing
        # Count the cached article summaries
        SUMMARY_COUNT=$(find "$CACHE_DIR" -name "summary_*.json" | wc -l)
        print_info "Found $SUMMARY_COUNT article summaries in local cache"
        
        # Count the final articles
        FINAL_COUNT=$(find "$CACHE_DIR" -name "final_article_*.json" | wc -l)
        print_info "Found $FINAL_COUNT final articles in local cache"
        
        if [ $FINAL_COUNT -eq 0 ]; then
            print_info "No final articles found. You may need to process articles first."
            return 0
        fi
        
        echo ""
        echo "Final Articles:"
        echo "----------------"
        
        # Extract and display article titles from final article files
        ARTICLE_NUMBER=1
        for ARTICLE_FILE in $(find "$CACHE_DIR" -name "final_article_*.json" | sort); do
            TITLE=$(grep -m 1 "^# " "$ARTICLE_FILE" | sed 's/^# //' || echo "Unknown Title")
            ID=$(basename "$ARTICLE_FILE" | sed 's/final_article_//' | sed 's/\.json//')
            echo "$ARTICLE_NUMBER. $TITLE ($ID)"
            ARTICLE_NUMBER=$((ARTICLE_NUMBER + 1))
        done
    fi
}

# Function to display help
show_help() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  cache                Cache articles without processing (only subjectizes)"
    echo "  process \"interests\"  Process articles with specified interests"
    echo "  list                 List all cached articles"
    echo "  help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 cache                                # Cache and subjectize articles only"
    echo "  $0 process \"technology, AI, science\"    # Process articles with interests"
    echo "  $0 list                                 # List all cached articles"
}

# Main script execution
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    cache)
        cache_articles
        ;;
    process)
        process_articles "$2"
        ;;
    list)
        list_articles
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac 