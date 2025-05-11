#!/bin/bash

# setup-edge-config.sh - Setup environment variables for Edge Config

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Export environment variables
export EDGE_CONFIG="https://edge-config.vercel.com/ecfg_xsczamr0q3eodjuagxzwjiznqxxs?token=854495e2-1208-47c1-84a6-213468e23510"
export USE_EDGE_CONFIG="1"

echo -e "${BLUE}Environment variables set:${NC}"
echo -e "${GREEN}EDGE_CONFIG:${NC} $EDGE_CONFIG"
echo -e "${GREEN}USE_EDGE_CONFIG:${NC} $USE_EDGE_CONFIG"

# Create/update a .env file (for applications that support it)
cat > .env << EOL
EDGE_CONFIG=https://edge-config.vercel.com/ecfg_xsczamr0q3eodjuagxzwjiznqxxs?token=854495e2-1208-47c1-84a6-213468e23510
USE_EDGE_CONFIG=1
EOL

echo -e "\n${GREEN}Created/updated .env file${NC}"

# Verify Node.js and @vercel/edge-config package are installed
if command -v node &> /dev/null; then
    if node -e "try { require('@vercel/edge-config'); console.log('Vercel Edge Config package is installed.'); } catch(e) { console.error('Error: @vercel/edge-config package is not installed.'); process.exit(1); }" &> /dev/null; then
        echo -e "\n${BLUE}Testing Edge Config access:${NC}"
        node -e "
        const { createClient } = require('@vercel/edge-config');
        
        async function testConfig() {
            try {
                const config = createClient('$EDGE_CONFIG');
                
                // Test if we can read and write
                console.log('Testing connection to Edge Config...');
                
                const testKey = 'test_connection_' + Date.now();
                const testValue = { status: 'connected', timestamp: Date.now() };
                
                // Get all existing values
                const allItems = await config.getAll();
                console.log('Successfully connected to Edge Config');
                
                // Add our test value
                await config.patch({
                    ...allItems,
                    [testKey]: testValue
                });
                console.log('Successfully wrote test value');
                
                // Read back our test value
                const readValue = await config.get(testKey);
                console.log('Successfully read test value:', readValue);
                
                // Remove our test value
                const { [testKey]: removed, ...rest } = allItems;
                await config.patch(rest);
                console.log('Successfully removed test value');
                
                return true;
            } catch (error) {
                console.error('Error accessing Edge Config:');
                console.error(error.message);
                return false;
            }
        }
        
        testConfig();" || echo -e "\n${RED}Edge Config access test failed. Please check your connection string.${NC}"
    else
        echo -e "\n${RED}Error: @vercel/edge-config package is not installed.${NC}"
        echo -e "${YELLOW}Please install it with: npm install @vercel/edge-config${NC}"
    fi
else
    echo -e "\n${RED}Error: Node.js is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install Node.js to use Edge Config${NC}"
fi

# Make the utilities executable
if [ -f "tools/edge-config-utils.js" ]; then
    chmod +x tools/edge-config-utils.js
    echo -e "\n${GREEN}Made Edge Config utilities executable${NC}"
else
    echo -e "\n${YELLOW}Warning: Edge Config utilities not found at tools/edge-config-utils.js${NC}"
fi

echo -e "\n${BLUE}Run the following command to use these environment variables in your shell:${NC}"
echo -e "${GREEN}source setup-edge-config.sh${NC}" 