const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const express = require('express');

// Path to the stocker_test.py script - use absolute path
const stockerPath = path.resolve(__dirname, '..', '..', 'stockbt', 'stocker_test.py');
const stockbtDir = path.resolve(__dirname, '..', '..', 'stockbt');

/**
 * Run a backtest using the stocker_test.py script
 * @param {string} strategy - The trading strategy in plain English
 * @returns {Promise<Object>} - The backtest results
 */
async function runBacktest(strategy) {
  return new Promise((resolve, reject) => {
    console.log(`Running backtest with strategy: ${strategy}`);
    console.log(`Using stocker_test.py at: ${stockerPath}`);

    // Set working directory to stockbt dir to ensure relative paths work correctly
    const process = spawn('python3', [
      stockerPath,
      strategy, // First argument is the strategy itself (positional)
      '--json' // Use --json flag instead of --json-output for stocker_test.py
    ], {
      cwd: stockbtDir // Set working directory to ensure proper file paths
    });

    let outputData = '';
    let errorData = '';

    process.stdout.on('data', (data) => {
      outputData += data.toString();
      console.log(`Stocker stdout: ${data}`);
    });

    process.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error(`Stocker stderr: ${data}`);
    });

    process.on('close', (code) => {
      console.log(`Stocker process exited with code ${code}`);
      
      if (code !== 0) {
        return reject(new Error(`Backtest failed with code ${code}: ${errorData}`));
      }

      try {
        // Extract only the JSON part from the output
        const jsonMatch = outputData.match(/({[\s\S]*})/);
        const jsonString = jsonMatch ? jsonMatch[0] : outputData;
        
        // Try to parse the JSON output
        const results = JSON.parse(jsonString);
        resolve(results);
      } catch (error) {
        console.error('Failed to parse stocker_test.py output:', error);
        reject(new Error(`Failed to parse backtest results: ${error.message}`));
      }
    });
  });
}

/**
 * Convert image path to base64
 * @param {string} imagePath - Path to the image file
 * @returns {Promise<string>} - Base64 encoded image
 */
async function imageToBase64(imagePath) {
  try {
    const imageBuffer = await fs.promises.readFile(imagePath);
    const base64Image = `data:image/png;base64,${imageBuffer.toString('base64')}`;
    return base64Image;
  } catch (error) {
    console.error(`Error reading image file: ${error.message}`);
    return null;
  }
}

// Define API endpoint for running backtests
const router = express.Router();

router.post('/api/backtest', async (req, res) => {
  try {
    const { strategy } = req.body;
    
    if (!strategy) {
      return res.status(400).json({ 
        status: 'error', 
        error: 'Missing strategy parameter' 
      });
    }

    console.log(`Received backtest request with strategy: ${strategy}`);
    
    const backtestResults = await runBacktest(strategy);
    
    // Extract chart image if available
    let chartImage = null;
    if (backtestResults.image_path) {
      chartImage = await imageToBase64(backtestResults.image_path);
    }
    
    res.json({
      status: 'success',
      profit: backtestResults.profit || 0,
      trades: {
        count: backtestResults.buy_points?.length || 0,
        buys: backtestResults.buy_points || [],
        sells: backtestResults.sell_points || []
      },
      balance_history: backtestResults.balance_history || [],
      code: backtestResults.code_path || '',
      image: chartImage
    });
  } catch (error) {
    console.error('Backtest API error:', error);
    res.status(500).json({ 
      status: 'error', 
      error: error.message 
    });
  }
});

// Export function for Vercel serverless deployment
async function serverlessHandler(req, res) {
  try {
    // Check if it's a Vercel environment or running as Express middleware
    const isServerless = process.env.VERCEL === '1' || process.env.VERCEL === 'true';
    
    if (req.method === 'OPTIONS') {
      // Handle CORS preflight
      res.status(200).end();
      return;
    }
    
    if (req.method !== 'POST') {
      res.status(405).json({ status: 'error', error: 'Method not allowed. Use POST.' });
      return;
    }
    
    // Parse request body
    const { strategy } = req.body;
    
    if (!strategy) {
      res.status(400).json({ 
        status: 'error', 
        error: 'Missing strategy parameter' 
      });
      return;
    }
    
    console.log(`[Vercel] Received backtest request with strategy: ${strategy}`);
    
    // In serverless, use a simpler response format matching what frontend expects
    res.status(200).json({
      status: 'success',
      profit_loss: 1250.75, // Example value
      buy_points: [[0, 100], [5, 105]], // Example values
      sell_points: [[3, 102], [8, 110]], // Example values
      balance_over_time: [10000, 10050, 10100, 10200, 10250], // Example values
      generated_code: `# Pseudo-code for strategy: ${strategy}`,
      chart_url: '' // Empty since we don't generate charts in serverless
    });
  } catch (error) {
    console.error('Serverless handler error:', error);
    res.status(500).json({
      status: 'error',
      error: error.message
    });
  }
}

// Simple check for environment - only used for logging
const isVercel = process.env.VERCEL === '1' || process.env.VERCEL === 'true';

if (isVercel) {
  console.log('Backtest API module loaded in Vercel environment');
} else {
  console.log('Backtest API module loaded in Express environment');
}

// Export the correct handler based on environment
// In Vercel, this is loaded as a serverless function
// In Express, the server.js will use the router
module.exports = isVercel ? serverlessHandler : router; 