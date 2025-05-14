const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const express = require('express');

// Path to the stocker.py script
const stockerPath = path.join(__dirname, '..', '..', 'stockbt', 'stocker.py');

/**
 * Run a backtest using the stocker.py script
 * @param {string} strategy - The trading strategy in plain English
 * @returns {Promise<Object>} - The backtest results
 */
async function runBacktest(strategy) {
  return new Promise((resolve, reject) => {
    console.log(`Running backtest with strategy: ${strategy}`);
    console.log(`Using stocker.py at: ${stockerPath}`);

    const process = spawn('python3', [
      stockerPath,
      '--strategy', strategy,
      '--save-chart',
      '--json-output'
    ]);

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
        // Try to parse the JSON output
        const results = JSON.parse(outputData);
        resolve(results);
      } catch (error) {
        console.error('Failed to parse stocker.py output:', error);
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
    if (backtestResults.chart_path) {
      chartImage = await imageToBase64(backtestResults.chart_path);
    }
    
    res.json({
      status: 'success',
      profit: backtestResults.profit_loss,
      trades: {
        count: backtestResults.buy_points?.length || 0,
        buys: backtestResults.buy_points || [],
        sells: backtestResults.sell_points || []
      },
      balance_history: backtestResults.balance_over_time || [],
      code: backtestResults.generated_code || '',
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

module.exports = router; 