import './style.css';
import './backtesting.css';
import Chart from 'chart.js/auto';

// Function to log messages from the frontend
function logFrontend(message: string, data?: any) {
    console.log(`[FRONTEND] ${message}`, data !== undefined ? data : '');
}

interface BacktestResult {
    status: string;
    profit_loss?: number;
    profit?: number; // Alternative field name used by Node backend
    buy_points?: Array<[number, number]>;
    sell_points?: Array<[number, number]>;
    balance_over_time?: number[];
    balance_history?: number[]; // Alternative field name used by Node backend
    generated_code?: string;
    code?: string; // Alternative field used by Node backend
    chart_url?: string;
    image?: string; // Alternative field used by Node backend
    trades?: {
        count: number;
        buys: Array<[number, number]>;
        sells: Array<[number, number]>;
    };
    error?: string;
    close?: number[]; // Added to receive price data
    dates?: string[]; // Optional date labels
}

class BacktestingController {
    private strategyInput: HTMLTextAreaElement | null;
    private runBacktestBtn: HTMLElement | null;
    private loadingIndicator: HTMLElement | null;
    private resultsContainer: HTMLElement | null;
    private profitLossValue: HTMLElement | null;
    private tradesCount: HTMLElement | null;
    private successRate: HTMLElement | null;
    private generatedCode: HTMLElement | null;
    private priceChart: Chart | null = null;
    private balanceChart: Chart | null = null;
    
    constructor() {
        logFrontend('BacktestingController constructor started');
        this.initElements();
        this.initEventListeners();
        this.initWebSocket();
        logFrontend('BacktestingController constructor finished');
    }
    
    private initElements(): void {
        logFrontend('initElements started');
        this.strategyInput = document.getElementById('strategy-input') as HTMLTextAreaElement;
        this.runBacktestBtn = document.getElementById('run-backtest-btn');
        this.loadingIndicator = document.getElementById('loading-indicator');
        this.resultsContainer = document.getElementById('results-container');
        this.profitLossValue = document.getElementById('profit-loss-value');
        this.tradesCount = document.getElementById('trades-count');
        this.successRate = document.getElementById('success-rate');
        this.generatedCode = document.getElementById('generated-code');
        logFrontend('initElements finished', {
            strategyInputExists: !!this.strategyInput,
            runBacktestBtnExists: !!this.runBacktestBtn
        });
    }
    
    private initEventListeners(): void {
        logFrontend('initEventListeners started');
        if (this.runBacktestBtn) {
            this.runBacktestBtn.addEventListener('click', () => {
                logFrontend('Run Backtest button clicked');
                this.runBacktest();
            });
            logFrontend('Event listener added to Run Backtest button');
        } else {
            logFrontend('Run Backtest button not found', 'WARN');
        }
        logFrontend('initEventListeners finished');
    }
    
    private async runBacktest(): Promise<void> {
        logFrontend('runBacktest started');
        if (!this.strategyInput || !this.strategyInput.value.trim()) {
            logFrontend('Strategy input is empty', 'WARN');
            alert('Please enter a trading strategy before running a backtest.');
            return;
        }
        
        const strategy = this.strategyInput.value.trim();
        logFrontend('Strategy to backtest:', strategy.substring(0, 100) + (strategy.length > 100 ? '...' : ''));
        
        this.showLoading(true);
        
        try {
            logFrontend('Calling callBacktestAPI');
            const result = await this.callBacktestAPI(strategy);
            logFrontend('callBacktestAPI returned', result);
            this.displayResults(result);
        } catch (error) {
            logFrontend('Error in runBacktest catch block', error);
            console.error('Error running backtest:', error);
            alert('An error occurred while running the backtest. Please try again.');
        } finally {
            logFrontend('runBacktest finally block');
            this.showLoading(false);
        }
        logFrontend('runBacktest finished');
    }
    
    private async callBacktestAPI(strategy: string): Promise<BacktestResult> {
        logFrontend('callBacktestAPI started for strategy:', strategy.substring(0, 100) + (strategy.length > 100 ? '...' : ''));
        try {
            logFrontend('Sending POST request to /api/backtest');
            const response = await fetch('/api/backtest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ strategy })
            });
            
            logFrontend('Response status from /api/backtest:', response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                logFrontend(`Server error: ${response.status} ${response.statusText}. Response text: ${errorText}`, 'ERROR');
                console.error(`Server error: ${response.status} ${response.statusText}`);
                console.error('Error response text:', errorText);
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            logFrontend('Parsed JSON result from /api/backtest:', result);
            return result;
        } catch (error) {
            logFrontend('Error in callBacktestAPI catch block', error);
            console.error('API call error:', error);
            return {
                status: 'error',
                error: error instanceof Error ? error.message : 'Unknown error occurred during API call'
            };
        }
    }
    
    private displayResults(result: BacktestResult): void {
        logFrontend('displayResults started with result:', result);
        if (!this.resultsContainer) {
            logFrontend('resultsContainer not found, cannot display results', 'ERROR');
            return;
        }
        
        if (result.status === 'error' || result.error) {
            logFrontend('Displaying error message from result', result.error);
            this.showError(result.error || 'An unknown error occurred during backtest processing.');
            return;
        }
        
        this.resultsContainer.style.display = 'block';
        logFrontend('Results container displayed');
        
        // Update profit/loss
        if (this.profitLossValue) {
            const profitValue = result.profit_loss ?? result.profit ?? 0;
            logFrontend('Profit/Loss value:', profitValue);
            
            const formattedProfitLoss = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(profitValue);
            
            this.profitLossValue.textContent = formattedProfitLoss;
            this.profitLossValue.className = 'result-value ' + 
                (profitValue >= 0 ? 'positive' : 'negative');
        }
        
        // Update trades count
        if (this.tradesCount) {
            // Get trades count from different possible response formats
            let totalTrades = 0;
            if (result.trades && result.trades.count) {
                totalTrades = result.trades.count;
            } else if (result.buy_points) {
                totalTrades = result.buy_points.length;
            }
            
            this.tradesCount.textContent = totalTrades.toString();
        }
        
        // Update success rate
        if (this.successRate) {
            let successfulTrades = 0;
            let buyPoints = result.buy_points || [];
            let sellPoints = result.sell_points || [];
            
            // Handle different response formats
            if (result.trades) {
                buyPoints = result.trades.buys || [];
                sellPoints = result.trades.sells || []; 
            }
            
            // Count trades where sell price > buy price
            if (buyPoints && sellPoints) {
                for (let i = 0; i < Math.min(buyPoints.length, sellPoints.length); i++) {
                    if (sellPoints[i][1] > buyPoints[i][1]) {
                        successfulTrades++;
                    }
                }
            }
            
            const buyPointsLength = buyPoints?.length || 0;
            const rate = buyPointsLength > 0 
                ? Math.round((successfulTrades / buyPointsLength) * 100) 
                : 0;
                
            this.successRate.textContent = `${rate}%`;
        }
        
        // Update charts
        logFrontend('Calling createCharts');
        this.createCharts(result);
        
        // Update generated code
        if (this.generatedCode) {
            const codeContent = result.generated_code || result.code || '';
            logFrontend('Generated code content length:', codeContent.length);
            this.generatedCode.textContent = codeContent;
        }
        logFrontend('displayResults finished');
    }

    private createCharts(result: BacktestResult): void {
        logFrontend('createCharts started with result:', result);
        // Get the prices data
        const close = result.close || [];
        logFrontend('Number of close prices for chart:', close.length);
        
        // Generate x-axis labels (either use dates from API or generate indices)
        let labels: string[] = [];
        if (result.dates && result.dates.length === close.length) {
            labels = result.dates;
        } else {
            labels = Array.from({ length: close.length }, (_, i) => `Day ${i+1}`);
        }
        
        // Get buy and sell points
        let buyPoints = result.buy_points || [];
        let sellPoints = result.sell_points || [];
        
        // Handle different response formats
        if (result.trades) {
            buyPoints = result.trades.buys || [];
            sellPoints = result.trades.sells || [];
        }
        
        // Get balance over time
        const balanceData = result.balance_over_time || result.balance_history || [];
        
        // Create price chart
        this.createPriceChart(labels, close, buyPoints, sellPoints);
        
        // Create balance chart
        this.createBalanceChart(labels, balanceData);
        logFrontend('createCharts finished');
    }
    
    private createPriceChart(labels: string[], prices: number[], buyPoints: Array<[number, number]>, sellPoints: Array<[number, number]>): void {
        const priceChartCanvas = document.getElementById('price-chart') as HTMLCanvasElement;
        if (!priceChartCanvas) return;
        
        // Destroy previous chart if it exists
        if (this.priceChart) {
            this.priceChart.destroy();
        }
        
        // Create buy points dataset
        const buyPointsData = Array(prices.length).fill(null);
        buyPoints.forEach(point => {
            const [index] = point;
            if (index >= 0 && index < buyPointsData.length) {
                buyPointsData[index] = prices[index];
            }
        });
        
        // Create sell points dataset
        const sellPointsData = Array(prices.length).fill(null);
        sellPoints.forEach(point => {
            const [index] = point;
            if (index >= 0 && index < sellPointsData.length) {
                sellPointsData[index] = prices[index];
            }
        });
        
        this.priceChart = new Chart(priceChartCanvas, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Stock Price',
                        data: prices,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1,
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'Buy Points',
                        data: buyPointsData,
                        backgroundColor: 'rgba(75, 192, 75, 1)',
                        borderColor: 'rgba(75, 192, 75, 1)',
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        showLine: false
                    },
                    {
                        label: 'Sell Points',
                        data: sellPointsData,
                        backgroundColor: 'rgba(255, 99, 132, 1)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        showLine: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Price ($)'
                        }
                    },
                    x: {
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Stock Price with Buy/Sell Points'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                }
            }
        });
    }
    
    private createBalanceChart(labels: string[], balanceData: number[]): void {
        const balanceChartCanvas = document.getElementById('balance-chart') as HTMLCanvasElement;
        if (!balanceChartCanvas) return;
        
        // Destroy previous chart if it exists
        if (this.balanceChart) {
            this.balanceChart.destroy();
        }
        
        this.balanceChart = new Chart(balanceChartCanvas, {
            type: 'line',
            data: {
                labels: labels.slice(0, balanceData.length),
                datasets: [
                    {
                        label: 'Portfolio Value',
                        data: balanceData,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderWidth: 2,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Value ($)'
                        },
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    },
                    x: {
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Portfolio Value Over Time'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += new Intl.NumberFormat('en-US', { 
                                        style: 'currency', 
                                        currency: 'USD' 
                                    }).format(context.parsed.y);
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }
    
    private showError(errorMessage: string): void {
        if (!this.resultsContainer) return;
        
        this.resultsContainer.innerHTML = `
            <div class="error-message">
                <h3>Error Running Backtest</h3>
                <p>${errorMessage}</p>
                <button class="retry-button" id="retry-button">Try Again</button>
            </div>
        `;
        
        this.resultsContainer.style.display = 'block';
        
        // Add event listener to retry button
        const retryButton = document.getElementById('retry-button');
        if (retryButton) {
            retryButton.addEventListener('click', () => {
                this.resetForm();
            });
        }
    }
    
    private resetForm(): void {
        if (this.resultsContainer) {
            this.resultsContainer.style.display = 'none';
        }
        
        if (this.loadingIndicator) {
            this.loadingIndicator.style.display = 'none';
        }
        
        // Destroy charts if they exist
        if (this.priceChart) {
            this.priceChart.destroy();
            this.priceChart = null;
        }
        
        if (this.balanceChart) {
            this.balanceChart.destroy();
            this.balanceChart = null;
        }
    }
    
    private showLoading(show: boolean): void {
        logFrontend(`showLoading called with: ${show}`);
        if (!this.loadingIndicator) return;
        
        this.loadingIndicator.style.display = show ? 'block' : 'none';
        
        // Hide results while loading
        if (show && this.resultsContainer) {
            this.resultsContainer.style.display = 'none';
        }
        
        // Disable or enable the submit button
        if (this.runBacktestBtn) {
            if (show) {
                this.runBacktestBtn.setAttribute('disabled', 'true');
                this.runBacktestBtn.classList.add('disabled');
            } else {
                this.runBacktestBtn.removeAttribute('disabled');
                this.runBacktestBtn.classList.remove('disabled');
            }
        }
    }

    private initWebSocket(): void {
        logFrontend('initWebSocket started (for log fetching)');
        // const ws = new WebSocket('ws://localhost:3000'); // WebSocket not used for this log fetching

        document.addEventListener('DOMContentLoaded', () => {
            logFrontend('DOMContentLoaded event fired for log fetching setup');
            const backtestLogOutputElement = document.getElementById('backtestLogOutput') as HTMLPreElement;
            logFrontend('backtestLogOutputElement found:', !!backtestLogOutputElement);

            async function fetchAndDisplayLogs() {
                // logFrontend('fetchAndDisplayLogs called'); // This would be too noisy
                if (!backtestLogOutputElement) {
                    // console.warn('[FRONTEND] backtestLogOutputElement not found in fetchAndDisplayLogs');
                    return;
                }

                try {
                    const response = await fetch('/api/backtest-logs');
                    if (!response.ok) {
                        const errorText = `Error fetching logs: ${response.status} ${response.statusText}`;
                        // logFrontend(errorText, 'ERROR');
                        if (backtestLogOutputElement.textContent !== errorText) { // Avoid spamming same error
                           backtestLogOutputElement.textContent = errorText;
                        }
                        return;
                    }
                    const logs = await response.text();
                    if (backtestLogOutputElement.textContent !== logs) { // Only update if content changed
                        // logFrontend('Logs received, length:', logs.length);
                        backtestLogOutputElement.textContent = logs;
                        backtestLogOutputElement.scrollTop = backtestLogOutputElement.scrollHeight;
                    }
                } catch (error) {
                    // logFrontend('Failed to fetch or display logs:', error);
                    // console.error('Failed to fetch or display logs:', error);
                    if (backtestLogOutputElement) {
                        const errorMessage = 'Failed to load logs. Check console for details.';
                        if (backtestLogOutputElement.textContent !== errorMessage) {
                           backtestLogOutputElement.textContent = errorMessage;
                        }
                    }
                }
            }

            if (backtestLogOutputElement) {
                logFrontend('Initial call to fetchAndDisplayLogs and setting interval');
                fetchAndDisplayLogs(); 
                setInterval(fetchAndDisplayLogs, 3000); 
            } else {
                logFrontend('backtestLogOutputElement not found, log polling not started', 'WARN');
            }
        });
        logFrontend('initWebSocket finished');
    }
}

logFrontend('Script loaded. Adding DOMContentLoaded listener for AppController.');
document.addEventListener('DOMContentLoaded', () => {
    logFrontend('DOMContentLoaded event fired. Initializing AppController.');
    new BacktestingController();
    logFrontend('AppController initialized.');
}); 