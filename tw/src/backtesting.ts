import './style.css';
import './backtesting.css';
import { renderBacktestCharts, runBacktestAndChart } from './utils/backtest-charts';
import Chart from 'chart.js/auto';

// Add Prism type definition
declare global {
    interface Window {
        Prism?: {
            highlightElement: (element: HTMLElement) => void;
        };
    }
}

// Function to log messages from the frontend
function logFrontend(message: string, data?: any) {
    console.log(`[FRONTEND] ${message}`, data !== undefined ? data : '');
}

// Backend API result interfaces 
interface BacktestSuccessResult {
    status: 'success';
    profit: number;
    buy_hold_profit: number;
    percent_above_buyhold: number;
    dataset: string;
    datasets_tested: number;
    buy_points: Array<[number, number]> | Array<{x: number, y: number}>;
    sell_points: Array<[number, number]> | Array<{x: number, y: number}>;
    balance_over_time: number[];
    close: number[];
    dates: string[];
    trades: {
        count: number;
        buys: Array<[number, number]> | Array<{x: number, y: number}>;
        sells: Array<[number, number]> | Array<{x: number, y: number}>;
    };
    code: string;
    all_iterations?: Array<{
        iteration: number;
        dataset: string;
        profit: number;
        percent_above_buyhold: number;
        trades_count: number;
    }>;
}

// For API error results
interface BacktestErrorResult {
    status: 'error';
    error: string;
}

// Combined type
type BacktestResult = BacktestSuccessResult | BacktestErrorResult;

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
        
        // Add event listeners for chart tabs
        const chartTabs = document.querySelectorAll('.chart-tab');
        chartTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs
                chartTabs.forEach(t => t.classList.remove('active'));
                // Add active class to clicked tab
                tab.classList.add('active');
                
                // Get target panel
                const targetPanelId = tab.getAttribute('data-target');
                if (targetPanelId) {
                    // Hide all panels
                    const panels = document.querySelectorAll('.chart-panel');
                    panels.forEach(panel => panel.classList.remove('active'));
                    
                    // Show target panel
                    const targetPanel = document.querySelector(`.${targetPanelId}`);
                    if (targetPanel) {
                        targetPanel.classList.add('active');
                    }
                }
            });
        });
        
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
            return result as BacktestResult;
        } catch (error: any) {
            logFrontend('Error in callBacktestAPI catch block', error);
            console.error('API call error:', error);
            return {
                status: 'error',
                error: error instanceof Error ? error.message : String(error)
            };
        }
    }
    
    private displayResults(result: BacktestResult): void {
        logFrontend('displayResults started with result:', result);
        if (!this.resultsContainer) {
            logFrontend('resultsContainer not found, cannot display results', 'ERROR');
            return;
        }
        
        if (result.status === 'error') {
            const errorResult = result as BacktestErrorResult;
            logFrontend('Displaying error message from result', errorResult.error);
            this.showError(errorResult.error || 'An unknown error occurred during backtest processing.');
            return;
        }
        
        this.resultsContainer.style.display = 'block';
        logFrontend('Results container displayed');
        
        // Update profit/loss
        if (this.profitLossValue) {
            const profitValue = result.profit;
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
            const totalTrades = result.trades?.count || result.buy_points.length;
            this.tradesCount.textContent = totalTrades.toString();
        }
        
        // Update success rate
        if (this.successRate) {
            const buyPoints = result.buy_points;
            const sellPoints = result.sell_points;
            
            let successfulTrades = 0;
            
            // Count trades where sell price > buy price
            if (buyPoints && sellPoints) {
                for (let i = 0; i < Math.min(buyPoints.length, sellPoints.length); i++) {
                    const buyPrice = buyPoints[i][1];
                    const sellPrice = sellPoints[i][1];
                    if (sellPrice > buyPrice) {
                        successfulTrades++;
                    }
                }
            }
            
            const totalTrades = Math.min(buyPoints.length, sellPoints.length);
            const successRateValue = totalTrades > 0 ? (successfulTrades / totalTrades) * 100 : 0;
            this.successRate.textContent = `${successRateValue.toFixed(1)}%`;
            
            // Add color based on success rate
            this.successRate.className = 'result-value ' + 
                (successRateValue >= 50 ? 'positive' : 'negative');
        }
        
        // Update "vs Buy & Hold" value
        const vsBuyHoldElement = document.getElementById('vs-buyhold');
        if (vsBuyHoldElement && 'percent_above_buyhold' in result) {
            const percentValue = result.percent_above_buyhold;
            vsBuyHoldElement.textContent = `${percentValue > 0 ? '+' : ''}${percentValue.toFixed(1)}%`;
            vsBuyHoldElement.className = 'result-value ' + 
                (percentValue >= 0 ? 'positive' : 'negative');
        }
        
        // Show generated code
        if (this.generatedCode) {
            this.generatedCode.textContent = result.code || '';
            
            // Add Prism.js syntax highlighting if it's available
            if (window.Prism) {
                window.Prism.highlightElement(this.generatedCode);
            }
        }
        
        // Update dataset info
        const datasetNameElement = document.getElementById('dataset-name');
        const datasetSizeElement = document.getElementById('dataset-size');
        
        if (datasetNameElement && result.dataset) {
            datasetNameElement.textContent = result.dataset || 'Unknown Dataset';
        }
        
        if (datasetSizeElement && result.close) {
            datasetSizeElement.textContent = `${result.close.length} data points`;
        }
        
        // Create charts
        this.createCharts(result as BacktestSuccessResult);
    }

    private createCharts(result: BacktestSuccessResult): void {
        logFrontend('createCharts started');
        
        // Get chart canvases
        const priceChartCanvas = document.getElementById('price-chart') as HTMLCanvasElement;
        const balanceChartCanvas = document.getElementById('balance-chart') as HTMLCanvasElement;
        
        if (!priceChartCanvas || !balanceChartCanvas) {
            logFrontend('Chart canvases not found', 'ERROR');
            console.error('Chart canvases not found:', {
                priceChartCanvas: !!priceChartCanvas,
                balanceChartCanvas: !!balanceChartCanvas
            });
            return;
        }
        
        // Convert tuple points to x,y objects if needed
        const processPoints = (points: Array<[number, number]> | Array<{x: number, y: number}>): Array<{x: number, y: number}> => {
            if (points.length === 0) return [];
            
            // Check if points are already in {x,y} format
            if (typeof points[0] === 'object' && 'x' in points[0] && 'y' in points[0]) {
                return points as Array<{x: number, y: number}>;
            }
            
            // Convert tuple format to {x,y} format
            return (points as Array<[number, number]>).map(point => ({
                x: point[0],
                y: point[1]
            }));
        };
        
        // Process points for charts
        const buyPoints = processPoints(result.buy_points);
        const sellPoints = processPoints(result.sell_points);
        
        logFrontend(`Processing ${result.buy_points.length} buy points and ${result.sell_points.length} sell points`);
        logFrontend('First few buy points:', buyPoints.slice(0, 3));
        logFrontend('First few sell points:', sellPoints.slice(0, 3));
        
        try {
            logFrontend('Creating price chart');
            
            // Destroy existing chart instances if they exist
            if (this.priceChart) {
                this.priceChart.destroy();
            }
            if (this.balanceChart) {
                this.balanceChart.destroy();
            }
            
            // Create charts using the utility functions
            const charts = renderBacktestCharts(
                result as any,  // Cast as any to avoid type issues
                priceChartCanvas,
                balanceChartCanvas
            );
            
            // Store the chart references
            this.priceChart = charts.priceChart;
            this.balanceChart = charts.balanceChart;

            // Update dataset information
            const datasetNameElement = document.getElementById('dataset-name');
            const datasetSizeElement = document.getElementById('dataset-size');
            
            if (datasetNameElement) {
                datasetNameElement.textContent = result.dataset || 'Unknown Dataset';
            }
            
            if (datasetSizeElement) {
                datasetSizeElement.textContent = `${result.close.length.toLocaleString()} data points`;
            }
            
            // Update vs buy & hold percentage
            const vsBuyholdElement = document.getElementById('vs-buyhold');
            if (vsBuyholdElement) {
                const percentage = result.percent_above_buyhold;
                vsBuyholdElement.textContent = `${percentage >= 0 ? '+' : ''}${percentage.toFixed(2)}%`;
                vsBuyholdElement.className = 'result-value ' + (percentage >= 0 ? 'positive' : 'negative');
            }
            
            logFrontend('Charts created successfully');
        } catch (error) {
            logFrontend('Error creating charts', error);
            console.error('Error creating charts:', error);
        }
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