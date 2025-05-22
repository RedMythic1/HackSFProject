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
    code_summary?: string;
}

// For API error results
interface BacktestErrorResult {
    status: 'error';
    error: string;
}

// Combined type
type BacktestResult = BacktestSuccessResult | BacktestErrorResult;

// Add interfaces for the stock prediction API response
interface StockPredictionSuccessResult {
    status: 'success';
    dataset_used: string;
    graphs: Array<{
        title: string;
        data: string; // Base64 encoded image
    }>;
    prediction_data: {
        pid_walk_preds: number[];
        pid_walk_actuals: number[];
        pid_mse: number;
        points_range: number[];
    };
    pid_tuning: {
        best_parameters: {
            Kp: number;
            Ki: number;
            Kd: number;
        };
        tuning_results: Array<{
            Kp: number;
            Ki: number;
            Kd: number;
            MSE: number;
        }>;
    };
    training_progress: Array<{
        epoch: number;
        predicted: number;
        actual: number;
        error: number;
        message?: string;
    }>;
    summary: string;
}

interface StockPredictionErrorResult {
    status: 'error';
    error: string;
}

type StockPredictionResult = StockPredictionSuccessResult | StockPredictionErrorResult;

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
    private testNewFeatureBtn: HTMLElement | null;
    
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
        
        // Add element for test new feature button
        this.testNewFeatureBtn = document.getElementById('test-new-feature-btn');
        
        logFrontend('initElements finished', {
            strategyInputExists: !!this.strategyInput,
            runBacktestBtnExists: !!this.runBacktestBtn,
            testNewFeatureExists: !!this.testNewFeatureBtn
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
        
        // Add event listener for Test New Feature button
        if (this.testNewFeatureBtn) {
            this.testNewFeatureBtn.addEventListener('click', () => {
                logFrontend('Test New Feature button clicked');
                this.runStockPrediction();
            });
            logFrontend('Event listener added to Test New Feature button');
        } else {
            logFrontend('Test New Feature button not found', 'WARN');
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
            logFrontend('Error in result:', errorResult.error);
            this.showError(errorResult.error);
            return;
        }
        
        // Show the results container
        this.resultsContainer.style.display = 'block';
        
        const successResult = result as BacktestSuccessResult;
        
        // Update the profit/loss value
        if (this.profitLossValue) {
            const profit = successResult.profit;
            this.profitLossValue.textContent = `$${profit.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            this.profitLossValue.className = 'result-value ' + (profit >= 0 ? 'positive' : 'negative');
        }
        
        // Update trades count
        if (this.tradesCount && successResult.trades) {
            const trades = successResult.trades.count;
            this.tradesCount.textContent = trades.toString();
        }
        
        // Create charts
        this.createCharts(successResult);
        
        // Update generated code, using syntax highlighting if Prism is available
        if (this.generatedCode && successResult.code) {
            this.generatedCode.textContent = successResult.code;
            this.generatedCode.className = 'code-block language-python';
            
            // Apply syntax highlighting with Prism if available
            if (window.Prism) {
                window.Prism.highlightElement(this.generatedCode);
            }
        }
        
        // Update code summary if available
        const codeSummaryElement = document.getElementById('code-summary');
        if (codeSummaryElement && successResult.code_summary) {
            codeSummaryElement.textContent = successResult.code_summary;
        } else if (codeSummaryElement) {
            codeSummaryElement.textContent = 'No code summary available.';
        }
        
        logFrontend('displayResults finished');
    }
    
    private createCharts(result: BacktestSuccessResult): void {
        try {
            logFrontend('createCharts started');
            
            const priceChartCanvas = document.getElementById('price-chart') as HTMLCanvasElement;
            const balanceChartCanvas = document.getElementById('balance-chart') as HTMLCanvasElement;
            
            if (!priceChartCanvas || !balanceChartCanvas) {
                logFrontend('Canvas elements not found, cannot create charts', 'ERROR');
                return;
            }
            
            // Destroy previous charts if they exist
            if (this.priceChart) {
                this.priceChart.destroy();
            }
            if (this.balanceChart) {
                this.balanceChart.destroy();
            }
            
            // Create charts using the utility functions
            const charts = renderBacktestCharts(
                result,
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

        document.addEventListener('DOMContentLoaded', () => {
            logFrontend('DOMContentLoaded event fired for log fetching setup');
            const backtestLogOutputElement = document.getElementById('backtestLogOutput') as HTMLPreElement;
            logFrontend('backtestLogOutputElement found:', !!backtestLogOutputElement);

            async function fetchAndDisplayLogs() {
                if (!backtestLogOutputElement) {
                    return;
                }

                try {
                    const response = await fetch('/api/backtest-logs');
                    if (!response.ok) {
                        const errorText = `Error fetching logs: ${response.status} ${response.statusText}`;
                        if (backtestLogOutputElement.textContent !== errorText) {
                           backtestLogOutputElement.textContent = errorText;
                        }
                        return;
                    }
                    const logs = await response.text();
                    if (backtestLogOutputElement.textContent !== logs) {
                        backtestLogOutputElement.textContent = logs;
                        backtestLogOutputElement.scrollTop = backtestLogOutputElement.scrollHeight;
                    }
                } catch (error) {
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

    private async runStockPrediction(): Promise<void> {
        logFrontend('runStockPrediction started');
        
        // Show prediction section and loading indicator
        const predictionSection = document.getElementById('prediction-section');
        const predictionLoadingIndicator = document.getElementById('prediction-loading-indicator');
        const predictionResultsContainer = document.getElementById('prediction-results-container');
        
        if (predictionSection) {
            predictionSection.style.display = 'block';
        }
        
        if (predictionLoadingIndicator) {
            predictionLoadingIndicator.style.display = 'block';
        }
        
        if (predictionResultsContainer) {
            predictionResultsContainer.style.display = 'none';
        }
        
        // Hide backtest results if they're showing
        if (this.resultsContainer) {
            this.resultsContainer.style.display = 'none';
        }
        
        // Scroll to prediction section
        predictionSection?.scrollIntoView({ behavior: 'smooth' });
        
        try {
            logFrontend('Calling fetchStockPrediction');
            const result = await this.fetchStockPrediction();
            logFrontend('fetchStockPrediction returned', result);
            this.displayStockPredictionResults(result);
        } catch (error) {
            logFrontend('Error in runStockPrediction catch block', error);
            console.error('Error running stock prediction:', error);
            alert('An error occurred while running stock prediction. Please try again.');
        } finally {
            logFrontend('runStockPrediction finally block');
            if (predictionLoadingIndicator) {
                predictionLoadingIndicator.style.display = 'none';
            }
        }
        logFrontend('runStockPrediction finished');
    }
    
    private async fetchStockPrediction(): Promise<StockPredictionResult> {
        logFrontend('fetchStockPrediction started');
        try {
            logFrontend('Sending GET request to /api/stock_prediction');
            const response = await fetch('/api/stock_prediction', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            logFrontend('Response status from /api/stock_prediction:', response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                logFrontend(`Server error: ${response.status} ${response.statusText}. Response text: ${errorText}`, 'ERROR');
                console.error(`Server error: ${response.status} ${response.statusText}`);
                console.error('Error response text:', errorText);
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            logFrontend('Parsed JSON result from /api/stock_prediction:', result);
            return result as StockPredictionResult;
        } catch (error: any) {
            logFrontend('Error in fetchStockPrediction catch block', error);
            console.error('API call error:', error);
            return {
                status: 'error',
                error: error instanceof Error ? error.message : String(error)
            };
        }
    }
    
    private displayStockPredictionResults(result: StockPredictionResult): void {
        logFrontend('displayStockPredictionResults started with result:', result);
        
        const predictionResultsContainer = document.getElementById('prediction-results-container');
        
        if (result.status === 'error') {
            this.showPredictionError(result.error);
            return;
        }
        
        // Display results
        if (predictionResultsContainer) {
            predictionResultsContainer.style.display = 'block';
        }
        
        // Set dataset used
        const datasetUsedElement = document.getElementById('dataset-used');
        if (datasetUsedElement) {
            datasetUsedElement.textContent = result.dataset_used;
        }
        
        // Set MSE
        const mseElement = document.getElementById('pid-mse');
        if (mseElement) {
            mseElement.textContent = result.prediction_data.pid_mse.toFixed(6);
        }
        
        // Set PID parameters
        const pidParamsElement = document.getElementById('pid-params');
        if (pidParamsElement) {
            const params = result.pid_tuning.best_parameters;
            pidParamsElement.textContent = `Kp=${params.Kp.toFixed(2)}, Ki=${params.Ki.toFixed(2)}, Kd=${params.Kd.toFixed(2)}`;
        }
        
        // Set graph
        const graphElement = document.getElementById('pid-graph') as HTMLImageElement;
        if (graphElement && result.graphs.length > 0) {
            graphElement.src = `data:image/png;base64,${result.graphs[0].data}`;
        }
        
        // Set explanation
        const explanationElement = document.getElementById('prediction-explanation');
        if (explanationElement) {
            explanationElement.textContent = result.summary;
        }
        
        logFrontend('displayStockPredictionResults finished');
    }
    
    private showPredictionError(errorMessage: string): void {
        logFrontend('showPredictionError started with message:', errorMessage);
        
        const predictionResultsContainer = document.getElementById('prediction-results-container');
        if (predictionResultsContainer) {
            predictionResultsContainer.innerHTML = `
                <div class="error-message">
                    <h3>Error Running Stock Prediction</h3>
                    <p>${errorMessage}</p>
                </div>
            `;
            predictionResultsContainer.style.display = 'block';
        }
        
        logFrontend('showPredictionError finished');
    }
}

logFrontend('Script loaded. Adding DOMContentLoaded listener for AppController.');
document.addEventListener('DOMContentLoaded', () => {
    logFrontend('DOMContentLoaded event fired. Initializing AppController.');
    new BacktestingController();
    logFrontend('AppController initialized.');
}); 