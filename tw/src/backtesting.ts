import './style.css';
import './backtesting.css';

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
}

class BacktestingController {
    private strategyInput: HTMLTextAreaElement | null;
    private runBacktestBtn: HTMLElement | null;
    private loadingIndicator: HTMLElement | null;
    private resultsContainer: HTMLElement | null;
    private profitLossValue: HTMLElement | null;
    private tradesCount: HTMLElement | null;
    private successRate: HTMLElement | null;
    private chartImage: HTMLImageElement | null;
    private generatedCode: HTMLElement | null;
    
    constructor() {
        this.initElements();
        this.initEventListeners();
    }
    
    private initElements(): void {
        this.strategyInput = document.getElementById('strategy-input') as HTMLTextAreaElement;
        this.runBacktestBtn = document.getElementById('run-backtest-btn');
        this.loadingIndicator = document.getElementById('loading-indicator');
        this.resultsContainer = document.getElementById('results-container');
        this.profitLossValue = document.getElementById('profit-loss-value');
        this.tradesCount = document.getElementById('trades-count');
        this.successRate = document.getElementById('success-rate');
        this.chartImage = document.getElementById('chart-image') as HTMLImageElement;
        this.generatedCode = document.getElementById('generated-code');
    }
    
    private initEventListeners(): void {
        if (this.runBacktestBtn) {
            this.runBacktestBtn.addEventListener('click', () => this.runBacktest());
        }
    }
    
    private async runBacktest(): Promise<void> {
        if (!this.strategyInput || !this.strategyInput.value.trim()) {
            alert('Please enter a trading strategy before running a backtest.');
            return;
        }
        
        const strategy = this.strategyInput.value.trim();
        
        // Show loading indicator
        this.showLoading(true);
        
        try {
            const result = await this.callBacktestAPI(strategy);
            this.displayResults(result);
        } catch (error) {
            console.error('Error running backtest:', error);
            alert('An error occurred while running the backtest. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }
    
    private async callBacktestAPI(strategy: string): Promise<BacktestResult> {
        try {
            const response = await fetch('/api/backtest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ strategy })
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API call error:', error);
            throw error;
        }
    }
    
    private displayResults(result: BacktestResult): void {
        if (!this.resultsContainer) return;
        
        // Show results container
        this.resultsContainer.style.display = 'block';
        
        // Update profit/loss
        if (this.profitLossValue) {
            // Handle different response formats (serverless vs node)
            const profitValue = result.profit_loss ?? result.profit ?? 0;
            
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
            let buyPoints = result.buy_points;
            let sellPoints = result.sell_points;
            
            // Handle different response formats
            if (result.trades) {
                buyPoints = result.trades.buys;
                sellPoints = result.trades.sells; 
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
        
        // Update chart
        if (this.chartImage) {
            // Handle different chart URL field names
            const chartUrl = result.chart_url || result.image;
            
            if (chartUrl) {
                this.chartImage.src = chartUrl;
                this.chartImage.style.display = 'block';
            } else {
                this.chartImage.style.display = 'none';
            }
        }
        
        // Update generated code
        if (this.generatedCode) {
            // Handle different code field names
            const codeContent = result.generated_code || result.code || '';
            
            if (codeContent) {
                this.generatedCode.textContent = codeContent;
            }
        }
    }
    
    private showLoading(show: boolean): void {
        if (this.loadingIndicator) {
            this.loadingIndicator.style.display = show ? 'block' : 'none';
        }
        
        if (this.resultsContainer) {
            this.resultsContainer.style.display = show ? 'none' : 'block';
        }
        
        if (this.runBacktestBtn) {
            this.runBacktestBtn.setAttribute('disabled', show.toString());
            if (show) {
                this.runBacktestBtn.textContent = 'Running...';
                this.runBacktestBtn.classList.add('disabled');
            } else {
                this.runBacktestBtn.textContent = 'Run Backtest';
                this.runBacktestBtn.classList.remove('disabled');
            }
        }
    }
}

// Initialize the controller when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new BacktestingController();
}); 