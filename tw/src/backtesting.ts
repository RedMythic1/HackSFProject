import './style.css';
import './backtesting.css';

interface BacktestResult {
    profit_loss: number;
    buy_points: Array<[number, number]>;
    sell_points: Array<[number, number]>;
    balance_over_time: number[];
    generated_code: string;
    chart_url: string;
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
            const formattedProfitLoss = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(result.profit_loss);
            
            this.profitLossValue.textContent = formattedProfitLoss;
            this.profitLossValue.className = 'result-value ' + 
                (result.profit_loss >= 0 ? 'positive' : 'negative');
        }
        
        // Update trades count
        if (this.tradesCount) {
            const totalTrades = result.buy_points.length;
            this.tradesCount.textContent = totalTrades.toString();
        }
        
        // Update success rate
        if (this.successRate) {
            let successfulTrades = 0;
            
            // Count trades where sell price > buy price
            for (let i = 0; i < Math.min(result.buy_points.length, result.sell_points.length); i++) {
                if (result.sell_points[i][1] > result.buy_points[i][1]) {
                    successfulTrades++;
                }
            }
            
            const rate = result.buy_points.length > 0 
                ? Math.round((successfulTrades / result.buy_points.length) * 100) 
                : 0;
                
            this.successRate.textContent = `${rate}%`;
        }
        
        // Update chart
        if (this.chartImage && result.chart_url) {
            this.chartImage.src = result.chart_url;
            this.chartImage.style.display = 'block';
        }
        
        // Update generated code
        if (this.generatedCode && result.generated_code) {
            this.generatedCode.textContent = result.generated_code;
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