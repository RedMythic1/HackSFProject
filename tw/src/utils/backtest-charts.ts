import { Chart, registerables, ChartTypeRegistry } from 'chart.js';

// Register all Chart.js components we'll need
Chart.register(...registerables);

// Function to log messages from the frontend
function logFrontend(message: string, data?: any) {
  console.log(`[CHARTS] ${message}`, data !== undefined ? data : '');
}

// Interface for backtest result data
interface BacktestResult {
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
  all_iterations?: {
    iteration: number;
    dataset: string;
    profit: number;
    percent_above_buyhold: number;
    trades_count: number;
  }[];
}

interface PointData {
  x: number;
  y: number;
}

/**
 * Converts points that may be in tuple form [x,y] or object form {x,y} to a consistent {x,y} format
 * @param points Points to convert
 * @returns Standardized array of {x,y} points
 */
function normalizePoints(points: Array<[number, number]> | Array<{x: number, y: number}>): PointData[] {
  if (!points || points.length === 0) return [];
  
  // Already in the right format
  if (typeof points[0] === 'object' && 'x' in points[0] && 'y' in points[0]) {
    return points as PointData[];
  }
  
  // Convert from tuple format
  return (points as Array<[number, number]>).map(point => ({
    x: point[0],
    y: point[1]
  }));
}

/**
 * Creates a price chart with buy/sell points
 * @param container The container element or ID where the chart should be rendered
 * @param backtestData The backtest result data from the API
 * @returns The created Chart instance
 */
export function createPriceChart(container: string | HTMLCanvasElement, backtestData: BacktestResult): Chart {
  // Get the canvas element
  const canvas = typeof container === 'string' 
    ? document.getElementById(container) as HTMLCanvasElement
    : container;
  
  if (!canvas) {
    throw new Error(`Canvas element ${container} not found`);
  }

  // Process data for chart
  const labels = backtestData.dates;
  const prices = backtestData.close;
  
  // Process buy and sell points - handle both array of tuples and array of objects
  const buyPoints = normalizePoints(backtestData.buy_points);
  const sellPoints = normalizePoints(backtestData.sell_points);
  
  console.log('[CHARTS] Processing buy/sell points for chart:', {
    buyPointsLength: buyPoints.length,
    sellPointsLength: sellPoints.length,
    sampleBuyPoints: buyPoints.slice(0, 3),
    sampleSellPoints: sellPoints.slice(0, 3)
  });

  // Prepare scaled balance for overlay
  let scaledBalance: number[] = [];
  if (backtestData.balance_over_time && backtestData.balance_over_time.length > 1) {
    const bal = backtestData.balance_over_time;
    const close0 = prices[0] || 1;
    const bal0 = bal[0] || 1;
    scaledBalance = bal.map(v => (v / bal0) * close0);
  }

  // Create the chart
  const priceChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Close Price',
          data: prices,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 1,
          fill: false
        },
        // Overlay scaled balance
        scaledBalance.length > 0 ? {
          label: 'Portfolio Value (scaled)',
          data: scaledBalance,
          borderColor: 'orange',
          backgroundColor: 'rgba(255, 165, 0, 0.1)',
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 1.5,
          fill: false,
          yAxisID: 'y',
        } : undefined,
        // Buy points as scatter plot
        {
          label: 'Buy Points',
          type: 'scatter' as any,
          data: buyPoints,
          backgroundColor: 'rgb(46, 204, 113)',  // Green
          borderColor: 'white',
          borderWidth: 1,
          pointRadius: 6,
          pointStyle: 'triangle',
          yAxisID: 'y',
          pointHoverRadius: 8,
          order: 0  // Make sure buy points appear on top
        },
        // Sell points as scatter plot
        {
          label: 'Sell Points',
          type: 'scatter' as any,
          data: sellPoints,
          backgroundColor: 'rgb(231, 76, 60)',  // Red
          borderColor: 'white',
          borderWidth: 1,
          pointRadius: 6,
          pointStyle: 'triangle',
          rotation: 180,  // Flip the triangle down
          yAxisID: 'y',
          pointHoverRadius: 8,
          order: 0  // Make sure sell points appear on top
        }
      ].filter(Boolean)
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Date'
          },
          ticks: {
            maxTicksLimit: 10
          }
        },
        y: {
          title: {
            display: true,
            text: 'Price'
          }
        }
      },
      plugins: {
        tooltip: {
          callbacks: {
            title: function(tooltipItems) {
              const item = tooltipItems[0];
              const index = item.dataIndex;
              const datasetIndex = item.datasetIndex;
              if (datasetIndex === 1 && scaledBalance.length > 0) return `Portfolio Value (scaled) at ${labels[index]}`;
              if (datasetIndex === 2) return `Buy Signal at ${labels[index]}`;
              if (datasetIndex === 3) return `Sell Signal at ${labels[index]}`;
              return labels[index];
            },
            label: function(context) {
              const label = context.dataset.label || '';
              const value = context.raw as number | { x: number, y: number };
              if (typeof value === 'object') {
                return `${label}: $${value.y.toFixed(2)}`;
              }
              return `${label}: $${value.toFixed(2)}`;
            }
          }
        },
        title: {
          display: true,
          text: `Price Chart - ${backtestData.dataset}`
        }
      }
    }
  });

  return priceChart;
}

/**
 * Creates a balance history chart
 * @param container The container element or ID where the chart should be rendered
 * @param backtestData The backtest result data from the API
 * @returns The created Chart instance
 */
export function createBalanceChart(container: string | HTMLCanvasElement, backtestData: BacktestResult): Chart {
  // Get the canvas element
  const canvas = typeof container === 'string' 
    ? document.getElementById(container) as HTMLCanvasElement
    : container;
  
  if (!canvas) {
    throw new Error(`Canvas element ${container} not found`);
  }

  // Prepare data
  const balanceData = backtestData.balance_over_time;
  const labels = Array.from({ length: balanceData.length }, (_, i) => i.toString());

  // Create the chart
  const balanceChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Portfolio Value',
          data: balanceData,
          borderColor: 'rgb(45, 152, 218)',
          backgroundColor: 'rgba(45, 152, 218, 0.1)',
          tension: 0.1,
          fill: true
        },
        {
          label: 'Initial Balance',
          data: Array(balanceData.length).fill(balanceData[0]),
          borderColor: 'rgba(149, 165, 166, 0.8)',
          backgroundColor: 'rgba(0, 0, 0, 0)',
          borderDash: [5, 5],
          pointRadius: 0,
          borderWidth: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Trading Days'
          },
          // Display fewer labels for better readability
          ticks: {
            maxTicksLimit: 10
          }
        },
        y: {
          title: {
            display: true,
            text: 'Portfolio Value ($)'
          },
          ticks: {
            callback: function(value) {
              return '$' + value.toLocaleString();
            }
          }
        }
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: function(context) {
              const label = context.dataset.label || '';
              const value = context.raw as number;
              return `${label}: $${value.toLocaleString()}`;
            }
          }
        },
        title: {
          display: true,
          text: 'Portfolio Value Over Time'
        }
      }
    }
  });

  return balanceChart;
}

/**
 * Renders all backtest charts in the specified containers
 * @param backtestData The backtest result data from the API
 * @param priceChartContainer The container for the price chart
 * @param balanceChartContainer The container for the balance chart
 * @returns An object containing all chart instances
 */
export function renderBacktestCharts(
  backtestData: BacktestResult,
  priceChartContainer: string | HTMLCanvasElement,
  balanceChartContainer: string | HTMLCanvasElement
) {
  // Create the charts
  const priceChart = createPriceChart(priceChartContainer, backtestData);
  const balanceChart = createBalanceChart(balanceChartContainer, backtestData);
  
  return {
    priceChart,
    balanceChart
  };
}

/**
 * Runs a backtest and renders the charts
 * @param strategy The trading strategy to test
 * @param priceChartContainer The container for the price chart
 * @param balanceChartContainer The container for the balance chart
 * @param statusCallback Optional callback to handle status updates
 * @returns A promise resolving to the backtest result and chart instances
 */
export async function runBacktestAndChart(
  strategy: string,
  priceChartContainer: string | HTMLCanvasElement,
  balanceChartContainer: string | HTMLCanvasElement,
  statusCallback?: (status: string) => void
) {
  if (statusCallback) statusCallback('Running backtest...');
  
  try {
    // Call the backtest API
    const response = await fetch('/api/backtest', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ strategy })
    });
    
    const data = await response.json();
    
    if (data.status !== 'success') {
      if (statusCallback) statusCallback(`Error: ${data.error || 'Unknown error'}`);
      throw new Error(data.error || 'Unknown error');
    }
    
    const backtestData = data as BacktestResult;
    
    if (statusCallback) statusCallback('Rendering charts...');
    
    // Render the charts
    const charts = renderBacktestCharts(
      backtestData,
      priceChartContainer,
      balanceChartContainer
    );
    
    if (statusCallback) statusCallback('Complete');
    
    return {
      backtestData,
      charts
    };
  } catch (error: any) {
    if (statusCallback) statusCallback(`Error: ${error.message || 'Unknown error'}`);
    throw error;
  }
}

function plotBacktestResults(close, buy_points, sell_points, balance_over_time, containerId = 'chart-container') {
  // Create container for the charts if it doesn't exist
  let container = document.getElementById(containerId);
  if (!container) {
    container = document.createElement('div');
    container.id = containerId;
    document.body.appendChild(container);
  }
  
  // Create canvases for the two charts
  const priceChartCanvas = document.createElement('canvas');
  priceChartCanvas.id = 'price-chart';
  const balanceChartCanvas = document.createElement('canvas');
  balanceChartCanvas.id = 'balance-chart';
  
  container.innerHTML = ''; // Clear existing content
  container.appendChild(priceChartCanvas);
  container.appendChild(balanceChartCanvas);
  
  // Scale buy/sell points as in the Python code
  const scaledBuyPoints = buy_points.map(point => ({
    x: point[0] * 19 - 1,
    y: point[1]
  }));
  
  const scaledSellPoints = sell_points.map(point => ({
    x: point[0] * 19,
    y: point[1]
  }));
  
  // Create price chart with buy/sell points
  const priceChart = new Chart(priceChartCanvas, {
    type: 'line',
    data: {
      labels: Array.from({ length: close.length }, (_, i) => i.toString()),
      datasets: [
        {
          label: 'Close Price',
          data: close,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 0.5,
          pointRadius: 0,
          fill: false
        },
        {
          label: 'Buy Points',
          data: scaledBuyPoints,
          backgroundColor: 'rgb(46, 204, 113)',
          borderColor: 'rgba(0, 0, 0, 0)',
          pointRadius: 3.5,
          pointStyle: 'circle',
          showLine: false
        },
        {
          label: 'Sell Points',
          data: scaledSellPoints,
          backgroundColor: 'rgb(231, 76, 60)',
          borderColor: 'rgba(0, 0, 0, 0)',
          pointRadius: 3.5,
          pointStyle: 'circle',
          showLine: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Time'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Price'
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Stock Close Price with Buy/Sell Points',
          font: {
            size: 16
          }
        },
        legend: {
          display: true
        }
      },
      layout: {
        padding: 20
      }
    }
  });
  
  // Create balance chart
  const balanceChart = new Chart(balanceChartCanvas, {
    type: 'line',
    data: {
      labels: Array.from({ length: balance_over_time.length }, (_, i) => i.toString()),
      datasets: [
        {
          label: 'Balance Over Time',
          data: balance_over_time,
          borderColor: 'rgb(45, 152, 218)',
          backgroundColor: 'rgba(45, 152, 218, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Time'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Balance'
          },
          ticks: {
            callback: function(value) {
              return '$' + value.toLocaleString();
            }
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Balance Over Time',
          font: {
            size: 16
          }
        },
        legend: {
          display: true
        }
      },
      layout: {
        padding: 20
      }
    }
  });
  
  return {
    priceChart,
    balanceChart
  };
}

// Example usage:
// const results = calculate_profit(close, upper, lower, fall_threshold, rise_threshold);
// plotBacktestResults(close, results.buy_points, results.sell_points, results.balance_over_time); 