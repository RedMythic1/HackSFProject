def handler(request, response):
    response.status_code = 405
    return {"error": "Use POST /api/backtest via JS endpoint."} 