from http.server import BaseHTTPRequestHandler
import json
import os
import sys

# Add parent directory to path so we can import server.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from server.py that we need
try:
    from server import process_articles_endpoint, get_article_endpoint, analyze_interests_endpoint
except ImportError:
    def fallback_response():
        return {"error": "Server module could not be imported"}

def handler(request, response):
    # Get the path and method from the request
    path = request.get("path", "")
    method = request.get("method", "").upper()
    
    # Extract query parameters
    query = request.get("query", {})
    
    # Handle different endpoints
    if path.startswith("/api/articles"):
        result = process_articles_endpoint(query)
        response.status_code = 200
        response.body = json.dumps(result)
    
    elif path.startswith("/api/article/"):
        article_id = path.split("/")[-1]
        result = get_article_endpoint(article_id)
        response.status_code = 200
        response.body = json.dumps(result)
    
    elif path == "/api/analyze-interests":
        # Parse body for POST requests
        if method == "POST":
            try:
                body = json.loads(request.get("body", "{}"))
                interests = body.get("interests", "")
                result = analyze_interests_endpoint(interests)
                response.status_code = 200
                response.body = json.dumps(result)
            except Exception as e:
                response.status_code = 400
                response.body = json.dumps({"error": str(e)})
        else:
            response.status_code = 405
            response.body = json.dumps({"error": "Method not allowed"})
    
    else:
        # Default fallback response
        response.status_code = 404
        response.body = json.dumps({"error": "Not found"})
    
    return response 