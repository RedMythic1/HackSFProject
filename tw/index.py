import json
import os
import sys

# Import server.py functionality if available
try:
    from server import process_articles_endpoint, get_article_endpoint, analyze_interests_endpoint
except ImportError:
    def process_articles_endpoint(query=None):
        return {"error": "Server module could not be imported"}
    def get_article_endpoint(article_id=None):
        return {"error": "Server module could not be imported"}
    def analyze_interests_endpoint(interests=None):
        return {"error": "Server module could not be imported"}

def handler(request):
    """Handle all incoming requests to the Python API"""
    # Get the path and method from the request
    path = request.get("path", "")
    method = request.get("method", "").upper()
    
    # Extract query parameters
    query = request.get("query", {})
    
    # Initialize response
    response = {"status": 404, "body": {"error": "Not found"}}
    
    # Handle different endpoints
    if path.startswith("/api/articles"):
        result = process_articles_endpoint(query)
        response = {"status": 200, "body": result}
    
    elif path.startswith("/api/article/"):
        article_id = path.split("/")[-1]
        result = get_article_endpoint(article_id)
        response = {"status": 200, "body": result}
    
    elif path == "/api/analyze-interests":
        # Parse body for POST requests
        if method == "POST":
            try:
                body = json.loads(request.get("body", "{}"))
                interests = body.get("interests", "")
                result = analyze_interests_endpoint(interests)
                response = {"status": 200, "body": result}
            except Exception as e:
                response = {"status": 400, "body": {"error": str(e)}}
        else:
            response = {"status": 405, "body": {"error": "Method not allowed"}}
    
    # Return response formatted for Vercel
    return {
        "statusCode": response["status"],
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(response["body"])
    } 