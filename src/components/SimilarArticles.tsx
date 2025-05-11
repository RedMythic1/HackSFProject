import React, { useEffect, useState } from 'react';
import { Article, findSimilarArticles, SimilarityWeights } from '../utils/articleSimilarity';

interface SimilarArticlesProps {
  currentArticle: Article;
  articleCollection: Article[];
  similarityThreshold?: number;
  maxArticles?: number;
  weights?: Partial<SimilarityWeights>;
  showSimilarityDetails?: boolean;
}

/**
 * Component to display articles similar to the current article
 * Uses hybrid similarity combining embeddings and term matching
 */
const SimilarArticles: React.FC<SimilarArticlesProps> = ({
  currentArticle,
  articleCollection,
  similarityThreshold = 0.5,
  maxArticles = 3,
  weights = {}, // Default weights will be applied in the findSimilarArticles function
  showSimilarityDetails = false
}) => {
  const [similarArticles, setSimilarArticles] = useState<Array<Article & { similarity: number }>>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    // Find similar articles when current article or collection changes
    if (currentArticle && articleCollection?.length) {
      setIsLoading(true);
      
      try {
        const similar = findSimilarArticles(
          currentArticle,
          articleCollection,
          similarityThreshold,
          maxArticles,
          weights
        );
        
        setSimilarArticles(similar);
      } catch (error) {
        console.error('Error finding similar articles:', error);
        setSimilarArticles([]);
      } finally {
        setIsLoading(false);
      }
    } else {
      setSimilarArticles([]);
      setIsLoading(false);
    }
  }, [currentArticle, articleCollection, similarityThreshold, maxArticles, weights]);

  if (isLoading) {
    return <div className="similar-articles-loading">Loading similar articles...</div>;
  }

  if (similarArticles.length === 0) {
    return null; // Don't show anything if no similar articles
  }

  return (
    <div className="similar-articles-container">
      <h3>Similar Articles</h3>
      <div className="similar-articles-list">
        {similarArticles.map((article) => (
          <div key={article.id || article.link} className="similar-article-item">
            <a 
              href={article.link} 
              target="_blank" 
              rel="noopener noreferrer"
              className="similar-article-link"
            >
              <h4>{article.title}</h4>
              <div className="similar-article-meta">
                {article.subject && (
                  <span className="similar-article-subject">{article.subject}</span>
                )}
                <span className="similar-article-similarity">
                  {(article.similarity * 100).toFixed(0)}% match
                </span>
              </div>
              {article.summary && (
                <p className="similar-article-summary">
                  {article.summary.length > 120 
                    ? `${article.summary.substring(0, 120)}...` 
                    : article.summary}
                </p>
              )}
              {showSimilarityDetails && (
                <div className="similarity-details">
                  <span className="similarity-score">
                    Match score: {(article.similarity * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </a>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SimilarArticles; 