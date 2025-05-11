import { cosineSimilarity, normalizeVector } from './embeddings';
import { extractTerms, jaccardSimilarity, termFrequencySimilarity, overlapSimilarity } from './textMatching';

export interface Article {
  id?: string;
  title: string;
  link: string;
  summary?: string;
  embedding?: number[];
  score?: number;
  subject?: string;
}

export interface SimilarityWeights {
  embedding: number;
  jaccard: number;
  termFrequency: number;
  overlap: number;
}

// Default weights for different similarity methods
const DEFAULT_WEIGHTS: SimilarityWeights = {
  embedding: 0.6,     // Primary weight for semantic similarity
  jaccard: 0.2,       // Weight for term set similarity
  termFrequency: 0.1, // Weight for term frequency similarity
  overlap: 0.1        // Weight for term overlap similarity
};

/**
 * Calculate hybrid similarity score between two articles 
 * combining embedding similarity and term matching
 */
export function calculateHybridSimilarity(
  article1: Article,
  article2: Article,
  weights: Partial<SimilarityWeights> = {}
): number {
  // Merge provided weights with defaults
  const finalWeights: SimilarityWeights = {
    ...DEFAULT_WEIGHTS,
    ...weights
  };
  
  // Normalize weights to ensure they sum to 1
  const totalWeight = Object.values(finalWeights).reduce((sum, w) => sum + w, 0);
  const normalizedWeights: SimilarityWeights = {
    embedding: finalWeights.embedding / totalWeight,
    jaccard: finalWeights.jaccard / totalWeight,
    termFrequency: finalWeights.termFrequency / totalWeight,
    overlap: finalWeights.overlap / totalWeight
  };
  
  // Calculate embedding similarity if available
  let embeddingSimilarity = 0;
  if (article1.embedding && article2.embedding && 
      article1.embedding.length > 0 && article2.embedding.length > 0) {
    const normalizedEmbedding1 = normalizeVector(article1.embedding);
    const normalizedEmbedding2 = normalizeVector(article2.embedding);
    embeddingSimilarity = cosineSimilarity(normalizedEmbedding1, normalizedEmbedding2);
  }
  
  // Extract terms from title and summary
  const text1 = `${article1.title} ${article1.subject || ''} ${article1.summary || ''}`;
  const text2 = `${article2.title} ${article2.subject || ''} ${article2.summary || ''}`;
  
  const terms1 = extractTerms(text1);
  const terms2 = extractTerms(text2);
  
  // Calculate term-based similarities
  const jaccardScore = jaccardSimilarity(terms1, terms2);
  const termFreqScore = termFrequencySimilarity(terms1, terms2);
  const overlapScore = overlapSimilarity(terms1, terms2);
  
  // Combine scores using weighted average
  const combinedScore = 
    (embeddingSimilarity * normalizedWeights.embedding) +
    (jaccardScore * normalizedWeights.jaccard) + 
    (termFreqScore * normalizedWeights.termFrequency) +
    (overlapScore * normalizedWeights.overlap);
    
  return combinedScore;
}

/**
 * Find similar articles based on hybrid similarity (embedding + term matching)
 * @param targetArticle The article to find similar articles for
 * @param articleCollection Collection of articles to search through
 * @param threshold Minimum similarity threshold (0 to 1)
 * @param limit Maximum number of similar articles to return
 * @param weights Optional weights for different similarity methods
 * @returns Array of similar articles with similarity scores
 */
export function findSimilarArticles(
  targetArticle: Article,
  articleCollection: Article[],
  threshold: number = 0.5,
  limit: number = 5,
  weights: Partial<SimilarityWeights> = {}
): Array<Article & { similarity: number }> {
  // Filter out the target article itself
  const candidateArticles = articleCollection.filter(article => 
    article.id !== targetArticle.id
  );
  
  // Calculate hybrid similarity for each article
  const articlesWithSimilarity = candidateArticles.map(article => {
    const similarity = calculateHybridSimilarity(targetArticle, article, weights);
    
    return {
      ...article,
      similarity
    };
  });
  
  // Filter by threshold, sort by similarity, and limit results
  return articlesWithSimilarity
    .filter(article => article.similarity >= threshold)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit);
}

/**
 * Generate recommendations based on a set of articles the user has engaged with
 * @param viewedArticles Articles the user has viewed/engaged with
 * @param articleCollection Full collection of articles to recommend from
 * @param limit Maximum number of recommendations to return
 * @param weights Optional weights for different similarity methods
 * @returns Array of recommended articles with similarity scores
 */
export function generateRecommendations(
  viewedArticles: Article[],
  articleCollection: Article[],
  limit: number = 10,
  weights: Partial<SimilarityWeights> = {}
): Array<Article & { relevanceScore: number }> {
  if (viewedArticles.length === 0) {
    console.error('No viewed articles provided');
    return [];
  }
  
  // Create a map to track cumulative scores and viewed article IDs
  const cumulativeScores: Map<string, number> = new Map();
  const viewedIds = new Set(viewedArticles.map(a => a.id).filter(Boolean));
  
  // For each article the user has viewed
  viewedArticles.forEach(viewedArticle => {
    // Find similar articles
    const similarArticles = findSimilarArticles(
      viewedArticle,
      articleCollection,
      0.3,  // Lower threshold for recommendations
      20,   // Higher limit to get more candidates
      weights
    );
    
    // Add similarity scores to cumulative scores
    similarArticles.forEach(similar => {
      if (similar.id && !viewedIds.has(similar.id)) {
        const currentScore = cumulativeScores.get(similar.id) || 0;
        cumulativeScores.set(similar.id, currentScore + similar.similarity);
      }
    });
  });
  
  // Convert to array and sort by cumulative score
  const recommendations = Array.from(cumulativeScores.entries())
    .map(([id, relevanceScore]) => {
      const article = articleCollection.find(a => a.id === id);
      if (!article) {
        console.error(`Article with id ${id} not found in collection`);
        return null;
      }
      return {
        ...article,
        relevanceScore
      };
    })
    .filter(Boolean) as Array<Article & { relevanceScore: number }>;
  
  // Sort by relevance score and limit results
  return recommendations
    .sort((a, b) => b.relevanceScore - a.relevanceScore)
    .slice(0, limit);
} 