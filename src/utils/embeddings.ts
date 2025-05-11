/**
 * Utility functions for working with embeddings
 */

/**
 * Calculate cosine similarity between two embedding vectors
 * Works with the 384-dimensional vectors from all-MiniLM-L6-v2 model
 */
export function cosineSimilarity(vec1: number[], vec2: number[]): number {
  // Validate vector dimensions
  if (vec1.length !== vec2.length) {
    throw new Error(`Vector dimensions don't match: ${vec1.length} vs ${vec2.length}`);
  }

  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (let i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }

  // Handle zero magnitude vectors
  if (norm1 === 0 || norm2 === 0) {
    return 0;
  }

  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

/**
 * Normalize a vector to unit length
 */
export function normalizeVector(vector: number[]): number[] {
  const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
  
  // Handle zero magnitude
  if (magnitude === 0) {
    return [...vector];
  }
  
  return vector.map(val => val / magnitude);
}

/**
 * Find the most similar vector to a query vector from a collection
 * @param queryVector The vector to compare against
 * @param vectorsCollection Array of vectors to search through
 * @param normalize Whether to normalize the vectors before comparison
 * @returns Index of the most similar vector and its similarity score
 */
export function findMostSimilarVector(
  queryVector: number[], 
  vectorsCollection: number[][],
  normalize: boolean = true
): { index: number, similarity: number } {
  let highestSimilarity = -1;
  let mostSimilarIndex = -1;
  
  // Normalize query vector if requested
  const normalizedQuery = normalize ? normalizeVector(queryVector) : queryVector;
  
  vectorsCollection.forEach((vector, index) => {
    // Normalize comparison vector if requested
    const normalizedVector = normalize ? normalizeVector(vector) : vector;
    
    const similarity = cosineSimilarity(normalizedQuery, normalizedVector);
    
    if (similarity > highestSimilarity) {
      highestSimilarity = similarity;
      mostSimilarIndex = index;
    }
  });
  
  return {
    index: mostSimilarIndex,
    similarity: highestSimilarity
  };
}

/**
 * Test function to demonstrate usage of vector similarity
 */
export function testVectorSimilarity(): void {
  // Create two random 384-dimensional vectors
  const vecA = Array(384).fill(0).map(() => Math.random());
  const vecB = Array(384).fill(0).map(() => Math.random());
  
  // Calculate similarity without normalization
  const rawSimilarity = cosineSimilarity(vecA, vecB);
  console.log("Raw Cosine Similarity:", rawSimilarity);
  
  // Calculate similarity with normalization
  const normalizedVecA = normalizeVector(vecA);
  const normalizedVecB = normalizeVector(vecB);
  const normalizedSimilarity = cosineSimilarity(normalizedVecA, normalizedVecB);
  console.log("Normalized Cosine Similarity:", normalizedSimilarity);
  
  // Verify the results are the same (they should be for cosine similarity)
  console.log("Difference:", Math.abs(rawSimilarity - normalizedSimilarity));
} 