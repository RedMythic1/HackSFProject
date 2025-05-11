/**
 * Utility functions for text-based matching between articles
 */

/**
 * Extract terms from text by tokenizing and filtering
 * @param text Text to extract terms from
 * @returns Array of normalized terms
 */
export function extractTerms(text: string): string[] {
  if (!text) return [];
  
  // Convert to lowercase
  const lowercaseText = text.toLowerCase();
  
  // Remove punctuation and split into words
  const words = lowercaseText
    .replace(/[^\w\s]/g, ' ')  // Replace punctuation with spaces
    .replace(/\s+/g, ' ')      // Replace multiple spaces with single space
    .trim()
    .split(' ');
  
  // Filter out common stop words and very short words
  const stopWords = new Set([
    'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
    'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
    'has', 'had', 'do', 'does', 'did', 'but', 'or', 'if', 'then', 'else', 
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 
    'more', 'most', 'some', 'such', 'that', 'than', 'these', 'this', 'those'
  ]);
  
  return words
    .filter(word => word.length > 2 && !stopWords.has(word))
    .slice(0, 100); // Limit to top 100 terms for efficiency
}

/**
 * Calculate Jaccard similarity between two sets of terms
 * @param terms1 First set of terms
 * @param terms2 Second set of terms
 * @returns Similarity score from 0 to 1
 */
export function jaccardSimilarity(terms1: string[], terms2: string[]): number {
  if (terms1.length === 0 && terms2.length === 0) return 1.0;
  if (terms1.length === 0 || terms2.length === 0) return 0.0;
  
  const set1 = new Set(terms1);
  const set2 = new Set(terms2);
  
  // Calculate intersection size
  const intersection = new Set([...set1].filter(term => set2.has(term)));
  
  // Calculate union size
  const union = new Set([...set1, ...set2]);
  
  // Jaccard similarity is the ratio of intersection size to union size
  return intersection.size / union.size;
}

/**
 * Calculate weighted term frequency similarity
 * @param terms1 First set of terms
 * @param terms2 Second set of terms
 * @returns Similarity score from 0 to 1
 */
export function termFrequencySimilarity(terms1: string[], terms2: string[]): number {
  if (terms1.length === 0 && terms2.length === 0) return 1.0;
  if (terms1.length === 0 || terms2.length === 0) return 0.0;
  
  // Count term frequencies
  const freqMap1 = countTerms(terms1);
  const freqMap2 = countTerms(terms2);
  
  // Find all unique terms across both documents
  const allTerms = new Set([...Object.keys(freqMap1), ...Object.keys(freqMap2)]);
  
  let dotProduct = 0;
  let magnitude1 = 0;
  let magnitude2 = 0;
  
  // Calculate dot product and magnitudes
  allTerms.forEach(term => {
    const freq1 = freqMap1[term] || 0;
    const freq2 = freqMap2[term] || 0;
    
    dotProduct += freq1 * freq2;
    magnitude1 += freq1 * freq1;
    magnitude2 += freq2 * freq2;
  });
  
  // Calculate cosine similarity
  if (magnitude1 === 0 || magnitude2 === 0) return 0;
  return dotProduct / (Math.sqrt(magnitude1) * Math.sqrt(magnitude2));
}

/**
 * Helper function to count term frequencies
 */
function countTerms(terms: string[]): Record<string, number> {
  const freqMap: Record<string, number> = {};
  
  terms.forEach(term => {
    freqMap[term] = (freqMap[term] || 0) + 1;
  });
  
  return freqMap;
}

/**
 * Calculate overlap coefficient between two sets of terms
 * Useful when sets differ significantly in size
 * @param terms1 First set of terms
 * @param terms2 Second set of terms
 * @returns Similarity score from 0 to 1
 */
export function overlapSimilarity(terms1: string[], terms2: string[]): number {
  if (terms1.length === 0 && terms2.length === 0) return 1.0;
  if (terms1.length === 0 || terms2.length === 0) return 0.0;
  
  const set1 = new Set(terms1);
  const set2 = new Set(terms2);
  
  // Calculate intersection size
  const intersection = new Set([...set1].filter(term => set2.has(term)));
  
  // Overlap coefficient is the ratio of intersection size to the size of the smaller set
  return intersection.size / Math.min(set1.size, set2.size);
} 