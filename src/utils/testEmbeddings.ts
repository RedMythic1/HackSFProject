import { cosineSimilarity, normalizeVector, findMostSimilarVector } from './embeddings';
import { 
  Article, 
  findSimilarArticles, 
  generateRecommendations, 
  calculateHybridSimilarity,
  SimilarityWeights 
} from './articleSimilarity';
import { extractTerms, jaccardSimilarity, termFrequencySimilarity } from './textMatching';

/**
 * Simple function to test the embedding similarity with random vectors
 */
function testBasicSimilarity() {
  console.log("=== Testing Basic Vector Similarity ===");
  
  // Generate random vectors of dimension 384 (the dimension used by all-MiniLM-L6-v2)
  const vecA = Array(384).fill(0).map(() => Math.random() * 2 - 1); // Random values between -1 and 1
  const vecB = Array(384).fill(0).map(() => Math.random() * 2 - 1);
  
  // Calculate similarity with and without normalization
  console.log("Raw similarity:", cosineSimilarity(vecA, vecB));
  
  const normA = normalizeVector(vecA);
  const normB = normalizeVector(vecB);
  console.log("Normalized similarity:", cosineSimilarity(normA, normB));
  
  // Verify vectors are normalized (length should be 1.0)
  const lengthA = Math.sqrt(normA.reduce((sum, val) => sum + val * val, 0));
  const lengthB = Math.sqrt(normB.reduce((sum, val) => sum + val * val, 0));
  
  console.log("Normalized vector A length:", lengthA);
  console.log("Normalized vector B length:", lengthB);
}

/**
 * Test term matching functions
 */
function testTermMatching() {
  console.log("\n=== Testing Term Matching ===");
  
  // Define some sample texts
  const text1 = "Machine Learning is transforming how we build artificial intelligence systems";
  const text2 = "AI and Machine Learning are revolutionizing technology systems";
  const text3 = "JavaScript frameworks like React and Vue make web development easier";
  
  // Extract terms
  const terms1 = extractTerms(text1);
  const terms2 = extractTerms(text2);
  const terms3 = extractTerms(text3);
  
  console.log("Text 1 terms:", terms1);
  console.log("Text 2 terms:", terms2);
  console.log("Text 3 terms:", terms3);
  
  // Calculate similarities
  console.log("Jaccard similarity (1-2):", jaccardSimilarity(terms1, terms2));
  console.log("Jaccard similarity (1-3):", jaccardSimilarity(terms1, terms3));
  console.log("Term frequency similarity (1-2):", termFrequencySimilarity(terms1, terms2));
  console.log("Term frequency similarity (1-3):", termFrequencySimilarity(terms1, terms3));
}

/**
 * Test finding similar articles using hybrid matching
 */
function testArticleSimilarity() {
  console.log("\n=== Testing Hybrid Article Similarity ===");
  
  // Create mock articles with both embeddings and text content
  const mockArticles: Article[] = [
    {
      id: "1",
      title: "Introduction to Machine Learning and AI Technologies",
      link: "https://example.com/ml-intro",
      subject: "Artificial Intelligence",
      summary: "This article covers the fundamentals of machine learning algorithms and their applications in modern AI systems. Learn about neural networks, deep learning, and how they're changing technology.",
      // Embedding biased toward AI topics
      embedding: Array(384).fill(0).map((_, i) => 
        i < 128 ? Math.random() * 0.8 + 0.2 : Math.random() * 0.3
      )
    },
    {
      id: "2",
      title: "Deep Learning Fundamentals and Neural Networks",
      link: "https://example.com/deep-learning",
      subject: "AI Research",
      summary: "An exploration of deep learning technologies including convolutional neural networks and transformer architectures. See how these systems are revolutionizing AI capabilities.",
      // Similar to article 1 (AI focused)
      embedding: Array(384).fill(0).map((_, i) => 
        i < 128 ? Math.random() * 0.7 + 0.3 : Math.random() * 0.3
      )
    },
    {
      id: "3",
      title: "JavaScript Frameworks Comparison: React vs Vue vs Angular",
      link: "https://example.com/js-frameworks",
      subject: "Web Development",
      summary: "Comparing the most popular JavaScript frameworks for frontend development. This guide helps developers choose the right tools for building modern web applications.",
      // Embedding biased toward web development topics
      embedding: Array(384).fill(0).map((_, i) => 
        i >= 128 && i < 256 ? Math.random() * 0.8 + 0.2 : Math.random() * 0.3
      )
    },
    {
      id: "4",
      title: "Frontend Development Best Practices and Performance Optimization",
      link: "https://example.com/frontend-best-practices",
      subject: "Web Development",
      summary: "Learn about the latest best practices in frontend development including code organization, performance optimization, and responsive design techniques for modern websites.",
      // Similar to article 3 (web development focused)
      embedding: Array(384).fill(0).map((_, i) => 
        i >= 128 && i < 256 ? Math.random() * 0.7 + 0.3 : Math.random() * 0.3
      )
    },
    {
      id: "5",
      title: "Quantum Computing Explained: Principles and Applications",
      link: "https://example.com/quantum-computing",
      subject: "Physics",
      summary: "An introduction to quantum computing concepts including qubits, superposition, and quantum algorithms. Discover how this technology will transform computing.",
      // Embedding biased toward quantum computing topics
      embedding: Array(384).fill(0).map((_, i) => 
        i >= 256 ? Math.random() * 0.8 + 0.2 : Math.random() * 0.3
      )
    },
    {
      id: "6",
      title: "Machine Learning Applications in Healthcare",
      link: "https://example.com/ml-healthcare",
      subject: "Healthcare Tech",
      summary: "How machine learning algorithms are being applied to healthcare problems like disease diagnosis, drug discovery, and personalized medicine. AI is revolutionizing patient care.",
      // Similar subject to article 1 but different focus
      embedding: Array(384).fill(0).map((_, i) => 
        i < 64 ? Math.random() * 0.8 + 0.2 : Math.random() * 0.3
      )
    }
  ];
  
  // Test different similarity weighting configurations
  const weights: Record<string, Partial<SimilarityWeights>> = {
    embeddingOnly: { embedding: 1.0, jaccard: 0, termFrequency: 0, overlap: 0 },
    termOnly: { embedding: 0, jaccard: 0.5, termFrequency: 0.5, overlap: 0 },
    balanced: { embedding: 0.5, jaccard: 0.25, termFrequency: 0.25, overlap: 0 },
    default: {} // Use default weights
  };
  
  // Find articles similar to the Machine Learning article
  const targetArticle = mockArticles[0];
  
  // Compare similarity methods
  Object.entries(weights).forEach(([name, weightConfig]) => {
    console.log(`\nTesting similarity with "${name}" configuration:`);
    
    const similarArticles = findSimilarArticles(
      targetArticle, 
      mockArticles, 
      0.3, 
      5, 
      weightConfig
    );
    
    console.log(`Articles similar to "${targetArticle.title}":`);
    similarArticles.forEach(article => {
      console.log(`- ${article.title} (Similarity: ${article.similarity.toFixed(4)})`);
    });
    
    // Show detailed similarity breakdown for the first result
    if (similarArticles.length > 0) {
      const firstSimilar = similarArticles[0];
      console.log(`\nDetailed similarity for "${firstSimilar.title}":`);
      
      // Calculate individual components
      const text1 = `${targetArticle.title} ${targetArticle.subject || ''} ${targetArticle.summary || ''}`;
      const text2 = `${firstSimilar.title} ${firstSimilar.subject || ''} ${firstSimilar.summary || ''}`;
      
      const terms1 = extractTerms(text1);
      const terms2 = extractTerms(text2);
      
      console.log(`- Term overlap: ${terms1.filter(t => terms2.includes(t)).length} terms`);
      console.log(`- Jaccard similarity: ${jaccardSimilarity(terms1, terms2).toFixed(4)}`);
      console.log(`- Term frequency similarity: ${termFrequencySimilarity(terms1, terms2).toFixed(4)}`);
      
      if (targetArticle.embedding && firstSimilar.embedding) {
        const embeddingSim = cosineSimilarity(
          normalizeVector(targetArticle.embedding),
          normalizeVector(firstSimilar.embedding)
        );
        console.log(`- Embedding similarity: ${embeddingSim.toFixed(4)}`);
      }
    }
  });
  
  // Generate recommendations based on viewed articles
  console.log("\n=== Testing Hybrid Recommendations ===");
  const viewedArticles = [mockArticles[0], mockArticles[3]]; // ML and Frontend articles
  const recommendations = generateRecommendations(viewedArticles, mockArticles, 3);
  
  console.log("Recommendations based on viewed articles:");
  viewedArticles.forEach(article => console.log(`- Viewed: ${article.title}`));
  console.log("\nRecommended articles:");
  recommendations.forEach(article => {
    console.log(`- ${article.title} (Relevance: ${article.relevanceScore.toFixed(4)})`);
  });
}

// Run the tests
export function runEmbeddingTests() {
  console.log("Running hybrid similarity tests...");
  testBasicSimilarity();
  testTermMatching();
  testArticleSimilarity();
  console.log("Tests completed!");
}

// Run tests if this file is executed directly
if (require.main === module) {
  runEmbeddingTests();
} 