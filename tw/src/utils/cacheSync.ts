// Browser-compatible cache synchronization module
// This uses API calls to the server instead of direct file system access

/**
 * Cache sync statistics interface
 */
interface CacheSyncStats {
  added: number;
  updated: number;
  skipped: number;
  errors: number;
  totalLocal: number;
}

/**
 * API response interface for cache operations
 */
interface CacheApiResponse<T = any> {
  status: 'success' | 'error';
  message?: string;
  data?: T;
  stats?: CacheSyncStats;
  source?: 'local' | 'main' | 'blob';
}

// File types to synchronize (for reference in UI)
export const FILE_TYPES = {
  SUMMARY: 'summary_',
  SEARCH: 'search_',
  FINAL_ARTICLE: 'final_article_'
};

/**
 * Synchronize the local cache with the main cache
 * @returns Promise with statistics about the synchronization
 */
export async function syncCache(): Promise<CacheSyncStats> {
  try {
    console.log('Initiating cache synchronization with Vercel Blob Storage...');
    const response = await fetch('/api/sync-cache', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      throw new Error(`Server returned error ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json() as CacheApiResponse;
    
    if (result.status === 'error') {
      throw new Error(result.message || 'Unknown server error');
    }
    
    console.log(`Cache sync complete: Using Vercel Blob Storage`);
    return result.stats || { added: 0, updated: 0, skipped: 0, errors: 0, totalLocal: 0 };
  } catch (err) {
    console.error(`Cache sync failed: ${err}`);
    return { added: 0, updated: 0, skipped: 0, errors: 1, totalLocal: 0 };
  }
}

/**
 * Generate a cache key from a string
 * @param input The input string to hash
 * @returns The key as a hex string
 */
export function generateCacheKey(input: string): string {
  // We'll calculate this on the server side
  // But we'll still encode any URI components for safety
  return encodeURIComponent(input);
}

/**
 * Get a cached file from the server
 * @param fileName The name of the file to get
 * @returns Promise with the file content or null if not found
 */
export async function getCachedFile<T = any>(fileName: string): Promise<T | null> {
  try {
    const response = await fetch(`/api/get-cached-file?file=${encodeURIComponent(fileName)}`);
    
    if (!response.ok) {
      if (response.status === 404) {
        console.log(`File not found in blob storage: ${fileName}`);
        return null;
      }
      throw new Error(`Server returned error ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json() as CacheApiResponse<T>;
    
    if (result.status === 'error') {
      throw new Error(result.message || 'Unknown server error');
    }
    
    // Add UI indicator for cache source
    if (result.source) {
      console.log(`Retrieved from ${result.source === 'blob' ? 'Vercel Blob Storage' : result.source}: ${fileName}`);
    }
    
    return result.data || null;
  } catch (err) {
    console.error(`Error getting cached file ${fileName}: ${err}`);
    return null;
  }
}

/**
 * Get a summary for an article from cache
 * @param articleId The article ID or URL
 * @returns Promise with the cached summary or null if not found
 */
export async function getCachedSummary(articleId: string): Promise<any> {
  try {
    const response = await fetch(`/api/get-summary?id=${encodeURIComponent(articleId)}`);
    
    if (!response.ok) {
      if (response.status === 404) {
        console.log(`Summary not found for article: ${articleId}`);
        return null;
      }
      throw new Error(`Server returned error ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json() as CacheApiResponse;
    
    if (result.status === 'error') {
      throw new Error(result.message || 'Unknown server error');
    }
    
    // Add UI indicator for cache source
    if (result.source) {
      console.log(`Retrieved summary from ${result.source === 'blob' ? 'Vercel Blob Storage' : result.source}`);
    }
    
    return result.data || null;
  } catch (err) {
    console.error(`Error getting cached summary for ${articleId}: ${err}`);
    return null;
  }
}

/**
 * Get search results from cache
 * @param searchQuery The search query
 * @returns Promise with the cached search results or null if not found
 */
export async function getCachedSearch(searchQuery: string): Promise<any> {
  try {
    const response = await fetch(`/api/get-summary?id=${encodeURIComponent(searchQuery)}`);
    
    if (!response.ok) {
      if (response.status === 404) {
        console.log(`Search results not found for query: ${searchQuery}`);
        return null;
      }
      throw new Error(`Server returned error ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json() as CacheApiResponse;
    
    if (result.status === 'error') {
      throw new Error(result.message || 'Unknown server error');
    }
    
    // Add UI indicator for cache source
    if (result.source) {
      console.log(`Retrieved search results from ${result.source === 'blob' ? 'Vercel Blob Storage' : result.source}`);
    }
    
    return result.data || null;
  } catch (err) {
    console.error(`Error getting cached search results for ${searchQuery}: ${err}`);
    return null;
  }
}

/**
 * Get a final article from cache
 * @param articleKey The article key
 * @returns Promise with the cached article or null if not found
 */
export async function getCachedArticle(articleKey: string): Promise<any> {
  try {
    const response = await fetch(`/api/get-article?key=${encodeURIComponent(articleKey)}`);
    
    if (!response.ok) {
      if (response.status === 404) {
        console.log(`Article not found for key: ${articleKey}`);
        return null;
      }
      throw new Error(`Server returned error ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json() as CacheApiResponse;
    
    if (result.status === 'error') {
      throw new Error(result.message || 'Unknown server error');
    }
    
    // Add UI indicator for cache source
    if (result.source) {
      console.log(`Retrieved article from ${result.source === 'blob' ? 'Vercel Blob Storage' : result.source}`);
    }
    
    return result.data || null;
  } catch (err) {
    console.error(`Error getting cached article for ${articleKey}: ${err}`);
    return null;
  }
}

// Initial cache synchronization
syncCache().catch(err => console.error('Initial cache sync failed:', err)); 