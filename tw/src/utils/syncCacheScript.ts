import { syncCache } from './cacheSync';

/**
 * This script can be run to manually sync the cache
 * Run with: npx ts-node src/utils/syncCacheScript.ts
 */

// Create an async main function
async function main() {
  console.log('Starting cache synchronization...');
  try {
    const stats = await syncCache();
    console.log('Cache synchronization complete!');
    console.log(`Stats: ${stats.added} added, ${stats.updated} updated, ${stats.skipped} skipped, ${stats.errors} errors`);
    console.log(`Total files in local cache: ${stats.totalLocal + stats.added}`);
  } catch (error) {
    console.error('Error during cache synchronization:', error);
  }
}

// Run the main function
main().catch(error => console.error('Unhandled error:', error)); 