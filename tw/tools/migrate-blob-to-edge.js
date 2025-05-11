#!/usr/bin/env node

/**
 * Migrate from Vercel Blob Storage to Edge Config
 * 
 * This tool migrates data from Vercel Blob Storage to Edge Config.
 * It will list all blobs with a specified prefix, then save them to Edge Config.
 */

const { list, get } = require('@vercel/blob');
const { createClient } = require('@vercel/edge-config');

// Initialize Edge Config client
const edgeConfigUrl = process.env.EDGE_CONFIG || 'https://edge-config.vercel.com/ecfg_xsczamr0q3eodjuagxzwjiznqxxs?token=854495e2-1208-47c1-84a6-213468e23510';
const edgeConfig = createClient(edgeConfigUrl);

// Define blob storage prefixes
const BLOB_PREFIX = 'articles/';
const BLOB_PREFIXES = [
  'articles/final_article_',
  'articles/summary_',
  'articles/search_'
];

/**
 * Migrate blobs from Vercel Blob Storage to Edge Config
 */
async function migrateBlobs(prefix = BLOB_PREFIX, options = {}) {
  try {
    console.log(`Starting migration from Blob Storage to Edge Config for prefix: ${prefix}`);
    console.log('Listing blobs...');
    
    // List all blobs with the specified prefix
    let { blobs, cursor } = await list({ prefix, limit: 100 });
    let allBlobs = [...blobs];
    
    // Continue listing if there are more blobs
    while (cursor) {
      const result = await list({ prefix, cursor, limit: 100 });
      allBlobs = [...allBlobs, ...result.blobs];
      cursor = result.cursor;
    }
    
    console.log(`Found ${allBlobs.length} blobs with prefix ${prefix}`);
    
    // Get all existing items in Edge Config
    console.log('Getting existing items from Edge Config...');
    const allItems = await edgeConfig.getAll();
    const patchData = { ...allItems };
    
    // Process blobs
    const results = {
      total: allBlobs.length,
      success: 0,
      failed: 0,
      skipped: 0,
      details: []
    };
    
    console.log('Processing blobs...');
    for (const blob of allBlobs) {
      try {
        // Skip if already processed and override is not enabled
        if (!options.override && patchData[blob.pathname]) {
          console.log(`Skipping ${blob.pathname} - already exists in Edge Config`);
          results.skipped++;
          results.details.push({
            blob: blob.pathname,
            status: 'skipped',
            reason: 'Already exists in Edge Config'
          });
          continue;
        }
        
        console.log(`Processing blob: ${blob.pathname}`);
        
        // Get blob content
        const content = await blob.text();
        let data;
        
        // Parse JSON data if possible
        try {
          data = JSON.parse(content);
        } catch (parseError) {
          // If content is not JSON, use it as-is
          data = content;
        }
        
        // Add to patch data
        patchData[blob.pathname] = data;
        
        results.success++;
        results.details.push({
          blob: blob.pathname,
          status: 'success'
        });
      } catch (blobError) {
        console.error(`Error processing blob ${blob.pathname}:`, blobError);
        results.failed++;
        results.details.push({
          blob: blob.pathname,
          status: 'failed',
          error: blobError.message
        });
      }
    }
    
    // Apply the patch to update Edge Config with all migrated items at once
    if (results.success > 0) {
      try {
        console.log(`Updating Edge Config with ${results.success} items...`);
        await edgeConfig.patch(patchData);
        console.log('Successfully updated Edge Config');
      } catch (patchError) {
        console.error('Error applying patch to Edge Config:', patchError);
        
        // Mark all previously "successful" items as failed
        results.failed += results.success;
        results.success = 0;
        
        for (const detail of results.details) {
          if (detail.status === 'success') {
            detail.status = 'failed';
            detail.error = 'Failed to update Edge Config: ' + patchError.message;
          }
        }
      }
    }
    
    console.log('\nMigration Summary:');
    console.log(`Total blobs: ${results.total}`);
    console.log(`Successfully migrated: ${results.success}`);
    console.log(`Failed: ${results.failed}`);
    console.log(`Skipped: ${results.skipped}`);
    
    return results;
  } catch (error) {
    console.error('Error migrating blobs:', error);
    return { error: error.message };
  }
}

/**
 * Migrate all blobs with specified prefixes
 */
async function migrateAllBlobPrefixes(prefixes = BLOB_PREFIXES, options = {}) {
  const overallResults = {
    total: 0,
    success: 0,
    failed: 0,
    skipped: 0,
    details: []
  };
  
  for (const prefix of prefixes) {
    console.log(`\n=== Migrating prefix: ${prefix} ===\n`);
    const result = await migrateBlobs(prefix, options);
    
    if (!result.error) {
      overallResults.total += result.total;
      overallResults.success += result.success;
      overallResults.failed += result.failed;
      overallResults.skipped += result.skipped;
      overallResults.details = overallResults.details.concat(result.details);
    } else {
      console.error(`Error migrating prefix ${prefix}:`, result.error);
    }
  }
  
  console.log('\n=== Overall Migration Summary ===');
  console.log(`Total blobs: ${overallResults.total}`);
  console.log(`Successfully migrated: ${overallResults.success}`);
  console.log(`Failed: ${overallResults.failed}`);
  console.log(`Skipped: ${overallResults.skipped}`);
  
  return overallResults;
}

// Command-line interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const prefix = args[0] || 'articles/';
  const override = args.includes('--override') || args.includes('-o');
  
  async function run() {
    try {
      if (prefix === 'all') {
        await migrateAllBlobPrefixes(BLOB_PREFIXES, { override });
      } else {
        await migrateBlobs(prefix, { override });
      }
      console.log('Migration completed');
    } catch (error) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  }
  
  run();
} else {
  // Export functions for use in other modules
  module.exports = {
    migrateBlobs,
    migrateAllBlobPrefixes
  };
} 