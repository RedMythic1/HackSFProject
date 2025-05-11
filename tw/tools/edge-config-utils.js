#!/usr/bin/env node

/**
 * Edge Config Utility Functions
 * 
 * This file contains utility functions for working with Vercel Edge Config
 * from both Node.js scripts and command line.
 */

const fs = require('fs');
const path = require('path');
const { createClient } = require('@vercel/edge-config');

// Initialize Edge Config client
const edgeConfigUrl = process.env.EDGE_CONFIG || 'https://edge-config.vercel.com/ecfg_xsczamr0q3eodjuagxzwjiznqxxs?token=854495e2-1208-47c1-84a6-213468e23510';
const edgeConfig = createClient(edgeConfigUrl);

// Define storage prefixes to maintain compatibility with previous blob storage approach
const STORAGE_PREFIX = 'articles/';
const ARTICLE_PREFIX = 'articles/final_article_';
const SUMMARY_PREFIX = 'articles/summary_';
const SEARCH_PREFIX = 'articles/search_';

/**
 * Get a key for Edge Config from file path (maintaining compatibility with blob storage naming)
 */
function getKeyFromPath(filePath) {
  if (!filePath) return null;
  
  const basename = path.basename(filePath);
  
  if (basename.startsWith('final_article_')) {
    return ARTICLE_PREFIX + basename.replace('final_article_', '');
  } else if (basename.startsWith('summary_')) {
    return SUMMARY_PREFIX + basename.replace('summary_', '');
  } else if (basename.startsWith('search_')) {
    return SEARCH_PREFIX + basename.replace('search_', '');
  } else {
    return STORAGE_PREFIX + basename;
  }
}

/**
 * Read data from Edge Config
 */
async function readConfig(key, defaultValue = null) {
  try {
    console.log(`Reading from Edge Config: ${key}`);
    const value = await edgeConfig.get(key);
    
    if (value === undefined) {
      console.log(`Key not found in Edge Config: ${key}`);
      return defaultValue;
    }
    
    return value;
  } catch (error) {
    console.error(`Error reading from Edge Config ${key}:`, error);
    return defaultValue;
  }
}

/**
 * Write data to Edge Config
 */
async function writeConfig(key, data) {
  try {
    console.log(`Writing to Edge Config: ${key}`);
    
    // Get existing items to patch
    const allItems = await edgeConfig.getAll();
    const patch = { 
      ...allItems, 
      [key]: data 
    };
    
    // Patch the Edge Config with new/updated data
    await edgeConfig.patch(patch);
    
    console.log(`Successfully wrote to Edge Config: ${key}`);
    return { success: true };
  } catch (error) {
    console.error(`Error writing to Edge Config ${key}:`, error);
    return { success: false, error: error.message };
  }
}

/**
 * List all keys with a specific prefix
 */
async function listKeys(prefix = STORAGE_PREFIX) {
  try {
    console.log(`Listing keys with prefix: ${prefix}`);
    const allItems = await edgeConfig.getAll();
    
    // Filter keys that start with the specified prefix
    const keys = Object.keys(allItems).filter(key => key.startsWith(prefix));
    
    console.log(`Found ${keys.length} keys with prefix ${prefix}`);
    return keys;
  } catch (error) {
    console.error(`Error listing keys with prefix ${prefix}:`, error);
    return [];
  }
}

/**
 * Delete a key from Edge Config
 */
async function deleteKey(key) {
  try {
    console.log(`Deleting key from Edge Config: ${key}`);
    
    // Get current items
    const allItems = await edgeConfig.getAll();
    
    // Create a new object without the key to delete
    const { [key]: removed, ...rest } = allItems;
    
    // Update Edge Config with the key removed
    await edgeConfig.patch(rest);
    
    console.log(`Successfully deleted key: ${key}`);
    return { success: true };
  } catch (error) {
    console.error(`Error deleting key ${key}:`, error);
    return { success: false, error: error.message };
  }
}

/**
 * Get all items from Edge Config that match a prefix
 */
async function getItemsWithPrefix(prefix = STORAGE_PREFIX) {
  try {
    console.log(`Getting all items with prefix: ${prefix}`);
    const allItems = await edgeConfig.getAll();
    
    // Filter items that have keys starting with the specified prefix
    const filteredItems = {};
    for (const [key, value] of Object.entries(allItems)) {
      if (key.startsWith(prefix)) {
        filteredItems[key] = value;
      }
    }
    
    console.log(`Found ${Object.keys(filteredItems).length} items with prefix ${prefix}`);
    return filteredItems;
  } catch (error) {
    console.error(`Error getting items with prefix ${prefix}:`, error);
    return {};
  }
}

/**
 * Upload local directory to Edge Config
 */
async function uploadDirectory(localDir, keyPrefix = '', options = {}) {
  try {
    if (!fs.existsSync(localDir)) {
      throw new Error(`Directory does not exist: ${localDir}`);
    }
    
    console.log(`Uploading directory: ${localDir} to Edge Config with prefix: ${keyPrefix}`);
    
    const files = fs.readdirSync(localDir);
    const results = {
      total: files.length,
      success: 0,
      failed: 0,
      skipped: 0,
      details: []
    };
    
    // Get all current items in Edge Config
    const allItems = await edgeConfig.getAll();
    const patchData = { ...allItems };
    
    for (const filename of files) {
      try {
        // Skip directories and non-JSON files unless all files are included
        const filePath = path.join(localDir, filename);
        if (fs.statSync(filePath).isDirectory()) {
          results.skipped++;
          results.details.push({
            file: filename,
            status: 'skipped',
            reason: 'Is a directory'
          });
          continue;
        }
        
        if (!options.includeAllFiles && !filename.endsWith('.json')) {
          results.skipped++;
          results.details.push({
            file: filename,
            status: 'skipped',
            reason: 'Not a JSON file'
          });
          continue;
        }
        
        // Read file content
        const fileContent = fs.readFileSync(filePath, 'utf8');
        let data;
        
        // If it's a JSON file, parse it to validate and format properly
        if (filename.endsWith('.json')) {
          try {
            data = JSON.parse(fileContent);
          } catch (parseError) {
            console.error(`Error parsing JSON for ${filename}:`, parseError);
            results.failed++;
            results.details.push({
              file: filename,
              status: 'failed',
              reason: 'Invalid JSON'
            });
            continue;
          }
        } else {
          data = fileContent;
        }
        
        // Get the key for Edge Config
        let key;
        if (options.useDirectKey) {
          // If a custom key transformer is provided, use it
          key = getKeyFromPath(filePath);
        } else {
          // Otherwise, just concatenate the prefix and filename
          key = `${keyPrefix}${filename}`;
        }
        
        // Add to patch data
        patchData[key] = data;
        
        results.success++;
        results.details.push({
          file: filename,
          status: 'success',
          key
        });
      } catch (fileError) {
        console.error(`Error processing file ${filename}:`, fileError);
        results.failed++;
        results.details.push({
          file: filename,
          status: 'failed',
          error: fileError.message
        });
      }
    }
    
    // Apply the patch to update Edge Config with all new items at once
    if (results.success > 0) {
      try {
        await edgeConfig.patch(patchData);
        console.log(`Successfully updated Edge Config with ${results.success} items`);
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
    
    console.log('\nUpload Summary:');
    console.log(`Total files: ${results.total}`);
    console.log(`Successfully uploaded: ${results.success}`);
    console.log(`Failed: ${results.failed}`);
    console.log(`Skipped: ${results.skipped}`);
    
    return results;
  } catch (error) {
    console.error('Error uploading directory:', error);
    return { error: error.message };
  }
}

/**
 * Download items from Edge Config to local directory
 */
async function downloadItems(prefix, localDir, options = {}) {
  try {
    if (!fs.existsSync(localDir)) {
      if (options.createDir) {
        fs.mkdirSync(localDir, { recursive: true });
      } else {
        throw new Error(`Directory does not exist: ${localDir}`);
      }
    }
    
    console.log(`Downloading items with prefix: ${prefix} to directory: ${localDir}`);
    
    // Get all items with the specified prefix
    const items = await getItemsWithPrefix(prefix);
    const keys = Object.keys(items);
    
    const results = {
      total: keys.length,
      success: 0,
      failed: 0,
      details: []
    };
    
    for (const key of keys) {
      try {
        console.log(`Writing item: ${key}`);
        const data = items[key];
        
        // Determine local file path
        let localPath;
        if (options.stripPrefixFromFilename) {
          // Strip the prefix from the filename
          const relativePath = key.substring(prefix.length);
          localPath = path.join(localDir, relativePath);
        } else {
          // Use the full key as the filename
          localPath = path.join(localDir, path.basename(key));
        }
        
        // Create nested directories if needed
        const localPathDir = path.dirname(localPath);
        if (!fs.existsSync(localPathDir)) {
          fs.mkdirSync(localPathDir, { recursive: true });
        }
        
        // Write the file
        const content = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
        fs.writeFileSync(localPath, content);
        
        results.success++;
        results.details.push({
          key,
          status: 'success',
          localPath
        });
      } catch (itemError) {
        console.error(`Error writing item ${key}:`, itemError);
        results.failed++;
        results.details.push({
          key,
          status: 'failed',
          error: itemError.message
        });
      }
    }
    
    console.log('\nDownload Summary:');
    console.log(`Total items: ${results.total}`);
    console.log(`Successfully downloaded: ${results.success}`);
    console.log(`Failed: ${results.failed}`);
    
    return results;
  } catch (error) {
    console.error('Error downloading items:', error);
    return { error: error.message };
  }
}

// Command-line interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0];
  
  async function run() {
    try {
      switch (command) {
        case 'list':
          const prefix = args[1] || STORAGE_PREFIX;
          const keys = await listKeys(prefix);
          console.log(JSON.stringify(keys, null, 2));
          break;
        
        case 'get':
          const key = args[1];
          if (!key) {
            throw new Error('Key is required');
          }
          const data = await readConfig(key);
          console.log(JSON.stringify(data, null, 2));
          break;
        
        case 'put':
          const putKey = args[1];
          const putFile = args[2];
          if (!putKey || !putFile) {
            throw new Error('Key and file path are required');
          }
          const content = fs.readFileSync(putFile, 'utf8');
          let dataToWrite;
          try {
            dataToWrite = JSON.parse(content);
          } catch (e) {
            dataToWrite = content;
          }
          await writeConfig(putKey, dataToWrite);
          break;
        
        case 'delete':
          const delKey = args[1];
          if (!delKey) {
            throw new Error('Key is required');
          }
          await deleteKey(delKey);
          break;
        
        case 'upload-dir':
          const uploadDir = args[1];
          const uploadPrefix = args[2] || '';
          if (!uploadDir) {
            throw new Error('Local directory path is required');
          }
          await uploadDirectory(uploadDir, uploadPrefix, { useDirectKey: true });
          break;
        
        case 'download':
          const downloadPrefix = args[1];
          const downloadDir = args[2];
          if (!downloadPrefix || !downloadDir) {
            throw new Error('Prefix and local directory path are required');
          }
          await downloadItems(downloadPrefix, downloadDir, { createDir: true });
          break;
        
        default:
          console.error('Unknown command:', command);
          console.log('Available commands: list, get, put, delete, upload-dir, download');
          process.exit(1);
      }
    } catch (error) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  }
  
  run();
} else {
  // Export functions for use in other modules
  module.exports = {
    getKeyFromPath,
    readConfig,
    writeConfig,
    listKeys,
    deleteKey,
    getItemsWithPrefix,
    uploadDirectory,
    downloadItems,
    STORAGE_PREFIX,
    ARTICLE_PREFIX,
    SUMMARY_PREFIX,
    SEARCH_PREFIX
  };
} 