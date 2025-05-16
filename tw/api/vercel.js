// tw/api/vercel.js
// This file should now primarily import the app from server.js and export it.

const app = require('./server'); // Require the fully configured app from server.js

// Potentially, you might need to re-apply specific Vercel middleware or configurations here
// if they were unique to the old vercel.js app instance, but generally, server.js should handle all.

// Example: If you had Vercel-specific CORS in the old vercel.js, you might consider if it's still needed
// or if server.js's CORS is sufficient.
// const cors = require('cors');
// app.use(cors({ origin: true, credentials: true })); // Example if needed

console.log('[vercel.js] Re-exporting app from server.js for Vercel deployment.');

module.exports = app; // Export the app for Vercel 