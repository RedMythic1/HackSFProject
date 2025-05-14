const { spawn } = require('child_process');

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed, use POST' });
  }

  let body = '';
  req.on('data', chunk => {
    body += chunk.toString();
  });

  req.on('end', () => {
    try {
      const { strategy } = JSON.parse(body || '{}');
      if (!strategy || typeof strategy !== 'string') {
        return res.status(400).json({ error: 'Missing "strategy" in request body' });
      }

      // Spawn the Python backtesting script
      const pyProcess = spawn('python3', ['stockbt/stocker_test.py', strategy, '--json'], {
        cwd: process.cwd(),
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
      });

      let pyOutput = '';
      let pyError = '';

      pyProcess.stdout.on('data', data => {
        pyOutput += data.toString();
      });

      pyProcess.stderr.on('data', data => {
        pyError += data.toString();
      });

      pyProcess.on('close', code => {
        if (code !== 0) {
          console.error('Python process exited with code', code, pyError);
          return res.status(500).json({ error: 'Backtest failed', details: pyError });
        }

        // The Python script prints JSON as the last line
        let resultJson;
        try {
          const lines = pyOutput.trim().split('\n');
          resultJson = JSON.parse(lines[lines.length - 1]);
        } catch (err) {
          console.error('Failed parsing Python output', err, pyOutput);
          return res.status(500).json({ error: 'Failed to parse results' });
        }

        const fs = require('fs');
        const path = require('path');

        // Embed code content
        let codeContent = '';
        if (resultJson.code_path && fs.existsSync(resultJson.code_path)) {
          codeContent = fs.readFileSync(resultJson.code_path, 'utf8');
        }

        // Embed image as base64
        let imageBase64 = '';
        if (resultJson.image_path && fs.existsSync(resultJson.image_path)) {
          const imgBuffer = fs.readFileSync(resultJson.image_path);
          imageBase64 = `data:image/png;base64,${imgBuffer.toString('base64')}`;
        }

        return res.status(200).json({
          profit: resultJson.profit,
          code: codeContent,
          image: imageBase64,
          buy_points: resultJson.buy_points,
          sell_points: resultJson.sell_points
        });
      });
    } catch (err) {
      return res.status(400).json({ error: 'Invalid JSON body', details: err.message });
    }
  });
}; 