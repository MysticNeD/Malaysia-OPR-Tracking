import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

// ES Modules 下获取 __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// Health check endpoint
app.get('/health', (req, res) => res.send('OK'));

// Serve React build files
const buildPath = path.join(__dirname, 'build');
app.use(express.static(buildPath));

// Catch-all route to serve index.html for React Router
app.get('*', (req, res) => {
  res.sendFile(path.join(buildPath, 'index.html'));
});

// Listen on the environment port or default 10000
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
