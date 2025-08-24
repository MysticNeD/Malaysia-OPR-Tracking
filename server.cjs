// server.cjs
const express = require('express');
const path = require('path');
const app = express();

// Health check endpoint
app.get('/health', (req, res) => res.send('OK'));

// Serve Vite build (dist) files
app.use(express.static(path.join(__dirname, 'dist')));

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
