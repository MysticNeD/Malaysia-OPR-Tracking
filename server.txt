// server.cjs
const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();

// Health check endpoint
app.get('/health', (req, res) => res.send('OK'));

// 代理 /api 请求到 Vercel 的无服务器函数
// 这个代理需要指向 Vercel 内部的无服务器函数运行时
app.use(
  '/api',
  createProxyMiddleware({
    target: 'http://localhost:3000', // Vercel 无服务器函数在本地运行时通常使用的端口
    changeOrigin: true,
  })
);

// Serve Vite build (dist) files
app.use(express.static(path.join(__dirname, 'dist')));

// This should be the last route. It serves the main HTML file for all other routes.
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));