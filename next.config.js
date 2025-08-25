/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [
      {
        source: '/(.*)', // 对所有页面生效
        headers: [
          {
            key: 'Content-Security-Policy',
            value: "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self' http://localhost:8000 http://localhost:10000 https://malaysia-opr-tracking.onrender.com; font-src 'self'; object-src 'none'; frame-ancestors 'none';"
          }
        ]
      }
    ];
  },
};

module.exports = nextConfig;
