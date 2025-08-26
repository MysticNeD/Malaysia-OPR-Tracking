/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Content-Security-Policy',
            value: "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self' https://malaysia-opr-tracking.onrender.com; font-src 'self'; object-src 'none'; frame-ancestors 'none';"
          }
        ]
      }
    ];
  },
};

module.exports = nextConfig;