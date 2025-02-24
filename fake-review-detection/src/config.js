const config = {
    development: {
        apiUrl: 'http://localhost:5000',
        wsUrl: 'ws://localhost:5000'
    },
    production: {
        apiUrl: 'https://your-api-domain.com',
        wsUrl: 'wss://your-api-domain.com'
    }
};

const environment = process.env.NODE_ENV || 'development';

export default {
    apiUrl: config[environment].apiUrl,
    wsUrl: config[environment].wsUrl,
    // Add any other configuration options here
    maxReviewLength: 5000,
    minReviewLength: 10,
    requestTimeout: 30000, // 30 seconds
    retryAttempts: 3
}; 