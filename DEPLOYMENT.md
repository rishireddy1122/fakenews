# Deployment Guide

This guide covers different deployment options for the Fake News Detection API.

## Local Development

```bash
# Start the Flask development server
python deployment/app.py
```

Access at: http://localhost:5000

## Production Deployment

### Using Gunicorn (Recommended)

```bash
# Install gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 deployment.app:app
```

### With Nginx Reverse Proxy

1. Configure Nginx:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

2. Start the application:

```bash
gunicorn --bind 127.0.0.1:5000 --workers 4 deployment.app:app
```

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t fakenews-api .

# Run container
docker run -d -p 5000:5000 -v $(pwd)/models:/app/models --name fakenews fakenews-api
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Cloud Deployment

### Heroku

1. Create Heroku app:
```bash
heroku create your-app-name
```

2. Add Procfile:
```
web: gunicorn --bind 0.0.0.0:$PORT --workers 4 deployment.app:app
```

3. Deploy:
```bash
git push heroku main
```

### AWS EC2

1. Launch an EC2 instance (Ubuntu)
2. SSH into the instance
3. Install dependencies:
```bash
sudo apt update
sudo apt install python3-pip nginx
pip3 install -r requirements.txt
```

4. Clone repository and train model
5. Configure Nginx as reverse proxy
6. Use systemd to manage the service:

```ini
# /etc/systemd/system/fakenews.service
[Unit]
Description=Fake News Detection API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/fakenews
ExecStart=/usr/local/bin/gunicorn --bind 127.0.0.1:5000 --workers 4 deployment.app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

### AWS Elastic Beanstalk

1. Install EB CLI:
```bash
pip install awsebcli
```

2. Initialize EB:
```bash
eb init -p python-3.9 fakenews-api
```

3. Create environment and deploy:
```bash
eb create fakenews-env
eb deploy
```

### Google Cloud Platform (Cloud Run)

1. Build container:
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/fakenews-api
```

2. Deploy:
```bash
gcloud run deploy --image gcr.io/PROJECT-ID/fakenews-api --platform managed
```

### Azure App Service

1. Create resource group:
```bash
az group create --name fakenews-rg --location eastus
```

2. Create app service plan:
```bash
az appservice plan create --name fakenews-plan --resource-group fakenews-rg --sku B1 --is-linux
```

3. Create web app:
```bash
az webapp create --resource-group fakenews-rg --plan fakenews-plan --name fakenews-api --runtime "PYTHON|3.9"
```

4. Deploy code:
```bash
az webapp up --name fakenews-api --resource-group fakenews-rg
```

## Environment Variables

Set these environment variables for production:

```bash
export FLASK_ENV=production
export MODEL_DIR=/path/to/models
export LOG_LEVEL=INFO
```

## Performance Optimization

### Caching
Consider adding Redis for caching predictions:

```python
import redis
cache = redis.Redis(host='localhost', port=6379)
```

### Load Balancing
Use multiple workers with a load balancer:

```bash
gunicorn --workers $(nproc) --bind 0.0.0.0:5000 deployment.app:app
```

### Model Optimization
- Use model quantization to reduce size
- Consider ONNX runtime for faster inference
- Implement request batching

## Monitoring

### Application Logs

```bash
# View logs
docker logs -f fakenews

# With systemd
journalctl -u fakenews -f
```

### Health Checks

Monitor the `/api/health` endpoint:

```bash
curl http://your-domain.com/api/health
```

### Metrics to Track
- Request latency
- Prediction accuracy
- Error rates
- CPU/Memory usage

## Security Considerations

1. **API Key Authentication**:
```python
from functools import wraps
from flask import request

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function
```

2. **Rate Limiting**:
```bash
pip install flask-limiter
```

3. **HTTPS**: Always use HTTPS in production
4. **CORS**: Configure CORS appropriately for your domain

## Scaling

### Horizontal Scaling
- Deploy multiple instances behind a load balancer
- Use container orchestration (Kubernetes, ECS)

### Vertical Scaling
- Increase CPU/memory resources
- Use GPU instances for deep learning models

## Backup and Recovery

1. **Model Backup**:
```bash
# Backup models directory
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/
```

2. **Automated Backups**:
```bash
# Add to crontab
0 2 * * * /path/to/backup-script.sh
```

## Troubleshooting

### Model Not Found
```bash
# Ensure model is trained
python train.py --save-model

# Check model directory
ls -la models/
```

### Out of Memory
```bash
# Reduce workers
gunicorn --workers 2 --bind 0.0.0.0:5000 deployment.app:app
```

### Slow Predictions
- Reduce model complexity
- Use caching
- Implement request batching

## Support

For issues or questions:
- Check logs for error messages
- Review the main README.md
- Open an issue on GitHub
