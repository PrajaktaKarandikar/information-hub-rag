# Information Hub - Production RAG API

A production-ready RAG (Retrieval Augmented Generation) API built using FastAPI and Docker. Loads documents from multiple sources (local PDFs, s3, web) and can be used for question answering using vector search and OpenAI.

## Features
- Multi-source document loading
- vector search witj FAISS and OpenAI embeddings
- Rest API with FastAPI
- Observabiliy with Prometheus metrics
- Docker containerization with multi-service orchestration

## Tech stack
- **Backend:** FastAPI, Python, REST API
- **Machine Learning:** Langchain, OpenAI API, FAISS
- **Infrastructure:** Docker, AWS S3, Prometheus
- **Deployment:** Render

## Project Structure
```
information-hub-rag/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── information_loader.py # Document loader (S3, web, PDF)
│   ├── rag_pipeline.py      # RAG logic
│   ├── observability.py     # Prometheus metrics
│   └── config.py             # Configuration
├── prometheus.yml            # Prometheus config
├── docker-compose.yml        # Multi-service orchestration
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── config.yaml               # App configuration
├── .env.example              # Environment variables template
├── .gitignore                
├── LICENSE                   
└── README.md                 # This file
```

## Quickstart

### Prerequisites
- Docker and Docker compose
- OpenAI API key
- AWS credentials

### Setup
```bash
# Clone the repository
git clone https://github.com/PrajaktaKarandikar/information-hub-rag.git
cd information-hub-rag

# Create .env file using example
cp .env.example .env
# Edit .env with actual API keys

# Run with docker
docker-compose up --build
```

Test the API
```bash
# Health check
curl http://localhost:8000/health

# Load a document
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d "{\"sources\":[\"https://en.wikipedia.org/wiki/Machine_learning\"],\"replace_existing\":true}"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is machine learning?\",\"return_sources\":true}"
```

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest` | Load documents |
| POST | `/query` | Ask questions |
| GET | `/metrics` | Prometheus metrics |

## Environment Variables
- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `AWS_ACCESS_KEY_ID` - AWS access key (optional, for S3)
- `AWS_SECRET_ACCESS_KEY` - AWS secret key (optional)
- `AWS_S3_BUCKET` - S3 bucket name (optional)
- `AWS_REGION` - AWS region (optional; default: us-east-1)

## Monitoring
- `/metrics` endpoint exposes Prometheus metrics
- Track query latency, error rates and request counts

When running locally:
- **Prometheus dashboard** at `http://localhost:9090`
- **Metrics endpoint** at `http://localhost:8000/metrics`

## Deployment
This application is designed to be deployed on Render.

**Live demo**: [https://rag-infohub.onrender.com/docs](https://rag-infohub.onrender.com/docs)
(*Note: This is hosted on a free tier that spins down after 15 mins inactivity. So first request after idle may take upto 30-60 secs. Thank you for your patience!*)

## References
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Prometheus Documentation](https://prometheus.io/)
- [OpenAI API Documentation](https://platform.openai.com/)
- [Docker Documentation](https://docs.docker.com/)

- [Render Documentation](https://render.com/docs)

