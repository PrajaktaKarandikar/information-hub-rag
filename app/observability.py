from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Define Prometheus metrics

# Query related metrics
QUERY_COUNT = Counter('rag_queries_total', 'Total queries processed by the RAG pipeline')
QUERY_LATENCY = Histogram('rag_query_latency_seconds', 'Latency of RAG queries in seconds')
QUERY_ERRORS = Counter('rag_query_errors_total', 'Total errors during RAG queries') 

# Token usage metrics for tracking
TOKENS_USED = Counter('rag_tokens_used_total', 'Total tokens used across all RAG queries')

# Vector store metrics
VECTOR_STORE_SIZE = Gauge('rag_vector_store_size', 'Number of documents in vector store')

def track_metrics(func):
    """ Decorator to track metrics for RAG pipeline queries. """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start timer for latency measurement
        try:
            result = func(*args, **kwargs)  # Call the original function
            QUERY_COUNT.inc()  # Increment query count
            return result
        except Exception as e:
            QUERY_ERRORS.inc()  # Increment error count if an exception occurs
            raise e  # Re-raise the exception after tracking it
        finally:
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            QUERY_LATENCY.observe(elapsed_time)  # Record latency in histogram
    return wrapper
