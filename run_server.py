import uvicorn
import logging
from server import app
from rag.logging_config import configure_logging

# Configure logging to ensure console output
configure_logging(level=logging.INFO)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7878, log_level="info")
