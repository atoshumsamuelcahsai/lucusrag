import uvicorn
import logging
from lucusrag.server import app

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7878, log_level="info")
