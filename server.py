from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import logging

from rag.query_processor import process_query

# Set up logging
logger = logging.getLogger(__name__)

app = FastAPI()

# Get the current directory
BASE_DIR = Path(__file__).resolve().parent

# Set up templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class Query(BaseModel):
    text: str
    ast_cache_dir: Optional[str] = "."


class QueryRequest(BaseModel):
    text: str
    ast_cache_dir: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "How do we compute inverse matrics",
                "ast_cache_dir": "./cache",
            }
        }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Handle code query requests."""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Query text cannot be empty")

        response = await process_query(request.text)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Add a health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Add error handlers
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {str(exc)}")
    return {
        "detail": "Invalid request format. Please send a POST request with JSON body: {'text': 'your question here', 'ast_cache_dir': 'path/to/ast/cache'}"
    }
