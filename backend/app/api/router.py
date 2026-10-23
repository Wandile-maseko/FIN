"""API layer: routes exposed to the React frontend and Streamlit app."""

from fastapi import APIRouter

from app.api.endpoints import insiders, learn, sentiment, stocks

api_router = APIRouter()
api_router.include_router(stocks.router, tags=["stocks"])
api_router.include_router(insiders.router, tags=["insiders"])
api_router.include_router(sentiment.router, tags=["sentiment"])
api_router.include_router(learn.router, tags=["learning"])
