from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings

# Import route modules
from api.routes import recommendations, health, routes_list


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events.
    """
    # 🟢 STARTUP: Initialize recommendation system
    print("🚀 Starting up recommendation API...")
    recommendations.initialize_recommendation_system()
    print("✅ Startup complete!")
    
    yield  # ⏸️ App runs here - handles all requests
    
    # 🔴 SHUTDOWN: Cleanup (if needed)
    print("🛑 Shutting down...")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(recommendations.router, tags=["Recommendations"])
app.include_router(routes_list.router, tags=["Routes"])


@app.get("/")
async def root():
    return {
        "message": "Product Recommendation System API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "routes": "/api/routes"
    }

