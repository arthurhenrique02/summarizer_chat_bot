from fastapi import FastAPI

from core.config import config_middlewares, config_routes


def create_app() -> FastAPI:
    app = FastAPI(title="Summarizer Chatbot", version="1.0.0")

    config_middlewares(app)
    config_routes(app)

    return app


app = create_app()
