from fastapi import FastAPI

from core.config import config_routes


def create_app() -> FastAPI:
    app = FastAPI(title="Soda Vendor API", version="1.0.0")

    config_routes(app)

    return app


app = create_app()
