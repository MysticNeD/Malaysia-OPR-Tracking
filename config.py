from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_secret_key: str
    db_user: str
    db_password: str
    db_host: str
    db_port: int

    class Config:
        env_file = ".env"

settings = Settings()
