# main.py
from fastapi import FastAPI

# Initialize the FastAPI application
app = FastAPI(title="Minimal FastAPI Server")

@app.get("/")
def read_root():
    """
    A simple health check endpoint.
    """
    return {"message": "API Server is running successfully!"}

@app.get("/hello/{name}")
def say_hello(name: str):
    """
    Greets the user by name.
    """
    # The response is automatically converted to JSON by FastAPI
    return {"greeting": f"Hello, {name.capitalize()}! Welcome to the minimal API."}

