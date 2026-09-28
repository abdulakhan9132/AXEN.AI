from fastapi import FastAPI

app = FastAPI(
    title="Private Job Agent",
    version="0.1.0"
)


@app.get("/")
def home():
    return {
        "app": "Private Job Agent",
        "status": "running",
        "version": "0.1.0"
    }


@app.get("/health")
def health():
    return {
        "status": "OK"
    }


@app.get("/profile")
def profile():
    return {
        "role": "UI/UX Designer",
        "location": "Bengaluru",
        "status": "ready"
    }