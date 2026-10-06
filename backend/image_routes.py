from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import image_service

router = APIRouter()

class ImageRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    negative_prompt: str | None = None
    width: int = Field(512, ge=256, le=768, multiple_of=8)
    height: int = Field(512, ge=256, le=768, multiple_of=8)

@router.post("/api/generate-image")
async def generate_image(req: ImageRequest):
    try:
        url = await run_in_threadpool(
            image_service.generate,
            req.prompt, req.negative_prompt, req.width, req.height,
        )
    except Exception as e:
        print(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail="Image generation failed")
    return {"url": url}
