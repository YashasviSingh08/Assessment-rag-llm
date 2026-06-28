from fastapi import HTTPException

def internal_error(e):
    raise HTTPException(
        status_code=500,
        detail=str(e)
    )