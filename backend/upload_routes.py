from pathlib import Path
import shutil
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse

from . import auth

router = APIRouter(prefix="/upload", tags=["CSV Upload"])

# Directory where raw CSV files are stored
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/csv", summary="Upload weekly sales CSV")
async def upload_csv(file: UploadFile = File(...),
                     current_user: dict = Depends(auth.require_manager_or_admin)):
    """Accept a CSV file and store it under `data/raw/weekly_sales.csv`.
    Overwrites any existing file. Returns a simple success payload.
    """
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail={"error": True, "message": "Only CSV files are allowed", "status_code": 400})
    destination = RAW_DATA_DIR / "weekly_sales.csv"
    try:
        with destination.open('wb') as out_file:
            shutil.copyfileobj(file.file, out_file)
    finally:
        await file.close()
    return JSONResponse(content={"success": True, "message": "CSV uploaded successfully"})
