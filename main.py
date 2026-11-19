from fastapi.responses import FileResponse, HTMLResponse

BASE_DIR = Path(__file__).resolve().parent


@app.get("/", include_in_schema=False)
def home():
    """Root URL (/) par website (index.html) serve karta hai."""
    page = BASE_DIR / "index.html"
    if not page.is_file():
        return HTMLResponse(
            "<h3>index.html nahi mili</h3><p>index.html ko main.py ke saath ek hi folder mein "
            "rakhein aur Dockerfile mein <code>COPY main.py index.html ./</code> likhein.</p>"
            "<p>API docs: <a href='/docs'>/docs</a></p>",
            status_code=500,
        )
    return FileResponse(page, media_type="text/html")
