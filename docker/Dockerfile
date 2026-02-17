FROM python:3.12-slim

WORKDIR /app

# Viewer assets (serve.py + index.html + js/)
COPY viewer/ /app/viewer/

# Baked tile data (~18 GB: pyramids.json + 38 brain pyramid dirs)
COPY data/mouselight/tiles/ /app/data/mouselight/tiles/

EXPOSE 8080

CMD ["python", "/app/viewer/serve.py", "--port", "8080", "--tiles-base", "/app/data/mouselight/tiles/"]
