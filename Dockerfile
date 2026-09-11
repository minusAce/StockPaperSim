FROM node:22-alpine AS frontend
WORKDIR /src/frontend

# Copy both package.json and package-lock.json (if it exists)
COPY frontend/package.json frontend/package-lock.json* ./

RUN npm install --no-audit --no-fund
COPY frontend/ .
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=frontend /src/frontend/dist /app/frontend/dist
EXPOSE 8000
CMD ["python", "run.py"]