# Paper Triage Agent: containerized for reproducible runs.
#
# Build:  docker build -t paper-triage .
# Run:    docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#                   paper-triage "Your research question here"

# Python slim image keeps the container small. We pin the minor version
# so future Python releases don't accidentally break the build.
FROM python:3.12-slim

# Install Node.js, which the Claude Agent SDK's bundled CLI binary needs.
# Combining apt-get update, install, and cleanup in one RUN keeps the layer
# small (cleanup is wasted if it's in a separate RUN).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container.
WORKDIR /app

# Copy ONLY the dependency manifest first, so the pip install layer caches.
# If we copied everything here, every code change would invalidate this layer.
COPY requirements.txt ./

# Install Python dependencies into the system Python (no venv inside a container;
# the container itself is the isolation boundary).
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the source. Changes to .py files only invalidate from here.
COPY . .

# Default to JSON logging in containers
ENV LOG_FORMAT=json

# Make sure stdout/stderr are unbuffered so logs appear immediately rather
# than being trapped in a Python buffer until the program exits.
ENV PYTHONUNBUFFERED=1

# When the container starts, run the CLI entry point. Anything passed to
# `docker run paper-triage <args>` becomes args to graph_app.py.
ENTRYPOINT ["python", "graph_app.py"]