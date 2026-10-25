# FaceGuard AI — Container Deployment

This directory contains container deployment orchestration assets for running FaceGuard AI in an isolated environment.

## Security and Privacy Boundary

1. **Zero Biometric Data in Images:** The Docker image contains strictly the software runtime and dependencies. Biometric templates, galleries, and model weights are **never baked into image layers**.
2. **Read-Only Mounts:** Private weights and gallery files under `models/private/` are mounted strictly read-only (`:ro`).
3. **Non-Root Execution:** The container executes as non-root user `faceguard` (UID 1000).
4. **No Privileged Mode:** Hardware camera passthrough is restricted to the specific device node (e.g. `/dev/video0`) without `--privileged`.

## Orchestration Files

- [`docker-compose.yml`](docker-compose.yml): Multi-container or single-service compose specification with secure volume mappings.
- [`run_container.sh`](run_container.sh): Command-line launcher supporting smoke checks, video files, and live camera streams.

## Quick Commands

### 1. Headless Smoke Check (Readiness Verification)
Runs model artifact loading and hash verification in an isolated container without camera:
```bash
deployment/run_container.sh smoke
```
Or with Docker Compose:
```bash
docker compose -f deployment/docker-compose.yml run --rm faceguard
```

### 2. Live Camera Session (Linux Host)
Passes `/dev/video0` and optional X11 display forwarding:
```bash
deployment/run_container.sh camera /dev/video0
```

### 3. Sequential Video File Verification
Runs headless verification against an MP4 file:
```bash
deployment/run_container.sh video /path/to/test_clip.mp4
```
