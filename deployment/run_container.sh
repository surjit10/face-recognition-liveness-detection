#!/usr/bin/env bash
# FaceGuard AI — Container Deployment Launcher
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="faceguard-runtime:prototype"
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video0}"

build_if_needed() {
    if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        echo "Building container image ${IMAGE_NAME}..."
        docker build -t "${IMAGE_NAME}" -f "${PROJECT_ROOT}/Dockerfile" "${PROJECT_ROOT}"
    fi
}

usage() {
    cat <<EOF
Usage: $0 [command] [options]

Commands:
  smoke                     Run headless model loading check (--check-only)
  camera [device]           Run live camera demo (default device: /dev/video0)
  video <file_path>         Run verification pipeline on a sequential video file
  build                     Build the container image

Options:
  --no-display              Run in headless mode without GUI window forwarding
EOF
}

COMMAND="${1:-smoke}"
shift || true

case "${COMMAND}" in
    build)
        docker build -t "${IMAGE_NAME}" -f "${PROJECT_ROOT}/Dockerfile" "${PROJECT_ROOT}"
        ;;
    smoke)
        build_if_needed
        echo "Executing container readiness smoke test..."
        docker run --rm \
            -v "${PROJECT_ROOT}/models/private:/app/models/private:ro" \
            "${IMAGE_NAME}" \
            --config configs/runtime.yaml \
            --check-only
        ;;
    camera)
        build_if_needed
        DEV="${1:-${CAMERA_DEVICE}}"
        if [ ! -e "${DEV}" ]; then
            echo "Error: Video device ${DEV} not found." >&2
            exit 2
        fi
        DISPLAY_ARGS=()
        if [ -n "${DISPLAY:-}" ] && [ -d "/tmp/.X11-unix" ]; then
            DISPLAY_ARGS=(
                -e "DISPLAY=${DISPLAY}"
                -v "/tmp/.X11-unix:/tmp/.X11-unix:ro"
            )
        fi
        echo "Starting container live camera session on ${DEV}..."
        docker run --rm -it \
            --device "${DEV}:${DEV}" \
            -v "${PROJECT_ROOT}/models/private:/app/models/private:ro" \
            -v "${PROJECT_ROOT}/reports:/app/reports:rw" \
            "${DISPLAY_ARGS[@]}" \
            "${IMAGE_NAME}" \
            --config configs/runtime.yaml \
            --camera 0
        ;;
    video)
        build_if_needed
        VIDEO_FILE="${1:-}"
        if [ -z "${VIDEO_FILE}" ] || [ ! -f "${VIDEO_FILE}" ]; then
            echo "Error: Video file not specified or missing." >&2
            exit 2
        fi
        ABS_VIDEO="$(cd "$(dirname "${VIDEO_FILE}")" && pwd)/$(basename "${VIDEO_FILE}")"
        echo "Starting container sequential video verification on ${ABS_VIDEO}..."
        docker run --rm -it \
            -v "${PROJECT_ROOT}/models/private:/app/models/private:ro" \
            -v "${ABS_VIDEO}:/app/input.mp4:ro" \
            "${IMAGE_NAME}" \
            --config configs/runtime.yaml \
            --video /app/input.mp4 \
            --no-display
        ;;
    *)
        usage
        exit 1
        ;;
esac
