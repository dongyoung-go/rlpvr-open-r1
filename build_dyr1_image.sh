#!/bin/sh

DOCKER_BASE_USERNAME="ro_base-nvidia"
DOCKER_BASE_PASSWORD="Nvidia1@#"

echo "$DOCKER_BASE_PASSWORD" | docker login reg.navercorp.com -u "$DOCKER_BASE_USERNAME" --password-stdin

NAVER_ID="KR21637"
IMAGE_NAME="dy-test"
TAG="0.0.1"
docker build --no-cache --label com.navercorp.image.author=${NAVER_ID} -t gaim.n3r.reg.navercorp.com/mlx/${IMAGE_NAME}:${TAG} -f Dockerfile .

DOCKER_N3R_USERNAME="rw_dongyoung"
DOCKER_N3R_PASSWORD="Gaim123!@#"
echo "$DOCKER_N3R_PASSWORD" | docker login gaim.n3r.reg.navercorp.com -u "$DOCKER_N3R_USERNAME" --password-stdin
docker push gaim.n3r.reg.navercorp.com/mlx/${IMAGE_NAME}:${TAG}
