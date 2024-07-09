docker run -it \
-v "$(pwd)/../":/workspace \
-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
-e DISPLAY=unix$DISPLAY \
--net=host \
--gpus all \
--privileged \
--name kalman_study \
leolixingyou/kalman_2d:20240709
