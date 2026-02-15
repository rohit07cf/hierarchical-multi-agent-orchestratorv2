# Default target
all: clean-docker

# Stop and remove all containers, then remove all images
clean-docker:
	-docker stop $$(docker ps -aq)
	-docker rm $$(docker ps -aq)
	-docker rmi $$(docker images -q)

# Optional: specifically just for containers
clean-containers:
	-docker stop $$(docker ps -aq)
	-docker rm $$(docker ps -aq)

# Optional: specifically just for images
clean-images:
	-docker rmi $$(docker images -q)