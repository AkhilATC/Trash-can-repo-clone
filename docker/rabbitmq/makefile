# --------------------------------------------
# Makefile for RabbitMQ (Docker)
# --------------------------------------------

CONTAINER_NAME = rabbitmq
IMAGE = rabbitmq:3-management
RABBITMQ_PORT = 5672
RABBITMQ_UI_PORT = 15672

# Default target
.PHONY: help
help:
	@echo ""
	@echo "RabbitMQ Docker Commands"
	@echo "------------------------"
	@echo "make up       - Start RabbitMQ container"
	@echo "make down     - Stop and remove container"
	@echo "make logs     - View container logs"
	@echo "make restart  - Restart container"
	@echo "make status   - Check if RabbitMQ is running"
	@echo ""

# Run RabbitMQ container
.PHONY: up
up:
	@echo "🚀 Starting RabbitMQ container..."
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(RABBITMQ_PORT):5672 \
		-p $(RABBITMQ_UI_PORT):15672 \
		$(IMAGE)
	@echo "✅ RabbitMQ is running!"
	@echo "Management UI: http://localhost:$(RABBITMQ_UI_PORT)"
	@echo "Login: guest / guest"

# Stop and remove container
.PHONY: down
down:
	@echo "🛑 Stopping and removing RabbitMQ container..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
	@echo "✅ Container removed."

# View logs
.PHONY: logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Restart container
.PHONY: restart
restart: down up

# Check container status
.PHONY: status
status:
	@docker ps --filter "name=$(CONTAINER_NAME)"

