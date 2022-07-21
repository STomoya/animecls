_v1_bin := $(shell command -v docker-compose 2> /dev/null)
ifdef _v1_bin
COMPOSE_COMMAND := "docker-compose"
else
COMPOSE_COMMAND := "docker compose"
endif

new:
	mkdir implementations/${name}
