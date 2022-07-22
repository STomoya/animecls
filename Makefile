_v1_bin := $(shell command -v docker-compose 2> /dev/null)
ifdef _v1_bin
COMPOSE_COMMAND := "docker-compose"
else
COMPOSE_COMMAND := "docker compose"
endif

new:
	touch animecls/models/${name}.py
	mkdir results/${name}
	cp results/template.md results/${name}/README.md
	mkdir results/${name}/imagenet_sketch
	mkdir results/${name}/animeface

run:
	${COMPOSE_COMMAND} run --rm torch python -m animecls ${args}

drun:
	${COMPOSE_COMMAND} run --rm -d torch python -m animecls ${args}
