# Run `make help` to list targets.

.PHONY: help all run test docs docs-deps clean-docker clean-containers clean-images

# Default target (kept for backwards compatibility)
all: clean-docker

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

run:  ## Launch the Streamlit app
	streamlit run main.py

test:  ## Run the test suite
	python -m pytest -q

docs-deps:  ## Install the doc-rendering toolchain (WeasyPrint)
	pip install -r docs/requirements-docs.txt

docs:  ## Regenerate the HLD PDF from docs/hld.html
	python docs/build_pdf.py

# --- Docker housekeeping (pre-existing) ---
clean-docker:  ## Stop & remove all containers, then remove all images
	-docker stop $$(docker ps -aq)
	-docker rm $$(docker ps -aq)
	-docker rmi $$(docker images -q)

clean-containers:  ## Stop & remove all containers
	-docker stop $$(docker ps -aq)
	-docker rm $$(docker ps -aq)

clean-images:  ## Remove all images
	-docker rmi $$(docker images -q)
