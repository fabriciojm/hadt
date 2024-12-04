REGION=europe-west1
GCP_PROJECT_ID=arrhythmia-442911
DOCKER_IMAGE_NAME=arrhythmia
TAG=prod
DOCKER_CONTAINER_NAME=arrhythmia
ARTIFACT_REPO_NAME=arrhythmia
ARTIFACT_REPO_LOCATION = $(REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(ARTIFACT_REPO_NAME)
ARTIFACT_IMAGE_NAME=$(ARTIFACT_REPO_LOCATION)/$(DOCKER_IMAGE_NAME):$(TAG)
PORT=8080
PATH_SERVICE_ACCOUNT_KEY=service-account.json

BUCKET_NAME=arrhythmia_raw_data
BUCKET_NAME_MODELS=arrhythmia-models

###################### TEST IMAGE DOCKER #############################
local_docker_build:
	docker build -t $(DOCKER_IMAGE_NAME) .

local_docker_run:
	docker run -p $(PORT):$(PORT) \
	-e PORT=$(PORT) \
	-e BUCKET_NAME=$(BUCKET_NAME) \
	-e BUCKET_NAME_MODELS=$(BUCKET_NAME_MODELS) \
	-e GCP_PROJECT_ID=$(GCP_PROJECT_ID) \
	-e GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json \
	--name $(DOCKER_CONTAINER_NAME) \
	$(DOCKER_IMAGE_NAME)

# local_docker_run_it:
# 	docker run -it \
# 	-p $(PORT):$(PORT) \
#  	-e PORT=$(PORT) \
# 	-e GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json \
# 	-e BUCKET_NAME=$(BUCKET_NAME) \
# 	-e BUCKET_NAME_MODELS=$(BUCKET_NAME_MODELS) \
# 	-e GCP_PROJECT_ID=$(GCP_PROJECT_ID) \
# 	--name $(DOCKER_CONTAINER_NAME) \
#   --entrypoint=/bin/bash \
#   $(DOCKER_IMAGE_NAME)

###################### BUILD IMAGE FOR ARTIFACT #############################

# mac arm64 based
local_docker_build_artifact:
	docker buildx build --no-cache --platform linux/amd64 -t $(ARTIFACT_IMAGE_NAME) .

cloud_docker_push_to_artifact:
	docker push $(ARTIFACT_IMAGE_NAME)

# Deploy Application to Google Cloud Run
cloud_run:
	gcloud run deploy arrhythmia \
		--image $(ARTIFACT_IMAGE_NAME) \
		--platform managed \
		--region $(REGION) \
		--allow-unauthenticated \
		--memory 2Gi \
		--max-instances 10 \
		--concurrency 80 \
		--set-env-vars " \
			GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json, \
			BUCKET_NAME=$(BUCKET_NAME), \
			BUCKET_NAME_MODELS=$(BUCKET_NAME_MODELS), \
			GCP_PROJECT_ID=$(GCP_PROJECT_ID)"
