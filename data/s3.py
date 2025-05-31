import logging

import boto3
import os
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from GitHub Actions or your local env
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_BUCKET = "adil-ml-models-prod"


def download_model_from_s3(key, local_path):
    if not os.path.exists(local_path):
        print("Downloading Model from s3")
        s3 = boto3.client('s3',
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                          region_name=AWS_REGION)

        s3.download_file(MODEL_BUCKET, key, local_path)
        logger.info("Model downloaded")
    else:
        logger.info("Model found locally.")
