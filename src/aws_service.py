import json
import logging
import os
import uuid
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class AWSService:
    def __init__(self):
        self.endpoint_url = os.getenv(
            "AWS_ENDPOINT_URL", "http://localhost:4566")
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

        config = {
            "endpoint_url": self.endpoint_url,
            "region_name": self.region,
            "aws_access_key_id": os.getenv(
                "AWS_ACCESS_KEY_ID",
                "test"),
            "aws_secret_access_key": os.getenv(
                "AWS_SECRET_ACCESS_KEY",
                "test"),
        }

        try:
            self.s3 = boto3.client("s3", **config)
            self.dynamodb = boto3.client("dynamodb", **config)
            self.sqs = boto3.client("sqs", **config)
            self.ssm = boto3.client("ssm", **config)
            self.lambda_client = boto3.client("lambda", **config)
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {e}")
            raise

        self.s3_bucket = os.getenv(
            "S3_BUCKET_NAME",
            "risk-assessment-documents")
        self.dynamodb_table = os.getenv(
            "DYNAMODB_TABLE_NAME", "risk-assessment-graphs")
        self.sqs_queue_name = os.getenv(
            "SQS_QUEUE_NAME", "document-processing-queue")

        try:
            response = self.sqs.get_queue_url(QueueName=self.sqs_queue_name)
            self.sqs_queue_url = response["QueueUrl"]
        except Exception as e:
            logger.warning(f"Could not get SQS queue URL: {e}")
            self.sqs_queue_url = None

    def upload_document_to_s3(self, file_content: bytes, filename: str) -> str:
        try:
            max_size_mb = int(os.getenv("S3_MAX_FILE_SIZE_MB", "50"))
            file_size_mb = len(file_content) / (1024 * 1024)

            if file_size_mb > max_size_mb:
                raise ValueError(
                    f"File size {file_size_mb:.1f}MB exceeds limit of {max_size_mb}MB"
                )

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            key = f"{os.getenv('S3_RAW_PREFIX', 'raw/')}{timestamp}_{filename}"

            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key=key,
                Body=file_content,
                ContentType="application/pdf",
                Metadata={
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "original_filename": filename,
                    "file_size_mb": str(round(file_size_mb, 2)),
                },
            )

            logger.info(f"Uploaded document to S3: {key}")
            return key

        except ClientError as e:
            logger.error(f"AWS Client error uploading to S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            raise

    def get_document_from_s3(self, key: str) -> bytes:
        try:
            response = self.s3.get_object(Bucket=self.s3_bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.error(f"Document not found in S3: {key}")
                raise FileNotFoundError(f"Document not found: {key}")
            logger.error(f"Error retrieving from S3: {e}")
            raise

    def move_to_processed(self, key: str) -> str:
        try:
            processed_key = key.replace(
                os.getenv("S3_RAW_PREFIX", "raw/"),
                os.getenv("S3_PROCESSED_PREFIX", "processed/"),
            )

            self.s3.copy_object(
                Bucket=self.s3_bucket,
                CopySource={"Bucket": self.s3_bucket, "Key": key},
                Key=processed_key,
                TaggingDirective="COPY",
                MetadataDirective="COPY",
            )

            self.s3.delete_object(Bucket=self.s3_bucket, Key=key)

            logger.info(f"Moved document to processed: {processed_key}")
            return processed_key

        except Exception as e:
            logger.error(f"Error moving to processed: {e}")
            return key

    def save_graph_metadata(self, graph_data: Dict[str, Any]) -> str:
        try:
            item = {
                "graph_id": {"S": graph_data["graph_id"]},
                "document_name": {"S": graph_data["document_name"]},
                "created_at": {"S": datetime.utcnow().isoformat()},
                "entity_count": {"N": str(graph_data["stats"]["total_entities"])},
                "relationship_count": {
                    "N": str(graph_data["stats"]["total_relationships"])
                },
                "extraction_method": {"S": graph_data["stats"]["extraction_method"]},
                "s3_key": {"S": graph_data.get("s3_key", "")},
                "processing_time": {
                    "N": str(round(graph_data.get("processing_time_seconds", 0), 2))
                },
                "chunk_percentage": {"N": str(graph_data.get("chunk_percentage", 100))},
                "status": {"S": "completed"},
                "ttl": {
                    "N": str(int((datetime.utcnow().timestamp() + 30 * 24 * 3600)))
                },
            }

            if "llm_provider" in graph_data:
                item["llm_provider"] = {"S": graph_data["llm_provider"]}

            self.dynamodb.put_item(TableName=self.dynamodb_table, Item=item)

            logger.info(f"Saved graph metadata: {graph_data['graph_id']}")
            return graph_data["graph_id"]

        except Exception as e:
            logger.error(f"Error saving to DynamoDB: {e}")
            raise

    def get_graph_metadata(self, graph_id: str) -> Optional[Dict]:
        try:
            response = self.dynamodb.get_item(
                TableName=self.dynamodb_table, Key={
                    "graph_id": {
                        "S": graph_id}})

            if "Item" not in response:
                return None

            item = response["Item"]
            return {
                "graph_id": item["graph_id"]["S"],
                "document_name": item["document_name"]["S"],
                "created_at": item["created_at"]["S"],
                "entity_count": int(
                    item["entity_count"]["N"]),
                "relationship_count": int(
                    item["relationship_count"]["N"]),
                "extraction_method": item["extraction_method"]["S"],
                "s3_key": item.get(
                    "s3_key",
                    {}).get(
                    "S",
                    ""),
                "processing_time": float(
                    item.get(
                        "processing_time",
                        {}).get(
                        "N",
                        0)),
                "chunk_percentage": int(
                    item.get(
                        "chunk_percentage",
                        {}).get(
                        "N",
                        100)),
                "status": item.get(
                    "status",
                    {}).get(
                    "S",
                    "unknown"),
                "llm_provider": item.get(
                    "llm_provider",
                    {}).get(
                    "S",
                    ""),
            }

        except Exception as e:
            logger.error(f"Error getting from DynamoDB: {e}")
            return None

    def list_recent_graphs(self, limit: int = 10) -> List[Dict]:
        try:
            response = self.dynamodb.scan(
                TableName=self.dynamodb_table,
                Limit=limit * 2,
                ProjectionExpression="graph_id, document_name, created_at, entity_count, relationship_count, extraction_method",
            )

            graphs = []
            for item in response.get("Items", []):
                try:
                    graphs.append(
                        {
                            "graph_id": item["graph_id"]["S"],
                            "document_name": item["document_name"]["S"],
                            "created_at": item["created_at"]["S"],
                            "entity_count": int(
                                item.get(
                                    "entity_count",
                                    {}).get(
                                    "N",
                                    0)),
                            "relationship_count": int(
                                item.get(
                                    "relationship_count",
                                    {}).get(
                                    "N",
                                    0)),
                            "extraction_method": item.get(
                                "extraction_method",
                                {}).get(
                                "S",
                                "Unknown"),
                        })
                except Exception as e:
                    logger.warning(f"Error parsing DynamoDB item: {e}")
                    continue

            graphs.sort(key=lambda x: x["created_at"], reverse=True)
            return graphs[:limit]

        except Exception as e:
            logger.error(f"Error listing graphs: {e}")
            return []

    def send_processing_message(self, message_data: Dict) -> str:
        try:
            if not self.sqs_queue_url:
                raise Exception("SQS queue URL not available")

            message_id = str(uuid.uuid4())
            message_data["message_id"] = message_id
            message_data["timestamp"] = datetime.utcnow().isoformat()

            dedup_id = f"{message_data['graph_id']}_{message_data['timestamp']}"

            response = self.sqs.send_message(
                QueueUrl=self.sqs_queue_url,
                MessageBody=json.dumps(message_data),
                MessageAttributes={
                    "document_type": {
                        "StringValue": message_data.get("document_type", "pdf"),
                        "DataType": "String",
                    },
                    "priority": {
                        "StringValue": message_data.get("priority", "normal"),
                        "DataType": "String",
                    },
                    "extraction_method": {
                        "StringValue": message_data.get("extraction_method", "unknown"),
                        "DataType": "String",
                    },
                },
            )

            logger.info(f"Sent message to SQS: {message_id}")
            return response["MessageId"]

        except Exception as e:
            logger.error(f"Error sending to SQS: {e}")
            return ""

    def get_api_key(self, provider: str) -> Optional[str]:
        try:
            if provider.lower() == "openai":
                param_name = os.getenv(
                    "SSM_OPENAI_KEY", "/risk-assessment/api-keys/openai"
                )
            elif provider.lower() == "anthropic":
                param_name = os.getenv(
                    "SSM_ANTHROPIC_KEY", "/risk-assessment/api-keys/anthropic"
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")

            response = self.ssm.get_parameter(
                Name=param_name, WithDecryption=True)

            value = response["Parameter"]["Value"]

            if value.startswith("sk-placeholder"):
                return None

            return value

        except self.ssm.exceptions.ParameterNotFound:
            logger.warning(f"API key not found in SSM for {provider}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving API key from SSM: {e}")
            return None

    def store_api_key(self, provider: str, api_key: str) -> bool:
        try:
            if not api_key or len(api_key) < 20:
                raise ValueError("Invalid API key format")

            if provider.lower() == "openai":
                param_name = os.getenv(
                    "SSM_OPENAI_KEY", "/risk-assessment/api-keys/openai"
                )
            elif provider.lower() == "anthropic":
                param_name = os.getenv(
                    "SSM_ANTHROPIC_KEY", "/risk-assessment/api-keys/anthropic"
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")

            self.ssm.put_parameter(
                Name=param_name,
                Value=api_key,
                Type="SecureString",
                Overwrite=True,
                Description=f"{provider.capitalize()} API key for Risk Assessment KG",
            )

            logger.info(f"Stored API key in SSM for {provider}")
            return True

        except Exception as e:
            logger.error(f"Error storing API key in SSM: {e}")
            return False

    def invoke_async_processing(
            self,
            function_name: str,
            payload: Dict) -> str:
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType="Event",
                Payload=json.dumps(payload),
            )

            request_id = response["ResponseMetadata"]["RequestId"]
            logger.info(
                f"Invoked Lambda function {function_name}: {request_id}")
            return request_id

        except Exception as e:
            logger.error(f"Error invoking Lambda: {e}")
            return ""

    def health_check(self) -> Dict[str, bool]:
        health = {
            "s3": False,
            "dynamodb": False,
            "sqs": False,
            "ssm": False,
            "lambda": False,
        }

        try:
            self.s3.head_bucket(Bucket=self.s3_bucket)
            health["s3"] = True
        except Exception as e:
            logger.debug(f"S3 health check failed: {e}")

        try:
            self.dynamodb.describe_table(TableName=self.dynamodb_table)
            health["dynamodb"] = True
        except Exception as e:
            logger.debug(f"DynamoDB health check failed: {e}")

        try:
            if self.sqs_queue_url:
                self.sqs.get_queue_attributes(
                    QueueUrl=self.sqs_queue_url, AttributeNames=["QueueArn"]
                )
                health["sqs"] = True
        except Exception as e:
            logger.debug(f"SQS health check failed: {e}")

        try:
            self.ssm.describe_parameters(MaxResults=1)
            health["ssm"] = True
        except Exception as e:
            logger.debug(f"SSM health check failed: {e}")

        try:
            self.lambda_client.list_functions(MaxItems=1)
            health["lambda"] = True
        except Exception as e:
            logger.debug(f"Lambda health check failed: {e}")

        return health

    def get_service_stats(self) -> Dict[str, Any]:
        stats = {"s3": {}, "dynamodb": {}, "sqs": {}}

        try:
            response = self.s3.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=os.getenv("S3_PROCESSED_PREFIX", "processed/"),
            )
            stats["s3"]["processed_documents"] = response.get("KeyCount", 0)
        except BaseException:
            pass

        try:
            response = self.dynamodb.describe_table(
                TableName=self.dynamodb_table)
            stats["dynamodb"]["item_count"] = response["Table"].get(
                "ItemCount", 0)
        except BaseException:
            pass

        try:
            if self.sqs_queue_url:
                response = self.sqs.get_queue_attributes(
                    QueueUrl=self.sqs_queue_url,
                    AttributeNames=["ApproximateNumberOfMessages"],
                )
                stats["sqs"]["messages_in_queue"] = int(
                    response["Attributes"].get(
                        "ApproximateNumberOfMessages", 0))
        except BaseException:
            pass

        return stats
