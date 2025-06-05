import json
import logging
import os
from datetime import datetime

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client(
    "s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
)
dynamodb = boto3.client(
    "dynamodb",
    endpoint_url=os.getenv(
        "AWS_ENDPOINT_URL",
        "http://localhost:4566"))
sns = boto3.client(
    "sns", endpoint_url=os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
)

S3_BUCKET = os.getenv("S3_BUCKET_NAME", "risk-assessment-documents")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE_NAME", "risk-assessment-graphs")


def lambda_handler(event, context):
    """
    Process document from SQS message

    Expected message format:
    {
        "graph_id": "uuid",
        "document_key": "s3_key",
        "document_name": "filename.pdf",
        "extraction_method": "Rule-Based|LLM-Based",
        "entity_count": 10,
        "relationship_count": 20
    }
    """

    processed_count = 0
    failed_count = 0

    for record in event.get("Records", []):
        try:
            message_body = json.loads(record["body"])
            graph_id = message_body.get("graph_id")
            document_key = message_body.get("document_key")
            document_name = message_body.get("document_name")
            extraction_method = message_body.get("extraction_method")

            logger.info(
                f"Processing document: {document_name} (Graph ID: {graph_id})")

            processing_result = process_document(
                graph_id=graph_id,
                document_key=document_key,
                document_name=document_name,
                extraction_method=extraction_method,
                metadata=message_body,
            )

            update_graph_status(graph_id, processing_result)

            if processing_result["status"] == "completed":
                send_completion_notification(
                    graph_id, document_name, processing_result)

            processed_count += 1
            logger.info(f"Successfully processed: {document_name}")

        except Exception as e:
            failed_count += 1
            logger.error(f"Error processing record: {str(e)}", exc_info=True)

            if "graph_id" in locals():
                update_graph_status(
                    graph_id,
                    {
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "Batch processing completed",
                "processed": processed_count,
                "failed": failed_count,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ),
    }


def process_document(
    graph_id, document_key, document_name, extraction_method, metadata
):
    """
    Perform additional processing on the document
    """
    try:
        processing_result = {
            "status": "processing",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {},
        }

        if document_key:
            try:
                s3.head_object(Bucket=S3_BUCKET, Key=document_key)
                processing_result["metadata"]["s3_validated"] = True
            except BaseException:
                processing_result["metadata"]["s3_validated"] = False

        entity_count = metadata.get("entity_count", 0)
        relationship_count = metadata.get("relationship_count", 0)

        quality_score = calculate_quality_score(
            entity_count, relationship_count)
        processing_result["metadata"]["quality_score"] = quality_score

        if quality_score < 0.5:
            processing_result["metadata"]["needs_review"] = True
            processing_result["metadata"]["review_reason"] = "Low quality score"
        elif entity_count < 5:
            processing_result["metadata"]["needs_review"] = True
            processing_result["metadata"][
                "review_reason"
            ] = "Too few entities extracted"
        else:
            processing_result["metadata"]["needs_review"] = False

        processing_result["processed_at"] = datetime.utcnow().isoformat()
        processing_result["status"] = "completed"

        if extraction_method == "LLM-Based":
            processing_result["metadata"]["processing_type"] = "advanced"
            processing_result["metadata"]["confidence_level"] = "high"
        else:
            processing_result["metadata"]["processing_type"] = "basic"
            processing_result["metadata"]["confidence_level"] = "medium"

        return processing_result

    except Exception as e:
        logger.error(f"Error in process_document: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


def calculate_quality_score(entity_count, relationship_count):
    entity_score = min(entity_count / 50, 1.0)
    relationship_score = min(
        relationship_count / 100,
        1.0)

    quality_score = 0.6 * entity_score + 0.4 * relationship_score

    return round(quality_score, 2)


def update_graph_status(graph_id, processing_result):
    """
    Update graph status in DynamoDB
    """
    try:
        update_expression = "SET #status = :status, #processed_at = :processed_at"
        expression_values = {
            ":status": {"S": processing_result["status"]},
            ":processed_at": {
                "S": processing_result.get(
                    "processed_at", datetime.utcnow().isoformat()
                )
            },
        }
        expression_names = {
            "#status": "status",
            "#processed_at": "processed_at"}

        if "metadata" in processing_result:
            update_expression += ", #metadata = :metadata"
            expression_values[":metadata"] = {
                "S": json.dumps(processing_result["metadata"])
            }
            expression_names["#metadata"] = "processing_metadata"

        if "error" in processing_result:
            update_expression += ", #error = :error"
            expression_values[":error"] = {"S": processing_result["error"]}
            expression_names["#error"] = "error_message"

        dynamodb.update_item(
            TableName=DYNAMODB_TABLE,
            Key={"graph_id": {"S": graph_id}},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_values,
            ExpressionAttributeNames=expression_names,
        )

        logger.info(
            f"Updated graph status for {graph_id}: {processing_result['status']}"
        )

    except Exception as e:
        logger.error(f"Error updating DynamoDB: {str(e)}")


def send_completion_notification(graph_id, document_name, processing_result):
    """
    Send completion notification via SNS (if topic exists)
    """
    try:
        topic_arn = os.getenv("SNS_TOPIC_ARN")
        if not topic_arn:
            return

        message = {
            "graph_id": graph_id,
            "document_name": document_name,
            "status": processing_result["status"],
            "quality_score": processing_result.get("metadata", {}).get(
                "quality_score", "N/A"
            ),
            "needs_review": processing_result.get("metadata", {}).get(
                "needs_review", False
            ),
            "processed_at": processing_result.get("processed_at"),
            "processing_type": processing_result.get("metadata", {}).get(
                "processing_type", "unknown"
            ),
        }

        sns.publish(
            TopicArn=topic_arn,
            Subject=f"Document Processing Complete: {document_name}",
            Message=json.dumps(message, indent=2),
        )

        logger.info(f"Sent completion notification for {graph_id}")

    except Exception as e:
        logger.warning(f"Could not send notification: {str(e)}")
