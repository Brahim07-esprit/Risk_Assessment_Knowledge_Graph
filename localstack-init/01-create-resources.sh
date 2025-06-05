#!/bin/bash
set -e

echo "Initializing LocalStack resources..."

# Give LocalStack more time to fully start
sleep 10

# Wait for LocalStack to be ready
echo "Waiting for LocalStack to be ready..."
until awslocal s3 ls &>/dev/null; do
  echo "Waiting for LocalStack..."
  sleep 2
done

echo "LocalStack is ready!"

# Create S3 bucket
echo "Creating S3 bucket..."
awslocal s3 mb s3://risk-assessment-documents || true
awslocal s3api put-bucket-versioning \
  --bucket risk-assessment-documents \
  --versioning-configuration Status=Enabled

# Create bucket folders
awslocal s3api put-object --bucket risk-assessment-documents --key raw/
awslocal s3api put-object --bucket risk-assessment-documents --key processed/

# Create DynamoDB table
echo "Creating DynamoDB table..."
awslocal dynamodb create-table \
  --table-name risk-assessment-graphs \
  --attribute-definitions \
    AttributeName=graph_id,AttributeType=S \
    AttributeName=document_name,AttributeType=S \
    AttributeName=created_at,AttributeType=S \
  --key-schema \
    AttributeName=graph_id,KeyType=HASH \
  --global-secondary-indexes \
    '[{"IndexName":"document-index","KeySchema":[{"AttributeName":"document_name","KeyType":"HASH"},{"AttributeName":"created_at","KeyType":"RANGE"}],"Projection":{"ProjectionType":"ALL"},"ProvisionedThroughput":{"ReadCapacityUnits":5,"WriteCapacityUnits":5}}]' \
  --provisioned-throughput \
    ReadCapacityUnits=5,WriteCapacityUnits=5 \
  || true

# Enable TTL on DynamoDB table
echo "Enabling TTL on DynamoDB table..."
awslocal dynamodb update-time-to-live \
  --table-name risk-assessment-graphs \
  --time-to-live-specification "Enabled=true,AttributeName=ttl" \
  || true

# Create SQS queues
echo "Creating SQS queues..."
# Dead Letter Queue
awslocal sqs create-queue \
  --queue-name document-processing-dlq \
  --attributes MessageRetentionPeriod=1209600 || true

# Get DLQ ARN
DLQ_ARN=$(awslocal sqs get-queue-attributes \
  --queue-url http://localhost:4566/000000000000/document-processing-dlq \
  --attribute-names QueueArn \
  --query 'Attributes.QueueArn' \
  --output text)

# Main Queue with DLQ
awslocal sqs create-queue \
  --queue-name document-processing-queue \
  --attributes "{\"MessageRetentionPeriod\":\"86400\",\"VisibilityTimeout\":\"300\",\"RedrivePolicy\":\"{\\\"deadLetterTargetArn\\\":\\\"${DLQ_ARN}\\\",\\\"maxReceiveCount\\\":\\\"3\\\"}\"}" \
  || true

# Create SSM Parameters
echo "Creating SSM parameters..."
# Store placeholder API keys
awslocal ssm put-parameter \
  --name /risk-assessment/api-keys/openai \
  --value "sk-placeholder-openai-key" \
  --type SecureString \
  --description "OpenAI API key placeholder" \
  --overwrite || true

awslocal ssm put-parameter \
  --name /risk-assessment/api-keys/anthropic \
  --value "sk-placeholder-anthropic-key" \
  --type SecureString \
  --description "Anthropic API key placeholder" \
  --overwrite || true

awslocal ssm put-parameter \
  --name /risk-assessment/credentials/neo4j-password \
  --value "${NEO4J_PASSWORD:-GenerateSecurePasswordHere123!@#}" \
  --type SecureString \
  --description "Neo4j database password" \
  --overwrite || true

# Create Lambda function
echo "Creating Lambda function..."

# Copy Lambda function if it exists
if [ -f "/docker-entrypoint-initaws.d/lambda_function.py" ]; then
  cp /docker-entrypoint-initaws.d/lambda_function.py /tmp/lambda_function.py
else
  # Use inline function if file doesn't exist
  cat > /tmp/lambda_function.py << 'EOF'
import json
import boto3
from datetime import datetime

def lambda_handler(event, context):
    """Process document from SQS message"""
    print(f"Received event: {json.dumps(event)}")
    
    # Extract message body
    for record in event.get('Records', []):
        message_body = json.loads(record['body'])
        document_key = message_body.get('document_key')
        graph_id = message_body.get('graph_id')
        
        print(f"Processing document: {document_key} for graph: {graph_id}")
        
        # Here would be actual processing logic
        # For now, just log and return success
        
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Document processed successfully',
            'timestamp': datetime.utcnow().isoformat()
        })
    }
EOF
fi

# Create deployment package
cd /tmp
zip lambda_function.zip lambda_function.py

# Create IAM role for Lambda
echo "Creating IAM role for Lambda..."
awslocal iam create-role \
  --role-name lambda-execution-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }' || true

# Attach policies to role
awslocal iam attach-role-policy \
  --role-name lambda-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole || true

# Create Lambda function
awslocal lambda create-function \
  --function-name process-document \
  --runtime python3.9 \
  --role arn:aws:iam::000000000000:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --timeout 300 \
  --memory-size 512 \
  --environment Variables="{
    AWS_ENDPOINT_URL=http://localhost:4566,
    S3_BUCKET_NAME=risk-assessment-documents,
    DYNAMODB_TABLE_NAME=risk-assessment-graphs
  }" \
  || true

# Create SQS to Lambda event source mapping
QUEUE_ARN=$(awslocal sqs get-queue-attributes \
  --queue-url http://localhost:4566/000000000000/document-processing-queue \
  --attribute-names QueueArn \
  --query 'Attributes.QueueArn' \
  --output text)

awslocal lambda create-event-source-mapping \
  --function-name process-document \
  --event-source-arn $QUEUE_ARN \
  --batch-size 1 \
  || true

# Create SNS topic for notifications (optional)
echo "Creating SNS topic..."
awslocal sns create-topic \
  --name risk-assessment-notifications || true

echo "LocalStack initialization complete!"

# List created resources
echo -e "\n=== Created Resources ==="
echo "S3 Buckets:"
awslocal s3 ls

echo -e "\nDynamoDB Tables:"
awslocal dynamodb list-tables

echo -e "\nSQS Queues:"
awslocal sqs list-queues

echo -e "\nSSM Parameters:"
awslocal ssm describe-parameters

echo -e "\nLambda Functions:"
awslocal lambda list-functions --query 'Functions[*].FunctionName'

echo -e "\nSNS Topics:"
awslocal sns list-topics --query 'Topics[*].TopicArn' 