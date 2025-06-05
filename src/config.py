import os
from typing import Dict


class Settings:
    """Application settings"""

    # App Settings
    APP_NAME: str = os.getenv("APP_NAME", "risk-assessment-kg")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_PORT: int = int(os.getenv("APP_PORT", "8501"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Neo4j Settings
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password123")

    # AWS Settings
    AWS_ENDPOINT_URL: str = os.getenv(
        "AWS_ENDPOINT_URL", "http://localhost:4566")
    AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "test")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "test")

    # S3 Settings
    S3_BUCKET_NAME: str = os.getenv(
        "S3_BUCKET_NAME",
        "risk-assessment-documents")
    S3_PROCESSED_PREFIX: str = os.getenv("S3_PROCESSED_PREFIX", "processed/")
    S3_RAW_PREFIX: str = os.getenv("S3_RAW_PREFIX", "raw/")

    # DynamoDB Settings
    DYNAMODB_TABLE_NAME: str = os.getenv(
        "DYNAMODB_TABLE_NAME", "risk-assessment-graphs"
    )
    DYNAMODB_GSI_NAME: str = os.getenv("DYNAMODB_GSI_NAME", "document-index")

    # SQS Settings
    SQS_QUEUE_NAME: str = os.getenv(
        "SQS_QUEUE_NAME",
        "document-processing-queue")
    SQS_DLQ_NAME: str = os.getenv("SQS_DLQ_NAME", "document-processing-dlq")
    SQS_VISIBILITY_TIMEOUT: int = int(
        os.getenv("SQS_VISIBILITY_TIMEOUT", "300"))
    SQS_MAX_RETRIES: int = int(os.getenv("SQS_MAX_RETRIES", "3"))

    # Lambda Settings
    LAMBDA_FUNCTION_NAME: str = os.getenv(
        "LAMBDA_FUNCTION_NAME", "process-document")
    LAMBDA_TIMEOUT: int = int(os.getenv("LAMBDA_TIMEOUT", "300"))
    LAMBDA_MEMORY_SIZE: int = int(os.getenv("LAMBDA_MEMORY_SIZE", "512"))

    # SSM Parameter Store Keys
    SSM_OPENAI_KEY: str = os.getenv(
        "SSM_OPENAI_KEY", "/risk-assessment/api-keys/openai"
    )
    SSM_ANTHROPIC_KEY: str = os.getenv(
        "SSM_ANTHROPIC_KEY", "/risk-assessment/api-keys/anthropic"
    )
    SSM_NEO4J_PASSWORD: str = os.getenv(
        "SSM_NEO4J_PASSWORD", "/risk-assessment/credentials/neo4j-password"
    )

    # Processing Settings
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "2000"))
    DEFAULT_CHUNK_PERCENTAGE: int = int(
        os.getenv("DEFAULT_CHUNK_PERCENTAGE", "50"))
    CONFIDENCE_THRESHOLD: float = float(
        os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    SPACY_MAX_LENGTH: int = int(os.getenv("SPACY_MAX_LENGTH", "100000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Graph Visualization Settings
    GRAPH_NODE_SIZE_MIN: int = int(os.getenv("GRAPH_NODE_SIZE_MIN", "5"))
    GRAPH_NODE_SIZE_MAX: int = int(os.getenv("GRAPH_NODE_SIZE_MAX", "20"))
    GRAPH_PHYSICS_ENABLED: bool = (
        os.getenv("GRAPH_PHYSICS_ENABLED", "true").lower() == "true"
    )
    GRAPH_STABILIZATION_TIME: int = int(
        os.getenv("GRAPH_STABILIZATION_TIME", "5000"))

    # Monitoring Settings
    ENABLE_METRICS: bool = os.getenv(
        "ENABLE_METRICS", "true").lower() == "true"
    METRICS_NAMESPACE: str = os.getenv("METRICS_NAMESPACE", "RiskAssessmentKG")
    CLOUDWATCH_LOG_GROUP: str = os.getenv(
        "CLOUDWATCH_LOG_GROUP", "/aws/lambda/risk-assessment-kg"
    )

    # Graph node colors
    GRAPH_NODE_COLORS: Dict[str, str] = {
        "RISK": "#E74C3C",
        "CONTROL": "#27AE60",
        "ASSET": "#3498DB",
        "STAKEHOLDER": "#F39C12",
        "IMPACT": "#E67E22",
        "COMPLIANCE": "#9B59B6",
        "PROCESS": "#1ABC9C",
        "THREAT": "#C0392B",
        "VULNERABILITY": "#D35400",
    }

    # Entity types for extraction
    ENTITY_TYPES = [
        "RISK",
        "CONTROL",
        "ASSET",
        "STAKEHOLDER",
        "IMPACT",
        "COMPLIANCE",
        "PROCESS",
        "THREAT",
        "VULNERABILITY",
    ]

    # Relationship types
    RELATIONSHIP_TYPES = [
        "MITIGATES",
        "CAUSES",
        "AFFECTS",
        "OWNS",
        "REQUIRES",
        "IMPLEMENTS",
        "MONITORS",
        "COMPLIES_WITH",
    ]


# Create singleton instance
settings = Settings()
