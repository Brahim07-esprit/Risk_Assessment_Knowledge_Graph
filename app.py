import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.aws_service import AWSService
from src.config import settings
from src.document_processor import DocumentProcessor
from src.graph_evaluator import GraphEvaluator
from src.graph_generator import GraphGenerator
from src.llm_graph_generator import LLMGraphGenerator
from src.neo4j_service import Neo4jService
from src.security import SecurityManager, get_user_id, sanitize_api_key
from src.visualizer import GraphVisualizer

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


st.set_page_config(
    page_title="Risk Assessment Knowledge Graph",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def init_services():
    try:
        doc_processor = DocumentProcessor()
        graph_generator = GraphGenerator()
        neo4j_service = Neo4jService()
        visualizer = GraphVisualizer()
        evaluator = GraphEvaluator()
        aws_service = AWSService()
        security_manager = SecurityManager()
        return (
            doc_processor,
            graph_generator,
            neo4j_service,
            visualizer,
            evaluator,
            aws_service,
            security_manager,
        )
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        st.error(f"Failed to initialize services: {str(e)}")
        st.stop()


def main():
    st.title("🎯 Risk Assessment Knowledge Graph Extractor")
    st.markdown(
        "Transform risk assessment documents into interactive knowledge graphs using AI"
    )

    services = init_services()
    if not services:
        return

    (
        doc_processor,
        graph_generator,
        neo4j_service,
        visualizer,
        evaluator,
        aws_service,
        security_manager,
    ) = services

    user_id = get_user_id()

    with st.sidebar:
        st.header("📁 Document Processing")

        st.subheader("🤖 Extraction Method")
        extraction_method = st.radio(
            "Choose extraction method:",
            ["Rule-Based (Fast, Free)", "LLM-Based (Intelligent, Requires API Key)"],
            help="Rule-based uses patterns, LLM-based uses GPT-4 or Claude for better accuracy",
        )

        llm_provider = None
        api_key = None
        chunk_percentage = 100

        if "LLM-Based" in extraction_method:
            llm_provider = st.selectbox(
                "LLM Provider:", ["OpenAI (GPT-4o)", "Anthropic (Claude)"]
            )

            env_key = (
                "OPENAI_API_KEY" if "OpenAI" in llm_provider else "ANTHROPIC_API_KEY")
            provider = "openai" if "OpenAI" in llm_provider else "anthropic"

            ssm_key = aws_service.get_api_key(provider)

            if ssm_key:
                st.success(f"✅ Using {llm_provider} API key from SSM")
                api_key = ssm_key
                os.environ[env_key] = ssm_key
            else:
                stored_key = os.getenv(env_key)

                if stored_key and stored_key != "sk-placeholder":
                    st.success(
                        f"✅ Using {llm_provider} API key from environment")
                    api_key = stored_key
                else:
                    api_key = st.text_input(
                        f"{llm_provider} API Key:",
                        type="password",
                        help=f"Enter your {llm_provider} API key",
                    )

                    if api_key:
                        if sanitize_api_key(api_key):
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(
                                    "💾 Save to SSM", use_container_width=True
                                ):
                                    if aws_service.store_api_key(
                                            provider, api_key):
                                        st.success("✅ API key saved to SSM")
                                        os.environ[env_key] = api_key
                                    else:
                                        st.error("Failed to save to SSM")
                            with col2:
                                os.environ[env_key] = api_key
                                st.info("Set for session")
                        else:
                            st.error("Invalid API key format")

            st.divider()
            chunk_percentage = st.slider(
                "📊 Document Processing %",
                min_value=10,
                max_value=100,
                value=int(
                    settings.DEFAULT_CHUNK_PERCENTAGE),
                step=10,
                help="Control how much of the document to process with LLM (affects API costs)",
            )

            st.info(
                f"Will process approximately {chunk_percentage}% of document chunks"
            )

            if api_key:
                cost_estimate = estimate_processing_cost(
                    chunk_percentage, llm_provider)
                st.caption(f"💰 Estimated cost: ${cost_estimate:.3f}")

        st.divider()

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a risk assessment document in PDF format",
        )

        if uploaded_file is not None:
            validation_result = security_manager.validate_file_upload(
                uploaded_file, uploaded_file.name
            )

            if validation_result["valid"]:
                st.success(f"✅ File validated: {uploaded_file.name}")
                st.text(
                    f"Size: {validation_result['file_info']['size_mb']:.2f} MB")

                for warning in validation_result["warnings"]:
                    st.warning(f"⚠️ {warning}")

                if "LLM-Based" in extraction_method:
                    if not api_key:
                        st.error(
                            "⚠️ Please enter an API key to use LLM extraction")
                    else:
                        st.info(
                            f"Using {llm_provider} for intelligent extraction")
                else:
                    st.info("Using rule-based pattern matching")

                if st.button(
                    "🚀 Process Document",
                    type="primary",
                        use_container_width=True):
                    if not security_manager.check_rate_limit(
                            user_id, "process"):
                        st.error(
                            "⚠️ Rate limit exceeded. Please wait a moment before processing another document."
                        )
                    elif "LLM-Based" in extraction_method and not api_key:
                        st.error(
                            "Please provide an API key for LLM-based extraction")
                    else:
                        process_document(
                            uploaded_file,
                            doc_processor,
                            graph_generator,
                            neo4j_service,
                            extraction_method,
                            llm_provider,
                            evaluator,
                            (
                                chunk_percentage
                                if "LLM-Based" in extraction_method
                                else 100
                            ),
                            aws_service,
                            security_manager,
                            user_id,
                        )
            else:
                st.error("❌ File validation failed:")
                for error in validation_result["errors"]:
                    st.error(f"• {error}")

        st.divider()

        st.header("🎛️ Graph Controls")

        if "current_graph" in st.session_state and st.session_state.current_graph:
            node_size = st.slider("Node Size", 5, 20, 10)
            link_distance = st.slider("Link Distance", 50, 200, 100)

            entity_types = st.session_state.current_graph.get(
                "entity_types", [])
            selected_types = st.multiselect(
                "Filter Entity Types", entity_types, default=entity_types
            )

            if st.button("🔄 Refresh Graph"):
                st.rerun()

        st.divider()

        st.header("☁️ AWS Services Status")

        with st.spinner("Checking services..."):
            health_status = aws_service.health_check()
            service_stats = aws_service.get_service_stats()

        status_icons = {True: "✅", False: "❌"}

        for service, is_healthy in health_status.items():
            st.write(f"{status_icons[is_healthy]} {service.upper()}")

        if any(service_stats.values()):
            st.caption("📊 Service Statistics:")
            if service_stats["dynamodb"].get("item_count"):
                st.caption(
                    f"• Graphs stored: {service_stats['dynamodb']['item_count']}"
                )
            if service_stats["s3"].get("processed_documents"):
                st.caption(
                    f"• Documents processed: {service_stats['s3']['processed_documents']}"
                )

        if st.button("🔄 Refresh Status", use_container_width=True):
            st.cache_resource.clear()
            rerun()

        st.divider()
        st.header("📊 Recent Graphs")

        recent_graphs = aws_service.list_recent_graphs(limit=5)

        if recent_graphs:
            for graph in recent_graphs:
                with st.expander(f"{graph['document_name'][:30]}..."):
                    st.write(f"**Graph ID:** {graph['graph_id']}")
                    st.write(f"**Created:** {graph['created_at']}")
                    st.write(f"**Entities:** {graph['entity_count']}")
                    st.write(
                        f"**Relationships:** {graph['relationship_count']}")
                    st.write(f"**Method:** {graph['extraction_method']}")

                    if st.button(
                        f"Load Graph",
                            key=f"load_{graph['graph_id']}"):
                        # Load graph from Neo4j
                        loaded_graph = neo4j_service.get_graph(
                            graph["graph_id"])
                        if loaded_graph:
                            st.session_state.current_graph = loaded_graph
                            rerun()
        else:
            st.info("No graphs found in DynamoDB")

        if "LLM-Based" in extraction_method and hasattr(
            st.session_state, "llm_generator"
        ):
            st.divider()
            st.header("💾 Cache Statistics")

            cache_stats = st.session_state.llm_generator.get_cache_stats()
            st.caption(
                f"• Cache enabled: {'Yes' if cache_stats['enabled'] else 'No'}")
            st.caption(f"• Memory entries: {cache_stats['memory_entries']}")
            st.caption(f"• Disk entries: {cache_stats['disk_entries']}")
            st.caption(f"• Total size: {cache_stats['total_size_mb']:.2f} MB")

            if st.button("🧹 Clear Expired Cache", use_container_width=True):
                cleared = st.session_state.llm_generator.clear_expired_cache()
                st.success(f"Cleared {cleared} expired entries")

    if "current_graph" not in st.session_state:
        st.info("👈 Upload a PDF document to begin")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            ### 🔧 Rule-Based Extraction
            - ✅ Fast and free
            - ✅ No API needed
            - ✅ Enhanced patterns
            - ✅ Better confidence scoring
            - ❌ Limited context understanding
            """
            )

        with col2:
            st.markdown(
                """
            ### 🧠 LLM-Based Extraction
            - ✅ Understands context
            - ✅ Provides reasoning
            - ✅ High accuracy
            - ✅ Confidence with evidence
            - ✅ Smart caching to reduce costs
            - ❌ Requires API key & costs
            """
            )

    else:
        display_results(visualizer, evaluator)


def estimate_processing_cost(chunk_percentage: int, provider: str) -> float:
    avg_chunks = 100
    chunks_to_process = int(avg_chunks * (chunk_percentage / 100))

    if "OpenAI" in provider:
        cost_per_1k = 0.002
    else:
        cost_per_1k = 0.001

    tokens_per_chunk = 1000
    total_tokens = chunks_to_process * tokens_per_chunk

    return (total_tokens / 1000) * cost_per_1k


def process_document(
    uploaded_file,
    doc_processor,
    graph_generator,
    neo4j_service,
    extraction_method,
    llm_provider,
    evaluator,
    chunk_percentage,
    aws_service,
    security_manager,
    user_id,
):
    progress_container = st.container()

    with progress_container:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        try:
            start_time = datetime.now()

            status_text.text("☁️ Uploading document to S3...")
            progress_bar.progress(0.10)

            try:
                s3_key = aws_service.upload_document_to_s3(
                    uploaded_file.getvalue(), uploaded_file.name
                )
            except ValueError as e:
                st.error(f"❌ {str(e)}")
                return
            except Exception as e:
                st.warning(f"⚠️ Could not upload to S3: {str(e)}")
                s3_key = None

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            status_text.text("📄 Extracting text from PDF...")
            progress_bar.progress(0.20)

            text, chunks = doc_processor.process_document(tmp_path)

            text = security_manager.sanitize_input(text)
            chunks = [security_manager.sanitize_input(
                chunk) for chunk in chunks]

            st.session_state.extracted_text = text
            st.session_state.chunks = chunks

            if "LLM-Based" in extraction_method:
                if not security_manager.check_rate_limit(user_id, "api_call"):
                    st.error(
                        "⚠️ API rate limit exceeded. Please wait before making more LLM calls."
                    )
                    return

                status_text.text(
                    f"🧠 Using {llm_provider} for intelligent extraction..."
                )
                progress_bar.progress(0.30)

                provider = "openai" if "OpenAI" in llm_provider else "anthropic"
                api_key = os.getenv(f"{provider.upper()}_API_KEY")

                if not api_key:
                    api_key = aws_service.get_api_key(provider)
                    if api_key:
                        os.environ[f"{provider.upper()}_API_KEY"] = api_key
                    else:
                        st.error(
                            f"No valid API key found for {provider}. Please set it in SSM Parameter Store."
                        )
                        return

                llm_generator = LLMGraphGenerator(provider=provider)
                st.session_state.llm_generator = llm_generator

                all_entities = []
                all_relationships = []
                global_entity_map = {}

                total_chunks = len(chunks)
                num_chunks = max(
                    1, int(total_chunks * (chunk_percentage / 100)))

                st.info(
                    f"Processing {num_chunks} out of {total_chunks} chunks ({chunk_percentage}%)"
                )

                total_tokens = 0
                cache_hits = 0

                for i, chunk in enumerate(chunks[:num_chunks]):
                    status_text.text(
                        f"🧠 Processing chunk {i+1}/{num_chunks} with {llm_provider}..."
                    )
                    progress_bar.progress(0.30 + (i / num_chunks * 0.20))

                    try:
                        entities, relationships = (
                            llm_generator.extract_entities_and_relationships(chunk))

                        for entity in entities:
                            key = entity["label"].lower()
                            if key not in global_entity_map:
                                global_entity_map[key] = entity
                                all_entities.append(entity)
                            else:
                                if entity.get(
                                    "confidence",
                                    0) > global_entity_map[key].get(
                                    "confidence",
                                        0):
                                    global_entity_map[key]["confidence"] = entity[
                                        "confidence"
                                    ]
                                    if "reasoning" in entity:
                                        global_entity_map[key]["reasoning"] = entity[
                                            "reasoning"
                                        ]

                        for rel in relationships:
                            source_exists = any(
                                e["id"] == rel["source"] for e in all_entities
                            )
                            target_exists = any(
                                e["id"] == rel["target"] for e in all_entities
                            )

                            if source_exists and target_exists:
                                all_relationships.append(rel)

                    except Exception as e:
                        st.warning(f"Error processing chunk {i+1}: {str(e)}")

                entities = all_entities
                relationships = all_relationships

                st.session_state.extraction_method = "LLM-Based"
                st.session_state.llm_provider = llm_provider

            else:
                status_text.text(
                    "🔍 Extracting entities using enhanced pattern matching..."
                )
                progress_bar.progress(0.30)

                entities = graph_generator.extract_entities(text, chunks)
                st.session_state.entities = entities

                status_text.text(
                    "🔗 Discovering relationships using improved rules...")
                progress_bar.progress(0.50)

                relationships = graph_generator.extract_relationships(
                    text, entities)
                st.session_state.relationships = relationships

                st.session_state.extraction_method = "Rule-Based"

            st.session_state.entities = entities
            st.session_state.relationships = relationships

            status_text.text("💾 Storing in graph database...")
            progress_bar.progress(0.70)

            graph_id = neo4j_service.create_graph(
                document_name=uploaded_file.name,
                entities=entities,
                relationships=relationships,
            )

            status_text.text("📊 Generating visualization...")
            progress_bar.progress(0.80)

            processing_time = (datetime.now() - start_time).total_seconds()

            graph_data = {
                "graph_id": graph_id,
                "document_name": uploaded_file.name,
                "entities": entities,
                "relationships": relationships,
                "entity_types": list(set(e["type"] for e in entities)),
                "s3_key": s3_key or "",
                "chunk_percentage": chunk_percentage,
                "processing_time_seconds": processing_time,
                "stats": {
                    "total_entities": len(entities),
                    "total_relationships": len(relationships),
                    "processing_time": datetime.now().isoformat(),
                    "extraction_method": extraction_method,
                },
            }

            if "LLM-Based" in extraction_method:
                graph_data["llm_provider"] = llm_provider

            if s3_key:
                status_text.text("💾 Saving metadata to DynamoDB...")
                progress_bar.progress(0.85)

                try:
                    aws_service.save_graph_metadata(graph_data)
                except Exception as e:
                    st.warning(f"Could not save metadata to DynamoDB: {e}")

            if s3_key:
                status_text.text("📤 Queuing for additional processing...")
                progress_bar.progress(0.90)

                sqs_message = {
                    "graph_id": graph_id,
                    "document_key": s3_key,
                    "document_name": uploaded_file.name,
                    "extraction_method": extraction_method,
                    "entity_count": len(entities),
                    "relationship_count": len(relationships),
                }

                try:
                    aws_service.send_processing_message(sqs_message)
                except Exception as e:
                    logger.warning(f"Could not send to processing queue: {e}")

            if s3_key:
                status_text.text("📁 Moving document to processed folder...")
                progress_bar.progress(0.95)

                try:
                    processed_key = aws_service.move_to_processed(s3_key)
                    graph_data["s3_key"] = processed_key
                except Exception as e:
                    logger.warning(f"Could not move to processed folder: {e}")

            st.session_state.current_graph = graph_data

            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")

            os.unlink(tmp_path)

            method_emoji = "🧠" if "LLM" in extraction_method else "🔧"
            success_msg = f"""
            {method_emoji} Successfully processed **{uploaded_file.name}** using **{extraction_method}**
            - Extracted **{len(entities)}** entities
            - Discovered **{len(relationships)}** relationships
            - Processing time: **{processing_time:.1f}** seconds
            """

            if "LLM-Based" in extraction_method:
                cache_stats = llm_generator.get_cache_stats()
                if cache_stats["memory_entries"] > 0:
                    success_msg += f"\n- Cache hits saved API calls!"

            st.success(success_msg)

            rerun()

        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Error processing document: {error_msg}",
                exc_info=True)

            if "Unauthorized" in error_msg or "AuthenticationRateLimit" in error_msg:
                st.cache_resource.clear()
            else:
                st.error(f"Error processing document: {error_msg}")
            progress_bar.empty()
            status_text.empty()


def display_results(visualizer, evaluator):

    graph_data = st.session_state.current_graph

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Entities", graph_data["stats"]["total_entities"])

    with col2:
        st.metric("Relationships", graph_data["stats"]["total_relationships"])

    with col3:
        st.metric("Entity Types", len(graph_data["entity_types"]))

    with col4:
        method = graph_data["stats"].get("extraction_method", "Unknown")
        emoji = "🧠" if "LLM" in method else "🔧"
        st.metric("Method", f"{emoji} {method.split()[0]}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Graph Visualization",
            "🔍 Entities",
            "🔗 Relationships",
            "📄 Text Analysis",
            "📈 Quality Metrics",
            "💾 Export",
        ]
    )

    with tab1:
        st.subheader("Interactive Knowledge Graph")

        graph_html = visualizer.create_3d_graph(
            graph_data["entities"], graph_data["relationships"]
        )

        st.components.v1.html(graph_html, height=600, scrolling=False)

        st.markdown("### 📈 Graph Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Most Connected Entities:**")
            top_entities = visualizer.get_top_entities(
                graph_data["entities"], graph_data["relationships"], n=5
            )
            for entity in top_entities:
                st.write(
                    f"- **{entity['label']}** ({entity['type']}) - {entity['connections']} connections"
                )

        with col2:
            st.markdown("**Entity Distribution:**")
            type_counts = {}
            for entity in graph_data["entities"]:
                type_counts[entity["type"]] = type_counts.get(
                    entity["type"], 0) + 1

            for entity_type, count in sorted(
                type_counts.items(), key=lambda x: x[1], reverse=True
            ):
                st.write(f"- {entity_type}: {count}")

    with tab2:
        st.subheader("Extracted Entities")

        if st.session_state.get("extraction_method") == "LLM-Based":
            st.info(
                f"🧠 Entities extracted using {st.session_state.get('llm_provider', 'LLM')} with reasoning"
            )

        selected_type = st.selectbox(
            "Filter by type", ["All"] + graph_data["entity_types"]
        )

        filtered_entities = graph_data["entities"]
        if selected_type != "All":
            filtered_entities = [
                e for e in filtered_entities if e["type"] == selected_type
            ]

        import pandas as pd

        if filtered_entities and "reasoning" in filtered_entities[0]:
            df_data = []
            for e in filtered_entities:
                df_data.append(
                    {
                        "Label": e["label"],
                        "Type": e["type"],
                        "Confidence": e.get("confidence", 1.0),
                        "Reasoning": (
                            e.get("reasoning", "N/A")[:100] + "..."
                            if len(e.get("reasoning", "")) > 100
                            else e.get("reasoning", "N/A")
                        ),
                    }
                )
            df = pd.DataFrame(df_data)
        else:
            df = pd.DataFrame(filtered_entities)
            if "label" in df.columns:
                df = df[["label", "type", "confidence"]].rename(
                    columns={
                        "label": "Label",
                        "type": "Type",
                        "confidence": "Confidence",
                    }
                )

        st.dataframe(
            df.sort_values(
                "Confidence",
                ascending=False),
            use_container_width=True)

        if filtered_entities and "reasoning" in filtered_entities[0]:
            st.subheader("Detailed Entity Analysis")
            for entity in filtered_entities[:5]:  # Show first 5
                with st.expander(f"{entity['label']} ({entity['type']})"):
                    st.write(
                        f"**Confidence:** {entity.get('confidence', 1.0):.2%}")
                    st.write(
                        f"**Reasoning:** {entity.get('reasoning', 'N/A')}")
                    if "evidence" in entity:
                        st.write(f"**Evidence:** {entity['evidence']}")

    with tab3:
        st.subheader("Discovered Relationships")

        relationships_data = []
        for rel in graph_data["relationships"]:
            source_label = next(
                (
                    e["label"]
                    for e in graph_data["entities"]
                    if e["id"] == rel["source"]
                ),
                rel["source"],
            )
            target_label = next(
                (
                    e["label"]
                    for e in graph_data["entities"]
                    if e["id"] == rel["target"]
                ),
                rel["target"],
            )

            rel_data = {
                "Source": source_label,
                "Relationship": rel["type"].upper(),
                "Target": target_label,
                "Confidence": rel.get("confidence", 1.0),
            }

            if "reasoning" in rel:
                rel_data["Reasoning"] = (
                    rel["reasoning"][:100] + "..."
                    if len(rel["reasoning"]) > 100
                    else rel["reasoning"]
                )

            relationships_data.append(rel_data)

        df_rel = pd.DataFrame(relationships_data)
        st.dataframe(
            df_rel.sort_values(
                "Confidence",
                ascending=False),
            use_container_width=True)

    with tab4:
        st.subheader("Text Analysis")

        if "extracted_text" in st.session_state:
            word_count = len(st.session_state.extracted_text.split())
            st.metric("Total Words", f"{word_count:,}")

            st.markdown("### Document Chunks")

            chunk_num = st.number_input(
                "Select chunk",
                min_value=1,
                max_value=len(st.session_state.chunks),
                value=1,
            )

            st.text_area(
                f"Chunk {chunk_num} of {len(st.session_state.chunks)}",
                st.session_state.chunks[chunk_num - 1],
                height=200,
            )

    with tab5:
        st.subheader("📈 Extraction Quality Metrics")

        quality_stats = evaluator._evaluate_quality_indicators(
            graph_data["entities"], graph_data["relationships"]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Entity Quality")
            st.metric(
                "Entities with Confidence",
                f"{quality_stats['entities_with_confidence']:.0%}",
            )
            st.metric(
                "High Confidence Entities",
                f"{quality_stats['high_confidence_entity_ratio']:.0%}",
            )

            if st.session_state.get("extraction_method") == "LLM-Based":
                with_reasoning = len(
                    [e for e in graph_data["entities"] if e.get("reasoning")]
                )
                st.metric(
                    "Entities with Reasoning",
                    f"{with_reasoning}/{len(graph_data['entities'])}",
                )

        with col2:
            st.markdown("### Relationship Quality")
            st.metric(
                "Relationships with Confidence",
                f"{quality_stats['relationships_with_confidence']:.0%}",
            )
            st.metric(
                "High Confidence Relationships",
                f"{quality_stats['high_confidence_relationship_ratio']:.0%}",
            )

        if st.session_state.get("extraction_method") == "LLM-Based":
            st.info(
                "🧠 LLM-based extraction provides reasoning and evidence for better explainability"
            )
        else:
            st.warning(
                "🔧 Rule-based extraction uses pattern matching - consider LLM for better accuracy"
            )

    with tab6:
        st.subheader("Export Options")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Download as JSON", use_container_width=True):
                json_data = json.dumps(graph_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )

        with col2:
            if st.button("📝 Get Neo4j Query", use_container_width=True):
                query = f"""
                // Query to retrieve this graph from Neo4j
                MATCH (n:Entity)-[r]-(m:Entity)
                WHERE n.graph_id = '{graph_data['graph_id']}'
                RETURN n, r, m
                """
                st.code(query, language="cypher")


if __name__ == "__main__":
    main()
