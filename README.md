# 🎯 Risk Assessment Knowledge Graph Extractor

A Streamlit application that transforms risk assessment documents into interactive knowledge graphs using Neo4j. Choose between rule-based extraction (free) or LLM-powered extraction (more accurate).

## 🚀 Features

- **Dual Extraction Methods**:
  - **Rule-Based**: Fast, free pattern matching
  - **LLM-Based**: Intelligent extraction with GPT-4 or Claude (optional)
- **PDF Processing**: Extract and analyze risk documents
- **Entity Recognition**: Identify risks, controls, assets, and stakeholders  
- **Relationship Discovery**: Map connections between entities
- **Interactive Visualization**: 3D graph with PyVis
- **Quality Metrics**: Evaluate extraction quality
- **Export Options**: JSON export and Neo4j queries

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB of available RAM
- Ports 8501, 7474, 7687, and 4566 available

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Risk-Assessment-KG-Streamlit
   ```

2. **Set Neo4j Password** (Optional - defaults to secure password)
   ```bash
   export NEO4J_PASSWORD=YourSecurePasswordHere
   ```

3. **Start the system**
   ```bash
   ./start.sh
   ```
   
   Or manually:
   ```bash
   docker compose up -d
   # Wait for services to be healthy, then:
   docker exec risk-kg-localstack sh /docker-entrypoint-initaws.d/01-create-resources.sh
   ```

4. **Access the application**
   - Streamlit App: http://localhost:8501
   - Neo4j Browser: http://localhost:7474
   - LocalStack: http://localhost:4566

## 📊 Extraction Methods

### 1. Rule-Based (Default)
- ✅ **Free** - No API costs
- ✅ **Fast** - Processes documents in seconds
- ❌ **Basic** - Pattern matching only
- Perfect for quick analysis and testing

### 2. LLM-Based (Optional)
- ✅ **Intelligent** - Understands context
- ✅ **Accurate** - 85-95% accuracy
- ✅ **Reasoning** - Explains why entities were extracted
- ❌ **Costs** - ~$0.01-0.05 per page

## 🔑 Using LLM Extraction

### Option A: Environment Variable
```bash
# Set your API key before starting
export OPENAI_API_KEY="your-key-here"
# OR
export ANTHROPIC_API_KEY="your-key-here"

docker compose up -d
```

### Option B: Enter in UI
1. Start the app normally
2. Select "LLM-Based" in sidebar
3. Enter your API key in the text field

### Getting API Keys
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/

## 📈 Usage Guide

1. **Upload PDF**: Click "Choose a PDF file" in sidebar
2. **Select Method**: 
   - Rule-Based for free, quick analysis
   - LLM-Based for detailed, accurate extraction
3. **Process**: Click "Process Document"
4. **Explore**: 
   - View interactive graph
   - Browse entities and relationships
   - Check quality metrics
   - Export results

## 🏗️ Project Structure

```
Risk-Assessment-KG-Streamlit/
├── app.py                       # Main Streamlit application
├── src/
│   ├── document_processor.py    # PDF text extraction
│   ├── graph_generator.py       # Rule-based extraction
│   ├── llm_graph_generator.py   # LLM-based extraction
│   ├── neo4j_service.py         # Graph database operations
│   ├── visualizer.py            # Graph visualization
│   └── graph_evaluator.py       # Quality metrics
├── data/                        # Place PDF files here
├── docker-compose.yml           # Docker configuration
├── Dockerfile                   # Container definition
└── requirements.txt             # Python dependencies
```

## 🧪 Example Output Comparison

### Rule-Based:
```json
{
  "label": "high risk",
  "type": "RISK",
  "confidence": 0.8
}
```

### LLM-Based:
```json
{
  "label": "Supply Chain Disruption Risk",
  "type": "RISK", 
  "confidence": 0.92,
  "reasoning": "Identified as critical operational risk affecting procurement",
  "evidence": "The supply chain disruption risk has increased by 40%..."
}
```

## 📊 Entity & Relationship Types

**Entities:**
- RISK: Threats, vulnerabilities, hazards
- CONTROL: Mitigations, safeguards
- ASSET: Systems, data, resources
- STAKEHOLDER: People, teams
- IMPACT: Consequences, effects
- COMPLIANCE: Standards, regulations

**Relationships:**
- MITIGATES: Control reduces risk
- AFFECTS: Risk impacts asset
- OWNS: Stakeholder owns asset/control
- REQUIRES: Dependencies

## 🛠️ Development Setup

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install openai anthropic  # For LLM support

# Download spaCy model
python -m spacy download en_core_web_sm

# Set Neo4j connection
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password123

# Run app
streamlit run app.py
```

### Docker Commands
```bash
# Build and start
docker compose up --build -d

# View logs
docker compose logs -f

# Stop
docker compose down

# Reset everything
docker compose down -v
```

## 💡 Tips & Best Practices

1. **Start with Rule-Based**: Test your document first with free extraction
2. **Use LLM for Important Docs**: When accuracy matters, use LLM extraction
3. **Cost Control**: LLM mode processes only first 5 chunks by default
4. **Privacy**: LLM mode sends content to OpenAI/Anthropic APIs

## 🔧 Troubleshooting

**Neo4j Connection Issues:**
```bash
# Check if Neo4j is running
docker compose ps
# View Neo4j logs
docker compose logs neo4j
```

**API Key Issues:**
- Ensure key has credits
- Check usage: OpenAI Dashboard / Anthropic Console

**Performance:**
- Increase chunk limit in app.py for longer documents
- Use GPU for faster spaCy processing

## 📝 Notes

- **Why Two Methods?** Rule-based is great for quick analysis and when you can't use external APIs. LLM-based provides superior accuracy when you need it.
- **Data Privacy**: Rule-based processing is 100% local. LLM-based sends data to API providers.
- **Customization**: Edit patterns in `graph_generator.py` or prompts in `llm_graph_generator.py`

## 🚀 Future Enhancements

- [ ] Hybrid extraction (rules + LLM)
- [ ] Custom entity types
- [ ] Batch processing
- [ ] Fine-tuned models
- [ ] Export to other formats

## 📄 License

This project is for educational/portfolio purposes.

---

*Built for AI Engineers and Risk Management Professionals* 