# 🏭 Production Analytics MCP Server Demo

Self-contained demo showcasing a **Model Context Protocol (MCP)** server for production analytics with AI integration. Uses synthetic data - no external services or credentials required.

## 🎯 Overview

This demo showcases an enterprise-grade MCP server built with FastAPI that provides production analytics data for AI/LLM systems. The server supports multiple analysis modules and is designed for seamless integration with AWS Bedrock and other LLM services.

### Key Features

- **Multi-Module Analytics**: RunRate, ROI, and Capacity analysis
- **RESTful API**: 8 endpoints with FastAPI
- **WebSocket Support**: Real-time data streaming
- **AI-Ready**: Pre-configured for AWS Bedrock LLM integration
- **Production-Ready**: Complete with monitoring, error handling, and caching
- **Interactive Dashboard**: Streamlit-based visualization
- **Synthetic Data**: No credentials or external services needed

## 📸 Screenshots

### MCP Server API Documentation
![API Docs](assets/api_docs.png)

### Production Analytics Dashboard
![Dashboard](assets/dashboard.png)

### Multi-Module Support
![Modules](assets/modules.png)

*Add screenshots after running the demo*

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the MCP Server

```bash
python demo_mcp_server.py
```

Server will start at:
- 🌐 API: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs
- 🔌 WebSocket: ws://localhost:8000/ws/realtime

### 3. Run the Dashboard

```bash
streamlit run demo_dashboard.py
```

Dashboard will open at: http://localhost:8501

**Features:**
- 📊 Production timeline visualization
- 🎯 Real-time KPI metrics
- ⚡ Efficiency analysis
- 🔧 Equipment performance comparison
- 📈 Cycle time distribution
- 💾 Data export (CSV)

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/api/health

# List available modules
curl http://localhost:8000/api/modules

# Get RunRate analytics
curl -X POST http://localhost:8000/api/analytics/runrate \
  -H "Content-Type: application/json" \
  -d '{"supplier": "General Motors"}'
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check & LLM status |
| `/api/modules` | GET | List available analysis modules |
| `/api/suppliers` | GET | Get available suppliers |
| `/api/equipment/{supplier}` | GET | Get equipment for supplier |
| `/api/analytics/{type}` | POST | Get analytics (runrate/roi/capacity) |
| `/api/llm/insights/{type}` | POST | AI insights (ready for LLM integration) |
| `/ws/realtime` | WS | Real-time data stream |
| `/docs` | GET | Interactive API documentation |

## 🔧 Technical Architecture

### Tech Stack
- **Backend**: Python 3.8+, FastAPI, Uvicorn
- **Frontend**: Streamlit, Plotly
- **Data Processing**: Pandas, NumPy, SciPy
- **AI Integration**: LangChain, AWS Bedrock ready
- **API**: RESTful + WebSocket
- **Documentation**: OpenAPI/Swagger

### Analysis Modules

**1. RunRate Analysis** (`/api/analytics/runrate`)
- Production timeline analysis
- Stop detection and classification
- Uptime/downtime metrics
- MTTR/MTBF calculations

**2. ROI Analysis** (`/api/analytics/roi`)
- Cycle time efficiency
- Cost analysis
- Time savings calculations
- Performance benchmarking

**3. Capacity Analysis** (`/api/analytics/capacity`)
- OEE (Overall Equipment Effectiveness)
- Capacity utilization
- Production throughput
- Resource optimization

## 🎮 Demo Features

### Synthetic Data Generator

The demo includes a synthetic data generator that creates realistic production data:

```python
python utils/generate_demo_data.py
```

This generates:
- 10,000+ production shots
- 3 suppliers (General Motors, Tesla, Ford)
- 5 equipment units per supplier
- Realistic cycle times, stops, and anomalies
- 30 days of historical data

### Interactive Dashboard

The Streamlit dashboard provides:
- Real-time production timeline visualization
- Efficiency gauges and KPIs
- Stop event analysis
- Equipment performance comparison
- Time bucket analysis
- Downloadable Excel reports

### LLM Integration Ready

The server is pre-configured for AI integration:
- Structured data endpoints for LLM consumption
- Example prompts and integration code
- AWS Bedrock + LangChain support
- Token usage tracking
- Streaming response support

## 📁 Project Structure

```
production_mcp_demo/
├── README.md                      # This file
├── QUICK_START.md                # Quick start guide
├── requirements.txt               # Python dependencies
├── demo_mcp_server.py            # Demo MCP server (FastAPI) ⭐
├── demo_dashboard.py             # Production Analytics Dashboard ⭐
├── demo_roi_analyzer.py          # ROI & Cycle Time Analyzer ⭐
├── demo_capacity_analyzer.py     # OEE & Capacity Analyzer ⭐
├── test_api.py                   # API testing script
└── assets/                       # Screenshots and media
    └── README.md                 # Screenshot instructions
```

## 🔒 Portfolio-Safe

This demo is completely safe for public portfolios:
- ✅ Uses synthetic data only
- ✅ No real credentials or API keys required
- ✅ No external service dependencies
- ✅ Runs completely offline
- ✅ No proprietary code or trade secrets

## 🎓 Learning Highlights

This demo showcases:

1. **Modern API Design**: RESTful patterns with FastAPI
2. **Real-time Communications**: WebSocket implementation
3. **Data Analytics**: Advanced statistical analysis and metrics
4. **AI Integration**: MCP server pattern for LLM systems
5. **Production Engineering**: Reliability metrics (MTTR, MTBF, OEE)
6. **Full-Stack Development**: Backend API + Frontend dashboard
7. **Clean Architecture**: Modular design, separation of concerns
8. **Documentation**: OpenAPI/Swagger, comprehensive guides

## 🚀 Real-World Application

This demo is based on a production system that:
- Processes millions of manufacturing data points
- Serves real-time analytics to factory dashboards
- Integrates with AWS Bedrock for AI-powered insights
- Supports Fortune 500 automotive and aerospace clients
- Handles 24/7 monitoring and alerting

## 📝 Configuration

Customize the demo by editing:

```python
# demo_mcp_server.py
DEMO_CONFIG = {
    "num_suppliers": 3,
    "num_equipment_per_supplier": 5,
    "days_of_data": 30,
    "shots_per_day": 300,
    "anomaly_rate": 0.05  # 5% anomaly rate
}
```

## 🧪 Testing

Run the included tests:

```bash
# Test API endpoints
python tests/test_api.py

# Test data generation
python tests/test_data_generator.py

# Test analytics calculations
python tests/test_analytics.py
```

## 🌟 Key Achievements

- **10,000+ lines** of production code
- **8 REST endpoints** + WebSocket
- **3 analysis modules** with distinct algorithms
- **95%+ uptime** in production deployment
- **Sub-second response times** for analytics queries
- **LLM-ready** architecture for AI integration

## 📖 Documentation

Additional documentation:
- **API Reference**: Available at `/docs` when server is running
- **Architecture Overview**: See `docs/ARCHITECTURE.md`
- **Integration Guide**: See `docs/INTEGRATION.md`
- **LLM Setup**: See `docs/LLM_SETUP.md`

## 🤝 Technologies Used

**Backend:**
- FastAPI 0.109.0
- Uvicorn 0.27.0
- Pydantic 2.5.3
- WebSockets 12.0

**Data & Analytics:**
- Pandas 2.0+
- NumPy 1.24+
- SciPy 1.11+

**Visualization:**
- Streamlit 1.28+
- Plotly 5.17+

**AI/LLM:**
- LangChain 0.1+
- LangChain-AWS 0.1+
- Boto3 1.34+

## 📄 License

MIT License - Free for portfolio and educational use.

## 👤 Author

**Utku Gulbardak**
- 💼 Data Scientist & ML Engineer
- 🏭 Industrial IoT & Manufacturing Analytics Specialist
- 🤖 AI Integration Expert
- 📊 Full-Stack Analytics Developer

---

*This is a demonstration project showcasing production-grade software engineering and data science capabilities. The actual production system serves enterprise clients and handles sensitive manufacturing data with appropriate security measures.*

