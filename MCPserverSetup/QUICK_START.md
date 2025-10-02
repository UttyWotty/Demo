# ⚡ Quick Start Guide - Production Analytics MCP Server Demo

Get the demo running in **3 easy steps**!

## 📦 Step 1: Install Dependencies

```bash
cd production_mcp_demo
pip install -r requirements.txt
```

## 🚀 Step 2: Start the Server (Backend)

```bash
python demo_mcp_server.py
```

You should see:
```
🎯 Production Analytics MCP Server (DEMO)
============================================================
📍 Server: http://0.0.0.0:8000
📚 API Docs: http://0.0.0.0:8000/docs
🔌 WebSocket: ws://0.0.0.0:8000/ws/realtime
============================================================

✅ Demo server ready!
💡 Using synthetic data - no credentials needed
🎨 Safe for portfolio showcase
```

## 📊 Step 2b: Start the Dashboards (Frontend)

**Choose one of the following dashboards:**

### Option 1: Production Analytics Dashboard
```bash
streamlit run demo_dashboard.py
```
Opens at: http://localhost:8501
- 🏭 Production timeline and efficiency analysis
- 📊 Equipment performance comparison
- 🎯 Real-time KPIs and metrics
- 💾 Data export options

### Option 2: ROI Analyzer
```bash
streamlit run demo_roi_analyzer.py
```
Opens at: http://localhost:8502
- 💰 Cycle time efficiency & ROI analysis
- 📈 Time savings/losses calculation
- 📊 FASTER/WITHIN/SLOWER classification
- 💵 Financial impact analysis

### Option 3: Capacity Risk Analyzer
```bash
streamlit run demo_capacity_analyzer.py
```
Opens at: http://localhost:8503
- ⚡ OEE (Overall Equipment Effectiveness)
- 📉 Availability, Performance, Quality metrics
- 🏭 Session-based capacity analysis
- 📊 Production losses breakdown

**Pro Tip:** Run all 3 dashboards simultaneously in separate terminals!

## 🧪 Step 3: Test the API

### Option A: Use the Interactive Docs
Open in browser: http://localhost:8000/docs

### Option B: Use cURL

```bash
# Health check
curl http://localhost:8000/api/health

# List modules
curl http://localhost:8000/api/modules

# Get suppliers
curl http://localhost:8000/api/suppliers

# Get analytics
curl -X POST http://localhost:8000/api/analytics/runrate \
  -H "Content-Type: application/json" \
  -d '{"supplier": "General Motors"}'
```

### Option C: Run the test script

```bash
python test_api.py
```

## 📸 Take Screenshots for Portfolio

### Priority Screenshots ⭐

1. **Production Analytics Dashboard** (Highest Priority)
   - Open: http://localhost:8501
   - Take full-page screenshot
   - Save as: `assets/production_dashboard.png`

2. **ROI Analyzer Dashboard**
   - Open: http://localhost:8502
   - Take full-page screenshot showing ROI metrics
   - Save as: `assets/roi_analyzer.png`

3. **Capacity Risk Analyzer Dashboard**
   - Open: http://localhost:8503
   - Take full-page screenshot showing OEE gauge
   - Save as: `assets/capacity_analyzer.png`

4. **API Documentation**
   - Open: http://localhost:8000/docs
   - Take screenshot
   - Save as: `assets/api_docs.png`

### Optional Screenshots

5. **Dashboard Charts Detail**
   - Navigate through tabs in each dashboard
   - Screenshot key visualizations
   - Save as: `assets/charts_detail.png`

6. **API Response Example**
   - Make a request to `/api/analytics/runrate`
   - Screenshot the JSON response
   - Save as: `assets/api_response.png`

## 🎯 What to Show in Portfolio

**Key Features to Highlight:**
- ✅ Multi-module support (3 analysis types)
- ✅ RESTful API with 8 endpoints
- ✅ WebSocket for real-time streaming
- ✅ Interactive API documentation
- ✅ AI/LLM integration ready
- ✅ Production-grade architecture
- ✅ Synthetic data (portfolio-safe)

**Metrics to Mention:**
- 10,000+ lines of production code
- Sub-second API response times
- 3 analysis modules (RunRate, ROI, Capacity)
- 8 REST endpoints + WebSocket
- AWS Bedrock integration ready

## 🔗 Add to Portfolio Website

Add this project card to your portfolio:

```html
<div class="project-card">
  <h3>🏭 Production Analytics MCP Server</h3>
  <p>Enterprise-grade MCP server with FastAPI, WebSocket, and AI integration</p>
  <ul>
    <li>Multi-module analytics (RunRate, ROI, Capacity)</li>
    <li>8 REST endpoints + real-time WebSocket</li>
    <li>AWS Bedrock LLM integration ready</li>
    <li>10,000+ lines of production code</li>
  </ul>
  <div class="tech-stack">
    <span>FastAPI</span>
    <span>WebSocket</span>
    <span>RESTful API</span>
    <span>AWS Bedrock</span>
    <span>Pandas</span>
  </div>
  <a href="demo/production_mcp_demo">View Demo</a>
  <a href="https://github.com/emoldino/RUN_RATE">GitHub</a>
</div>
```

## 💡 Tips

- **Live Demo**: Run the server during interviews to show the interactive API docs
- **Code Review**: Highlight the clean architecture and modular design
- **Scalability**: Mention it's production-ready and currently serves real clients
- **AI Integration**: Emphasize the LLM-ready architecture and AWS Bedrock support

## 🎨 Customization

Edit `demo_mcp_server.py` to customize:

```python
DEMO_CONFIG = {
    "suppliers": ["Your Company 1", "Your Company 2"],
    "equipment_per_supplier": 3,
    "days_of_data": 60,
    "shots_per_day_range": (100, 500),
}
```

## ✅ Checklist for Portfolio

- [ ] Server runs without errors
- [ ] API documentation loads at /docs
- [ ] All endpoints return 200 status
- [ ] Screenshots added to assets/ folder
- [ ] README.md updated with screenshots
- [ ] Added to main portfolio page
- [ ] GitHub link updated (if public)
- [ ] Demo video recorded (optional)

---

**Ready to showcase!** 🎉

