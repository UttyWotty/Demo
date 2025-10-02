#!/usr/bin/env python3
"""
Production Analytics MCP Server - Portfolio Demo
==============================================

Self-contained demo server with synthetic data.
No external dependencies or credentials required.

Run with: python demo_mcp_server.py
Visit: http://localhost:8000/docs
"""

import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import random

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ===========================
# Configuration
# ===========================

DEMO_CONFIG = {
    "suppliers": ["General Motors", "Tesla", "Ford"],
    "equipment_per_supplier": 5,
    "days_of_data": 30,
    "shots_per_day_range": (200, 400),
}

# ===========================
# Data Models
# ===========================


class AnalyticsRequest(BaseModel):
    supplier: str
    equipment_code: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class AnalyticsResponse(BaseModel):
    supplier: str
    equipment_code: Optional[str]
    date_range: Dict[str, str]
    metrics: Dict[str, Any]
    raw_data: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    llm_enabled: bool


# ===========================
# FastAPI App
# ===========================

app = FastAPI(
    title="Production Analytics MCP Server (Demo)",
    description="Demo MCP server with synthetic data - Portfolio safe!",
    version="1.0.0-demo",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# Module Registry
# ===========================

ANALYSIS_MODULES = {
    "runrate": {
        "name": "Run Rate Analysis",
        "description": "Production timeline and stop analysis",
    },
    "roi": {
        "name": "ROI Analysis",
        "description": "Cycle time efficiency and ROI analysis",
    },
    "capacity": {
        "name": "Capacity Risk Analysis",
        "description": "OEE and capacity analysis",
    },
}

# ===========================
# Synthetic Data Generator
# ===========================


def generate_synthetic_production_data(
    supplier: str, equipment_code: Optional[str] = None, num_days: int = 30
) -> List[Dict]:
    """Generate realistic synthetic production data"""

    data = []
    base_date = datetime.now() - timedelta(days=num_days)

    # Generate equipment list
    if equipment_code:
        equipment_list = [equipment_code]
    else:
        equipment_list = [f"{supplier[:3].upper()}-{i:03d}" for i in range(1, 6)]

    for equipment in equipment_list:
        # Equipment-specific characteristics
        base_ct = random.uniform(25, 45)  # Base cycle time
        efficiency = random.uniform(0.85, 0.98)  # Equipment efficiency

        for day in range(num_days):
            current_date = base_date + timedelta(days=day)
            shots_today = random.randint(*DEMO_CONFIG["shots_per_day_range"])

            for shot in range(shots_today):
                timestamp = current_date + timedelta(
                    hours=random.randint(6, 22),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59),
                )

                # Realistic cycle time with variations
                actual_ct = base_ct + random.gauss(0, 2)

                # Occasionally introduce anomalies (stops)
                if random.random() > efficiency:
                    actual_ct = 999.9  # Stop marker

                data.append(
                    {
                        "SUPPLIER_NAME": supplier,
                        "EQUIPMENT_CODE": equipment,
                        "LOCAL_SHOT_TIME": timestamp.isoformat(),
                        "ACTUAL_CT": max(10, actual_ct),
                        "APPROVED_CT": base_ct,
                    }
                )

    return sorted(data, key=lambda x: x["LOCAL_SHOT_TIME"])


def calculate_runrate_metrics(data: List[Dict]) -> Dict[str, Any]:
    """Calculate RunRate analysis metrics"""

    if not data:
        return {}

    valid_data = [d for d in data if d["ACTUAL_CT"] < 999]
    stop_count = len([d for d in data if d["ACTUAL_CT"] >= 999])

    return {
        "module": "runrate",
        "total_shots": len(data),
        "normal_shots": len(valid_data),
        "stop_count": stop_count,
        "avg_cycle_time": sum(d["ACTUAL_CT"] for d in valid_data) / len(valid_data)
        if valid_data
        else 0,
        "efficiency_pct": (len(valid_data) / len(data) * 100) if data else 0,
        "equipment_count": len(set(d["EQUIPMENT_CODE"] for d in data)),
        "time_span_hours": (
            (
                datetime.fromisoformat(data[-1]["LOCAL_SHOT_TIME"])
                - datetime.fromisoformat(data[0]["LOCAL_SHOT_TIME"])
            ).total_seconds()
            / 3600
        )
        if len(data) > 1
        else 0,
    }


def calculate_roi_metrics(data: List[Dict]) -> Dict[str, Any]:
    """Calculate ROI analysis metrics"""

    if not data:
        return {}

    within_tolerance = sum(
        1
        for d in data
        if 0.95 * d["APPROVED_CT"] <= d["ACTUAL_CT"] <= 1.05 * d["APPROVED_CT"]
    )
    faster = sum(1 for d in data if d["ACTUAL_CT"] < 0.95 * d["APPROVED_CT"])
    slower = sum(1 for d in data if d["ACTUAL_CT"] > 1.05 * d["APPROVED_CT"])

    time_diff = sum(d["APPROVED_CT"] - d["ACTUAL_CT"] for d in data)

    return {
        "module": "roi",
        "total_shots": len(data),
        "within_tolerance": within_tolerance,
        "faster_than_approved": faster,
        "slower_than_approved": slower,
        "efficiency_pct": (within_tolerance / len(data) * 100) if data else 0,
        "time_savings_hours": max(0, time_diff / 3600),
        "time_loss_hours": abs(min(0, time_diff / 3600)),
    }


def calculate_capacity_metrics(data: List[Dict]) -> Dict[str, Any]:
    """Calculate Capacity/OEE metrics"""

    if not data:
        return {}

    total_time = (
        (
            datetime.fromisoformat(data[-1]["LOCAL_SHOT_TIME"])
            - datetime.fromisoformat(data[0]["LOCAL_SHOT_TIME"])
        ).total_seconds()
        if len(data) > 1
        else 0
    )

    avg_actual = sum(d["ACTUAL_CT"] for d in data) / len(data)
    avg_approved = sum(d["APPROVED_CT"] for d in data) / len(data)

    production_time = len(data) * avg_actual
    availability = (production_time / total_time * 100) if total_time > 0 else 0
    performance = (avg_approved / avg_actual * 100) if avg_actual > 0 else 0
    oee = (availability * performance) / 10000

    return {
        "module": "capacity",
        "total_shots": len(data),
        "availability_pct": min(100, availability),
        "performance_pct": min(100, performance),
        "oee_pct": min(100, oee),
        "total_time_hours": total_time / 3600,
        "production_time_hours": production_time / 3600,
    }


# ===========================
# API Endpoints
# ===========================


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Production Analytics MCP Server (Demo)",
        "version": "1.0.0-demo",
        "mode": "synthetic_data",
        "modules": list(ANALYSIS_MODULES.keys()),
        "endpoints": {
            "health": "/api/health (GET)",
            "modules": "/api/modules (GET)",
            "suppliers": "/api/suppliers (GET)",
            "equipment": "/api/equipment/{supplier} (GET)",
            "analytics": "/api/analytics/{analysis_type} (POST)",
            "llm_insights": "/api/llm/insights/{analysis_type} (POST)",
            "docs": "/docs",
        },
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0-demo",
        llm_enabled=False,  # Demo mode
    )


@app.get("/api/suppliers", response_model=List[str])
async def list_suppliers():
    """Get list of available suppliers (synthetic)"""
    return DEMO_CONFIG["suppliers"]


@app.get("/api/equipment/{supplier}", response_model=List[str])
async def list_equipment(supplier: str):
    """Get equipment codes for a supplier (synthetic)"""
    if supplier not in DEMO_CONFIG["suppliers"]:
        raise HTTPException(404, f"Supplier '{supplier}' not found")

    return [f"{supplier[:3].upper()}-{i:03d}" for i in range(1, 6)]


@app.get("/api/modules", response_model=Dict[str, Any])
async def list_analysis_modules():
    """List all available analysis modules"""
    return ANALYSIS_MODULES


@app.post("/api/analytics/{analysis_type}", response_model=AnalyticsResponse)
async def get_analytics_by_type(analysis_type: str, request: AnalyticsRequest):
    """
    Get analytics data for specified analysis type (with synthetic data)

    Examples:
        POST /api/analytics/runrate
        POST /api/analytics/roi
        POST /api/analytics/capacity
    """
    print(f"📊 {analysis_type.upper()} analytics request: {request.supplier}")

    # Validate analysis type
    if analysis_type not in ANALYSIS_MODULES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown analysis type: {analysis_type}. Available: {list(ANALYSIS_MODULES.keys())}",
        )

    # Generate synthetic data
    data = generate_synthetic_production_data(
        supplier=request.supplier, equipment_code=request.equipment_code, num_days=30
    )

    if not data:
        raise HTTPException(
            status_code=404, detail="No data found for the specified criteria"
        )

    # Calculate metrics
    if analysis_type == "runrate":
        metrics = calculate_runrate_metrics(data)
    elif analysis_type == "roi":
        metrics = calculate_roi_metrics(data)
    elif analysis_type == "capacity":
        metrics = calculate_capacity_metrics(data)
    else:
        metrics = {}

    # Add module info
    metrics["analysis_type"] = analysis_type
    metrics["module_name"] = ANALYSIS_MODULES[analysis_type]["name"]
    metrics["data_mode"] = "synthetic"

    # Sample data for response
    sample_data = {
        "sample_size": min(20, len(data)),
        "total_size": len(data),
        "sample_data": data[:20],
    }

    response = AnalyticsResponse(
        supplier=request.supplier,
        equipment_code=request.equipment_code,
        date_range={
            "start": data[0]["LOCAL_SHOT_TIME"] if data else "N/A",
            "end": data[-1]["LOCAL_SHOT_TIME"] if data else "N/A",
        },
        metrics=metrics,
        raw_data=sample_data,
    )

    return response


@app.post("/api/llm/insights/{analysis_type}")
async def get_llm_insights_by_type(analysis_type: str, request: AnalyticsRequest):
    """
    🤖 LLM Integration Endpoint (Demo - returns mock insights)

    In production, this would integrate with AWS Bedrock/LangChain
    """
    if analysis_type not in ANALYSIS_MODULES:
        raise HTTPException(
            status_code=400, detail=f"Unknown analysis type: {analysis_type}"
        )

    return {
        "status": "demo_mode",
        "analysis_type": analysis_type,
        "module_name": ANALYSIS_MODULES[analysis_type]["name"],
        "message": "This endpoint would integrate with AWS Bedrock in production",
        "mock_insight": f"📊 Analysis complete for {request.supplier}. "
        f"Based on {analysis_type} analysis, equipment performance is within "
        f"expected ranges. Consider reviewing high-variance periods for optimization.",
        "llm_integration_ready": True,
    }


# ===========================
# WebSocket for Real-time Data
# ===========================


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming (demo)"""
    await manager.connect(websocket)
    print("🔌 WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "subscribe":
                supplier = message.get("supplier", "General Motors")
                await websocket.send_json(
                    {
                        "type": "subscribed",
                        "supplier": supplier,
                        "message": f"Subscribed to {supplier} updates (demo mode)",
                        "mode": "synthetic_data",
                    }
                )

            elif message.get("type") == "get_data":
                supplier = message.get("supplier", "General Motors")
                data = generate_synthetic_production_data(supplier, num_days=1)
                await websocket.send_json(
                    {
                        "type": "data",
                        "supplier": supplier,
                        "data": data[:10],  # Send first 10 shots
                        "count": len(data),
                    }
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("🔌 WebSocket client disconnected")


# ===========================
# Server Startup
# ===========================


def start_demo_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the demo MCP server"""
    print("\n" + "=" * 60)
    print("🎯 Production Analytics MCP Server (DEMO)")
    print("=" * 60)
    print(f"📍 Server: http://{host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"🔌 WebSocket: ws://{host}:{port}/ws/realtime")
    print("=" * 60)
    print("\n✅ Demo server ready!")
    print("💡 Using synthetic data - no credentials needed")
    print("🎨 Safe for portfolio showcase\n")

    print("📊 Test endpoints:")
    print(f"   curl http://{host}:{port}/api/health")
    print(f"   curl http://{host}:{port}/api/modules")
    print(f"   curl http://{host}:{port}/api/suppliers\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_demo_server()
