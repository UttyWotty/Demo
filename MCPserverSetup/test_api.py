#!/usr/bin/env python3
"""
Quick API test script for the Production Analytics MCP Server Demo

Run this after starting the server to verify all endpoints work.
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint"""
    print("\n🔍 Testing /api/health...")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_modules():
    """Test modules list endpoint"""
    print("\n🔍 Testing /api/modules...")
    response = requests.get(f"{BASE_URL}/api/modules")
    print(f"   Status: {response.status_code}")
    print(f"   Modules: {list(response.json().keys())}")
    return response.status_code == 200


def test_suppliers():
    """Test suppliers list endpoint"""
    print("\n🔍 Testing /api/suppliers...")
    response = requests.get(f"{BASE_URL}/api/suppliers")
    print(f"   Status: {response.status_code}")
    print(f"   Suppliers: {response.json()}")
    return response.status_code == 200


def test_equipment():
    """Test equipment list endpoint"""
    print("\n🔍 Testing /api/equipment/Tesla...")
    response = requests.get(f"{BASE_URL}/api/equipment/Tesla")
    print(f"   Status: {response.status_code}")
    print(f"   Equipment: {response.json()}")
    return response.status_code == 200


def test_analytics_runrate():
    """Test RunRate analytics endpoint"""
    print("\n🔍 Testing /api/analytics/runrate...")
    payload = {
        "supplier": "General Motors",
        "equipment_code": None,
        "start_date": None,
        "end_date": None,
    }
    response = requests.post(f"{BASE_URL}/api/analytics/runrate", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Total Shots: {data['metrics'].get('total_shots', 0)}")
        print(f"   Efficiency: {data['metrics'].get('efficiency_pct', 0):.1f}%")
    return response.status_code == 200


def test_analytics_roi():
    """Test ROI analytics endpoint"""
    print("\n🔍 Testing /api/analytics/roi...")
    payload = {"supplier": "Tesla"}
    response = requests.post(f"{BASE_URL}/api/analytics/roi", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Total Shots: {data['metrics'].get('total_shots', 0)}")
        print(f"   Efficiency: {data['metrics'].get('efficiency_pct', 0):.1f}%")
    return response.status_code == 200


def test_analytics_capacity():
    """Test Capacity analytics endpoint"""
    print("\n🔍 Testing /api/analytics/capacity...")
    payload = {"supplier": "Ford"}
    response = requests.post(f"{BASE_URL}/api/analytics/capacity", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   OEE: {data['metrics'].get('oee_pct', 0):.1f}%")
        print(f"   Availability: {data['metrics'].get('availability_pct', 0):.1f}%")
    return response.status_code == 200


def test_llm_insights():
    """Test LLM insights endpoint"""
    print("\n🔍 Testing /api/llm/insights/runrate...")
    payload = {"supplier": "General Motors"}
    response = requests.post(f"{BASE_URL}/api/llm/insights/runrate", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Status: {data.get('status', 'unknown')}")
        print(f"   LLM Ready: {data.get('llm_integration_ready', False)}")
    return response.status_code == 200


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🧪 Testing Production Analytics MCP Server Demo")
    print("=" * 60)

    tests = [
        ("Health Check", test_health),
        ("Modules List", test_modules),
        ("Suppliers List", test_suppliers),
        ("Equipment List", test_equipment),
        ("RunRate Analytics", test_analytics_runrate),
        ("ROI Analytics", test_analytics_roi),
        ("Capacity Analytics", test_analytics_capacity),
        ("LLM Insights", test_llm_insights),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("📋 Test Results Summary")
    print("=" * 60)

    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status} - {name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n   Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Server is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check server status.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to server at " + BASE_URL)
        print("   Make sure the server is running:")
        print("   python demo_mcp_server.py\n")
