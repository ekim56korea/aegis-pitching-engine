#!/bin/bash

# Quick connectivity test for Streamlit dashboard
# This script checks if the dashboard can be launched and accessed

echo "🧪 Testing Aegis Dashboard Connectivity..."
echo ""

# Start dashboard in background
echo "1️⃣  Starting dashboard in background..."
./launch_dashboard.sh > /dev/null 2>&1 &
DASHBOARD_PID=$!

# Wait for startup
echo "2️⃣  Waiting for server to start (10 seconds)..."
sleep 10

# Test connectivity
echo "3️⃣  Testing connectivity..."
echo ""

# Test localhost
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ localhost:8501 - OK"
else
    echo "❌ localhost:8501 - FAILED"
fi

# Test 127.0.0.1
if curl -s http://127.0.0.1:8501 > /dev/null; then
    echo "✅ 127.0.0.1:8501 - OK"
else
    echo "❌ 127.0.0.1:8501 - FAILED"
fi

echo ""
echo "4️⃣  Recommended URL: http://127.0.0.1:8501"
echo ""

# Stop dashboard
echo "5️⃣  Stopping test dashboard..."
kill $DASHBOARD_PID 2>/dev/null

# Check if streamlit is still running
sleep 2
if pgrep -f "streamlit run" > /dev/null; then
    echo "⚠️  Cleaning up remaining processes..."
    pkill -f "streamlit run"
fi

echo ""
echo "✅ Test complete!"
echo ""
echo "To launch dashboard normally:"
echo "  ./launch_dashboard.sh"
