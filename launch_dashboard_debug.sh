#!/bin/bash

# Aegis Strategy Room - Troubleshooting Launcher
# ===============================================
# Use this script if the normal launcher fails to connect

echo "🔧 Aegis Dashboard - Troubleshooting Mode"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check directory
if [ ! -f "src/dashboard/app.py" ]; then
    echo "❌ Error: Must run from project root"
    exit 1
fi

# Function to check if port is in use
check_port() {
    if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo "⚠️  Port 8501 is already in use!"
        echo "   Existing process:"
        lsof -Pi :8501 -sTCP:LISTEN
        echo ""
        echo "   To kill existing process:"
        echo "   kill -9 \$(lsof -t -i:8501)"
        echo ""
        read -p "   Kill existing process? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kill -9 $(lsof -t -i:8501) 2>/dev/null
            echo "   ✅ Process killed"
            sleep 1
        else
            echo "   ❌ Aborting"
            exit 1
        fi
    fi
}

# Function to get local IP
get_local_ip() {
    LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)
    echo "$LOCAL_IP"
}

# Step 1: Check port availability
echo "Step 1: Checking port 8501..."
check_port
echo "✅ Port 8501 is available"
echo ""

# Step 2: Get network info
echo "Step 2: Network information"
LOCAL_IP=$(get_local_ip)
echo "   Local IP: ${LOCAL_IP:-Not detected}"
echo ""

# Step 3: Display connection methods
echo "Step 3: Connection URLs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Try these URLs in order:"
echo ""
echo "   1️⃣  http://localhost:8501          (Standard)"
echo "   2️⃣  http://127.0.0.1:8501          (IPv4 direct)"
if [ -n "$LOCAL_IP" ]; then
    echo "   3️⃣  http://${LOCAL_IP}:8501        (Network IP)"
fi
echo ""
echo "💡 Recommended: Use option 2 (127.0.0.1) for best compatibility"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 4: Launch with verbose output
echo "Step 4: Launching dashboard with verbose logging..."
echo "   Press Ctrl+C to stop"
echo ""

# Run with maximum compatibility settings
STREAMLIT_LOG_LEVEL=debug streamlit run src/dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless false \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.enableWebsocketCompression true \
    --browser.gatherUsageStats false \
    --logger.level debug \
    2>&1 | tee streamlit_debug.log

# Capture exit status
EXIT_STATUS=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ Dashboard closed normally"
else
    echo "❌ Dashboard exited with error (code: $EXIT_STATUS)"
    echo ""
    echo "Debug log saved to: streamlit_debug.log"
    echo ""
    echo "Common issues:"
    echo "   • Missing dependencies: pip install -r requirements-dashboard.txt"
    echo "   • Module not found: Ensure you're in project root directory"
    echo "   • Port in use: Run 'lsof -ti:8501 | xargs kill -9'"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
