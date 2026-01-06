#!/bin/bash

# Aegis Strategy Room - Dashboard Launcher
# =========================================

echo "🚀 Starting Aegis Strategy Room..."
echo ""

# Check if in correct directory
if [ ! -f "src/dashboard/app.py" ]; then
    echo "❌ Error: Must run from project root directory"
    echo "   cd /Users/ekim56/Desktop/aegis-pitching-engine"
    exit 1
fi

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Streamlit not installed. Installing dependencies..."
    pip install -r requirements-dashboard.txt
fi

# Display connection options
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📡 Dashboard will be available at multiple addresses:"
echo ""
echo "   ✅ Primary:   http://localhost:8501"
echo "   ✅ IPv4:      http://127.0.0.1:8501"
echo "   ✅ Network:   http://[Your-Local-IP]:8501"
echo ""
echo "💡 If localhost doesn't work, try 127.0.0.1 instead"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔄 Starting Streamlit server..."
echo "   Press Ctrl+C to stop"
echo ""

# Launch dashboard with enhanced network binding
# --server.address=0.0.0.0 allows access from all network interfaces
# This resolves IPv4/IPv6 conflicts and allows external access
streamlit run src/dashboard/app.py \
    --server.port 8501 \
    --server.address=0.0.0.0 \
    --server.headless false \
    --browser.gatherUsageStats false \
    --server.enableCORS false \
    --server.enableXsrfProtection false

# If the script reaches here, streamlit has stopped
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  Dashboard stopped"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
