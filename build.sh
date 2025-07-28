#!/bin/bash
# Build script for deployment

echo "🔧 Installing build dependencies..."
pip install --upgrade setuptools wheel pip

echo "📦 Installing requirements..."
pip install -r requirements.txt

echo "✅ Build complete!"
