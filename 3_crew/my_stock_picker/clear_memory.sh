#!/bin/bash
# Clear all CrewAI memory for my_stock_picker

echo "Clearing CrewAI memory..."

if [ -d "memory" ]; then
    rm -rf memory/
    echo "✅ Memory cleared successfully!"
    echo ""
    echo "Deleted:"
    echo "  - Long-term memory (SQLite)"
    echo "  - Short-term memory (FAISS)"
    echo "  - Entity memory (FAISS)"
    echo ""
    echo "Memory will be recreated on next 'crewai run'"
else
    echo "⚠️  No memory folder found (already clear)"
fi
