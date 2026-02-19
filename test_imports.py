# test_imports.py
print("🔍 Testing imports...\n")

try:
    from app.information_loader import InformationLoader
    print("✅ information_loader - OK")
except Exception as e:
    print(f"❌ information_loader - Failed: {e}")

try:
    from app.rag_pipeline import ProductionRAGPipeline
    print("✅ rag_pipeline - OK")
except Exception as e:
    print(f"❌ rag_pipeline - Failed: {e}")

try:
    from app.config import CONFIG
    print("✅ config - OK")
except Exception as e:
    print(f"❌ config - Failed: {e}")

try:
    from app.main import app
    print("✅ main - OK")
except Exception as e:
    print(f"❌ main - Failed: {e}")

print("\n✅ Import test complete!")