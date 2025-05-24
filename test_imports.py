# test_imports.py
try:
    # Test basic imports
    from utils.logger import get_logger
    print("✓ Logger import successful")

    from clients.base_client import BaseClient
    print("✓ Base client import successful")

    from client_factory import ClientFactory
    print("✓ Client factory import successful")

    from test_executor import TestExecutor
    print("✓ Test executor import successful")

    print("\n✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")