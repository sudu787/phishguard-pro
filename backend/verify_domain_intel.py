import asyncio
import json
from app.services.domain_intel import DomainIntelService

async def main():
    service = DomainIntelService()
    print("Testing Domain Intelligence Service...")
    
    # 1. Test a safe domain (Google)
    print("\n--- Test: Safe Domain (google.com) ---")
    res1 = await service.analyze("https://google.com")
    print(json.dumps(res1, indent=2))
    
    # 2. Test a missing MX domain (example.com)
    print("\n--- Test: Missing MX (example.com) ---")
    res2 = await service.analyze("https://example.com")
    print(json.dumps(res2, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
