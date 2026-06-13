import asyncio
import json
from app.services.soc_agent import SOCAgentService

async def main():
    agent = SOCAgentService()
    print("Testing AI SOC Analyst Agent...\n")

    # 1. Test a safe scenario
    safe_payload = {
        "url": "https://google.com/search?q=hello",
        "url_analysis": {"score": 0.01, "verdict": "safe"},
        "domain_intel": {"age_days": 10000, "is_nrd": False, "is_suspicious_registrar": False},
        "html_analysis": {"has_password_form": False, "criticalThreat": False},
        "vision_analysis": {"detected_logos": ["Google"]}
    }
    
    print("--- Scenario 1: Safe Search Page (google.com) ---")
    res1 = await agent.analyze(safe_payload)
    print(json.dumps(res1, indent=2))
    
    # 2. Test a phishing scenario (Microsoft credential harvester)
    phish_payload = {
        "url": "https://microsoft-secure-login.xyz/auth",
        "url_analysis": {"score": 0.85, "verdict": "phishing"},
        "domain_intel": {"age_days": 2, "is_nrd": True, "is_suspicious_registrar": True},
        "html_analysis": {"has_password_form": True, "floating_password_input": True},
        "vision_analysis": {"detected_logos": ["Microsoft"], "has_login_prompt": True}
    }
    
    print("\n--- Scenario 2: Microsoft Phishing Attack ---")
    res2 = await agent.analyze(phish_payload)
    print(json.dumps(res2, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
