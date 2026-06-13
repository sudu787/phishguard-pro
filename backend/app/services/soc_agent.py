"""
app/services/soc_agent.py

AI SOC Analyst Agent Service.
Fuses URL, Domain, HTML, and Vision signals to generate an explainable risk report
using Groq Llama-3 (or a mock fallback if no API key is provided).
"""

import json
import logging
import os
from typing import Any

from pydantic import BaseModel, Field

# Try to import groq, fallback to mock mode if not installed or no key
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

logger = logging.getLogger("phishguard.soc_agent")

class SOCReportSchema(BaseModel):
    risk_score: int = Field(description="Overall risk score from 0 to 100")
    reasons: list[str] = Field(description="List of brief technical reasons explaining the score")
    user_warning: str = Field(description="A user-friendly, concise warning or reassuring message")

class SOCAgentService:
    def __init__(self):
        from app.core.config import settings
        self.api_key = settings.GROQ_API_KEY
        self.use_mock = not HAS_GROQ or not self.api_key
        
        if self.use_mock:
            logger.warning("SOCAgentService: Running in MOCK mode (no GROQ_API_KEY found or groq not installed).")
        else:
            self.client = Groq(api_key=self.api_key)
            logger.info("SOCAgentService: Initialized with Groq Llama-3.")

    async def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze the fused payload and return a structured SOC report.
        """
        if self.use_mock:
            return self._mock_analyze(payload)
            
        system_instruction = """
        You are an expert Security Operations Center (SOC) Analyst and Threat Hunter.
        You will receive a JSON payload containing intelligence from four distinct modules analyzing a URL:
        1. URL Analysis (ML score, typosquatting)
        2. Domain Intel (WHOIS age, Registrars, DNS anomalies)
        3. HTML Analysis (Hidden forms, suspicious scripts)
        4. Vision Analysis (Detected logos, UI text)

        Your job is to fuse this context, weigh the evidence objectively, and output a structured assessment.
        Beware of false positives: a newly registered domain is suspicious, but if it has no phishing keywords or logos, it might be safe. 
        A known brand logo (e.g., Microsoft) on a non-Microsoft domain (e.g., domain age 2 days, weird URL) is a CRITICAL phishing indicator.

        Return a JSON object with strictly these keys:
        - risk_score: integer from 0 to 100
        - reasons: A list of 3-5 technical string bullet points justifying the score.
        - user_warning: A single, non-technical string sentence advising the user.
        Do NOT output any markdown blocks, only raw JSON.
        """

        prompt = f"Analyze the following threat intelligence payload:\n\n{json.dumps(payload, indent=2)}"

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            result_dict = json.loads(response.choices[0].message.content)
            
            # Ensure required keys exist to match expected schema
            if "risk_score" not in result_dict: result_dict["risk_score"] = 50
            if "reasons" not in result_dict: result_dict["reasons"] = ["Analysis completed."]
            if "user_warning" not in result_dict: result_dict["user_warning"] = "Proceed with caution."
            
            return result_dict
            
        except Exception as e:
            logger.error(f"SOCAgent Groq API Error: {e}", exc_info=True)
            # Fallback on error
            return {
                "risk_score": 50,
                "reasons": [f"Analysis failed due to an internal API error: {str(e)}"],
                "user_warning": "Unable to complete advanced AI analysis. Proceed with standard caution."
            }

    def _mock_analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Deterministic mock logic for testing without an API key."""
        url = payload.get("url", "").lower()
        vision = payload.get("vision_analysis", {})
        domain = payload.get("domain_intel", {})
        
        is_phish = False
        reasons = []
        
        if vision.get("detected_logos") and len(vision["detected_logos"]) > 0:
            brand = vision["detected_logos"][0]
            if payload.get("url_analysis", {}).get("verdict") == "phishing" or domain.get("age_days", 999) < 30:
                is_phish = True
                reasons.append(f"Detected {brand} logo, but URL structure or domain age is highly suspicious.")
                
        if payload.get("url_analysis", {}).get("verdict") == "phishing":
            is_phish = True
            reasons.append("URL ML model flagged this as phishing (typosquatting or malicious patterns).")
            
        if domain.get("age_days", 999) < 30:
            reasons.append(f"Domain is very new ({domain.get('age_days')} days old).")
            if is_phish:
                pass 
                
        if payload.get("html_analysis", {}).get("has_password_form"):
            reasons.append("Password input field detected on the page.")
            
        if is_phish:
            return {
                "risk_score": 95,
                "reasons": reasons,
                "user_warning": "Warning: This site is attempting to impersonate a known brand to steal your credentials."
            }
        else:
            return {
                "risk_score": 5,
                "reasons": ["No immediate impersonation detected.", "Domain appears established or benign."],
                "user_warning": "This site appears safe, but always verify the URL before entering passwords."
            }
