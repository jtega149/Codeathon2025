
# To run use uvicorn main:app --reload
# or uvicorn main:app --reload --port 8000

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Load environment variables from project root
import pathlib
project_root = pathlib.Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / ".env")

# Helper function to replace NaN values with None for JSON serialization
def clean_dict_for_json(data):
    """Recursively replace NaN values with None in dictionaries"""
    if isinstance(data, dict):
        return {k: clean_dict_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_dict_for_json(item) for item in data]
    elif pd.isna(data):
        return None
    elif isinstance(data, (np.integer, np.floating)):
        if pd.isna(data):
            return None
        return data.item() if hasattr(data, 'item') else data
    else:
        return data

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset once at startup
PATH_TO_CSV = "../data/final_ds.csv" # Adjust path as needed
df = pd.read_csv(PATH_TO_CSV)

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print(f"Gemini API key loaded (key length: {len(GEMINI_API_KEY)})")
        
        # List available models to see what's accessible
        print("Checking available models...")
        models = genai.list_models()
        available_models = []
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                model_name = m.name.split('/')[-1]  # Get just the model name
                available_models.append(model_name)
                print(f"  - {model_name}")
        
        if not available_models:
            print("WARNING: No models found with generateContent support!")
        else:
            # Try different model names in order of preference
            # Use models that are actually available from the list (prioritize these!)
            model_names_to_try = [
                'gemini-2.5-flash',      # Latest stable flash model (AVAILABLE)
                'gemini-2.0-flash',      # Alternative flash model (AVAILABLE)
                'gemini-flash-latest',   # Latest flash alias (AVAILABLE)
                'gemini-2.5-pro',        # Latest pro model (AVAILABLE)
                'gemini-pro-latest',     # Latest pro alias (AVAILABLE)
            ]
            
            # Also try any models from the available list that we haven't tried
            for model_name in available_models:
                if model_name not in model_names_to_try and 'flash' in model_name.lower():
                    model_names_to_try.append(model_name)
            
            for model_name in model_names_to_try:
                try:
                    test_model = genai.GenerativeModel(model_name)
                    # Actually test the model with a simple request to verify it works
                    try:
                        test_response = test_model.generate_content("test")
                        gemini_model = test_model
                        print(f"✓ Successfully initialized and tested model: {model_name}")
                        break
                    except Exception as test_error:
                        print(f"✗ Model {model_name} initialized but failed test: {str(test_error)[:100]}")
                        continue
                except Exception as e:
                    print(f"✗ Failed to initialize {model_name}: {str(e)[:100]}")
                    continue
            
            if not gemini_model:
                print("ERROR: Could not initialize any Gemini model!")
                print("This might indicate:")
                print("  1. API key is invalid or expired")
                print("  2. API key doesn't have access to these models")
                print("  3. Model names have changed")
    except Exception as e:
        print(f"ERROR configuring Gemini API: {str(e)}")
        print("This might indicate an invalid or expired API key")
else:
    print("Warning: GEMINI_API_KEY not found in environment variables")

@app.get("/")
def root():
    return {"message": "FastAPI backend is running!"}

@app.get("/entity_ids")
def get_entity_ids():
    ids = df["entity_id"].tolist()
    return {"entity_ids": ids}

@app.get("/submission-data")
def get_submission_data():
    """Returns the submission.csv data for scatter plot visualization"""
    submission_path = project_root / "notebooks" / "submission.csv"
    if not submission_path.exists():
        raise HTTPException(status_code=404, detail="Submission data not found")
    submission_df = pd.read_csv(submission_path)
    records = submission_df.to_dict(orient="records")
    return {"data": clean_dict_for_json(records)}

@app.get("/company/{entity_id}")
def get_company(entity_id: int):
    # Returns the first row corresponding to the given entity_id
    # (some entities may have multiple rows)
    row = df[df["entity_id"] == entity_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Entity not found")
    data = row.iloc[0].to_dict()
    return clean_dict_for_json(data)

@app.get("/comparisons/{entity_id}")
def get_comparisons(entity_id: int, n: int = 5):
    """
    Returns n random records from the dataset excluding the current entity_id.
    Default is 5 comparisons.
    """
    other_records = df[df["entity_id"] != entity_id]
    sample = other_records.sample(n=min(n, len(other_records)))  # handle small datasets
    records = sample.to_dict(orient="records")
    return {"comparisons": clean_dict_for_json(records)}


# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    entity_id: int
    message: str
    conversation_history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    response: str


@app.post("/ai/chat", response_model=ChatResponse)
def chat_with_ai(request: ChatRequest):
    """
    Chat endpoint that uses Gemini to answer questions about a specific entity's emissions data.
    """
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini API not configured")
    
    # Get the entity record
    entity_row = df[df["entity_id"] == request.entity_id]
    if entity_row.empty:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    entity_data = entity_row.iloc[0].to_dict()
    
    # Helper function to safely format values for context
    def safe_format(value, default='N/A', format_type='str'):
        if pd.isna(value) or value is None:
            return default
        if format_type == 'float':
            try:
                return f"{float(value):,.0f}"
            except (ValueError, TypeError):
                return default
        return str(value)
    
    # Format entity data for context
    context = f"""
You are an AI sustainability advisor specialized in analyzing corporate emissions data and providing actionable recommendations.

Current Entity Data:
- Entity ID: {safe_format(entity_data.get('entity_id'))}
- Region: {safe_format(entity_data.get('region_name'))}
- Country: {safe_format(entity_data.get('country_name'))}
- Revenue: ${safe_format(entity_data.get('revenue'), format_type='float')}
- Overall Score: {safe_format(entity_data.get('overall_score'))}
- Environmental Score: {safe_format(entity_data.get('environmental_score'))}
- Social Score: {safe_format(entity_data.get('social_score'))}
- Governance Score: {safe_format(entity_data.get('governance_score'))}
- Target Scope 1 Emissions: {safe_format(entity_data.get('target_scope_1'))} tCO₂e
- Target Scope 2 Emissions: {safe_format(entity_data.get('target_scope_2'))} tCO₂e
- Industry (NACE Level 1): {safe_format(entity_data.get('nace_level_1_name'))}
- Industry (NACE Level 2): {safe_format(entity_data.get('nace_level_2_name'))}
- Activity Type: {safe_format(entity_data.get('activity_type'))}
- Revenue Percentage: {safe_format(entity_data.get('revenue_pct'))}
- Environmental Score Adjustment: {safe_format(entity_data.get('env_score_adjustment'))}

Your role is to:
1. Understand and explain the emissions data for this company
2. Suggest improvement strategies based on their current performance
3. Answer questions about sustainability goals and best practices
4. Provide specific, actionable recommendations

Be concise, professional, and data-driven in your responses.
"""
    
    # Build conversation history
    conversation_text = context + "\n\n"
    
    # Add conversation history if provided
    if request.conversation_history:
        for msg in request.conversation_history[-5:]:  # Keep last 5 messages for context
            conversation_text += f"{msg.role.capitalize()}: {msg.content}\n"
    
    # Add current user message
    conversation_text += f"User: {request.message}\n\nAssistant:"
    
    try:
        response = gemini_model.generate_content(conversation_text)
        # Handle different response formats
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            response_text = response.candidates[0].content.parts[0].text
        else:
            response_text = str(response)
        return ChatResponse(response=response_text)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in chat_with_ai: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


class RecommendationResponse(BaseModel):
    recommendations: List[dict]


@app.get("/ai/recommendations/{entity_id}", response_model=RecommendationResponse)
def get_ai_recommendations(entity_id: int):
    """
    Generate automatic AI recommendations for a specific entity.
    Returns up to 3 recommendations with hardcoded impact values for proof of concept.
    """
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini API not configured")
    
    # Get the entity record
    entity_row = df[df["entity_id"] == entity_id]
    if entity_row.empty:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    entity_data = entity_row.iloc[0].to_dict()
    
    # Helper function to safely format values for context
    def safe_format(value, default='N/A', format_type='str'):
        if pd.isna(value) or value is None:
            return default
        if format_type == 'float':
            try:
                return f"{float(value):,.0f}"
            except (ValueError, TypeError):
                return default
        return str(value)
    
    # Format entity data for context
    context = f"""
You are an AI sustainability advisor. Analyze the following company data and provide exactly 3 specific, actionable recommendations to improve their sustainability performance.

Company Data:
- Entity ID: {safe_format(entity_data.get('entity_id'))}
- Region: {safe_format(entity_data.get('region_name'))}
- Country: {safe_format(entity_data.get('country_name'))}
- Revenue: ${safe_format(entity_data.get('revenue'), format_type='float')}
- Overall Score: {safe_format(entity_data.get('overall_score'))}
- Environmental Score: {safe_format(entity_data.get('environmental_score'))}
- Social Score: {safe_format(entity_data.get('social_score'))}
- Governance Score: {safe_format(entity_data.get('governance_score'))}
- Target Scope 1 Emissions: {safe_format(entity_data.get('target_scope_1'))} tCO₂e
- Target Scope 2 Emissions: {safe_format(entity_data.get('target_scope_2'))} tCO₂e
- Industry (NACE Level 1): {safe_format(entity_data.get('nace_level_1_name'))}
- Industry (NACE Level 2): {safe_format(entity_data.get('nace_level_2_name'))}
- Activity Type: {safe_format(entity_data.get('activity_type'))}

Provide exactly 3 recommendations. For each recommendation, provide:
1. A clear, concise title (max 10 words)
2. A detailed description explaining why this recommendation is relevant and what actions to take (2-3 sentences)
3. A category (choose from: Energy, Operations, Governance, Transport, Supply Chain, Waste Management, or Other)

Format your response as a JSON array with this exact structure:
[
  {{
    "title": "Recommendation title",
    "description": "Detailed description of the recommendation and why it's relevant",
    "category": "Category name"
  }},
  ...
]

Return ONLY the JSON array, no additional text.
"""
    
    try:
        response = gemini_model.generate_content(context)
        # Handle different response formats
        if hasattr(response, 'text'):
            response_text = response.text.strip()
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            response_text = response.candidates[0].content.parts[0].text.strip()
        else:
            response_text = str(response).strip()
        
        # Clean up the response (remove markdown code blocks if present)
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        recommendations = json.loads(response_text)
        
        # Ensure we have exactly 3 recommendations
        if len(recommendations) > 3:
            recommendations = recommendations[:3]
        
        # Add hardcoded impact values for proof of concept
        impact_levels = ["high", "medium", "low"]
        estimated_reductions = ["-2,800 tCO₂e/year", "-1,500 tCO₂e/year", "-900 tCO₂e/year"]
        
        for i, rec in enumerate(recommendations):
            rec["impact"] = impact_levels[i] if i < len(impact_levels) else "medium"
            rec["estimatedReduction"] = estimated_reductions[i] if i < len(estimated_reductions) else "-500 tCO₂e/year"
        
        return RecommendationResponse(recommendations=recommendations)
    except json.JSONDecodeError as e:
        # Fallback if JSON parsing fails
        return RecommendationResponse(recommendations=[
            {
                "title": "Optimize Energy Consumption",
                "description": "Based on your current emissions profile, implementing energy efficiency measures could significantly reduce your carbon footprint.",
                "category": "Energy",
                "impact": "high",
                "estimatedReduction": "-2,800 tCO₂e/year"
            },
            {
                "title": "Enhance Environmental Governance",
                "description": "Improving your governance structure around environmental management could help you achieve better sustainability scores.",
                "category": "Governance",
                "impact": "medium",
                "estimatedReduction": "-1,500 tCO₂e/year"
            },
            {
                "title": "Supply Chain Optimization",
                "description": "Engaging with suppliers to reduce upstream emissions can improve your overall sustainability performance.",
                "category": "Supply Chain",
                "impact": "low",
                "estimatedReduction": "-900 tCO₂e/year"
            }
        ])
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_ai_recommendations: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")