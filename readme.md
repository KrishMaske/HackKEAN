# ⚡ PRODUCTINTEL SPOTLIGHT
> **The Goated Product Placement & ROI Intelligence Engine.**

ProductIntel Spotlight is a high-fidelity analytics platform designed to quantify the impact of product placements in video content. By combining **Llama-4** vision tracking with **Gemma-4** marketing intelligence, Spotlight transforms raw video into actionable executive strategy.

![Spotlight Dashboard](https://raw.githubusercontent.com/KrishMaske/HackKEAN/main/docs/assets/dashboard_preview.png) *(Placeholder: Update with real screenshot)*

---

## 🚀 Key Features

### 1. Neural Masking Pipeline (SAM3)
*   **Keyframe Interpolation**: Uses `meta-llama/llama-4-scout` to identify and lock onto products every 1 second.
*   **Temporal Precision**: High-accuracy bounding box interpolation handles scene cuts, camera pans, and motion blur.
*   **Neural Alpha Masks**: Generates frame-by-frame transparency masks for every tracked product.

### 2. Marketing Intelligence (Agentic ROI)
*   **Scene Analyst**: Examines character-product interactions and emotional context.
*   **Market Analyst**: Quantifies visibility rate, screen coverage, and airtime.
*   **Ad Strategist**: Brainstorms campaign optimizations and revenue attribution ideas.
*   *Powered by `gemma-4-31b-it` (Dense Architecture).*

### 3. Executive Dashboard
*   **Attention Retention Analysis**: Interactive second-by-second exposure charts.
*   **High-Density Metrics**: Visibility scores, Peak Frame Coverage, and Airtime stats.
*   **Executive Strategy HUD**: AI-generated strategic insights and campaign optimization logs.

---

## 🛠️ Technology Stack

*   **Backend**: FastAPI, Python 3.10+
*   **Frontend**: Next.js 14, TailwindCSS, Framer Motion
*   **AI Engine**: LangGraph (Orchestration), Groq (Vision), Google Vertex/Gemma 4 (Reasoning)
*   **Vision**: OpenCV, PIL, Llama-4-Scout

---

## 🚦 Getting Started

### 1. Requirements
*   Python 3.10+
*   Node.js 18+
*   Groq API Key
*   Google Cloud Credentials (for Gemma 4)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/KrishMaske/HackKEAN.git

# Setup Backend
cd backend
pip install -r requirements.txt
cp .env.example .env # Add your keys

# Setup Frontend
cd ../frontend
npm install
```

### 3. Execution
```powershell
# Run Backend (Port 8000)
cd backend
python main.py

# Run Frontend (Port 3000)
cd frontend
npm run dev
```

---

## 🧪 Pipeline Flow
1.  **Ingestion**: Scene is uploaded; `Llama-4` identifies the most prominent product.
2.  **Rendering**: Neural mask video is generated via keyframe interpolation.
3.  **Analysis**: ROI stats are calculated from the rendered masks.
4.  **Strategy**: `Gemma-4` agents brainstorm executive insights based on the ROI data.
5.  **Delivery**: Dashboard populates with the complete "Proper" analytical story.

---

## 🏛️ Project Structure
*   `backend/app/agents/`: Marketing Agent definitions.
*   `backend/app/services/masking.py`: The core temporal tracking engine.
*   `backend/app/services/orchestrator.py`: Agentic workflow management.
*   `frontend/src/app/page.tsx`: The high-fidelity React dashboard.

---
*Created with ⚡ by the ProductIntel Team.*
