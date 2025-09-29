import os
import re
import cv2
import tempfile
import numpy as np
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# FastAPI and web framework imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# ML and processing imports
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from textblob import TextBlob
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

try:
    from PIL import Image
    from sklearn.ensemble import IsolationForest
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False


# Initialize FastAPI app
app = FastAPI(
    title="Simple AI/Human Detector",
    description="Single-page AI content detector for text, images, and videos",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
text_classifier = None
image_anomaly_detector = None


class SimpleDetector:
    """Combined detector class for all content types"""

    def __init__(self):
        self.initialized = False

    async def initialize(self):
        """Initialize all detection models"""
        global text_classifier, image_anomaly_detector

        print("🚀 Initializing AI/Human Detector...")

        # Initialize text classifier
        if TRANSFORMERS_AVAILABLE:
            try:
                text_classifier = pipeline(
                    "text-classification",
                    model="Hello-SimpleAI/chatgpt-detector-roberta",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("✅ Text classifier loaded (Transformer)")
            except Exception as e:
                print(f"⚠️  Transformer model failed: {e}")
                text_classifier = None
        else:
            print("⚠️  Transformers not available, using heuristic text analysis")

        # Initialize NLTK data
        if NLTK_AVAILABLE:
            try:
                # Attempt to ensure some common datasets are present
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    print("📥 Downloading punkt...")
                    nltk.download('punkt', quiet=True)

                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    print("📥 Downloading stopwords...")
                    nltk.download('stopwords', quiet=True)

                # vader_lexicon used by some sentiment packages; try to fetch if missing
                try:
                    nltk.data.find('sentiment/vader_lexicon')
                except LookupError:
                    print("📥 Downloading vader_lexicon...")
                    nltk.download('vader_lexicon', quiet=True)

                print("✅ NLTK data ready")
            except Exception as e:
                print(f"⚠️  NLTK setup failed: {e}")

        # Initialize image anomaly detector
        if ML_AVAILABLE:
            try:
                image_anomaly_detector = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=50
                )
                # Pre-fit with dummy data
                dummy_features = np.random.rand(50, 10)
                image_anomaly_detector.fit(dummy_features)
                print("✅ Image anomaly detector ready")
            except Exception as e:
                print(f"⚠️  Image anomaly init failed: {e}")
                image_anomaly_detector = None

        self.initialized = True
        print("🎉 Detector initialization complete!")

    def detect_text(self, text: str) -> Dict[str, Any]:
        """Detect if text is AI-generated"""
        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Clean text
        cleaned_text = re.sub(r'\s+', ' ', text.strip())

        # Try transformer model first
        if text_classifier:
            try:
                # Limit text length for model
                model_text = cleaned_text[:2000]
                result = text_classifier(model_text)

                if isinstance(result, list) and result:
                    result = result[0]

                label = result.get('label') if isinstance(result, dict) else None
                confidence = float(result.get('score', 0.0)) if isinstance(result, dict) else 0.0

                # Heuristic mapping for label names (model labels vary)
                is_ai = False
                if label:
                    label_norm = str(label).lower()
                    if any(k in label_norm for k in ['ai', 'generated', 'fake', '1', 'gpt']):
                        is_ai = True

                return {
                    'prediction': 'AI' if is_ai else 'HUMAN',
                    'confidence': round(confidence, 3),
                    'ai_probability': round(confidence if is_ai else 1 - confidence, 3),
                    'human_probability': round(1 - confidence if is_ai else confidence, 3),
                    'method': 'transformer',
                    'features': self._extract_text_features(text)
                }
            except Exception as e:
                print(f"Transformer failed, using heuristics: {e}")

        # Fallback to heuristic analysis
        return self._heuristic_text_analysis(text)

    def _heuristic_text_analysis(self, text: str) -> Dict[str, Any]:
        """Simple heuristic-based text analysis"""
        features = self._extract_text_features(text)

        ai_score = 0.0
        reasons = []

        # Check various indicators
        if features.get('avg_sentence_length', 0) > 25:
            ai_score += 0.2
            reasons.append("Long average sentence length")

        if features.get('vocabulary_diversity', 1.0) < 0.3:
            ai_score += 0.3
            reasons.append("Low vocabulary diversity")

        if features.get('formal_word_count', 0) > 3:
            ai_score += 0.2
            reasons.append("High formal language usage")

        # Repetition check
        words = [w for w in re.findall(r"\w+", text.lower())]
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.7:
                ai_score += 0.3
                reasons.append("High word repetition")

        # clamp
        ai_score = min(1.0, ai_score)
        # confidence mapping (ensure at least 0.6 as baseline)
        if ai_score > 0.5:
            confidence = max(0.6, ai_score)
        else:
            confidence = max(0.6, 1 - ai_score)

        return {
            'prediction': 'AI' if ai_score > 0.5 else 'HUMAN',
            'confidence': round(confidence, 3),
            'ai_probability': round(ai_score, 3),
            'human_probability': round(1 - ai_score, 3),
            'method': 'heuristic',
            'reasoning': reasons,
            'features': features
        }

    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract basic text features"""
        words = re.findall(r"\w+", text)
        sentences = re.split(r'[.!?]+', text)

        formal_words = [
            'however', 'therefore', 'furthermore', 'moreover',
            'consequently', 'specifically', 'particularly', 'essentially'
        ]
        formal_count = sum(1 for word in formal_words if word in text.lower())

        features = {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': float(np.mean([len(w) for w in words])) if words else 0.0,
            'avg_sentence_length': (len(words) / len([s for s in sentences if s.strip()])) if sentences and any(s.strip() for s in sentences) else 0.0,
            'vocabulary_diversity': (len(set(w.lower() for w in words)) / len(words)) if words else 0.0,
            'formal_word_count': formal_count
        }

        # Add sentiment if TextBlob available
        if NLTK_AVAILABLE:
            try:
                blob = TextBlob(text)
                features['sentiment_polarity'] = float(blob.sentiment.polarity)
                features['sentiment_subjectivity'] = float(blob.sentiment.subjectivity)
            except Exception:
                pass

        return features

    def detect_image(self, image_path: str) -> Dict[str, Any]:
        """Detect if image is AI-generated"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract features
            features = self._extract_image_features(image_rgb)

            # Add file size if possible
            try:
                features['file_size_mb'] = os.path.getsize(image_path) / (1024 * 1024)
            except Exception:
                features['file_size_mb'] = 0.0

            # Simple rule-based analysis
            ai_indicators = []
            human_indicators = []

            # Check various indicators
            if features.get('color_diversity', 0) < 0.3:
                ai_indicators.append("Low color diversity")
            else:
                human_indicators.append("Good color diversity")

            if features.get('edge_density', 0) > 0.15:
                ai_indicators.append("High edge density")

            if features.get('file_size_mb', 0) < 0.5:
                ai_indicators.append("Very small file size")

            # Use anomaly detector if available
            anomaly_score = 0.0
            if image_anomaly_detector is not None and ML_AVAILABLE:
                try:
                    feature_vals = [features.get(k, 0) for k in
                                    ['color_diversity', 'edge_density', 'contrast',
                                     'r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std', 'file_size_mb']]
                    feature_vector = np.array(feature_vals).reshape(1, -1)
                    anomaly_score = float(image_anomaly_detector.decision_function(feature_vector)[0])

                    if anomaly_score < -0.1:
                        ai_indicators.append("Anomalous statistical patterns")
                except Exception:
                    pass

            # Calculate final prediction
            ai_score = len(ai_indicators) / (len(ai_indicators) + len(human_indicators) + 1)
            if anomaly_score < -0.1:
                ai_score += 0.3

            ai_score = min(1.0, ai_score)
            confidence = max(0.6, ai_score) if ai_score > 0.5 else max(0.6, 1 - ai_score)

            return {
                'prediction': 'AI' if ai_score > 0.5 else 'HUMAN',
                'confidence': round(confidence, 3),
                'ai_probability': round(ai_score, 3),
                'human_probability': round(1 - ai_score, 3),
                'method': 'statistical_analysis',
                'ai_indicators': ai_indicators,
                'human_indicators': human_indicators,
                'features': features,
                'anomaly_score': round(anomaly_score, 3)
            }

        except Exception as e:
            raise ValueError(f"Image analysis failed: {str(e)}")

    def _extract_image_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract basic image features"""
        features = {}

        try:
            height, width, channels = image.shape
            features['width'] = int(width)
            features['height'] = int(height)
            features['aspect_ratio'] = float(width / height) if height != 0 else 0.0

            # Color statistics
            for i, channel in enumerate(['r', 'g', 'b']):
                channel_data = image[:, :, i]
                features[f'{channel}_mean'] = float(np.mean(channel_data))
                features[f'{channel}_std'] = float(np.std(channel_data))

            # Color diversity
            unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
            features['color_diversity'] = float(unique_colors) / float(width * height) if width * height > 0 else 0.0

            # Edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = float(np.sum(edges > 0) / (width * height)) if width * height > 0 else 0.0

            # Contrast
            features['contrast'] = float(np.std(gray))

        except Exception as e:
            print(f"Feature extraction error: {e}")

        return features

    def detect_video(self, video_path: str) -> Dict[str, Any]:
        """Detect if video is AI-generated (simplified frame analysis)"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video file")

            # Get basic metadata
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = (frame_count / fps) if fps > 0 else 0.0

            # Analyze first few frames
            frame_analyses = []
            max_frames = min(5, frame_count) if frame_count > 0 else 0

            for i in range(max_frames):
                # position evenly across video
                pos = int(i * (frame_count // max_frames)) if max_frames > 0 else 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame temporarily and analyze
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, frame)
                    try:
                        frame_result = self.detect_image(tmp_file.name)
                        frame_analyses.append(frame_result.get('ai_probability', 0.0))
                    except Exception:
                        pass
                    finally:
                        try:
                            os.unlink(tmp_file.name)
                        except Exception:
                            pass

            cap.release()

            # Aggregate results
            if frame_analyses:
                avg_ai_prob = float(np.mean(frame_analyses))
                confidence = max(0.6, avg_ai_prob) if avg_ai_prob > 0.5 else max(0.6, 1 - avg_ai_prob)
            else:
                avg_ai_prob = 0.5
                confidence = 0.5

            # File-based indicators
            try:
                file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            except Exception:
                file_size_mb = 0.0

            filename = os.path.basename(video_path).lower()

            ai_filename_indicators = sum(1 for keyword in ['ai', 'generated', 'deepfake', 'synthetic']
                                        if keyword in filename)

            reasoning = []
            if avg_ai_prob > 0.7:
                reasoning.append("High AI probability in frame analysis")
            if duration < 10 and duration > 0:
                reasoning.append("Very short video duration")
                avg_ai_prob = min(1.0, avg_ai_prob + 0.1)
            if file_size_mb < 5 and file_size_mb > 0:
                reasoning.append("Small file size for video")
            if ai_filename_indicators > 0:
                reasoning.append("AI-related keywords in filename")
                avg_ai_prob = min(1.0, avg_ai_prob + 0.2)

            return {
                'prediction': 'AI' if avg_ai_prob > 0.5 else 'HUMAN',
                'confidence': round(confidence, 3),
                'ai_probability': round(avg_ai_prob, 3),
                'human_probability': round(1 - avg_ai_prob, 3),
                'method': 'frame_analysis',
                'frames_analyzed': len(frame_analyses),
                'reasoning': reasoning,
                'video_info': {
                    'duration_seconds': round(duration, 2),
                    'resolution': f"{width}x{height}",
                    'fps': round(fps, 1),
                    'file_size_mb': round(file_size_mb, 2)
                }
            }

        except Exception as e:
            raise ValueError(f"Video analysis failed: {str(e)}")


# Initialize detector
detector = SimpleDetector()


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    await detector.initialize()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Home page with simple UI"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI/Human Detector</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 10px 0; }
            input, textarea, button { margin: 10px 0; padding: 10px; width: 100%; box-sizing: border-box; }
            button { background: #007cba; color: white; border: none; cursor: pointer; border-radius: 4px; }
            button:hover { background: #005a9e; }
            .result { background: #e8f4fd; padding: 15px; border-radius: 4px; margin: 10px 0; }
            .ai { background: #ffe6e6; }
            .human { background: #e6ffe6; }
        </style>
    </head>
    <body>
        <h1>🤖 AI/Human Content Detector</h1>

        <div class="container">
            <h2>📝 Text Analysis</h2>
            <textarea id="textInput" placeholder="Enter text to analyze..." rows="4"></textarea>
            <button onclick="analyzeText()">Analyze Text</button>
            <div id="textResult"></div>
        </div>

        <div class="container">
            <h2>🖼️ Image Analysis</h2>
            <input type="file" id="imageInput" accept="image/*">
            <button onclick="analyzeImage()">Analyze Image</button>
            <div id="imageResult"></div>
        </div>

        <div class="container">
            <h2>🎥 Video Analysis</h2>
            <input type="file" id="videoInput" accept="video/*">
            <button onclick="analyzeVideo()">Analyze Video</button>
            <div id="videoResult"></div>
        </div>

        <script>
        async function analyzeText() {
            const text = document.getElementById('textInput').value;
            if (!text.trim()) return alert('Please enter some text');

            try {
                const formData = new FormData();
                formData.append('text', text);

                const response = await fetch('/detect/text', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                displayResult('textResult', result);
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        async function analyzeImage() {
            const file = document.getElementById('imageInput').files[0];
            if (!file) return alert('Please select an image');

            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/detect/image', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                displayResult('imageResult', result);
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        async function analyzeVideo() {
            const file = document.getElementById('videoInput').files[0];
            if (!file) return alert('Please select a video');

            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/detect/video', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                displayResult('videoResult', result);
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        function displayResult(elementId, result) {
            const element = document.getElementById(elementId);
            const isAI = result.prediction === 'AI';
            const className = isAI ? 'ai' : 'human';

            element.innerHTML = `
                <div class="result ${className}">
                    <h3>${isAI ? '🤖' : '👤'} Prediction: ${result.prediction}</h3>
                    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                    <p><strong>AI Probability:</strong> ${(result.ai_probability * 100).toFixed(1)}%</p>
                    <p><strong>Method:</strong> ${result.method}</p>
                    ${result.reasoning ? `<p><strong>Reasoning:</strong> ${Array.isArray(result.reasoning) ? result.reasoning.join(', ') : result.reasoning}</p>` : ''}
                </div>
            `;
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_available": {
            "transformers": TRANSFORMERS_AVAILABLE,
            "nltk": NLTK_AVAILABLE,
            "ml_libraries": ML_AVAILABLE
        },
        "initialized": detector.initialized
    }


@app.post("/detect/text")
async def detect_text_endpoint(text: str = Form(...)):
    """Analyze text content"""
    try:
        result = detector.detect_text(text)
        return {
            "status": "success",
            "content_type": "text",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/detect/image")
async def detect_image_endpoint(file: UploadFile = File(...)):
    """Analyze image content"""
    try:
        # Validate file
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            result = detector.detect_image(tmp_file_path)

            # Ensure file_size is in features (some code paths compute it)
            result['features']['file_size_mb'] = len(content) / (1024 * 1024)

            return {
                "status": "success",
                "content_type": "image",
                "filename": file.filename,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                **result
            }
        finally:
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/video")
async def detect_video_endpoint(file: UploadFile = File(...)):
    """Analyze video content"""
    try:
        # Validate file
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')):
            raise HTTPException(status_code=400, detail="Invalid video format")

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            result = detector.detect_video(tmp_file_path)

            return {
                "status": "success",
                "content_type": "video",
                "filename": file.filename,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                **result
            }
        finally:
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 Starting Simple AI/Human Detector...")
    print("📚 This single-file version combines all detection capabilities")
    print("🌐 Web interface will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("\n" + "=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")
