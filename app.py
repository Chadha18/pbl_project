from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os
import traceback
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for API calls

# Global variable to store loaded models
MODELS = None
MODEL_FEATURE_ORDER = None

def check_dependencies():
    """Check if all required libraries are installed"""
    required_libs = ['flask', 'pandas', 'numpy', 'flask_cors', 'joblib']
    missing_libs = []
    
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"✅ {lib} - OK")
        except ImportError:
            missing_libs.append(lib)
            print(f"❌ {lib} - MISSING")
    
    if missing_libs:
        print(f"\n⚠️  Missing libraries: {missing_libs}")
        print("Install with: pip install " + " ".join(missing_libs))
    
    return len(missing_libs) == 0

def extract_feature_names(model):
    """Extract feature names from a trained model"""
    # For sklearn models with feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    
    # For XGBoost models
    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
        if hasattr(booster, 'feature_names'):
            return booster.feature_names
    
    # For pipeline models
    if hasattr(model, 'named_steps'):
        for step_name, step in reversed(model.named_steps.items()):
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    
    # Default fallback order based on your error message
    return ['Depth', 'RxoRt', 'RILD', 'MN', 'GR', 'CNLS', 'RHOC']

def load_models():
    """Load all the trained models from joblib files"""
    global MODEL_FEATURE_ORDER
    
    print("\n" + "="*60)
    print("🔄 LOADING MACHINE LEARNING MODELS")
    print("="*60)
    
    models = {}
    
    # Updated model file names to match your actual files
    model_files = {
        'knn_phi': 'best_model_knn_phi.pkl',
        'knn_sw': 'best_model_knn_sw.pkl',
        'rf_phi': 'best_model_rf_phi.pkl',
        'rf_sw': 'best_model_rf_sw.pkl',
        'xgb_phi': 'best_model_xgb_phi.pkl',
        'xgb_sw': 'best_model_xgb_sw.pkl'
    }
    
    print("📁 Checking for model files in current directory:")
    print(f"   Current working directory: {os.getcwd()}")
    
    # Check if files exist
    missing_files = []
    existing_files = []
    
    for model_name, filename in model_files.items():
        file_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"   ✅ {filename} ({file_size:,} bytes)")
            existing_files.append((model_name, filename))
        else:
            print(f"   ❌ {filename} - FILE NOT FOUND")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} model files:")
        for file in missing_files:
            print(f"     - {file}")
        print("\n💡 Make sure all .pkl files are in the same directory as app.py")
    
    if not existing_files:
        print("\n❌ NO MODEL FILES FOUND! Cannot proceed.")
        return None
    
    # Load existing models
    print(f"\n🔄 Loading {len(existing_files)} available models...")
    loaded_count = 0
    failed_count = 0
    feature_orders = {}
    
    for model_name, filename in existing_files:
        try:
            print(f"   Loading {filename}...", end=" ")
            model = joblib.load(filename)
            
            # Basic model validation
            if hasattr(model, 'predict'):
                models[model_name] = model
                
                # Extract feature order from this model
                try:
                    feature_order = extract_feature_names(model)
                    feature_orders[model_name] = feature_order
                    print(f"✅ SUCCESS ({type(model).__name__}) - Features: {feature_order}")
                except Exception as e:
                    print(f"✅ SUCCESS ({type(model).__name__}) - Could not extract features: {e}")
                    feature_orders[model_name] = ['Depth', 'RxoRt', 'RILD', 'MN', 'GR', 'CNLS', 'RHOC']
                
                loaded_count += 1
            else:
                print(f"❌ ERROR: Not a valid ML model (no predict method)")
                failed_count += 1
                
        except FileNotFoundError:
            print(f"❌ FILE NOT FOUND")
            failed_count += 1
        except Exception as e:
            print(f"❌ JOBLIB ERROR: {str(e)}")
            failed_count += 1
    
    # Determine the consistent feature order
    if feature_orders:
        # Use the first model's feature order as the standard
        first_model = list(feature_orders.keys())[0]
        MODEL_FEATURE_ORDER = feature_orders[first_model]
        
        print(f"\n🎯 FEATURE ORDER ANALYSIS:")
        print(f"   Standard order (from {first_model}): {MODEL_FEATURE_ORDER}")
        
        # Check if all models have the same feature order
        consistent = True
        for model_name, order in feature_orders.items():
            if order != MODEL_FEATURE_ORDER:
                print(f"   ⚠️  {model_name} has different order: {order}")
                consistent = False
        
        if consistent:
            print("   ✅ All models have consistent feature order")
        else:
            print("   ⚠️  Models have different feature orders - using first model's order")
    else:
        # Fallback order based on your error message
        MODEL_FEATURE_ORDER = ['Depth', 'RxoRt', 'RILD', 'MN', 'GR', 'CNLS', 'RHOC']
        print(f"   Using fallback feature order: {MODEL_FEATURE_ORDER}")
    
    # Summary
    print(f"\n📊 LOADING SUMMARY:")
    print(f"   ✅ Successfully loaded: {loaded_count} models")
    print(f"   ❌ Failed to load: {failed_count} models")
    print(f"   📦 Available models: {list(models.keys())}")
    print(f"   🎯 Feature order: {MODEL_FEATURE_ORDER}")
    
    if loaded_count == 0:
        print("\n❌ CRITICAL: No models could be loaded!")
        return None
    elif loaded_count < len(model_files):
        print(f"\n⚠️  WARNING: Only partial models loaded ({loaded_count}/{len(model_files)})")
    else:
        print(f"\n🎉 SUCCESS: All {loaded_count} models loaded successfully!")
    
    return models

def prepare_features(data):
    """Prepare features as DataFrame for prediction with correct order"""
    print(f"\n🔧 PREPARING FEATURES:")
    print(f"   Input data: {data}")
    
    try:
        # Create feature dictionary first
        feature_dict = {
            'Depth': float(data['Depth']),
            'RxoRt': float(data['RxoRt']),
            'RILD': float(data['RILD']),
            'MN': float(data['MN']),
            'GR': float(data['GR']),
            'CNLS': float(data['CNLS']),
            'RHOC': float(data['RHOC']),
        }
        
        # Create DataFrame with the correct feature order
        if MODEL_FEATURE_ORDER:
            features = pd.DataFrame([feature_dict])[MODEL_FEATURE_ORDER]
            print(f"   ✅ Features reordered to match training: {MODEL_FEATURE_ORDER}")
        else:
            features = pd.DataFrame([feature_dict])
            print(f"   ⚠️  Using default order (MODEL_FEATURE_ORDER not set)")
        
        print(f"   📊 Final feature values:")
        for col, val in features.iloc[0].items():
            print(f"      {col}: {val}")
        
        print(f"   📏 Shape: {features.shape}")
        print(f"   🏷️  Columns: {list(features.columns)}")
        
        return features
        
    except ValueError as e:
        print(f"   ❌ VALUE ERROR: {str(e)}")
        raise ValueError(f"Invalid numeric input: {str(e)}")
    except KeyError as e:
        print(f"   ❌ MISSING FEATURE: {str(e)}")
        raise ValueError(f"Missing required feature: {str(e)}")
    except Exception as e:
        print(f"   ❌ UNEXPECTED ERROR: {str(e)}")
        raise

def make_individual_predictions(features):
    """Make individual predictions using all available models"""
    print(f"\n🤖 MAKING INDIVIDUAL PREDICTIONS:")
    
    if MODELS is None:
        print("   ❌ ERROR: No models loaded")
        return None
    
    print(f"   📦 Available models: {list(MODELS.keys())}")
    print(f"   🎯 Input features shape: {features.shape}")
    print(f"   🎯 Feature order: {list(features.columns)}")
    
    predictions = {}
    successful_predictions = 0
    failed_predictions = 0
    
    try:
        # Individual model predictions for PHI (Porosity)
        phi_predictions = {}
        
        # KNN PHI Prediction
        if 'knn_phi' in MODELS:
            try:
                print("   🔄 Making KNN PHI prediction...")
                knn_phi_pred = MODELS['knn_phi'].predict(features)[0]
                phi_predictions['KNN'] = round(float(knn_phi_pred), 4)
                print(f"   ✅ KNN PHI: {phi_predictions['KNN']}")
                successful_predictions += 1
            except Exception as e:
                print(f"   ❌ KNN PHI prediction failed: {str(e)}")
                failed_predictions += 1
        
        # Random Forest PHI Prediction
        if 'rf_phi' in MODELS:
            try:
                print("   🔄 Making Random Forest PHI prediction...")
                rf_phi_pred = MODELS['rf_phi'].predict(features)[0]
                phi_predictions['Random_Forest'] = round(float(rf_phi_pred), 4)
                print(f"   ✅ RF PHI: {phi_predictions['Random_Forest']}")
                successful_predictions += 1
            except Exception as e:
                print(f"   ❌ Random Forest PHI prediction failed: {str(e)}")
                failed_predictions += 1
        
        # XGBoost PHI Prediction
        if 'xgb_phi' in MODELS:
            try:
                print("   🔄 Making XGBoost PHI prediction...")
                xgb_phi_pred = MODELS['xgb_phi'].predict(features)[0]
                phi_predictions['XGBoost'] = round(float(xgb_phi_pred), 4)
                print(f"   ✅ XGB PHI: {phi_predictions['XGBoost']}")
                successful_predictions += 1
            except Exception as e:
                print(f"   ❌ XGBoost PHI prediction failed: {str(e)}")
                failed_predictions += 1
        
        # Individual model predictions for Sw (Water Saturation)
        sw_predictions = {}
        
        # KNN Sw Prediction
        if 'knn_sw' in MODELS:
            try:
                print("   🔄 Making KNN Sw prediction...")
                knn_sw_pred = MODELS['knn_sw'].predict(features)[0]
                sw_predictions['KNN'] = round(float(knn_sw_pred), 4)
                print(f"   ✅ KNN Sw: {sw_predictions['KNN']}")
                successful_predictions += 1
            except Exception as e:
                print(f"   ❌ KNN Sw prediction failed: {str(e)}")
                failed_predictions += 1
        
        # Random Forest Sw Prediction
        if 'rf_sw' in MODELS:
            try:
                print("   🔄 Making Random Forest Sw prediction...")
                rf_sw_pred = MODELS['rf_sw'].predict(features)[0]
                sw_predictions['Random_Forest'] = round(float(rf_sw_pred), 4)
                print(f"   ✅ RF Sw: {sw_predictions['Random_Forest']}")
                successful_predictions += 1
            except Exception as e:
                print(f"   ❌ Random Forest Sw prediction failed: {str(e)}")
                failed_predictions += 1
        
        # XGBoost Sw Prediction
        if 'xgb_sw' in MODELS:
            try:
                print("   🔄 Making XGBoost Sw prediction...")
                xgb_sw_pred = MODELS['xgb_sw'].predict(features)[0]
                sw_predictions['XGBoost'] = round(float(xgb_sw_pred), 4)
                print(f"   ✅ XGB Sw: {sw_predictions['XGBoost']}")
                successful_predictions += 1
            except Exception as e:
                print(f"   ❌ XGBoost Sw prediction failed: {str(e)}")
                failed_predictions += 1
        
        # Organize predictions by target variable
        predictions = {
            'PHI_predictions': phi_predictions,
            'Sw_predictions': sw_predictions
        }
        
        # Calculate average predictions if multiple models available
        if phi_predictions:
            avg_phi = sum(phi_predictions.values()) / len(phi_predictions)
            predictions['PHI_average'] = round(avg_phi, 4)
        
        if sw_predictions:
            avg_sw = sum(sw_predictions.values()) / len(sw_predictions)
            predictions['Sw_average'] = round(avg_sw, 4)
        
        # Summary
        print(f"\n   📊 PREDICTION SUMMARY:")
        print(f"      ✅ Successful: {successful_predictions}")
        print(f"      ❌ Failed: {failed_predictions}")
        print(f"      📦 PHI models: {list(phi_predictions.keys())}")
        print(f"      📦 Sw models: {list(sw_predictions.keys())}")
        
        if successful_predictions == 0:
            print("   ❌ CRITICAL: No predictions could be made!")
            return None
        
        return predictions
        
    except Exception as e:
        print(f"   ❌ UNEXPECTED ERROR in make_individual_predictions: {str(e)}")
        traceback.print_exc()
        return None

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index .html')

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """API endpoint for making predictions"""
    print("\n" + "="*80)
    print("🎯 NEW PREDICTION REQUEST")
    print("="*80)
    
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            error_msg = "No JSON data provided in request"
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        print(f"📥 Received data keys: {list(data.keys())}")
        print(f"📥 Received data values: {data}")
        
        # Validate required fields
        required_fields = ['Depth', 'RxoRt', 'RILD', 'MN', 'GR', 'CNLS', 'RHOC']
        missing_fields = []
        empty_fields = []
        
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
            elif data[field] == '' or data[field] is None:
                empty_fields.append(field)
        
        if missing_fields:
            error_msg = f'Missing required fields: {missing_fields}'
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        if empty_fields:
            error_msg = f'Empty values for required fields: {empty_fields}'
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # Check if models are loaded
        if MODELS is None:
            error_msg = 'Models not loaded. Check server console for model loading errors.'
            print(f"❌ {error_msg}")
            return jsonify({
                'error': error_msg,
                'details': 'No machine learning models could be loaded from .pkl files'
            }), 500
        
        # Prepare features
        try:
            features = prepare_features(data)
        except Exception as e:
            error_msg = f'Error preparing features: {str(e)}'
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # Make individual predictions
        predictions = make_individual_predictions(features)
        
        if predictions is None:
            error_msg = 'Prediction failed. Check server console for detailed error information.'
            print(f"❌ {error_msg}")
            return jsonify({
                'error': error_msg,
                'details': 'All prediction attempts failed'
            }), 500
        
        # Success response
        response_data = {
            'success': True,
            'predictions': predictions,
            'input_features': data,
            'available_models': {
                'PHI': list(predictions.get('PHI_predictions', {}).keys()),
                'Sw': list(predictions.get('Sw_predictions', {}).keys())
            },
            'feature_order_used': MODEL_FEATURE_ORDER,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        print(f"✅ PREDICTION SUCCESSFUL!")
        print(f"   PHI models used: {list(predictions.get('PHI_predictions', {}).keys())}")
        print(f"   Sw models used: {list(predictions.get('Sw_predictions', {}).keys())}")
        print("="*80)
        
        return jsonify(response_data)
        
    except ValueError as e:
        error_msg = f'Invalid input values: {str(e)}'
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 400
        
    except Exception as e:
        error_msg = f'Unexpected server error: {str(e)}'
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return jsonify({
            'error': error_msg,
            'type': type(e).__name__
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed status"""
    print("\n🏥 HEALTH CHECK REQUEST")
    
    model_status = {
        'status': 'healthy' if MODELS is not None else 'unhealthy',
        'models_loaded': MODELS is not None,
        'available_models': list(MODELS.keys()) if MODELS else [],
        'model_count': len(MODELS) if MODELS else 0,
        'feature_order': MODEL_FEATURE_ORDER,
        'timestamp': pd.Timestamp.now().isoformat(),
        'python_version': sys.version,
        'working_directory': os.getcwd()
    }
    
    print(f"   Status: {model_status['status']}")
    print(f"   Models loaded: {model_status['model_count']}")
    print(f"   Available: {model_status['available_models']}")
    print(f"   Feature order: {model_status['feature_order']}")
    
    return jsonify(model_status)

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get detailed information about loaded models"""
    if MODELS is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    info = {
        'feature_order': MODEL_FEATURE_ORDER or ['Depth', 'RxoRt', 'RILD', 'MN', 'GR', 'CNLS', 'RHOC'],
        'target_variables': ['PHI', 'Sw'],
        'available_algorithms': {
            'KNN': 'K-Nearest Neighbors',
            'XGBoost': 'XGBoost Gradient Boosting',
            'Random_Forest': 'Random Forest'
        },
        'feature_descriptions': {
            'Depth': 'Well depth measurement (feet)',
            'RxoRt': 'Resistivity ratio (Ω⋅m)',
            'RILD': 'Deep induction resistivity (Ω⋅m)', 
            'MN': 'Neutron porosity measurement (API units)',
            'GR': 'Gamma ray measurement (API units)',
            'CNLS': 'Compensated neutron-density porosity (porosity units)',
            'RHOC': 'Bulk density measurement (g/cm³)'
        },
        'loaded_models': list(MODELS.keys()),
        'model_count': len(MODELS),
        'prediction_targets': {
            'PHI': 'Porosity - measure of rock porosity',
            'Sw': 'Water Saturation - fraction of pore space filled with water'
        },
        'prediction_structure': {
            'individual_predictions': 'Each model provides separate predictions',
            'average_predictions': 'Average of all available model predictions',
            'output_format': {
                'PHI_predictions': 'Individual PHI predictions by model',
                'Sw_predictions': 'Individual Sw predictions by model',
                'PHI_average': 'Average PHI prediction',
                'Sw_average': 'Average Sw prediction'
            }
        }
    }
    return jsonify(info)

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint with system information"""
    debug_data = {
        'system_info': {
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'flask_debug': app.debug
        },
        'model_info': {
            'models_loaded': MODELS is not None,
            'model_count': len(MODELS) if MODELS else 0,
            'available_models': list(MODELS.keys()) if MODELS else [],
            'feature_order': MODEL_FEATURE_ORDER
        },
        'file_info': {
            'pkl_files': [f for f in os.listdir('.') if f.endswith('.pkl')],
            'template_files': [f for f in os.listdir('./templates') if f.endswith('.html')] if os.path.exists('./templates') else []
        }
    }
    return jsonify(debug_data)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /',
            'POST /api/predict',
            'GET /api/health',
            'GET /api/model-info',
            'GET /api/debug'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'Check server console for detailed error information'
    }), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 FLASK APPLICATION STARTUP")
    print("="*80)
    
    # Check dependencies
    print("\n📦 CHECKING DEPENDENCIES:")
    if not check_dependencies():
        print("❌ Missing required libraries. Please install them first.")
        print("Run: pip install flask pandas numpy flask-cors joblib")
        sys.exit(1)
    
    # Load models
    MODELS = load_models()
    
    # Startup summary
    print("\n" + "="*80)
    print("📊 STARTUP SUMMARY")
    print("="*80)
    
    if MODELS is None:
        print("❌ CRITICAL WARNING: No models loaded!")
        print("   The application will start but predictions will fail.")
        print("   Please ensure all .pkl files are in the same directory as app.py:")
        print("   - best_model_knn_phi.pkl")
        print("   - best_model_knn_sw.pkl")
        print("   - best_model_rf_phi.pkl")
        print("   - best_model_rf_sw.pkl")
        print("   - best_model_xgb_phi.pkl")
        print("   - best_model_xgb_sw.pkl")
    else:
        print("✅ APPLICATION READY!")
        print(f"   Loaded {len(MODELS)} models: {list(MODELS.keys())}")
        print(f"   Feature order: {MODEL_FEATURE_ORDER}")
    
    print("\n🌐 AVAILABLE ENDPOINTS:")
    print("   - GET  http://localhost:8000/                 - Main web interface")
    print("   - POST http://localhost:8000/api/predict      - Make predictions (JSON)")
    print("   - GET  http://localhost:8000/api/health       - System health check")
    print("   - GET  http://localhost:8000/api/model-info   - Model information")
    print("   - GET  http://localhost:8000/api/debug        - Debug information")
    
    print("\n🔧 TESTING SUGGESTIONS:")
    print("   1. Visit http://localhost:8000/api/health to check model status")
    print("   2. Visit http://localhost:8000/api/debug for system information")
    print("   3. Use the main interface at http://localhost:8000")
    
    print("="*80)
    print("🚀 Starting Flask development server...")
    print("   Host: 0.0.0.0 (accessible from other devices)")
    print("   Port: 8000")
    print("   Debug: True")
    print("="*80)
    
    app.run(debug=True, host='0.0.0.0', port=8000)