"""
Debug the benchmark prediction errors to identify what's causing the failures
"""

import sys
from pathlib import Path
import traceback

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict import GlucosePredictor

def debug_single_benchmark():
    """Debug a single benchmark preset to see the exact error"""
    print("="*80)
    print("DEBUGGING BENCHMARK PREDICTION FAILURE")
    print("="*80)
    
    predictor = GlucosePredictor()
    
    # Test with Healthy Adult preset
    test_input = {
        "saliva_ph": 7.35, 
        "temperature_c": 36.6, 
        "hr_bpm": 66.0,
        "hrv_sdnn": 58.0, 
        "hrv_rmssd": 52.0, 
        "hrv_pnn50": 26.0, 
        "hrv_lf_hf_ratio": 1.10,
        "perfusion_index": 0.81, 
        "pulse_width_ms": 285.0,
        "ppg_raw_dc_baseline": 178000.0, 
        "ppg_raw_ac_p2p": 1450.0,
        "vpg_max": 4800.0, 
        "apg_a": 85.0, 
        "apg_b": -55.0,
        "age": 34.0, 
        "bmi": 21.0, 
        "diabetes_diagnosis": "None", 
        "fasting": 1,
        "family_history": 0, 
        "smoking": 0, 
        "gender": "Female"
    }
    
    print("Testing Healthy Adult preset...")
    print("Input parameters:")
    for key, value in test_input.items():
        print(f"  {key}: {value}")
    
    try:
        print(f"\nTesting prediction...")
        result = predictor.predict_full_sensor(test_input)
        
        print(f"✅ Prediction successful!")
        print(f"  Predicted BGL: {result['predicted_bgl_mg_dl']:.1f} mg/dL")
        print(f"  Confidence interval: [{result['confidence_interval_5th_95th'][0]:.1f}, {result['confidence_interval_5th_95th'][1]:.1f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        
        # Try to identify which step failed
        print(f"\nDebugging individual steps...")
        
        try:
            print("1. Testing feature preparation...")
            features_df = predictor._prepare_full_sensor_features(test_input)
            print(f"   ✓ Features prepared successfully: {features_df.shape}")
            
            print("2. Testing model loading...")
            if predictor.quantile_bundle is None:
                print("   ❌ Quantile bundle not loaded")
                return False
            else:
                print("   ✓ Quantile bundle loaded")
            
            print("3. Testing individual model predictions...")
            feature_cols = predictor.quantile_bundle["feature_list"]
            X = features_df[feature_cols]
            
            # Test each model individually
            q05_pred = predictor.quantile_bundle["q05_model"].predict(X)
            print(f"   ✓ Q05 model: {q05_pred[0]:.1f}")
            
            q95_pred = predictor.quantile_bundle["q95_model"].predict(X)  
            print(f"   ✓ Q95 model: {q95_pred[0]:.1f}")
            
            # Check if there's a point model
            if "point_model" in predictor.quantile_bundle:
                point_pred = predictor.quantile_bundle["point_model"].predict(X)
                print(f"   ✓ Point model: {point_pred[0]:.1f}")
            else:
                print("   ⚠️  No point model found, may be using mean of quantiles")
            
        except Exception as inner_e:
            print(f"   ❌ Step failed: {inner_e}")
            traceback.print_exc()
        
        return False

def test_simple_prediction():
    """Test with a very simple input to see if basic prediction works"""
    print(f"\n" + "="*80)
    print("TESTING SIMPLE PREDICTION")
    print("="*80)
    
    predictor = GlucosePredictor()
    
    # Minimal input
    simple_input = {
        "saliva_ph": 7.25,
        "temperature_c": 36.6,
        "hr_bpm": 72.0,
        "age": 45.0,
        "bmi": 26.0,
        "diabetes_diagnosis": "None",
        "fasting": 1,
        "ppg_raw_dc_baseline": 175000.0,
        "ppg_raw_ac_p2p": 1500.0
    }
    
    try:
        result = predictor.predict_full_sensor(simple_input)
        print(f"✅ Simple prediction successful!")
        print(f"  Predicted BGL: {result['predicted_bgl_mg_dl']:.1f} mg/dL")
        return True
        
    except Exception as e:
        print(f"❌ Simple prediction failed: {e}")
        traceback.print_exc()
        return False

def check_predict_py_syntax():
    """Check if there are syntax errors in the modified predict.py"""
    print(f"\n" + "="*80)
    print("CHECKING PREDICT.PY SYNTAX")
    print("="*80)
    
    predict_py_path = BASE_DIR / "scripts" / "predict.py"
    
    try:
        with open(predict_py_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Try to compile the code
        compile(content, predict_py_path, 'exec')
        print("✅ predict.py syntax is valid")
        
        # Look for our recent changes
        if "apg_a_val" in content:
            print("✅ Found apg_a_val calculation")
        else:
            print("❌ apg_a_val calculation not found")
        
        # Check for specific fixes
        fixes_found = []
        if "apg_a_val * 0.25" in content:
            fixes_found.append("apg_c")
        if "apg_a_val * -0.25" in content:
            fixes_found.append("apg_d")  
        if "apg_a_val * 0.15" in content:
            fixes_found.append("apg_e")
        if "raw_ac * 1.65" in content:
            fixes_found.append("vpg_max")
        if "raw_ac * 1.353" in content:
            fixes_found.append("vpg_min")
        
        print(f"✅ Found fixes for: {', '.join(fixes_found)}")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in predict.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking predict.py: {e}")
        return False

def main():
    """Main debugging pipeline"""
    print("DEBUGGING BENCHMARK PREDICTION FAILURES")
    print("="*80)
    
    # Check syntax first
    syntax_ok = check_predict_py_syntax()
    
    if not syntax_ok:
        print("🚫 Fix syntax errors first")
        return
    
    # Test simple prediction
    simple_ok = test_simple_prediction()
    
    if not simple_ok:
        print("🚫 Basic prediction failing - fundamental issue")
        return
    
    # Test benchmark preset
    benchmark_ok = debug_single_benchmark()
    
    if benchmark_ok:
        print(f"\n🎉 Benchmark predictions working!")
        print(f"   The error in the verification script may be a parsing issue")
    else:
        print(f"\n❌ Benchmark prediction issues remain")
        print(f"   Need to investigate model loading or feature processing")

if __name__ == "__main__":
    main()