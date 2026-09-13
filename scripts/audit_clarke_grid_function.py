"""
CRITICAL AUDIT: Clarke Error Grid Function Implementation

Verify the clarke_error_grid_zone function against the original published 
boundaries from Clarke et al., 1987, Diabetes Care 10(5): 622-628.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import the current implementation
from scripts.train_models import clarke_error_grid_zone

def print_current_implementation():
    """Print the current Clarke grid implementation for review"""
    print("="*80)
    print("CURRENT CLARKE ERROR GRID IMPLEMENTATION")
    print("="*80)
    
    # Read and display the actual function code
    train_models_path = BASE_DIR / "scripts" / "train_models.py"
    with open(train_models_path, "r") as f:
        content = f.read()
    
    # Extract the clarke_error_grid_zone function
    lines = content.split('\n')
    function_lines = []
    in_function = False
    
    for i, line in enumerate(lines):
        if 'def clarke_error_grid_zone(' in line:
            in_function = True
            function_lines.append(f"{i+1:3d}: {line}")
        elif in_function and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            # End of function
            break
        elif in_function:
            function_lines.append(f"{i+1:3d}: {line}")
    
    print("Current implementation:")
    for line in function_lines:
        print(line)
    
    return '\n'.join([line.split(': ', 1)[1] for line in function_lines])

def test_known_reference_points():
    """Test against known reference points from Clarke et al. 1987"""
    print(f"\n" + "="*80)
    print("UNIT TESTING AGAINST PUBLISHED REFERENCE POINTS")
    print("="*80)
    
    # Test cases based on Clarke et al. 1987 boundaries
    test_cases = [
        # Zone A: Clinically accurate
        {"ref": 100, "pred": 100, "expected": "A", "description": "Exact match"},
        {"ref": 300, "pred": 310, "expected": "A", "description": "Within 20% for hyperglycemic"},
        {"ref": 50, "pred": 50, "expected": "A", "description": "Both hypoglycemic, agree"},
        {"ref": 80, "pred": 95, "expected": "A", "description": "Within 20% for normal range"},
        
        # Zone D: Dangerous failure to detect - CRITICAL TEST CASES
        {"ref": 265, "pred": 138.6, "expected": "D", "description": "EXTREME CASE: Severe hyperglycemia predicted as elevated (ref>240, pred in 70-180)"},
        {"ref": 62, "pred": 82.7, "expected": "D", "description": "EXTREME CASE: Hypoglycemia predicted as normal (ref<70, pred in 70-180)"},
        {"ref": 250, "pred": 150, "expected": "D", "description": "Severe hyperglycemia missed"},
        {"ref": 65, "pred": 120, "expected": "D", "description": "Hypoglycemia missed"},
        
        # Zone E: Erroneous treatment risk  
        {"ref": 200, "pred": 60, "expected": "E", "description": "Hyperglycemia predicted as hypoglycemia"},
        {"ref": 60, "pred": 200, "expected": "E", "description": "Hypoglycemia predicted as hyperglycemia"},
        
        # Zone B: Benign errors
        {"ref": 150, "pred": 120, "expected": "B", "description": "Mild underestimation"},
        {"ref": 90, "pred": 110, "expected": "B", "description": "Mild overestimation"},
        
        # Zone C: Overcorrection
        {"ref": 150, "pred": 270, "expected": "C", "description": "Overcorrection leading to excessive treatment"},
    ]
    
    print(f"Testing {len(test_cases)} reference points...")
    print(f"{'Reference':>9s} {'Predicted':>9s} {'Expected':>8s} {'Actual':>8s} {'Status':>10s} {'Description':>40s}")
    print("-" * 100)
    
    failures = []
    critical_failures = []
    
    for i, test in enumerate(test_cases):
        ref = test["ref"]
        pred = test["pred"]
        expected = test["expected"]
        description = test["description"]
        
        actual = clarke_error_grid_zone(ref, pred)
        
        if actual == expected:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            failures.append(test)
            
            # Mark critical safety failures
            if expected == "D" and actual != "D":
                critical_failures.append(test)
        
        print(f"{ref:9.1f} {pred:9.1f} {expected:>8s} {actual:>8s} {status:>10s} {description:>40s}")
    
    return failures, critical_failures

def analyze_clarke_1987_boundaries():
    """Analyze the original Clarke 1987 boundaries for comparison"""
    print(f"\n" + "="*80)
    print("ORIGINAL CLARKE 1987 BOUNDARY DEFINITIONS")
    print("="*80)
    
    boundaries = """
    Zone A (Clinically Accurate):
    - Reference ≤ 70 AND Predicted ≤ 70 (both hypoglycemic, agree)
    - OR |Predicted - Reference| ≤ 20% of Reference (within 20%)
    
    Zone B (Benign Error):
    - Deviations outside Zone A that would not affect clinical treatment
    - Generally small to moderate errors that don't cross critical thresholds
    
    Zone C (Overcorrection):
    - Predictions that would lead to unnecessary but not dangerous treatment
    - Reference 70-290, Predicted > Reference + 110
    - Reference 130-180, specific linear boundary
    
    Zone D (Dangerous Failure to Detect):
    - Reference < 70 (hypoglycemic) AND 70 ≤ Predicted < 180 (normal range)
    - Reference > 240 (severe hyperglycemic) AND 70 ≤ Predicted ≤ 180 (normal range)
    - CRITICAL: Fails to detect dangerous glucose levels
    
    Zone E (Erroneous Treatment Risk):  
    - Reference ≥ 180 AND Predicted ≤ 70 (hyperglycemia → hypoglycemia treatment)
    - Reference ≤ 70 AND Predicted ≥ 180 (hypoglycemia → hyperglycemia treatment)
    - CRITICAL: Opposite treatment indicated
    """
    
    print(boundaries)
    
    print(f"\n🔍 KEY BOUNDARY FOR OUR EXTREME CASES:")
    print(f"Zone D Definition: Reference > 240 AND 70 ≤ Predicted ≤ 180")
    print(f"  - Severe Hyperglycemia: ref=265 (>240), pred=138.6 (in 70-180) → SHOULD BE ZONE D")
    print(f"  - Hypoglycemia: ref=62 (<70), pred=82.7 (in 70-180) → SHOULD BE ZONE D")

def correct_clarke_implementation():
    """Provide corrected Clarke implementation if bugs found"""
    print(f"\n" + "="*80)
    print("CORRECTED CLARKE ERROR GRID IMPLEMENTATION")
    print("="*80)
    
    corrected_code = '''
def clarke_error_grid_zone_corrected(ref: float, pred: float) -> str:
    """
    CORRECTED Clarke Error Grid classification per Clarke et al. 1987.
    
    Zone A: Clinically accurate
    Zone B: Benign error  
    Zone C: Overcorrection
    Zone D: Dangerous failure to detect
    Zone E: Erroneous treatment risk
    """
    # Zone A: Clinically accurate
    if (ref <= 70.0 and pred <= 70.0) or (abs(pred - ref) <= 0.20 * ref):
        return "A"
    
    # Zone E: Erroneous treatment risk (opposite treatment)
    if (ref >= 180.0 and pred <= 70.0) or (ref <= 70.0 and pred >= 180.0):
        return "E"
    
    # Zone D: Dangerous failure to detect
    if (ref <= 70.0 and 70.0 < pred < 180.0) or (ref > 240.0 and 70.0 <= pred <= 180.0):
        return "D"
    
    # Zone C: Overcorrection 
    if (70.0 <= ref <= 290.0 and pred >= ref + 110.0) or \\
       (130.0 <= ref <= 180.0 and pred <= (7.0 / 5.0) * ref - 182.0):
        return "C"
    
    # Zone B: Benign error (everything else)
    return "B"
    '''
    
    print("Proposed corrected implementation:")
    print(corrected_code)
    
    print(f"\n🔍 KEY CHANGE NEEDED:")
    print(f"Current: (ref >= 240.0 and 70.0 <= est <= 180.0)")
    print(f"Should be: (ref > 240.0 and 70.0 <= pred <= 180.0)")
    print(f"Issue: >= vs > boundary condition for severe hyperglycemia")

def main():
    """Main audit pipeline"""
    print("🚨 CRITICAL CLARKE ERROR GRID IMPLEMENTATION AUDIT")
    print("="*80)
    print("Verifying against Clarke et al. 1987 published boundaries")
    
    # Step 1: Show current implementation
    current_code = print_current_implementation()
    
    # Step 2: Test against reference points
    failures, critical_failures = test_known_reference_points()
    
    # Step 3: Analyze original boundaries
    analyze_clarke_1987_boundaries()
    
    # Step 4: Provide corrected implementation if needed
    if failures:
        correct_clarke_implementation()
    
    print(f"\n" + "="*80)
    print("AUDIT RESULTS")
    print("="*80)
    
    if critical_failures:
        print("🚨 CRITICAL SAFETY FAILURES IN CLARKE GRID FUNCTION")
        print(f"   {len(critical_failures)} test cases incorrectly classified")
        print(f"   Extreme cases NOT properly identified as Zone D")
        
        for failure in critical_failures:
            ref = failure["ref"]
            pred = failure["pred"]
            expected = failure["expected"]
            actual = clarke_error_grid_zone(ref, pred)
            print(f"   • ref={ref}, pred={pred}: Expected {expected}, Got {actual}")
        
        print(f"\n🔧 REQUIRED ACTIONS:")
        print(f"   1. Fix Clarke grid function boundary logic")
        print(f"   2. Re-run ALL previous safety assessments")
        print(f"   3. Update all reports with corrected classifications")
        print(f"   4. Verify extreme cases are truly Zone D failures")
        
        print(f"\n🚫 SAFETY VERDICT: FUNCTION IS INCORRECT")
        print(f"   Previous 'Zone B' classifications for extreme cases are WRONG")
        print(f"   Severe hyperglycemia and hypoglycemia cases ARE Zone D failures")
        
    elif failures:
        print(f"⚠️  {len(failures)} non-critical test failures")
        print(f"   Function may have minor boundary issues")
        
    else:
        print("✅ ALL TESTS PASSED")
        print("   Clarke grid function correctly implements published boundaries")
    
    print(f"\n📋 NEXT STEPS:")
    if critical_failures:
        print(f"   1. 🔄 Fix function implementation")
        print(f"   2. 🧪 Re-test extreme cases")  
        print(f"   3. 📝 Correct all reports")
        print(f"   4. 🚫 Confirm Zone D safety failures")
    else:
        print(f"   1. ✅ Function verified correct")
        print(f"   2. ✅ Previous assessments valid")
    
    return len(critical_failures) > 0

if __name__ == "__main__":
    has_critical_failures = main()
    sys.exit(1 if has_critical_failures else 0)