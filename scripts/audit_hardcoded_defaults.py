"""
Comprehensive audit of hardcoded defaults vs real derivations in predict.py

This investigates whether APG/VPG features have the same bug pattern as ppg_signal_energy:
hardcoded defaults instead of proper calculations from raw sensor data.
"""

import sys
import numpy as np
import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

def extract_predict_py_defaults():
    """Extract all raw_dict.get() default values from predict.py"""
    print("="*80)
    print("EXTRACTING PREDICT.PY DEFAULT VALUES")
    print("="*80)
    
    predict_py_path = BASE_DIR / "scripts" / "predict.py"
    
    with open(predict_py_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find all raw_dict.get() patterns
    pattern = r'"([^"]+)":\s*float\(raw_dict\.get\("([^"]+)",\s*([^)]+)\)\)'
    matches = re.findall(pattern, content)
    
    defaults = {}
    
    print("Found raw_dict.get() default values:")
    print(f"{'Feature':25s} {'Default Value':>20s}")
    print("-" * 50)
    
    for target_field, source_field, default_value in matches:
        # Clean up the default value
        default_clean = default_value.strip()
        defaults[target_field] = default_clean
        print(f"{target_field:25s} {default_clean:>20s}")
    
    return defaults

def extract_synth_generator_calculations():
    """Extract how features are calculated in synth_generator.py"""
    print(f"\n" + "="*80)
    print("EXTRACTING SYNTH_GENERATOR CALCULATIONS")
    print("="*80)
    
    synth_path = BASE_DIR / "scripts" / "synth_generator.py"
    
    with open(synth_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Look for feature calculations
    target_features = ["apg_a", "apg_b", "apg_c", "apg_d", "apg_e", "vpg_min", "vpg_max"]
    
    calculations = {}
    
    for feature in target_features:
        # Find lines that assign to this feature
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if f'"{feature}"' in line and '=' in line:
                # Get context around the assignment
                start = max(0, i-5)
                end = min(len(lines), i+3)
                
                context = []
                for j in range(start, end):
                    marker = ">>> " if j == i else "    "
                    context.append(f"{marker}{lines[j]}")
                
                calculations[feature] = {
                    "line": i+1,
                    "assignment": line.strip(),
                    "context": context
                }
                break
    
    print("Found feature calculations in synth_generator.py:")
    for feature, calc_info in calculations.items():
        print(f"\n{feature} (line {calc_info['line']}):")
        print(f"  Assignment: {calc_info['assignment']}")
    
    return calculations

def analyze_apg_vpg_patterns():
    """Analyze specific APG and VPG calculation patterns"""
    print(f"\n" + "="*80) 
    print("ANALYZING APG/VPG CALCULATION PATTERNS")
    print("="*80)
    
    synth_path = BASE_DIR / "scripts" / "synth_generator.py"
    
    with open(synth_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Look for APG/VPG blocks
    lines = content.split('\n')
    
    apg_section = []
    vpg_section = []
    in_apg = False
    in_vpg = False
    
    for i, line in enumerate(lines):
        if 'APG' in line and 'Derivatives' in line:
            in_apg = True
            apg_section.append(f"{i+1:4d}: {line}")
            continue
        elif 'VPG' in line and 'Derivatives' in line:
            in_vpg = True
            vpg_section.append(f"{i+1:4d}: {line}")
            continue
        elif line.strip().startswith('#') and (in_apg or in_vpg):
            # New section started
            in_apg = False
            in_vpg = False
        elif in_apg and ('apg_' in line.lower() or 'vpg_' in line.lower() or line.strip() == ''):
            apg_section.append(f"{i+1:4d}: {line}")
        elif in_vpg and ('vpg_' in line.lower() or line.strip() == ''):
            vpg_section.append(f"{i+1:4d}: {line}")
    
    print("APG calculation section:")
    for line in apg_section[:20]:  # First 20 lines
        print(line)
    
    print(f"\nVPG calculation section:")
    for line in vpg_section[:15]:  # First 15 lines  
        print(line)
    
    return apg_section, vpg_section

def check_dashboard_preset_completeness():
    """Check if dashboard presets populate all required fields"""
    print(f"\n" + "="*80)
    print("CHECKING DASHBOARD PRESET COMPLETENESS") 
    print("="*80)
    
    dashboard_path = BASE_DIR / "app" / "dashboard.py"
    
    with open(dashboard_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find preset definitions
    preset_pattern = r'"([^"]+)":\s*{[^}]+}'
    presets = re.findall(preset_pattern, content, re.DOTALL)
    
    print(f"Found {len(presets)} preset definitions")
    
    # Look for specific preset content
    lines = content.split('\n')
    preset_section = []
    in_presets = False
    
    for i, line in enumerate(lines):
        if 'BENCHMARK_PRESETS' in line or 'presets' in line.lower():
            in_presets = True
            preset_section.append(f"{i+1:4d}: {line}")
        elif in_presets and line.strip().startswith('}'):
            preset_section.append(f"{i+1:4d}: {line}")
            break
        elif in_presets:
            preset_section.append(f"{i+1:4d}: {line}")
    
    print("Preset section from dashboard.py:")
    for line in preset_section[:30]:  # First 30 lines
        print(line)
    
    # Check if problematic features are mentioned
    problematic_features = ["ppg_signal_energy", "apg_a", "apg_c", "apg_d", "apg_e", "vpg_min", "vpg_max"]
    
    print(f"\nChecking if problematic features are in presets:")
    for feature in problematic_features:
        if feature in content:
            print(f"  ✓ {feature}: Found in dashboard.py")
        else:
            print(f"  ❌ {feature}: NOT found in dashboard.py")
    
    return preset_section

def compare_defaults_vs_calculations(defaults, calculations):
    """Compare predict.py defaults against synth_generator calculations"""
    print(f"\n" + "="*80)
    print("COMPARING DEFAULTS VS REAL CALCULATIONS")
    print("="*80)
    
    problematic_features = ["apg_a", "apg_c", "apg_d", "apg_e", "vpg_min", "vpg_max"]
    
    print(f"{'Feature':12s} {'Predict.py Default':>20s} {'Real Calculation':>30s} {'Status':>15s}")
    print("-" * 80)
    
    bugs_found = []
    
    for feature in problematic_features:
        default_val = defaults.get(feature, "NOT FOUND")
        calculation = calculations.get(feature, {})
        calc_desc = calculation.get("assignment", "NOT FOUND")
        
        # Analyze if it's a hardcoded default bug
        if default_val != "NOT FOUND" and "(" not in default_val:
            # Simple numeric default - likely hardcoded bug
            status = "🚨 HARDCODED"
            bugs_found.append({
                "feature": feature,
                "default": default_val,
                "calculation": calc_desc
            })
        elif default_val == "NOT FOUND":
            status = "❌ MISSING"
        elif "raw_" in default_val or "sys_" in default_val or "dias_" in default_val:
            # Uses other variables - likely real calculation
            status = "✅ DERIVED"
        else:
            status = "⚠️ UNCLEAR"
        
        # Truncate long descriptions
        calc_short = calc_desc[:25] + "..." if len(calc_desc) > 25 else calc_desc
        
        print(f"{feature:12s} {default_val:>20s} {calc_short:>30s} {status:>15s}")
    
    print(f"\n" + "="*60)
    print("BUG ASSESSMENT")
    print("="*60)
    
    if len(bugs_found) > 0:
        print(f"❌ FOUND {len(bugs_found)} HARDCODED DEFAULT BUGS:")
        for i, bug in enumerate(bugs_found):
            print(f"  {i+1}. {bug['feature']}: using default {bug['default']} instead of calculation")
            print(f"      Should calculate: {bug['calculation'][:60]}...")
        
        print(f"\n🔧 REQUIRED FIXES:")
        print(f"   1. Replace hardcoded defaults with real calculations")
        print(f"   2. Use same formulas as synth_generator.py")
        print(f"   3. Re-test coverage expecting significant improvement")
    else:
        print(f"✅ NO OBVIOUS HARDCODED DEFAULT BUGS")
        print(f"   The feature mismatch may be due to:")
        print(f"   - Different calculation methods")
        print(f"   - Scaling/rounding differences")
        print(f"   - Input data preprocessing variations")
    
    return bugs_found

def main():
    """Main audit pipeline"""
    print("COMPREHENSIVE HARDCODED DEFAULT AUDIT")
    print("="*80)
    print("Investigating whether APG/VPG features have the same bug as ppg_signal_energy:")
    print("hardcoded defaults instead of proper derivation from raw sensor data.")
    
    # Step 1: Extract predict.py defaults
    defaults = extract_predict_py_defaults()
    
    # Step 2: Extract synth_generator calculations
    calculations = extract_synth_generator_calculations()
    
    # Step 3: Analyze APG/VPG patterns specifically
    apg_section, vpg_section = analyze_apg_vpg_patterns()
    
    # Step 4: Check dashboard preset completeness
    preset_section = check_dashboard_preset_completeness()
    
    # Step 5: Compare and identify bugs
    bugs_found = compare_defaults_vs_calculations(defaults, calculations)
    
    print(f"\n" + "="*80)
    print("FINAL AUDIT RESULTS")
    print("="*80)
    
    if len(bugs_found) >= 3:
        print(f"❌ SYSTEMATIC BUG PATTERN CONFIRMED")
        print(f"   {len(bugs_found)} features use hardcoded defaults instead of real calculations")
        print(f"   This explains the persistent 14-point coverage gap")
        print(f"   Same root cause as ppg_signal_energy bug")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Fix all {len(bugs_found)} hardcoded defaults using synth_generator formulas")
        print(f"   2. Re-test coverage (expect 85-90% after fixes)")
        print(f"   3. Only then consider production readiness")
        
    elif len(bugs_found) > 0:
        print(f"⚠️  PARTIAL BUG PATTERN")
        print(f"   {len(bugs_found)} features may have hardcoded defaults")
        print(f"   Worth fixing but may not fully close coverage gap")
        
    else:
        print(f"✅ NO SYSTEMATIC HARDCODED DEFAULT PATTERN")
        print(f"   The coverage gap may be due to more complex preprocessing differences")
        print(f"   Consider accepting 76% coverage as realistic for synthetic training data")
    
    print(f"\n⚠️  PRODUCTION STATUS: BLOCKED")
    print(f"   Cannot deploy until preprocessing audit is complete")
    print(f"   Current 76% vs 90% target gap is unacceptable")

if __name__ == "__main__":
    main()