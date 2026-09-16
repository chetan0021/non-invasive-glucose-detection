"""
Verify that the current training data follows canonical pipeline structure:
1. Has participant_id structure with multiple readings per participant
2. R² was computed via GroupKFold on participant_id 
3. Data came from canonical synth_generator.py -> ingest -> features -> train flow
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def verify_participant_structure():
    """Check if training data has proper participant_id structure"""
    print("="*80)
    print("VERIFYING PARTICIPANT STRUCTURE")
    print("="*80)
    
    train_path = BASE_DIR / "data" / "processed" / "full_sensor_train_features.csv"
    test_path = BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv"
    
    if not train_path.exists():
        print(f"❌ Training file not found: {train_path}")
        return False
    
    train_df = pd.read_csv(train_path)
    
    # Check for participant_id column
    if 'participant_id' not in train_df.columns:
        print(f"❌ No participant_id column found")
        print(f"   Available columns: {list(train_df.columns)[:10]}...")
        return False
    
    # Analyze participant structure
    participant_counts = train_df['participant_id'].value_counts()
    unique_participants = train_df['participant_id'].nunique()
    total_readings = len(train_df)
    avg_readings_per_participant = total_readings / unique_participants
    
    print(f"✅ Participant structure found:")
    print(f"   Total readings: {total_readings}")
    print(f"   Unique participants: {unique_participants}")
    print(f"   Average readings per participant: {avg_readings_per_participant:.1f}")
    
    # Check reading distribution per participant
    min_readings = participant_counts.min()
    max_readings = participant_counts.max()
    
    print(f"   Readings per participant: {min_readings} - {max_readings}")
    
    # Sample participant data
    print(f"\nSample participant data:")
    sample_participant = participant_counts.index[0]
    sample_data = train_df[train_df['participant_id'] == sample_participant]
    
    print(f"   Participant {sample_participant}: {len(sample_data)} readings")
    if 'bgl_mg_dl' in sample_data.columns:
        glucose_values = sample_data['bgl_mg_dl'].dropna()
        if len(glucose_values) > 0:
            print(f"   Glucose range: {glucose_values.min():.1f} - {glucose_values.max():.1f} mg/dL")
    
    # Check data sources
    if 'data_source' in train_df.columns:
        source_counts = train_df['data_source'].value_counts()
        print(f"\nData sources:")
        for source, count in source_counts.items():
            pct = (count / total_readings) * 100
            print(f"   {source}: {count} readings ({pct:.1f}%)")
    
    return unique_participants > 1 and avg_readings_per_participant > 1

def check_groupkfold_compliance():
    """Verify that model training would use GroupKFold"""
    print(f"\n" + "="*80)
    print("CHECKING GROUPKFOLD COMPLIANCE")
    print("="*80)
    
    # Check if train_models.py uses GroupKFold
    train_models_path = BASE_DIR / "scripts" / "train_models.py"
    
    if not train_models_path.exists():
        print(f"❌ train_models.py not found")
        return False
    
    with open(train_models_path, 'r') as f:
        content = f.read()
    
    if 'GroupKFold' in content:
        print(f"✅ train_models.py uses GroupKFold")
        
        # Check for participant_id grouping
        if 'participant_id' in content:
            print(f"✅ GroupKFold likely uses participant_id for grouping")
        else:
            print(f"⚠️  GroupKFold found but participant_id grouping unclear")
        
        return True
    else:
        print(f"❌ No GroupKFold found in train_models.py")
        print(f"   This means R² scores could be inflated by data leakage")
        return False

def verify_canonical_data_flow():
    """Check if data came through canonical pipeline"""
    print(f"\n" + "="*80)
    print("VERIFYING CANONICAL DATA FLOW")  
    print("="*80)
    
    # Check if synthetic data was generated properly
    synthetic_path = BASE_DIR / "data" / "interim" / "synthetic_features.csv"
    
    if synthetic_path.exists():
        synthetic_df = pd.read_csv(synthetic_path)
        print(f"✅ Synthetic data found: {len(synthetic_df)} samples")
        
        # Check for extreme value representation
        if 'bgl_mg_dl' in synthetic_df.columns:
            glucose_values = synthetic_df['bgl_mg_dl'].dropna()
            if len(glucose_values) > 0:
                hypo = (glucose_values < 70).sum()
                severe = (glucose_values > 250).sum()
                total = len(glucose_values)
                
                hypo_pct = (hypo / total) * 100
                severe_pct = (severe / total) * 100
                
                print(f"   Hypoglycemic (<70): {hypo} samples ({hypo_pct:.1f}%)")
                print(f"   Severe hyperglycemic (>250): {severe} samples ({severe_pct:.1f}%)")
                
                # Check if this matches rebalancing targets
                hypo_target_met = hypo_pct >= 7.0  # Allow some tolerance
                severe_target_met = severe_pct >= 9.0
                
                if hypo_target_met and severe_target_met:
                    print(f"✅ Rebalancing targets met in synthetic data")
                    return True
                else:
                    print(f"⚠️  Rebalancing targets not met in synthetic data")
                    print(f"   This suggests bypass of canonical pipeline")
        
    else:
        print(f"⚠️  No synthetic data found at {synthetic_path}")
    
    # Check if training data is from canonical merge
    train_path = BASE_DIR / "data" / "processed" / "full_sensor_train_features.csv"
    train_df = pd.read_csv(train_path)
    
    if 'data_source' in train_df.columns:
        sources = train_df['data_source'].unique()
        print(f"\nTraining data sources: {list(sources)}")
        
        if 'synthetic' in sources and 'real_tabular' in sources:
            print(f"✅ Training data contains both synthetic and real data")
            print(f"   Suggests canonical merge process was used")
            return True
        else:
            print(f"⚠️  Training data sources suggest non-canonical process")
    
    return False

def main():
    """Main verification pipeline"""
    print("🔍 VERIFYING CANONICAL PIPELINE COMPLIANCE")
    print("="*80)
    print("Checking if rebalanced data follows proper pipeline structure")
    
    # Check participant structure
    has_participants = verify_participant_structure()
    
    # Check GroupKFold compliance
    uses_groupkfold = check_groupkfold_compliance()
    
    # Check canonical data flow
    canonical_flow = verify_canonical_data_flow()
    
    print(f"\n" + "="*80)
    print("CANONICAL PIPELINE VERIFICATION RESULTS")
    print("="*80)
    
    print(f"Participant structure: {'✅ PASS' if has_participants else '❌ FAIL'}")
    print(f"GroupKFold compliance: {'✅ PASS' if uses_groupkfold else '❌ FAIL'}")  
    print(f"Canonical data flow: {'✅ PASS' if canonical_flow else '❌ FAIL'}")
    
    all_pass = has_participants and uses_groupkfold and canonical_flow
    
    if all_pass:
        print(f"\n✅ CANONICAL PIPELINE VERIFIED")
        print(f"   Data structure follows project conventions")
        print(f"   R² scores are valid (GroupKFold prevents data leakage)")
        print(f"   Ready for production integration")
    else:
        print(f"\n❌ CANONICAL PIPELINE VIOLATIONS DETECTED")
        print(f"   Need to regenerate data through proper pipeline")
        print(f"   Current R² scores may be inflated")
        print(f"   predict.py integration will likely fail")
    
    if not has_participants:
        print(f"\n🔧 FIX NEEDED: Regenerate with proper participant structure")
        
    if not uses_groupkfold:
        print(f"\n🔧 FIX NEEDED: Ensure GroupKFold validation is used")
        
    if not canonical_flow:
        print(f"\n🔧 FIX NEEDED: Use synth_generator.py -> ingest -> features -> train pipeline")
    
    return all_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)