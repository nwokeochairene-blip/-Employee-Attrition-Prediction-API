# test_model.py
import pickle
import joblib
import os

print("=" * 50)
print("Testing Model Files")
print("=" * 50)

# Check if files exist
files_to_check = ['model.pkl', 'scaler.pkl', 'label_encoders.pkl']
for file in files_to_check:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024  # Size in KB
        print(f"✅ {file} exists (Size: {size:.2f} KB)")
    else:
        print(f"❌ {file} NOT FOUND!")

print("\n" + "=" * 50)
print("Loading Model...")
print("=" * 50)

# Test model
try:
    # Try pickle first
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully using pickle!")
    print(f"   Model type: {type(model)}")
    
    # Try to get number of features if available
    if hasattr(model, 'n_features_in_'):
        print(f"   Features expected: {model.n_features_in_}")
    elif hasattr(model, 'n_features_'):
        print(f"   Features: {model.n_features_}")
        
except Exception as e:
    print(f"❌ Pickle load failed: {e}")
    
    # Try joblib as fallback
    try:
        model = joblib.load('model.pkl')
        print("✅ Model loaded successfully using joblib!")
        print(f"   Model type: {type(model)}")
    except Exception as e2:
        print(f"❌ Joblib load also failed: {e2}")

print("\n" + "=" * 50)
print("Loading Scaler...")
print("=" * 50)

# Test scaler
try:
    scaler = joblib.load('scaler.pkl')
    print("✅ Scaler loaded successfully!")
    print(f"   Scaler type: {type(scaler)}")
except Exception as e:
    print(f"❌ Scaler load failed: {e}")

print("\n" + "=" * 50)
print("Loading Label Encoders...")
print("=" * 50)

# Test label encoders
try:
    label_encoders = joblib.load('label_encoders.pkl')
    print("✅ Label encoders loaded successfully!")
    print(f"   Encoded columns: {list(label_encoders.keys())}")
    
    # Show sample of what each encoder contains
    for col, encoder in label_encoders.items():
        print(f"   - {col}: {list(encoder.classes_)}")
        
except Exception as e:
    print(f"❌ Label encoders load failed: {e}")

print("\n" + "=" * 50)
print("Test Complete!")
print("=" * 50)