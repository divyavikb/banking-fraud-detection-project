"""
Test script for Fraud Detection API
Run the API first: python app.py
Then run this: python test_api.py
"""

import requests
import json

# API URL
BASE_URL = "http://localhost:5000"


def test_health():
    """Test health check"""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction():
    """Test single transaction prediction"""
    print("\n" + "="*70)
    print("TEST 2: Single Transaction Prediction")
    print("="*70)
    
    # Example transaction (potentially fraudulent - high amount at night)
    transaction = {
        "TX_AMOUNT": 500.0,
        "TX_TIME_SECONDS": 82800,
        "TX_TIME_DAYS": 100,
        "TX_DURING_NIGHT": 1,
        "CUSTOMER_ID_NB_TX_1DAY_WINDOW": 5,
        "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW": 450.0,
        "CUSTOMER_ID_NB_TX_7DAY_WINDOW": 10,
        "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW": 400.0,
        "CUSTOMER_ID_NB_TX_30DAY_WINDOW": 25,
        "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW": 350.0,
        "TERMINAL_ID_NB_TX_1DAY_WINDOW": 100,
        "TERMINAL_ID_RISK_1DAY_WINDOW": 0.15,
        "TERMINAL_ID_NB_TX_7DAY_WINDOW": 500,
        "TERMINAL_ID_RISK_7DAY_WINDOW": 0.12,
        "TERMINAL_ID_NB_TX_30DAY_WINDOW": 2000,
        "TERMINAL_ID_RISK_30DAY_WINDOW": 0.10,
        "hour": 23,
        "day": 15,
        "month": 4,
        "weekday": 3,
        "is_weekend": 0,
        "is_night": 1
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=transaction,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*70)
    print("TEST 3: Batch Prediction")
    print("="*70)
    
    # Multiple transactions
    transactions = {
        "transactions": [
            {
                "TX_AMOUNT": 50.0,
                "TX_TIME_SECONDS": 50400,
                "TX_TIME_DAYS": 100,
                "TX_DURING_NIGHT": 0,
                "CUSTOMER_ID_NB_TX_1DAY_WINDOW": 2,
                "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW": 45.0,
                "CUSTOMER_ID_NB_TX_7DAY_WINDOW": 8,
                "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW": 50.0,
                "CUSTOMER_ID_NB_TX_30DAY_WINDOW": 20,
                "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW": 55.0,
                "TERMINAL_ID_NB_TX_1DAY_WINDOW": 150,
                "TERMINAL_ID_RISK_1DAY_WINDOW": 0.02,
                "TERMINAL_ID_NB_TX_7DAY_WINDOW": 800,
                "TERMINAL_ID_RISK_7DAY_WINDOW": 0.03,
                "TERMINAL_ID_NB_TX_30DAY_WINDOW": 3000,
                "TERMINAL_ID_RISK_30DAY_WINDOW": 0.02,
                "hour": 14,
                "day": 15,
                "month": 4,
                "weekday": 3,
                "is_weekend": 0,
                "is_night": 0
            },
            {
                "TX_AMOUNT": 1000.0,
                "TX_TIME_SECONDS": 3600,
                "TX_TIME_DAYS": 100,
                "TX_DURING_NIGHT": 1,
                "CUSTOMER_ID_NB_TX_1DAY_WINDOW": 10,
                "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW": 800.0,
                "CUSTOMER_ID_NB_TX_7DAY_WINDOW": 15,
                "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW": 750.0,
                "CUSTOMER_ID_NB_TX_30DAY_WINDOW": 30,
                "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW": 700.0,
                "TERMINAL_ID_NB_TX_1DAY_WINDOW": 50,
                "TERMINAL_ID_RISK_1DAY_WINDOW": 0.25,
                "TERMINAL_ID_NB_TX_7DAY_WINDOW": 200,
                "TERMINAL_ID_RISK_7DAY_WINDOW": 0.20,
                "TERMINAL_ID_NB_TX_30DAY_WINDOW": 800,
                "TERMINAL_ID_RISK_30DAY_WINDOW": 0.18,
                "hour": 1,
                "day": 15,
                "month": 4,
                "weekday": 3,
                "is_weekend": 0,
                "is_night": 1
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=transactions,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*70)
    print("TEST 4: Model Info")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


if __name__ == "__main__":
    print("\n🧪 Testing Fraud Detection API")
    print("="*70)
    print("Make sure the API is running: python app.py")
    print("="*70)
    
    try:
        # Run all tests
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_model_info()
        
        print("\n" + "="*70)
        print("✅ All tests completed!")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("💡 Make sure the API is running: python app.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")