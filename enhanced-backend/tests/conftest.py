"""
Test configuration and fixtures for the SuperHacker Enhanced Backend
"""

import os
import pytest
import pandas as pd
from app import create_app, db
from app.models.dataset import Dataset

@pytest.fixture
def app():
    """Create and configure a test Flask application instance"""
    # Create the app with testing config
    app = create_app('testing')
    
    # Create test database and tables
    with app.app_context():
        db.create_all()
        
        # Create a test dataset
        dataset = Dataset(
            name="Test Dataset",
            filename="sample_data.csv",
            file_path="../sample_data.csv",
            description="Test dataset for API testing",
            file_type="csv",
            mime_type="text/csv",
            status="ready"
        )
        db.session.add(dataset)
        db.session.commit()
        
        yield app
        
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    """Create a test client"""
    return app.test_client()

@pytest.fixture
def test_dataset(app):
    """Create a test dataset record with proper session handling"""
    with app.app_context():
        dataset = Dataset.query.first()
        if not dataset:
            dataset = Dataset(
                name="Test Dataset",
                filename="sample_data.csv",
                file_path="../sample_data.csv",
                description="Test dataset for API testing",
                file_type="csv",
                mime_type="text/csv",
                status="ready"
            )
            db.session.add(dataset)
            db.session.commit()
        return dataset.id

@pytest.fixture
def sample_data():
    """Load sample data from CSV"""
    sample_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_data.csv')
    return pd.read_csv(sample_file)
