"""
SuperHacker Enhanced Backend - Main Application Entry Point
"""

import os
from app import create_app, db
from app.models import Dataset, MLModel, Workflow, Visualization, SystemLog

# Create Flask application
app = create_app()
with app.app_context():
    db.create_all()


@app.cli.command()
def init_db():
    """Initialize the database"""
    db.create_all()
    print("Database initialized!")

@app.cli.command()
def reset_db():
    """Reset the database"""
    db.drop_all()
    db.create_all()
    print("Database reset!")

@app.cli.command()
def seed_db():
    """Seed the database with sample data"""
    # Create sample datasets
    sample_dataset = Dataset(
        name="Sample Sales Data",
        filename="sales_data.csv",
        file_path="/tmp/sample_sales.csv",
        description="Sample sales dataset for testing",
        file_type="csv",
        rows_count=1000,
        columns_count=8,
        data_quality_score=85.5,
        status="ready"
    )
    
    # Create sample ML model
    sample_model = MLModel(
        name="Sales Prediction Model",
        algorithm="random_forest",
        model_type="regression",
        description="Random Forest model for sales prediction",
        accuracy=0.87,
        status="trained"
    )
    
    # Create sample workflow
    sample_workflow = Workflow(
        name="Data Processing Pipeline",
        description="Standard data processing and analysis workflow",
        status="ready"
    )
    
    # Create sample visualization
    sample_viz = Visualization(
        name="Sales Dashboard",
        chart_type="dashboard",
        description="Interactive sales performance dashboard",
        is_dashboard=True
    )
    
    db.session.add_all([sample_dataset, sample_model, sample_workflow, sample_viz])
    db.session.commit()
    
    print("Database seeded with sample data!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    #app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
