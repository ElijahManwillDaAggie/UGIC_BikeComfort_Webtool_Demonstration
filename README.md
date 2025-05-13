# Bike Route Planner

A web application for planning comfortable bike routes in Logan, Utah, using comfort level data from shapefiles.

## Features

- Interactive map interface for route planning
- Comfort-based route calculation
- Address geocoding
- Distance calculations
- Multiple route options based on comfort levels

## Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd bike-route-planner
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place the required shapefile (`COMFORT_LOGAN.shp` and associated files) in the project root directory.

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Python Version: 3.11.0

4. Add the following environment variables:
   - `SHAPE_RESTORE_SHX`: YES

5. Add a disk:
   - Name: data
   - Mount Path: /opt/render/project/src/data
   - Size: 1 GB

6. Upload the shapefile and associated files to the disk after deployment

## Required Files

The following files must be present in the project directory:
- `COMFORT_LOGAN.shp`
- `COMFORT_LOGAN.shx`
- `COMFORT_LOGAN.dbf`
- `COMFORT_LOGAN.prj`
- `COMFORT_LOGAN.cpg`
- `COMFORT_LOGAN.sbx`
- `COMFORT_LOGAN.sbn`
- `COMFORT_LOGAN.shp.xml`

## API Endpoints

- `GET /`: Serves the main web interface
- `GET /bike_comfort_roads.geojson`: Returns the bike comfort road data
- `GET /health`: Health check endpoint
- `POST /calculate_route`: Calculates a route between two addresses
- `POST /calculate_distance`: Calculates the distance between two points

## Dependencies

- Flask
- GeoPandas
- Shapely
- NetworkX
- PyProj
- Pandas
- NumPy
- Requests
- Gunicorn

## License

[Your License Here] 