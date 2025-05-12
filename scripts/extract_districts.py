import json

# Path to the TopoJSON file
file_path = 'data/Portugal-Distritos-Ilhas_TopoJSON.json'

# Read the file content
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        topojson_data = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {file_path}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the file: {e}")
    exit()

# Extract district names
district_names = set()

try:
    # Navigate to the geometries list
    # Adjust the key if your actual object name differs from 'ilhasGeo2'
    if 'objects' in topojson_data and 'ilhasGeo2' in topojson_data['objects']:
        geometries = topojson_data['objects']['ilhasGeo2'].get('geometries', [])
        
        for geometry in geometries:
            properties = geometry.get('properties', {})
            # Check if it's classified as a 'Distrito'
            if properties.get('TYPE_1') == 'Distrito':
                name = properties.get('NAME_1')
                if name:
                    district_names.add(name)
    else:
        print("Error: Could not find 'objects' or 'ilhasGeo2' key in the TopoJSON structure.")

except Exception as e:
    print(f"An error occurred while processing the geometries: {e}")

# Print the unique district names found
if district_names:
    print("Found District Names (NAME_1 where TYPE_1 == 'Distrito'):")
    # Sort for consistent output
    for name in sorted(list(district_names)):
        print(f"- {name}")
else:
    print("No district names found matching the criteria.") 