import argparse
import carla

def main():
    parser = argparse.ArgumentParser(description="Convert an OSM map file to OpenDRIVE (.xodr) using CARLA's Osm2Odr.")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input .osm file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save the output .xodr file"
    )

    args = parser.parse_args()

    # Read the .osm data
    with open(args.input, 'r', encoding='utf-8') as f:
        osm_data = f.read()

    # Define the desired settings
    settings = carla.Osm2OdrSettings()
    settings.set_osm_way_types([
        "motorway", "motorway_link",
        "trunk", "trunk_link",
        "primary", "primary_link",
        "secondary", "secondary_link",
        "tertiary", "tertiary_link",
        "unclassified", "residential",
        "service",               # <-- добавить
        "living_street" ,         # <-- опционально
    ])

    # Convert to .xodr
    xodr_data = carla.Osm2Odr.convert(osm_data, settings)

    # Save OpenDRIVE file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(xodr_data)

    print(f"Successfully converted '{args.input}' to '{args.output}'")

if __name__ == "__main__":
    main()