import xml.etree.ElementTree as ET
import argparse
import os
import copy

def wrap_visual_geoms_with_bodies(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()

    def wrap_geom(parent):
        for i, elem in enumerate(parent):
            if elem.tag == "geom" and elem.attrib.get("class") == "visual":
                # Create a new <body> element
                body = ET.Element("body")
                # Give it a name based on mesh if available
                mesh_name = elem.attrib.get("mesh", f"{i}")
                body.set("name", f"body_{mesh_name}")
                # Deep copy of the geom to keep attributes
                body.append(copy.deepcopy(elem))
                # Replace <geom> with <body> in parent
                parent.remove(elem)
                parent.insert(i, body)
            else:
                wrap_geom(elem)  # Recurse into children

    wrap_geom(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

# Example usage
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True, default="./panda.xml")
args = parser.parse_args()

input_file = args.file
output_file = f"{os.path.splitext(input_file)[0]}_new.xml"
wrap_visual_geoms_with_bodies(input_file, output_file)
