import json

def convert_enre_to_depends(enre_data):
    depends_data = {"variables": [], "cells": []}
    file_variable_mapping = {}
    all_variable_mapping = {}
    file_num = 0

    # Convert variables
    for enre_variable in enre_data["variables"]:
        category = enre_variable.get("category", "Unknown")
        if category != "Macro":
            qualified_name = enre_variable["qualifiedName"]
            # 实体所属的文件
            entity_file = enre_variable.get("entityFile")
            parent_id = enre_variable.get("parentId")
            startLine = enre_variable.get("startLine")
            file_name = enre_variable.get("File", "Unknown")
            all_variable_mapping[enre_variable["id"]] = {
                "qualifiedName": qualified_name,
                "category": category,
                "entityFile": entity_file,
                "parentID": parent_id,
                "startLine": startLine,
                "File": file_name
            }

            if category == "File":
                depends_data["variables"].append(qualified_name)
                file_variable_mapping[file_num] = qualified_name
                file_num += 1

    # Convert cells
    cell_mapping = {}
    for enre_cell in enre_data["cells"]:
        src_id = enre_cell["src"]
        dest_id = enre_cell["dest"]

        src_object = all_variable_mapping.get(src_id, {}).get("qualifiedName", None)
        dest_object = all_variable_mapping.get(dest_id, {}).get("qualifiedName", None)

        src_file = all_variable_mapping.get(src_id, {}).get("File", "Unknown")
        dest_file = all_variable_mapping.get(dest_id, {}).get("File", "Unknown")

        # Find the qualified name for src_file_name
        src_file_name = None
        for var_id, var_data in all_variable_mapping.items():
            if var_data["category"] == "File" and var_data["File"] == src_file:
                src_file_name = var_data["qualifiedName"]
                break

        # Find the qualified name for dest_file_name
        dest_file_name = None
        for var_id, var_data in all_variable_mapping.items():
            if var_data["category"] == "File" and var_data["File"] == dest_file:
                dest_file_name = var_data["qualifiedName"]
                break

        # 找到将所有file重新编号后的id
        src_file_id_new = None
        dest_file_id_new = None

        if src_file_name is not None:
            for key, value in file_variable_mapping.items():
                if value == src_file_name:
                    src_file_id_new = key

        if dest_file_name is not None:
            for key, value in file_variable_mapping.items():
                if value == dest_file_name:
                    dest_file_id_new = key

        if all_variable_mapping.get(src_id, {}).get("startLine"):
            src_line_num = all_variable_mapping[src_id]["startLine"]
        else:
            src_line_num = 0

        if all_variable_mapping.get(dest_id, {}).get("startLine"):
            dest_line_num = all_variable_mapping[dest_id]["startLine"]
        else:
            dest_line_num = 0

        if src_id in all_variable_mapping and dest_id in all_variable_mapping and src_id != dest_id:
            if src_file_id_new is not None and dest_file_id_new is not None and src_file_id_new != dest_file_id_new:
                # 是不同文件之间的实体依赖
                for relation_type in enre_cell["values"]:
                    if relation_type != "loc":
                        cell_key = (src_file_id_new, dest_file_id_new)

                        if cell_key not in cell_mapping:
                            cell_mapping[cell_key] = {  # 外层src和dest为实体所属file
                                "src": src_file_id_new,
                                "dest": dest_file_id_new,
                                "values": {},
                                "details": []
                            }

                        cell_mapping[cell_key]["details"].append({
                            "src": {
                                "object": src_object,
                                "File": src_file_name,
                                "type": all_variable_mapping[src_id].get("category", "Unknown"),
                                "lineNumber": src_line_num
                            },
                            "dest": {
                                "object": dest_object,
                                "File": dest_file_name,
                                "type": all_variable_mapping[dest_id].get("category", "Unknown"),
                                "lineNumber": dest_line_num
                            },
                            "type": relation_type
                        })
                        if relation_type in cell_mapping[cell_key]["values"]:
                            cell_mapping[cell_key]["values"][relation_type] += 1.0
                        else:
                            cell_mapping[cell_key]["values"][relation_type] = 1.0

    depends_data["cells"] = list(cell_mapping.values())

    return depends_data

def main():
    # enre JSON文件
    enre_json_path = ".\\jabref\\jabref-out.json"
    # 生成的depends JSON文件
    output_depends_path = ".\\jabref\\jabref_out_depends.json"

    with open(enre_json_path, "r", encoding="utf-8") as enre_file:
        enre_data = json.load(enre_file)

    depends_data = convert_enre_to_depends(enre_data)

    with open(output_depends_path, "w", encoding="utf-8") as depends_file:
        json.dump(depends_data, depends_file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()