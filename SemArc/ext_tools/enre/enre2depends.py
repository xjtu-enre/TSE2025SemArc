import json

def convert_enre_to_depends(enre_data):
    depends_data = {"variables": [], "cells": []}
    file_variable_mapping = {}
    all_variable_mapping = {}
    file_num=0

    # Convert variables
    for entry in enre_data:
        for enre_variable in entry["variables"]:
            if enre_variable["category"]!="Macro":
                qualified_name = enre_variable["qualifiedName"]
                category = enre_variable["category"]
                # 实体所属的文件
                entity_file = enre_variable.get("entityFile")
                parent_id = enre_variable.get("parentID")
                startLine=enre_variable.get("startLine")
                all_variable_mapping[enre_variable["id"]] = {"qualifiedName": qualified_name, "category": category, "entityFile": entity_file, "parentID": parent_id,"startLine":startLine}

                if category == "File":
                    depends_data["variables"].append(qualified_name)
                    file_variable_mapping[file_num] = qualified_name
                    file_num+=1

    # Convert cells
    cell_mapping = {}
    for entry in enre_data:
        for enre_cell in entry["relations"]:
            src_id = enre_cell["from"]
            dest_id = enre_cell["to"]

            src_object = all_variable_mapping.get(src_id, {}).get("qualifiedName", None)
            dest_object = all_variable_mapping.get(dest_id, {}).get("qualifiedName", None)

            if all_variable_mapping.get(src_id, {}).get("entityFile"):
                src_file_id = all_variable_mapping[src_id]["entityFile"]
                src_file_name = all_variable_mapping[src_file_id]["qualifiedName"]
            elif all_variable_mapping.get(src_id, {}).get("parentID"):
                src_file_id = all_variable_mapping[src_id]["parentID"]
                src_file_name = all_variable_mapping[src_file_id]["qualifiedName"]
            else:
                src_file_id = src_id
                src_file_name = src_object

            if all_variable_mapping.get(dest_id, {}).get("entityFile"):
                dest_file_id = all_variable_mapping[dest_id]["entityFile"]
                dest_file_name = all_variable_mapping[dest_file_id]["qualifiedName"]
            elif all_variable_mapping.get(dest_id, {}).get("parentID"):
                dest_file_id = all_variable_mapping[dest_id]["parentID"]
                dest_file_name = all_variable_mapping[dest_file_id]["qualifiedName"]
            else:
                dest_file_id = dest_id
                dest_file_name = dest_object

            # 初始化新文件ID
            src_file_id_new = None
            dest_file_id_new = None

            # 遍历 file_variable_mapping 寻找 src_file_name 和 dest_file_name 的新ID
            for key, value in file_variable_mapping.items():
                if value == src_file_name:
                    src_file_id_new = key

            for key, value in file_variable_mapping.items():
                if value == dest_file_name:
                    dest_file_id_new = key

            # 检查是否找到映射，未找到时跳过当前循环
            if src_file_id_new is None or dest_file_id_new is None:
                print(f"Warning: Mapping not found for src '{src_file_name}' or dest '{dest_file_name}'. Skipping...")
                continue

            # 如果找到映射，继续执行以下逻辑
            if src_id in all_variable_mapping and dest_id in all_variable_mapping and src_id != dest_id:
                if src_file_id in all_variable_mapping and dest_file_id in all_variable_mapping and src_file_id != dest_file_id:
                    if enre_cell["category"] != "Define":
                        cell_key = (src_file_id, dest_file_id)

                        if cell_key not in cell_mapping:
                            cell_mapping[cell_key] = {
                                "src": src_file_id_new,
                                "dest": dest_file_id_new,
                                "values": {},
                                "details": []
                            }

                        cell_mapping[cell_key]["details"].append({
                            "src": {
                                "object": src_object,
                                "file": src_file_name,
                                "type": all_variable_mapping[src_id].get("category", "Unknown"),
                                "lineNumber": all_variable_mapping.get(src_id, {}).get("startLine", 0)
                            },
                            "dest": {
                                "object": dest_object,
                                "file": dest_file_name,
                                "type": all_variable_mapping[dest_id].get("category", "Unknown"),
                                "lineNumber": all_variable_mapping.get(dest_id, {}).get("startLine", 0)
                            },
                            "type": enre_cell["category"]
                        })
                        if enre_cell["category"] in cell_mapping[cell_key]["values"]:
                            cell_mapping[cell_key]["values"][enre_cell["category"]] += 1.0
                        else:
                            cell_mapping[cell_key]["values"][enre_cell["category"]] = 1.0

    depends_data["cells"] = list(cell_mapping.values())

    return depends_data

def main():
    #enre JSON文件
    enre_json_path = "D:\\enre\\blink-114\\blink-114_out.json"
    # 生成的depends JSON文件
    output_depends_path = "D:\\enre\\blink-114\\blink-114_out_depends.json"

    with open(enre_json_path, "r", encoding="utf-8") as enre_file:
        enre_data = json.load(enre_file)

    depends_data = convert_enre_to_depends(enre_data)

    with open(output_depends_path, "w", encoding="utf-8") as depends_file:
        json.dump(depends_data, depends_file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()