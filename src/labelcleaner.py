import os

def filter_label_file(input_path, output_path, class_to_remove):
    with open(input_path, "r") as f:
        lines = f.readlines()

    # Filtere alle Zeilen, die NICHT die zu entfernende Klasse enthalten
    filtered_lines = [line for line in lines if not line.strip().startswith(class_to_remove + " ")]

    # Schreibe die gefilterten Zeilen
    with open(output_path, "w") as f:
        f.writelines(filtered_lines)

def process_all_labels(label_dir, output_dir, class_to_remove):
    for filename in os.listdir(label_dir):
        if not filename.endswith(".txt"):
            continue
        input_path = os.path.join(label_dir, filename)
        output_path = os.path.join(output_dir, filename)
        filter_label_file(input_path, output_path, class_to_remove)

    print(f"Fertig. Klasse '{class_to_remove}' wurde aus den Labels entfernt.")


