import os
import xml.etree.ElementTree as ET
import pandas as pd

# Path to MedQuAD folder
DATASET_PATH = "Data\MedQuAD-master"          # Change if your folder name is different

records = []

# Traverse all XML files
for root_dir, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".xml"):
            file_path = os.path.join(root_dir, file)

            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                # Document level information
                document_id = root.attrib.get("id", "")
                source = root.attrib.get("source", "")
                url = root.attrib.get("url", "")
                focus = root.findtext("Focus", default="")

                # Extract all QA pairs
                qa_pairs = root.find("QAPairs")

                if qa_pairs is not None:
                    for qa in qa_pairs.findall("QAPair"):

                        question_element = qa.find("Question")
                        answer_element = qa.find("Answer")

                        question = ""
                        qtype = ""
                        answer = ""

                        if question_element is not None:
                            question = question_element.text.strip() if question_element.text else ""
                            qtype = question_element.attrib.get("qtype", "")

                        if answer_element is not None:
                            answer = answer_element.text.strip() if answer_element.text else ""

                        records.append({
                            "document_id": document_id,
                            "focus": focus,
                            "qtype": qtype,
                            "question": question,
                            "answer": answer,
                            "source": source,
                            "url": url,
                            "xml_file": file
                        })

            except Exception as e:
                print(f"Error reading {file_path}")
                print(e)

# Convert to DataFrame
df = pd.DataFrame(records)

# Save CSV
df.to_csv("medquad.csv", index=False)

print("=" * 50)
print(f"Total QA Pairs : {len(df)}")
print("CSV Saved As   : medquad.csv")
print("=" * 50)