import xml.etree.ElementTree as ET
import csv

if __name__ == "__main__":
    tree = ET.parse("./data/ABSA16_Laptops_Train_SB1_v2.xml")
    reviews = tree.getroot()

    aspects = set()
    polarities = set()
    for aspect in reviews.iter("Opinion"):
        aspects.add(aspect.attrib["category"])
        polarities.add(aspect.attrib["polarity"])

    print(f"Number of different aspects = {len(aspects)}\n")
    print(aspects)
    print("\n\n")
    print(f"Number of different polarities = {len(polarities)}\n")
    print(polarities)
    print("\n\n")


    dataset = []
    for sentence_nr, sentence in enumerate(reviews.iter("sentence")):
        for aspect in aspects:
            record = {}
            record["sentence_id"] = sentence_nr
            record["aspect"] = aspect.replace("#", " ").replace("_", " ").lower()
            for text in sentence.iter("text"):
                record["text"] = text.text

            record["sentiment"] = "no"
            for opinion in sentence.iter("Opinion"):
                if opinion.attrib["category"] == aspect:
                    record["sentiment"] = opinion.attrib["polarity"]
            dataset.append(record)

    print(f"Dataset length = {len(dataset)}")

    with open("./data/dataset.csv", "w") as f:
        writer = csv.DictWriter(f, ["sentence_id", "text", "aspect", "sentiment"])
        writer.writeheader()
        writer.writerows(dataset)
    print("Dataset created")