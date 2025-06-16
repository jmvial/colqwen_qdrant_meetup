import os
import re
from pathlib import Path
import torch
from PIL import Image
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from colqwen_qdrant_demo.src.vision_model_loader import get_colqwen_model, get_colqwen_processor, device
from dotenv import load_dotenv

load_dotenv("colqwen_qdrant_meetup/.env")

colqwen_model = get_colqwen_model()
colqwen_processor = get_colqwen_processor()

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")

image_dir = Path(DATA_FOLDER_PATH)
image_paths = list(image_dir.glob("*.png"))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
BATCH_SIZE = 5

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=128,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
    ),
)

points = []

for idx, image_path in enumerate(image_paths):
    image = Image.open(image_path).convert("RGB")
    image_input = colqwen_processor(images=image)
    image_input = {k: v.to(device) for k, v in image_input.items()}
    
    with torch.no_grad():
        image_embeddings = colqwen_model(**image_input).embeddings  # (1, 731, 128)
    
    image_embedding = image_embeddings.squeeze(0).cpu().tolist()  # (731, 128)

    match = re.match(r"(.+)_([0-9]+)\.png", image_path.name)
    if not match:
        continue

    base_name = match.group(1)
    page_number = int(match.group(2))

    # ✅ vector como lista de vectores (no diccionario)
    points.append(PointStruct(
        id=idx,
        vector=image_embedding,
        payload={
            "filename": image_path.name,
            "base": base_name,
            "page": page_number
        }
    ))


for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i+BATCH_SIZE]
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=batch)

