import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from mmpretrain import TextToImageRetrievalInferencer

# Add current path to sys.path to ensure module imports work
current_path = os.getcwd()
if current_path not in sys.path:
    sys.path.append(current_path)

try:
    from src import datasets
    from src import models
except ImportError:
    print("Warning: Custom modules from 'src' directory not found. Please ensure 'src' exists and contains necessary __init__.py.")


class Settings:
    """Configuration and parameters for the retrieval and re-ranking process."""
    debug = False
    
    config_file = 'configs/exp/xvlm_1xb24_hccm_geotext1652.py'
    checkpoint_file = "pretrain/epoch_6.pth" # download from https://drive.google.com/file/d/1p468glkjTqxuE7YhzdXnC1Kx4xEJQEM3/view?usp=sharing
    
    # Modify to your data path
    data_root = "../dataset"
    image_gallery_dir = f"{data_root}/GeoText1652_Dataset/Track_4_Phase_II_Images/"
    query_file = f"{data_root}/GeoText1652_Dataset/PhaseII-queries.txt"

    output_retrieval_file = './topk256.txt'
    reranked_output_file = './topk256_diver.txt'
    
    top_k_for_rerank = 256
    top_k_for_direct_output = 10
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    final_k_rerank = 10
    similarity_threshold = 0.85


def get_image_paths_from_dir(directory: str) -> tuple[list, list]:
    """Retrieves all image filenames and full paths from a specified directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    filenames, full_paths = [], []
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname)[1].lower() in image_extensions:
            filenames.append(fname)
            full_paths.append(os.path.join(directory, fname))
    return filenames, full_paths


class RetrievalAndRerankRunner:
    """Encapsulates the text-to-image retrieval and diversity re-ranking workflow."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.queries = self._load_queries()
        self.inferencer = self._setup_inferencer()
        self.all_image_filenames, _ = get_image_paths_from_dir(self.settings.image_gallery_dir)
        self.filename_to_idx = {os.path.splitext(fname)[0]: i for i, fname in enumerate(self.all_image_filenames)}
        self.basename_to_fullname_map = {os.path.splitext(fname)[0]: fname for fname in self.all_image_filenames}
        self.all_image_features = self.inferencer.prototype['image_feat'].cpu().numpy()

    def _load_queries(self) -> list:
        """Loads queries from a text file (query_id\tquery_text format)."""
        query_file_path = Path(self.settings.query_file)
        if not query_file_path.exists():
            print(f"Error: Query file '{query_file_path}' not found.")
            sys.exit(1)

        queries = []
        with open(self.settings.query_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    query_id, text = line.split('\t', 1)
                    queries.append({'id': query_id, 'text': text})
                except ValueError:
                    print(f"Warning: Skipping malformed line -> '{line}'")
        
        if self.settings.debug:
            queries = queries[:5]
        
        return queries

    def _setup_inferencer(self) -> TextToImageRetrievalInferencer:
        """Builds and initializes the TextToImageRetrievalInferencer."""
        inferencer = TextToImageRetrievalInferencer(
            model=self.settings.config_file,
            prototype=self.settings.image_gallery_dir,
            fast_match=False,
            pretrained=self.settings.checkpoint_file,
            device=self.settings.device
        )
        return inferencer

    def _format_output_line(self, query_id_raw: str, retrieved_images: list) -> str:
        """Formats a single output line for submission."""
        try:
            query_numeric_part = int(query_id_raw.split('_')[-1])
            formatted_query_id = f"q{query_numeric_part}"
        except ValueError:
            formatted_query_id = query_id_raw

        retrieved_img_ids = [
            os.path.splitext(self.basename_to_fullname_map.get(img_basename, img_basename + '.jpeg'))[0]
            for img_basename in retrieved_images
        ]
        return f"{formatted_query_id} {' '.join(retrieved_img_ids)}"

    def run_retrieval_and_rerank(self):
        """Executes the full retrieval, re-ranking process, and saves results."""
        if not self.queries:
            return

        query_texts = [q['text'] for q in self.queries]
        query_ids_raw = [q['id'] for q in self.queries]

        retrieval_results_full = self.inferencer(
            query_texts, 
            batch_size=self.settings.batch_size, 
            topk=self.settings.top_k_for_rerank
        )
        
        with open(self.settings.output_retrieval_file, 'w', encoding='utf-8') as f_direct_out:
            for query_id_raw, retrieved_matches in zip(query_ids_raw, retrieval_results_full):
                retrieved_img_basenames = [
                    os.path.splitext(match['sample']['img_path'].split('/')[-1])[0]
                    for match in retrieved_matches[:self.settings.top_k_for_direct_output]
                ]
                line = self._format_output_line(query_id_raw, retrieved_img_basenames)
                f_direct_out.write(line + '\n')

        new_reranked_results = {}

        for i, query_id_raw in tqdm(enumerate(query_ids_raw), total=len(query_ids_raw), desc="Re-ranking queries"):
            candidates_full_list = [
                os.path.splitext(match['sample']['img_path'].split('/')[-1])[0]
                for match in retrieval_results_full[i]
            ]
            
            # Diversity Pass
            diverse_images = []
            diverse_features = []
            
            for candidate_name in candidates_full_list:
                if len(diverse_images) >= self.settings.final_k_rerank:
                    break
                    
                candidate_idx = self.filename_to_idx.get(candidate_name)
                if candidate_idx is None:
                    continue
                
                candidate_feat = self.all_image_features[candidate_idx]
                
                if not diverse_images:
                    diverse_images.append(candidate_name)
                    diverse_features.append(candidate_feat)
                    continue
                
                similarities = np.dot(np.array(diverse_features), candidate_feat)
                
                if np.max(similarities) < self.settings.similarity_threshold:
                    diverse_images.append(candidate_name)
                    diverse_features.append(candidate_feat)
                    
            reranked_images = diverse_images

            # Fill-in Pass
            if len(reranked_images) < self.settings.final_k_rerank:
                selected_set = set(reranked_images)
                
                for candidate_name in candidates_full_list:
                    if len(reranked_images) >= self.settings.final_k_rerank:
                        break
                    
                    if candidate_name not in selected_set:
                        reranked_images.append(candidate_name)
            
            new_reranked_results[query_id_raw] = reranked_images

        with open(self.settings.reranked_output_file, 'w', encoding='utf-8') as f_rerank_out:
            for query_id_raw, image_list in new_reranked_results.items():
                line = self._format_output_line(query_id_raw, image_list)
                f_rerank_out.write(line + '\n')


if __name__ == "__main__":
    config = Settings()
    runner = RetrievalAndRerankRunner(config)
    runner.run_retrieval_and_rerank()