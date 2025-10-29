"""
dt.py - Load Indian Law Dataset from Hugging Face
Dataset: viber1/indian-law-dataset
"""

from datasets import load_dataset
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndianLawDatasetLoader:
    """Load and save Indian law dataset from Hugging Face"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = None
    
    def load_dataset(self, dataset_name: str = "viber1/indian-law-dataset"):
        """Load dataset from Hugging Face"""
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            # Load the dataset
            self.dataset = load_dataset(dataset_name)
            
            logger.info("Dataset loaded successfully!")
            logger.info(f"Available splits: {list(self.dataset.keys())}")
            
            # Print dataset info
            for split_name, split_data in self.dataset.items():
                logger.info(f"{split_name}: {len(split_data)} examples")
                logger.info(f"Features: {split_data.features}")
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def explore_dataset(self):
        """Explore dataset structure"""
        if not self.dataset:
            logger.error("Dataset not loaded. Call load_dataset() first.")
            return
        
        print("\n" + "="*60)
        print("DATASET EXPLORATION")
        print("="*60)
        
        # Check each split
        for split_name, split_data in self.dataset.items():
            print(f"\n--- Split: {split_name} ---")
            print(f"Total examples: {len(split_data)}")
            print(f"Features: {list(split_data.features.keys())}")
            
            # Show first example
            if len(split_data) > 0:
                print(f"\nFirst example:")
                first_example = split_data[0]
                for key, value in first_example.items():
                    value_preview = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                    print(f"  {key}: {value_preview}")
    
    def convert_to_json(self, split: str = "train", max_examples: int = None):
        """Convert dataset to simple JSON format with id, Instruction, Response"""
        if not self.dataset:
            logger.error("Dataset not loaded. Call load_dataset() first.")
            return []
        
        if split not in self.dataset:
            logger.error(f"Split '{split}' not found in dataset")
            return []
        
        split_data = self.dataset[split]
        
        # Limit examples if specified
        if max_examples:
            split_data = split_data.select(range(min(max_examples, len(split_data))))
        
        logger.info(f"Converting {len(split_data)} examples from '{split}' split...")
        
        converted_data = []
        skipped = 0
        
        for idx, example in enumerate(split_data):
            try:
                # Debug: Print first example structure
                if idx == 0:
                    logger.info(f"Example structure: {list(example.keys())}")
                    logger.info(f"First example preview:")
                    for key, value in example.items():
                        preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        logger.info(f"  {key}: {preview}")
                
                # Handle Instruction-Response format (viber1/indian-law-dataset)
                if 'Instruction' in example and 'Response' in example:
                    instruction = str(example['Instruction']).strip()
                    response = str(example['Response']).strip()
                    
                    # Skip if either is too short
                    if len(instruction) < 10 or len(response) < 50:
                        skipped += 1
                        continue
                    
                    # Keep original format with id, Instruction, Response
                    doc = {
                        'id': idx,
                        'Instruction': instruction,
                        'Response': response
                    }
                    
                    # Add the document
                    converted_data.append(doc)
                    
                    # Progress update every 1000
                    if (idx + 1) % 1000 == 0:
                        logger.info(f"Processed {idx + 1} examples, converted {len(converted_data)}, skipped {skipped}")
                
                else:
                    # For other formats, try to adapt to Instruction-Response format
                    instruction = None
                    response = None
                    
                    if 'question' in example and 'answer' in example:
                        instruction = str(example['question']).strip()
                        response = str(example['answer']).strip()
                    elif 'text' in example:
                        # Try to split if it contains Q&A format
                        text = str(example['text']).strip()
                        if len(text) > 100:
                            instruction = example.get('title', f"Document {idx}")
                            response = text
                    
                    if instruction and response and len(instruction) >= 10 and len(response) >= 50:
                        doc = {
                            'id': idx,
                            'Instruction': instruction,
                            'Response': response
                        }
                        converted_data.append(doc)
                    else:
                        skipped += 1
                
            except Exception as e:
                logger.warning(f"Error converting example {idx}: {e}")
                skipped += 1
                continue
        
        logger.info(f"Successfully converted {len(converted_data)} documents (skipped {skipped})")
        return converted_data
    
    def save_to_json(self, data: list, filename: str = "huggingface_legal_data.json"):
        """Save converted data to JSON file"""
        filepath = self.output_dir / filename
        
        logger.info(f"Saving {len(data)} documents to {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Saved successfully!")
        return filepath
    
    def process_all_splits(self, max_per_split: int = None):
        """Process all available splits and save to separate files"""
        if not self.dataset:
            logger.error("Dataset not loaded. Call load_dataset() first.")
            return
        
        all_data = []
        
        for split_name in self.dataset.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing split: {split_name}")
            logger.info(f"{'='*60}")
            
            data = self.convert_to_json(split=split_name, max_examples=max_per_split)
            
            if data:
                # Save split-specific file
                filename = f"legal_data_{split_name}.json"
                self.save_to_json(data, filename)
                
                # Add to combined data
                all_data.extend(data)
        
        # Save combined file
        if all_data:
            logger.info(f"\n{'='*60}")
            logger.info(f"Saving combined dataset")
            logger.info(f"{'='*60}")
            self.save_to_json(all_data, "legal_data_all.json")
        
        return all_data

def main():
    """Main function"""
    print("="*60)
    print("INDIAN LAW DATASET LOADER")
    print("="*60)
    print("\nLoading from Hugging Face: viber1/indian-law-dataset")
    print("This may take a few minutes on first run...\n")
    
    # Initialize loader
    loader = IndianLawDatasetLoader(output_dir="data/raw")
    
    try:
        # Load dataset
        dataset = loader.load_dataset("viber1/indian-law-dataset")
        
        # Explore structure
        loader.explore_dataset()
        
        # Ask user how many examples to process
        print("\n" + "="*60)
        print("DATASET LOADED SUCCESSFULLY!")
        print("="*60)
        
        choice = input("\nProcess all examples? (y/n): ").lower().strip()
        
        if choice == 'y':
            max_examples = None
        else:
            max_input = input("Enter max examples per split (or 'all'): ").strip()
            if max_input.lower() == 'all' or max_input == '':
                max_examples = None
            else:
                try:
                    max_examples = int(max_input)
                except ValueError:
                    print(f"Invalid input '{max_input}'. Processing all examples.")
                    max_examples = None
        
        # Process and save
        print("\n" + "="*60)
        print("PROCESSING DATASET")
        print("="*60)
        
        all_data = loader.process_all_splits(max_per_split=max_examples)
        
        # Summary
        print("\n" + "="*60)
        print("COMPLETE!")
        print("="*60)
        print(f"✅ Total documents processed: {len(all_data)}")
        print(f"✅ Files saved in: data/raw/")
        print("\nNext steps:")
        print("1. Run: python data_preprocess.py")
        print("2. Run: python models/vector_db.py")
        print("3. Test: python models/rag_pipeline.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install datasets: pip install datasets")
        print("2. Check internet connection")
        print("3. Try a different dataset name")

if __name__ == "__main__":
    main()