import numpy as np
from typing import List
from numpy.typing import NDArray
import os
import torch
from transformers import AutoTokenizer, AutoModel

# Initialize tokenizer and model once at module level
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model.eval()  # Set to evaluation mode for inference

def string_to_vector(text: str) -> torch.Tensor:
    """
    Convert a string into a vector using a proper transformer model.
    This follows the LLM pipeline: str -> tokenizer -> embedding model -> vector
    
    Args:
        text (str): Input text to convert
        
    Returns:
        torch.Tensor: Vector representation of the text
    """
    # Tokenize the text
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**tokens)
        # Use the [CLS] token embedding as the sentence representation
        embedding = outputs.last_hidden_state[:, 0, :]
    
    # Normalize the embedding to unit length
    embedding = embedding.squeeze(0)  # Remove batch dimension
    norm = torch.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

def print_vector(vector: torch.Tensor) -> None:
    """
    Print the vector in a readable format.
    
    Args:
        vector (torch.Tensor): The vector to print
    """
    print("\nFull Vector:")
    print("=" * 80)
    
    # Print header
    header = "Dimension | Value"
    print(header)
    print("-" * len(header))
    
    # Print each dimension and its value
    for i, value in enumerate(vector):
        print(f"dim_{i:9} | {value:+.6f}")
    
    print("=" * 80)

def compute_angle(vector1: torch.Tensor, vector2: torch.Tensor) -> float:
    """
    Compute the angle between two vectors in degrees.
    
    Args:
        vector1 (torch.Tensor): First vector
        vector2 (torch.Tensor): Second vector
        
    Returns:
        float: Angle between vectors in degrees
    """
    # Compute dot product (cosine similarity since vectors are normalized)
    cos_angle = torch.dot(vector1, vector2)
    
    # Ensure cos_angle is within valid range [-1, 1]
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
    # Compute angle in radians and convert to degrees
    angle_radians = torch.acos(cos_angle)
    angle_degrees = torch.rad2deg(angle_radians)
    
    return angle_degrees.item()

def analyze_vectors(text1: str, text2: str):
    """
    Analyze and compare two text vectors with detailed information.
    """
    print(f"\nComparing: '{text1}' vs '{text2}'")
    print("=" * 80)
    
    # Convert texts to vectors
    vector1 = string_to_vector(text1)
    vector2 = string_to_vector(text2)
    
    # Print vector information
    print(f"Vector shapes: {vector1.shape}, {vector2.shape}")
    
    # Compute angle and cosine similarity
    angle = compute_angle(vector1, vector2)
    cos_sim = torch.dot(vector1, vector2).item()
    
    print(f"Angle between vectors: {angle:.2f}°")
    print(f"Cosine similarity: {cos_sim:.4f}")
    
    # Print vector norms
    print(f"Vector 1 norm: {torch.norm(vector1).item():.4f}")
    print(f"Vector 2 norm: {torch.norm(vector2).item():.4f}")
    
    # Print first few dimensions of each vector
    print("\nFirst 5 dimensions of each vector:")
    print("Vector 1:", vector1[:5].tolist())
    print("Vector 2:", vector2[:5].tolist())
    
    print("=" * 80)

def main():
    print("Welcome to the Text Vectorizer!")
    print("This program will analyze semantic relationships between two texts.")
    print("=" * 80)
    
    # Get input strings from user
    text1 = input("\nEnter the first text: ")
    text2 = input("Enter the second text: ")
    
    # Analyze the vectors
    analyze_vectors(text1, text2)

if __name__ == "__main__":
    main()
