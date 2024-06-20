import argparse
from Model.recommender import HealthcareRecommender

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Healthcare Provider Recommendation System')
    
    parser.add_argument('--user_id', type=int, required=True,
                        help='Patient ID to generate recommendations for')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of recommendations to generate')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing data files')
    
    return parser.parse_args()

def main():
    """Main entry point for CLI."""
    args = parse_args()
    
    print(f"Initializing recommender system using data from '{args.data_dir}'...")
    recommender = HealthcareRecommender(data_dir=args.data_dir)
    
    print(f"\nGenerating top {args.top_n} recommendations for patient {args.user_id}...")
    recommendations = recommender.recommend(args.user_id, top_n=args.top_n)
    
    print("\nTop recommended healthcare providers:")
    print(recommendations[['provider_id', 'quality_score', 'cost', 'specialty', 
                          'predicted_rating']].to_string(index=False))

if __name__ == '__main__':
    main()
