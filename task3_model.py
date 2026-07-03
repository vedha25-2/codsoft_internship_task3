import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# 1. THE DATA ENGINE (Products & User Profiles)
# =====================================================================
# Content Data (Item Descriptions/Genres)
movies_metadata = {
    'movie_id': [101, 102, 103, 104, 105],
    'title': ['The Matrix', 'Inception', 'Interstellar', 'Toy Story', 'Finding Nemo'],
    'metadata': [
        'Sci-Fi action simulation virtual reality cyberpunk keanu',
        'Sci-Fi thriller dream heist mind-bending Nolan',
        'Sci-Fi space exploration gravity time-travel Nolan',
        'Animation children pixar toys family comedy',
        'Animation children pixar fish ocean family adventure'
    ]
}
df_items = pd.DataFrame(movies_metadata)

# Collaborative Data (Historical User Ratings: 1 to 5 stars)
ratings_matrix = pd.DataFrame({
    'User_A': [5, 4, 4, 1, 0],  # Likes Sci-Fi
    'User_B': [1, 0, 2, 5, 4],  # Likes Animation
    'User_C': [4, 5, 0, 2, 1],  # Likes Sci-Fi
}, index=df_items['title'])  # Indexed by movie title for clean matrix math


# =====================================================================
# 2. ADVANCED HYBRID RECOMMENDER CLASS
# =====================================================================
class HybridRecommender:
    def __init__(self, items_df, ratings_df):
        self.items_df = items_df
        self.ratings_df = ratings_df
        self.tfidf_matrix = None
        self.content_sim = None
        self._build_content_engine()

    def _build_content_engine(self):
        """Constructs the fallback content-based vector matrix."""
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.items_df['metadata'])
        self.content_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def _get_content_recommendations(self, target_item, top_n=2):
        """Fallback Logic: Content-based filtering using Cosine Similarity."""
        idx = self.items_df[self.items_df['title'] == target_item].index[0]
        sim_scores = list(enumerate(self.content_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Pick top items skipping itself
        chosen_indices = [i[0] for i in sim_scores[1:top_n+1]]
        return self.items_df['title'].iloc[chosen_indices].tolist()

    def _get_collaborative_recommendations(self, user_id, top_n=2):
        """Primary Logic: User-User Collaborative Filtering via Pearson Correlation."""
        # Calculate user similarities across shared item ratings
        user_corr = self.ratings_df.corr(method='pearson')
        
        # Find the most similar user to our target
        similar_users = user_corr[user_id].sort_values(ascending=False)
        most_similar_user = similar_users.index[1] # index 0 is the user themselves
        
        # Find items the target user hasn't rated, but the similar user rated highly
        target_user_history = self.ratings_df[user_id]
        similar_user_history = self.ratings_df[most_similar_user]
        
        unwatched_by_target = target_user_history[target_user_history == 0].index
        
        # Sort unviewed items by the peer user's score metrics
        recommendations = similar_user_history[unwatched_by_target].sort_values(ascending=False)
        return recommendations.head(top_n).index.tolist()

    def recommend(self, user_id=None, last_watched_item=None, top_n=2):
        """
        Unique Switch Logic:
        If user data exists -> Deploy collaborative behavioral mapping.
        If user is anonymous/new -> Deploy textual item content similarity maps.
        """
        # Scenario A: User Profile Exists & has rated items
        if user_id in self.ratings_df.columns and not (self.ratings_df[user_id] == 0).all():
            print(f" [Collaborative Engine] Routing based on historical trends of {user_id}:")
            return self._get_collaborative_recommendations(user_id, top_n)
        
        # Scenario B: Cold-start / New User (Relies strictly on item metadata)
        elif last_watched_item:
            print(f" [Content Engine] Cold-Start detected. Routing similarities for '{last_watched_item}':")
            return self._get_content_recommendations(last_watched_item, top_n)
        
        else:
            return ["Default Top Trending Items: The Matrix, Inception"]


# =====================================================================
# 3. LIVE PIPELINE DEMONSTRATION
# =====================================================================
if __name__ == "__main__":
    # Initialize our system framework
    engine = HybridRecommender(items_df=df_items, ratings_df=ratings_matrix)

    # TEST CASE 1: Active User profile (Collaborative Tracking)
    # User_C likes Sci-Fi, should map closely to User_A's taste profile.
    rec_user = engine.recommend(user_id='User_C', top_n=1)
    print(f" Recommendations: {rec_user}\n")

    # TEST CASE 2: New/Anonymous User (Content / Metadata Fallback)
    # User has no history profile, but just finished watching 'Toy Story'.
    rec_anonymous = engine.recommend(last_watched_item='Toy Story', top_n=1)
    print(f"Recommendations: {rec_anonymous}\n")
