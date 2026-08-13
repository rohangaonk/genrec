"""
GenRec Verbalizer Module
Translates structured user interaction history, item metadata, and contextual signals 
into natural-language prompts for the backbone LLM.
"""

from typing import List, Dict, Any, Optional

class GenRecVerbalizer:
    def __init__(self, max_history_length: int = 10):
        self.max_history_length = max_history_length

    def verbalize(
        self,
        user_history: List[Dict[str, Any]],
        context: Dict[str, Any],
        candidate_item: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Formats user history, context, and optional candidate item into a prompt string.

        Args:
            user_history: List of dicts containing 'title', 'genres', 'rating', 'year'
            context: Dict containing 'device', 'time_of_day', 'locale'
            candidate_item: Optional target item dict containing 'title', 'genres'

        Returns:
            Formatted prompt string.
        """
        # 1. Verbalize Context Signals
        device = context.get("device", "TV")
        time_of_day = context.get("time_of_day", "evening")
        locale = context.get("locale", "US")

        context_str = f"Context: User is watching on a {device} in the {time_of_day} ({locale}).\n"

        # 2. Verbalize Engagement History (Context Compaction: Keep top/recent N high-signal items)
        recent_history = user_history[-self.max_history_length:]
        
        history_items = []
        for item in recent_history:
            title = item.get("title", "Unknown Title")
            genres = item.get("genres", "General")
            rating = item.get("rating", None)
            
            if rating is not None:
                item_text = f"• \"{title}\" ({genres}) - Rated {rating}/5★"
            else:
                item_text = f"• \"{title}\" ({genres})"
            history_items.append(item_text)

        history_str = "Watch History:\n" + "\n".join(history_items) + "\n"

        # 3. Verbalize Candidate Item (if provided for pairwise/pointwise scoring)
        candidate_str = ""
        if candidate_item:
            title = candidate_item.get("title", "Unknown Title")
            genres = candidate_item.get("genres", "General")
            candidate_str = f"\nCandidate Item to Score: \"{title}\" ({genres})\n"

        prompt = (
            f"{context_str}\n"
            f"{history_str}"
            f"{candidate_str}\n"
            f"Task: Predict user affinity score for candidate item based on historical preferences and context."
        )

        return prompt.strip()


if __name__ == "__main__":
    # Self-test / demonstration
    verbalizer = GenRecVerbalizer(max_history_length=5)
    
    sample_history = [
        {"title": "The Matrix", "genres": "Sci-Fi|Action", "rating": 5, "year": 1999},
        {"title": "Inception", "genres": "Sci-Fi|Thriller", "rating": 5, "year": 2010},
        {"title": "Interstellar", "genres": "Sci-Fi|Drama", "rating": 4, "year": 2014},
    ]
    
    sample_context = {
        "device": "Smart TV",
        "time_of_day": "Late Night",
        "locale": "US-East"
    }

    sample_candidate = {
        "title": "Dune: Part One",
        "genres": "Sci-Fi|Adventure"
    }

    prompt = verbalizer.verbalize(sample_history, sample_context, sample_candidate)
    print("--- DEMO VERBALIZED PROMPT ---")
    print(prompt)
