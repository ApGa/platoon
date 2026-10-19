"""
Recipe loader for synthetic TextCraft-Synth recipes.

Adapts the SynthRecipeDatabase to match the interface of the original RecipeDatabase,
allowing the synthetic dataset to use the same environment code.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set

from .synth_recipe_generator import SynthRecipe, SynthRecipeDatabase


class SynthRecipe_Compat:
    """Wrapper to make SynthRecipe compatible with the original Recipe interface."""

    def __init__(self, synth_recipe: SynthRecipe):
        self._synth_recipe = synth_recipe
        self.result_item = synth_recipe.result_item
        self.result_count = synth_recipe.result_count
        self.ingredients = synth_recipe.ingredients.copy()
        self.filename = f"{synth_recipe.result_item}.json"
        self.type = "synth:crafting"

    def __repr__(self) -> str:
        return f"SynthRecipe({self.result_item} <- {self.ingredients})"


class SynthRecipeLoader:
    """
    Recipe database for synthetic recipes.

    Provides the same interface as RecipeDatabase for compatibility
    with the existing TextCraft environment.
    """

    def __init__(self, recipes_dir: Optional[Path] = None, seed: int = 42, items_per_domain_tier: int = 25):
        """
        Initialize the synthetic recipe database.

        Args:
            recipes_dir: Directory to save/load recipes (optional, used for caching)
            seed: Random seed for recipe generation
            items_per_domain_tier: Number of items per domain per tier (default 25 for single-target dataset)
        """
        self._synth_db = SynthRecipeDatabase()
        self._synth_db.generate_all_recipes(seed=seed, items_per_domain_tier=items_per_domain_tier)

        # Build compatible interface
        self.recipes: Dict[str, List[SynthRecipe_Compat]] = {}
        for item_name, synth_recipes in self._synth_db.recipes.items():
            self.recipes[item_name] = [SynthRecipe_Compat(r) for r in synth_recipes]

        self.all_items: Set[str] = self._synth_db.all_items.copy()
        self.base_items: Set[str] = self._synth_db.base_items.copy()
        self.craftable_items: Set[str] = set(self.recipes.keys())

        # No tags in synthetic recipes
        self.all_tags: Set[str] = set()
        self.tag_to_items: Dict[str, Set[str]] = {}

        # Store for reference
        self.item_depths = self._synth_db.item_depths.copy()
        self.items_by_tier = self._synth_db.items_by_tier

    def get_items_for_tag(self, tag_name: str) -> List[str]:
        """Get items for tag (not used in synthetic recipes)."""
        return []

    def get_recipes_for_item(self, item: str) -> List[SynthRecipe_Compat]:
        """Get all recipes that produce the given item."""
        return self.recipes.get(item, [])

    def can_craft(self, item: str) -> bool:
        """Check if an item can be crafted."""
        return item in self.recipes

    def is_base_item(self, item: str) -> bool:
        """Check if an item is a base material."""
        return item in self.base_items

    def get_crafting_depth(self, item: str, visited: Optional[Set[str]] = None) -> int:
        """Get the crafting depth of an item (pre-computed for synthetic recipes)."""
        return self.item_depths.get(item, 0)

    def find_hierarchical_recipes(self, min_depth: int = 2, max_depth: int = 12) -> List[str]:
        """Find items that require hierarchical crafting."""
        hierarchical = []
        for item, depth in self.item_depths.items():
            if min_depth <= depth <= max_depth:
                hierarchical.append(item)
        return hierarchical

    def get_recipe_chain(self, item: str, visited: Optional[Set[str]] = None) -> List[tuple]:
        """Get a chain of recipes needed to craft an item."""
        if visited is None:
            visited = set()

        if item in visited or self.is_base_item(item):
            return []

        if item not in self.recipes:
            return []

        visited.add(item)
        recipes = self.recipes[item]
        if not recipes:
            return []

        recipe = recipes[0]
        chain = [(item, recipe)]

        for ingredient in recipe.ingredients.keys():
            sub_chain = self.get_recipe_chain(ingredient, visited.copy())
            chain.extend(sub_chain)

        return chain


def create_synth_recipe_database(seed: int = 42, items_per_domain_tier: int = 25) -> SynthRecipeLoader:
    """Create a synthetic recipe database."""
    return SynthRecipeLoader(seed=seed, items_per_domain_tier=items_per_domain_tier)
