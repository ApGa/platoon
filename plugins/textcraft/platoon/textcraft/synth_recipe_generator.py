"""
Synthetic recipe generator for TextCraft-Synth dataset.

Creates a procedurally generated crafting system with configurable depth,
designed for much deeper hierarchies than the original Minecraft recipes.

The synthetic world consists of multiple material domains, each with their
own crafting progression from raw materials to ultimate items.
"""
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class MaterialDomain(Enum):
    """Different material domains for variety."""
    METAL = "metal"
    CRYSTAL = "crystal"
    ORGANIC = "organic"
    ARCANE = "arcane"
    TECH = "tech"


# Domain-specific naming prefixes
# Two modes: semantic (for debugging) and generic (for clean benchmarking)
DOMAIN_PREFIXES_SEMANTIC = {
    MaterialDomain.METAL: ["iron", "copper", "gold", "silver", "titanium", "bronze", "steel", "platinum", "mithril", "adamant"],
    MaterialDomain.CRYSTAL: ["ruby", "sapphire", "emerald", "diamond", "quartz", "opal", "topaz", "amethyst", "jade", "onyx"],
    MaterialDomain.ORGANIC: ["wood", "fiber", "vine", "leaf", "bark", "root", "moss", "mushroom", "petal", "seed"],
    MaterialDomain.ARCANE: ["ether", "void", "flux", "aura", "mana", "rune", "sigil", "glyph", "soul", "spirit"],
    MaterialDomain.TECH: ["circuit", "chip", "wire", "cell", "core", "module", "unit", "array", "grid", "matrix"],
}

# Generic prefixes: no semantic meaning, forces agent to use get_info()
DOMAIN_PREFIXES_GENERIC = {
    MaterialDomain.METAL: [f"m{i}" for i in range(10)],      # m0, m1, m2...
    MaterialDomain.CRYSTAL: [f"c{i}" for i in range(10)],    # c0, c1, c2...
    MaterialDomain.ORGANIC: [f"o{i}" for i in range(10)],    # o0, o1, o2...
    MaterialDomain.ARCANE: [f"a{i}" for i in range(10)],     # a0, a1, a2...
    MaterialDomain.TECH: [f"t{i}" for i in range(10)],       # t0, t1, t2...
}

# Default to generic for cleaner benchmarking
DOMAIN_PREFIXES = DOMAIN_PREFIXES_GENERIC


@dataclass
class SynthRecipe:
    """A synthetic crafting recipe."""
    result_item: str
    result_count: int
    ingredients: Dict[str, int]  # item_name -> count
    depth: int  # crafting depth of this item
    domain: MaterialDomain
    
    def to_minecraft_format(self) -> dict:
        """Convert to Minecraft-like JSON format."""
        # Create a shaped pattern that accommodates the ingredients
        # For simplicity, we'll use shapeless recipes
        return {
            "type": "minecraft:crafting_shapeless",
            "ingredients": [
                {"item": f"synth:{item}"} 
                for item, count in self.ingredients.items()
                for _ in range(count)
            ],
            "result": {
                "item": f"synth:{self.result_item}",
                "count": self.result_count
            },
            "synth_metadata": {
                "depth": self.depth,
                "domain": self.domain.value
            }
        }


@dataclass  
class SynthRecipeDatabase:
    """Database of synthetic recipes organized by depth tier."""
    
    # Tier definitions with (name_suffix, typical_ingredients_count, result_count_range)
    # Higher result counts at lower tiers to reduce the explosion of required base materials
    # 
    # Two naming modes:
    # - Semantic: ("raw", "refined", "processed"...) - human readable but creates LLM prior conflicts
    # - Generic: ("i0", "i1", "i2"...) - no semantic meaning, cleaner for benchmarking
    TIER_DEFINITIONS_SEMANTIC = {
        0: ("raw", 0, (1, 1)),           # Base materials - no recipe
        1: ("refined", 1, (2, 4)),       # 1 ingredient -> 2-4 refined
        2: ("processed", 2, (2, 3)),     # 2 ingredients -> 2-3 processed
        3: ("component", 2, (1, 2)),     # 2 ingredients -> 1-2 component
        4: ("part", 2, (1, 2)),          # 2 ingredients -> 1-2 part  
        5: ("module", 2, (1, 2)),        # 2 ingredients -> 1-2 module
        6: ("assembly", 2, (1, 1)),      # 2 ingredients -> 1 assembly
        7: ("system", 2, (1, 1)),        # 2-3 ingredients -> 1 system
        8: ("complex", 3, (1, 1)),       # 3 ingredients -> 1 complex
        9: ("advanced", 3, (1, 1)),      # 3 ingredients -> 1 advanced
        10: ("elite", 3, (1, 1)),        # 3 ingredients -> 1 elite
        11: ("ultimate", 3, (1, 1)),     # 3 ingredients -> 1 ultimate
        12: ("legendary", 4, (1, 1)),    # 4 ingredients -> 1 legendary
    }
    
    # Generic tier names: i0, i1, i2... (no semantic meaning)
    TIER_DEFINITIONS_GENERIC = {
        0: ("i0", 0, (1, 1)),
        1: ("i1", 1, (2, 4)),
        2: ("i2", 2, (2, 3)),
        3: ("i3", 2, (1, 2)),
        4: ("i4", 2, (1, 2)),
        5: ("i5", 2, (1, 2)),
        6: ("i6", 2, (1, 1)),
        7: ("i7", 2, (1, 1)),
        8: ("i8", 3, (1, 1)),
        9: ("i9", 3, (1, 1)),
        10: ("i10", 3, (1, 1)),
        11: ("i11", 3, (1, 1)),
        12: ("i12", 4, (1, 1)),
    }
    
    # Default to generic for cleaner benchmarking
    TIER_DEFINITIONS = TIER_DEFINITIONS_GENERIC
    
    recipes: Dict[str, List[SynthRecipe]] = field(default_factory=dict)
    base_items: Set[str] = field(default_factory=set)
    all_items: Set[str] = field(default_factory=set)
    item_depths: Dict[str, int] = field(default_factory=dict)
    items_by_tier: Dict[int, List[str]] = field(default_factory=lambda: {i: [] for i in range(13)})
    items_by_domain: Dict[MaterialDomain, Dict[int, List[str]]] = field(default_factory=dict)
    
    def __post_init__(self):
        for domain in MaterialDomain:
            self.items_by_domain[domain] = {i: [] for i in range(13)}
    
    def generate_all_recipes(self, seed: int = 42, items_per_domain_tier: int = 8):
        """Generate the complete recipe database."""
        random.seed(seed)
        
        # Generate items for each domain
        for domain in MaterialDomain:
            self._generate_domain_items(domain, items_per_domain_tier)
        
        # Generate cross-domain items (high tier items that combine domains)
        self._generate_cross_domain_items()
    
    def _generate_domain_items(self, domain: MaterialDomain, items_per_tier: int):
        """Generate items for a single domain."""
        prefixes = DOMAIN_PREFIXES[domain]
        
        # Tier 0: Base materials (no recipes)
        for i in range(items_per_tier):
            prefix = prefixes[i % len(prefixes)]
            item_name = f"{prefix}_ore" if domain in [MaterialDomain.METAL, MaterialDomain.CRYSTAL] else f"raw_{prefix}"
            
            self.base_items.add(item_name)
            self.all_items.add(item_name)
            self.item_depths[item_name] = 0
            self.items_by_tier[0].append(item_name)
            self.items_by_domain[domain][0].append(item_name)
        
        # Tiers 1-12: Craftable items
        for tier in range(1, 13):
            suffix, num_ingredients, result_count_range = self.TIER_DEFINITIONS[tier]
            num_items = max(1, items_per_tier - tier // 3)  # Fewer items at higher tiers
            
            for i in range(num_items):
                prefix = prefixes[i % len(prefixes)]
                item_name = f"{prefix}_{suffix}"
                
                # Ensure unique names by adding tier number for high tiers
                if tier >= 6:
                    item_name = f"{prefix}_{suffix}_{tier}"
                
                # Skip if already exists
                if item_name in self.all_items:
                    item_name = f"{item_name}_{i}"
                
                # Generate recipe
                recipe = self._generate_recipe_for_tier(
                    item_name, tier, domain, num_ingredients, result_count_range
                )
                
                if recipe:
                    self.recipes.setdefault(item_name, []).append(recipe)
                    self.all_items.add(item_name)
                    self.item_depths[item_name] = tier
                    self.items_by_tier[tier].append(item_name)
                    self.items_by_domain[domain][tier].append(item_name)
    
    def _generate_recipe_for_tier(
        self, 
        item_name: str, 
        tier: int, 
        domain: MaterialDomain,
        base_num_ingredients: int,
        result_count_range: Tuple[int, int]
    ) -> Optional[SynthRecipe]:
        """Generate a recipe for a specific tier item."""
        
        # Collect possible ingredients from lower tiers
        # Higher tier items should use ingredients from multiple lower tiers
        possible_ingredients = []
        
        # Primary ingredients from tier-1 and tier-2
        for t in range(max(0, tier - 2), tier):
            domain_items = self.items_by_domain[domain][t]
            if domain_items:
                possible_ingredients.extend([(item, tier - t) for item in domain_items])
        
        # Also allow some cross-tier ingredients for variety
        if tier >= 4:
            cross_tier = random.choice(range(max(0, tier - 4), tier - 2))
            if self.items_by_domain[domain][cross_tier]:
                item = random.choice(self.items_by_domain[domain][cross_tier])
                possible_ingredients.append((item, tier - cross_tier))
        
        if not possible_ingredients:
            return None
        
        # Select ingredients
        num_ingredients = base_num_ingredients + random.randint(0, min(2, tier // 3))
        num_ingredients = min(num_ingredients, len(possible_ingredients), 6)  # Cap at 6
        
        # Weight selection towards closer tiers
        weights = [1.0 / (dist + 1) for _, dist in possible_ingredients]
        
        selected = []
        available = list(range(len(possible_ingredients)))
        
        for _ in range(num_ingredients):
            if not available:
                break
            
            # Weighted random selection
            total_weight = sum(weights[i] for i in available)
            r = random.random() * total_weight
            cumsum = 0
            for idx in available:
                cumsum += weights[idx]
                if cumsum >= r:
                    selected.append(possible_ingredients[idx][0])
                    available.remove(idx)
                    break
        
        if not selected:
            return None
        
        # Create ingredient counts - keep them small to avoid exponential explosion
        # Lower tiers: 1-2 each, higher tiers: 1 each
        ingredients = {}
        for item in selected:
            max_count = 2 if tier <= 4 else 1
            ingredients[item] = random.randint(1, max_count)
        
        result_count = random.randint(*result_count_range)
        
        return SynthRecipe(
            result_item=item_name,
            result_count=result_count,
            ingredients=ingredients,
            depth=tier,
            domain=domain
        )
    
    def _generate_cross_domain_items(self):
        """Generate high-tier items that combine multiple domains."""
        
        # Generic names for cross-domain items (x = cross-domain)
        cross_domain_items = [
            # Tier 10 cross-domain
            ("x10_mc", 10, [MaterialDomain.METAL, MaterialDomain.CRYSTAL]),
            ("x10_ot", 10, [MaterialDomain.ORGANIC, MaterialDomain.TECH]),
            ("x10_am", 10, [MaterialDomain.ARCANE, MaterialDomain.METAL]),
            ("x10_ct", 10, [MaterialDomain.CRYSTAL, MaterialDomain.TECH]),
            
            # Tier 11 cross-domain
            ("x11_mct", 11, [MaterialDomain.METAL, MaterialDomain.CRYSTAL, MaterialDomain.TECH]),
            ("x11_oam", 11, [MaterialDomain.ORGANIC, MaterialDomain.ARCANE, MaterialDomain.METAL]),
            ("x11_act", 11, [MaterialDomain.ARCANE, MaterialDomain.CRYSTAL, MaterialDomain.TECH]),
            ("x11_tca", 11, [MaterialDomain.TECH, MaterialDomain.CRYSTAL, MaterialDomain.ARCANE]),
            
            # Tier 12 items (combine 4+ domains)
            ("x12_mcat", 12, [MaterialDomain.METAL, MaterialDomain.CRYSTAL, MaterialDomain.ARCANE, MaterialDomain.TECH]),
            ("x12_oacm", 12, [MaterialDomain.ORGANIC, MaterialDomain.ARCANE, MaterialDomain.CRYSTAL, MaterialDomain.METAL]),
            ("x12_tcoa", 12, [MaterialDomain.TECH, MaterialDomain.CRYSTAL, MaterialDomain.ORGANIC, MaterialDomain.ARCANE]),
            ("x12_all", 12, list(MaterialDomain)),  # All domains
        ]
        
        for item_name, tier, domains in cross_domain_items:
            ingredients = {}
            
            # Get high-tier ingredients from each domain
            for domain in domains:
                # Use items from tier-1 to tier-3
                for t in range(max(1, tier - 3), tier):
                    if self.items_by_domain[domain][t]:
                        ing = random.choice(self.items_by_domain[domain][t])
                        ingredients[ing] = random.randint(1, 2)
                        break
            
            if len(ingredients) >= 2:
                recipe = SynthRecipe(
                    result_item=item_name,
                    result_count=1,
                    ingredients=ingredients,
                    depth=tier,
                    domain=domains[0]  # Primary domain
                )
                
                self.recipes.setdefault(item_name, []).append(recipe)
                self.all_items.add(item_name)
                self.item_depths[item_name] = tier
                self.items_by_tier[tier].append(item_name)
    
    def get_items_by_depth_range(self, min_depth: int, max_depth: int) -> List[str]:
        """Get all items within a depth range."""
        items = []
        for depth in range(min_depth, max_depth + 1):
            items.extend(self.items_by_tier.get(depth, []))
        return items
    
    def get_crafting_depth(self, item: str) -> int:
        """Get the crafting depth of an item."""
        return self.item_depths.get(item, 0)
    
    def is_base_item(self, item: str) -> bool:
        """Check if an item is a base material."""
        return item in self.base_items
    
    def can_craft(self, item: str) -> bool:
        """Check if an item can be crafted."""
        return item in self.recipes
    
    def get_recipes_for_item(self, item: str) -> List[SynthRecipe]:
        """Get all recipes for an item."""
        return self.recipes.get(item, [])
    
    def save_to_directory(self, output_dir: Path):
        """Save all recipes to a directory as JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for item_name, recipes in self.recipes.items():
            for idx, recipe in enumerate(recipes):
                filename = f"{item_name}.json" if idx == 0 else f"{item_name}_{idx}.json"
                filepath = output_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(recipe.to_minecraft_format(), f, indent=2)
        
        # Also save metadata
        metadata = {
            "base_items": sorted(self.base_items),
            "all_items": sorted(self.all_items),
            "item_depths": self.item_depths,
            "items_by_tier": {k: v for k, v in self.items_by_tier.items()},
            "total_items": len(self.all_items),
            "total_recipes": sum(len(r) for r in self.recipes.values()),
            "max_depth": max(self.item_depths.values()) if self.item_depths else 0,
        }
        
        with open(output_dir / "_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {metadata['total_recipes']} recipes for {metadata['total_items']} items")
        print(f"Max crafting depth: {metadata['max_depth']}")
        print(f"Items per tier: {[(k, len(v)) for k, v in sorted(self.items_by_tier.items())]}")


def set_naming_mode(semantic: bool = False):
    """
    Set naming mode for recipe generation.
    
    Args:
        semantic: If True, use human-readable names (iron_refined, etc.)
                  If False (default), use generic names (m0_i1, etc.) for clean benchmarking
    """
    global DOMAIN_PREFIXES
    if semantic:
        SynthRecipeDatabase.TIER_DEFINITIONS = SynthRecipeDatabase.TIER_DEFINITIONS_SEMANTIC
        globals()['DOMAIN_PREFIXES'] = DOMAIN_PREFIXES_SEMANTIC
    else:
        SynthRecipeDatabase.TIER_DEFINITIONS = SynthRecipeDatabase.TIER_DEFINITIONS_GENERIC
        globals()['DOMAIN_PREFIXES'] = DOMAIN_PREFIXES_GENERIC


def generate_synth_recipes(output_dir: Path, seed: int = 42, items_per_domain_tier: int = 8, semantic_names: bool = False):
    """Generate and save synthetic recipes.
    
    Args:
        output_dir: Directory to save recipe JSON files
        seed: Random seed for reproducibility
        items_per_domain_tier: Number of items per domain per tier
        semantic_names: If True, use human-readable names. If False (default), use generic names.
    """
    set_naming_mode(semantic=semantic_names)
    db = SynthRecipeDatabase()
    db.generate_all_recipes(seed=seed, items_per_domain_tier=items_per_domain_tier)
    db.save_to_directory(output_dir)
    return db


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic recipes for TextCraft-Synth")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "synth_recipes",
                        help="Output directory for recipe JSON files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--items-per-tier", type=int, default=8, 
                        help="Number of items per domain per tier")
    parser.add_argument("--semantic-names", action="store_true",
                        help="Use semantic names (iron_refined) instead of generic (m0_i1). "
                             "Generic names are recommended for benchmarking to avoid LLM prior conflicts.")
    
    args = parser.parse_args()
    
    print(f"Using {'semantic' if args.semantic_names else 'generic'} naming mode")
    generate_synth_recipes(args.output_dir, args.seed, args.items_per_tier, args.semantic_names)
