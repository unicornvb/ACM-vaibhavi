import numpy as np
import pandas as pd
import re

# Enhanced keyword list for competitive programming
KEYWORDS = [
    # Graph algorithms
    "graph", "dfs", "bfs", "dijkstra", "bellman", "floyd", "warshall",
    "mst", "kruskal", "prim", "shortest path", "topological", "scc",
    "bipartite", "matching", "flow", "max flow", "min cut", "lca",
    "tarjan", "kosaraju", "articulation", "bridge",
    
    # Dynamic programming
    "dp", "dynamic programming", "memoization", "memo", "knapsack",
    "lis", "lcs", "subsequence", "subarray", "bitmask", "digit dp",
    "tree dp", "interval dp", "state machine",
    
    # Data structures
    "segment tree", "fenwick", "bit", "binary indexed", "sparse table",
    "trie", "suffix", "dsu", "union find", "disjoint set", "heap",
    "priority queue", "stack", "queue", "deque", "monotonic",
    "persistent", "hld", "heavy light",
    
    # String algorithms
    "string", "kmp", "z algorithm", "manacher", "rabin karp",
    "aho corasick", "suffix array", "suffix tree", "hashing",
    "palindrome", "pattern matching",
    
    # Mathematics
    "math", "number theory", "modulo", "mod", "gcd", "lcm", "prime",
    "sieve", "factorization", "euler", "totient", "crt", "chinese remainder",
    "combinatorics", "permutation", "combination", "probability",
    "matrix", "exponentiation", "fibonacci", "linear algebra",
    
    # Geometry
    "geometry", "convex hull", "line segment", "polygon", "distance",
    "intersection", "computational geometry",
    
    # Sorting & searching
    "binary search", "ternary search", "sort", "merge sort", "quick sort",
    "radix sort", "counting sort",
    
    # Greedy
    "greedy", "interval scheduling", "activity selection",
    
    # Game theory
    "game theory", "nim", "grundy", "sprague",
    
    # General techniques
    "recursion", "backtrack", "divide conquer", "two pointer",
    "sliding window", "prefix sum", "difference array"
]

# Grouped keyword categories
KEYWORD_GROUPS = {
    "graph": [
        "graph", "dfs", "bfs", "dijkstra", "bellman", "floyd", "warshall",
        "mst", "kruskal", "prim", "shortest path", "topological", "scc",
        "bipartite", "matching", "flow", "max flow", "min cut", "lca",
        "tarjan", "kosaraju", "articulation", "bridge", "dsu", "union find"
    ],
    "dp": [
        "dp", "dynamic programming", "memoization", "memo", "knapsack",
        "lis", "lcs", "subsequence", "bitmask", "digit dp", "tree dp",
        "interval dp", "state machine"
    ],
    "advanced_ds": [
        "segment tree", "fenwick", "bit", "binary indexed", "sparse table",
        "trie", "suffix", "persistent", "hld", "heavy light"
    ],
    "math": [
        "math", "number theory", "modulo", "mod", "gcd", "lcm", "prime",
        "sieve", "factorization", "euler", "totient", "crt", "chinese remainder",
        "combinatorics", "permutation", "combination", "matrix", "exponentiation"
    ],
    "string": [
        "string", "kmp", "z algorithm", "manacher", "rabin karp",
        "aho corasick", "suffix array", "suffix tree", "hashing", "palindrome"
    ],
    "geometry": [
        "geometry", "convex hull", "line segment", "polygon", "distance",
        "intersection", "computational geometry"
    ],
    "greedy": [
        "greedy", "interval scheduling", "activity selection"
    ]
}

_STOPWORDS = {
    "the", "and", "a", "an", "in", "on", "of", "to", "for", "with", "is", "are",
    "be", "by", "this", "that", "we", "you", "it", "as", "from", "at", "or",
    "if", "else", "while", "return", "will", "can", "all", "each", "every"
}


def _safe_col(name: str) -> str:
    return re.sub(r"\W+", "_", name.strip().lower())


def _extract_max_n(text: str):
    """Extract maximum constraint value from text."""
    patterns = [
        r"\b1\s*<=\s*n\s*<=\s*(\d+)",
        r"\b1\s*≤\s*n\s*≤\s*(\d+)",
        r"\bn\s*(?:<=|≤|is at most|up to)\s*([0-9]{2,})",
        r"\b([0-9]{2,})\s*(?:<=|≤)\s*n\b",
        r"\bn\s*(?:<=|≤|up to|is at most)\s*10\^(\d+)"
    ]
    
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            try:
                val = int(m.group(1))
                if pat.endswith(r"10\^(\d+)"):
                    return 10 ** val
                return val
            except Exception:
                continue
    
    # Fallback: look for "n" and nearby numbers
    for m in re.finditer(r"\bn\b(.{0,30})", text):
        segment = m.group(1)
        nums = re.findall(r"(\d{2,})", segment)
        if nums:
            return int(nums[0])
    
    return 0


def _has_big_o(text: str):
    """Check if big-O notation appears."""
    return 1 if re.search(r"\bO\([^)]+\)", text) else 0


def _avg_word_len_and_ttr(text: str):
    """Calculate average word length and type-token ratio."""
    toks = re.findall(r"[A-Za-z0-9_]+", text)
    if not toks:
        return 0.0, 0.0, 0
    
    avg_len = np.mean([len(t) for t in toks])
    uniq = len(set(toks))
    ttr = uniq / len(toks)
    return avg_len, ttr, len(toks)


def _stopword_ratio(s: str):
    """Calculate ratio of stopwords in text."""
    toks = re.findall(r"[A-Za-z0-9_]+", s.lower())
    if not toks:
        return 0.0
    stop = sum(1 for t in toks if t in _STOPWORDS)
    return stop / len(toks)


def add_meta_features(df):
    """Add comprehensive meta-features for competitive programming problems."""
    text = df.get("combined_text", pd.Series([""] * len(df))).fillna("").astype(str)

    # Length and structure
    df["text_len"] = text.str.len()
    df["text_len_log"] = np.log1p(df["text_len"])
    df["num_lines_est"] = text.str.count(r"\n") + 1

    # Math and constraints
    df["num_math_symbols"] = text.str.count(r"[=<>+\-*/%^]")
    df["num_constraints"] = text.str.count(
        r"constraint|limit|≤|≥|<=|>=|time limit|memory limit", flags=re.I
    )

    # Examples and I/O
    df["num_examples"] = text.str.count(r"\bexample\b", flags=re.I)
    df["sample_io_count"] = text.str.count(
        r"sample input|sample output|example input|example output", flags=re.I
    )
    df["num_sample_pairs"] = df["sample_io_count"]

    # Numeric statistics
    nums_series = text.str.findall(r"-?\d+")
    df["num_numerics"] = nums_series.apply(len)
    
    def safe_max_log(lst):
        try:
            vals = [abs(int(x)) for x in lst]
            return float(np.log1p(max(vals))) if vals else 0.0
        except:
            return 0.0
    
    def safe_mean_log(lst):
        try:
            vals = [abs(int(x)) for x in lst]
            return float(np.mean([np.log1p(v) for v in vals])) if vals else 0.0
        except:
            return 0.0
    
    df["max_number_log"] = nums_series.apply(safe_max_log)
    df["mean_number_log"] = nums_series.apply(safe_mean_log)

    # Estimated input size
    df["estimated_max_n"] = text.apply(_extract_max_n)

    # Complexity hints
    df["has_big_o"] = text.apply(_has_big_o)

    # Code detection
    df["code_like_lines"] = text.str.count(
        r"^\s*(#|def\s+|\w+\s*:=|for\s+|while\s+|if\s+).*$", flags=re.I | re.M
    )

    # Token statistics
    single_letter_tokens = text.str.findall(r"\b[a-zA-Z]\b")
    df["single_letter_vars"] = single_letter_tokens.apply(len)
    
    avg_word_len_list = []
    ttr_list = []
    token_count_list = []
    for s in text.tolist():
        avg_len, ttr, token_count = _avg_word_len_and_ttr(s)
        avg_word_len_list.append(avg_len)
        ttr_list.append(ttr)
        token_count_list.append(token_count)
    
    df["avg_word_len"] = avg_word_len_list
    df["type_token_ratio"] = ttr_list
    df["token_count"] = token_count_list

    # Linguistic features
    df["stopword_ratio"] = text.apply(_stopword_ratio)
    df["punctuation_count"] = text.str.count(r"[.,:;(){}\[\]]")
    df["question_mark"] = text.str.contains(r"\?").astype(int)
    df["has_directive_find"] = text.str.contains(
        r"\b(find|determine|compute|output|construct|print)\b", flags=re.I
    ).astype(int)

    # Keyword flags
    for kw in KEYWORDS:
        col = f"has_{_safe_col(kw)}"
        pattern = rf"\b{re.escape(kw)}\b" if " " not in kw else re.escape(kw)
        df[col] = text.str.contains(pattern, case=False, regex=True).astype(int)

    # Grouped keyword counts
    for group, kws in KEYWORD_GROUPS.items():
        df[f"{group}_kw_count"] = sum(
            text.str.count(
                (rf"\b{re.escape(kw)}\b" if " " not in kw else re.escape(kw)),
                flags=re.I
            ) for kw in kws
        )

    # Aggregate signals
    has_cols = [c for c in df.columns if c.startswith("has_")]
    df["num_keywords"] = df[has_cols].sum(axis=1)

    df.fillna(0, inplace=True)
    return df