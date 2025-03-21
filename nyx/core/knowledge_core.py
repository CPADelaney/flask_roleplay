# nyx/core/knowledge_core.py

import asyncio
import json
import logging
import os
import math
import networkx as nx
import numpy as np

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Try to import a real embedding model from sentence-transformers or transformers
# If you prefer a different approach (like OpenAI embeddings), adapt the code below.

try:
    from sentence_transformers import SentenceTransformer
    _EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    _EMBEDDING_MODEL_AVAILABLE = False

logger = logging.getLogger(__name__)


###############################################################################
#                           KNOWLEDGE NODE & RELATION
###############################################################################

class KnowledgeNode:
    """
    Represents a node in the knowledge graph, storing a piece of knowledge.
    """

    def __init__(self,
                 id: str,
                 type: str,
                 content: Dict[str, Any],
                 source: str,
                 confidence: float = 0.5,
                 timestamp: Optional[datetime] = None):
        """
        :param id: Unique ID for the node.
        :param type: The type/category (e.g. 'concept', 'fact', 'rule', 'hypothesis').
        :param content: Arbitrary dictionary storing the knowledge data.
        :param source: Which system or user provided this knowledge.
        :param confidence: Float in [0.0..1.0] indicating how reliable or certain this knowledge is.
        :param timestamp: Creation timestamp. Defaults to now if None.
        """
        self.id = id
        self.type = type
        self.content = content
        self.source = source
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now()
        self.last_accessed = self.timestamp
        self.access_count = 0

        # Track nodes that conflict or support this node
        self.conflicting_nodes: List[str] = []
        self.supporting_nodes: List[str] = []

        # Arbitrary metadata (e.g., merged_from info, domain tags, etc.)
        self.metadata: Dict[str, Any] = {}

    def update(self,
               new_content: Dict[str, Any],
               new_confidence: Optional[float] = None,
               source: Optional[str] = None) -> None:
        """
        Merge new content into the node’s existing content, optionally adjusting confidence and source.

        :param new_content: Additional or overriding fields for the node’s content.
        :param new_confidence: If provided, set the node’s confidence to this new value.
        :param source: If provided, update the node’s source.
        """
        self.content.update(new_content)
        if new_confidence is not None:
            self.confidence = new_confidence
        if source:
            self.source = source
        self.last_accessed = datetime.now()

    def access(self) -> None:
        """
        Record an “access” operation on the node, for usage statistics or confidence decay logic.
        """
        self.last_accessed = datetime.now()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this node to a dictionary for JSON serialization or logging.

        :return: Dictionary representation.
        """
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "conflicting_nodes": self.conflicting_nodes,
            "supporting_nodes": self.supporting_nodes,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """
        Recreate a KnowledgeNode from a dictionary (e.g., loading from JSON).

        :param data: Dictionary with node fields.
        :return: KnowledgeNode instance.
        """
        node = cls(
            id=data["id"],
            type=data["type"],
            content=data["content"],
            source=data["source"],
            confidence=data["confidence"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        node.last_accessed = datetime.fromisoformat(data["last_accessed"])
        node.access_count = data["access_count"]
        node.conflicting_nodes = data["conflicting_nodes"]
        node.supporting_nodes = data["supporting_nodes"]
        node.metadata = data["metadata"]
        return node


class KnowledgeRelation:
    """
    Represents a directed relation (edge) between two knowledge nodes in the graph.
    """

    def __init__(self,
                 source_id: str,
                 target_id: str,
                 type: str,
                 weight: float = 1.0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        :param source_id: ID of the node that is the source of the relation (directed edge).
        :param target_id: ID of the node that is the target of the relation (directed edge).
        :param type: The type/category of the relation (e.g. 'supports', 'contradicts', 'similar_to').
        :param weight: Float representing strength or importance of the relation.
        :param metadata: Arbitrary dictionary storing extra info about the relation.
        """
        self.source_id = source_id
        self.target_id = target_id
        self.type = type
        self.weight = weight
        self.timestamp = datetime.now()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this relation to a dictionary for JSON serialization or logging.

        :return: Dictionary representation.
        """
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "weight": self.weight,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeRelation':
        """
        Recreate a KnowledgeRelation from a dictionary (e.g., loading from JSON).

        :param data: Dictionary with relation fields.
        :return: KnowledgeRelation instance.
        """
        relation = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=data["type"],
            weight=data["weight"],
            metadata=data["metadata"]
        )
        relation.timestamp = datetime.fromisoformat(data["timestamp"])
        return relation


###############################################################################
#                       REASONING SYSTEM (SIMPLE STUB)
###############################################################################

class SimpleReasoningSystem:
    """
    A minimal example of a “reasoning” component that attempts to resolve contradictions.
    Replace/extend with your custom logic or external reasoner calls.
    """

    async def resolve_contradiction(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> Dict[str, Any]:
        """
        A naive contradiction-resolution approach:
        - Always pick the node with higher confidence as “preferred”.
        - Return the other node with reduced confidence.

        :param node1: Node dict (from node.to_dict()).
        :param node2: Node dict.
        :return: A dictionary indicating resolution status and which node is preferred.
        """
        # Example: just pick the higher confidence node as “preferred”
        confidence1 = node1["confidence"]
        confidence2 = node2["confidence"]

        if abs(confidence1 - confidence2) < 0.05:
            # If they are almost the same, say we “didn’t” resolve it fully
            return {
                "resolved": False,
                "preferred_node": None
            }
        else:
            if confidence1 >= confidence2:
                return {
                    "resolved": True,
                    "preferred_node": node1["id"]
                }
            else:
                return {
                    "resolved": True,
                    "preferred_node": node2["id"]
                }


###############################################################################
#                           KNOWLEDGE CORE (MAIN CLASS)
###############################################################################

class KnowledgeCore:
    """
    Main system that integrates knowledge across different components, storing data
    in a directed graph and using embeddings for similarity detection, plus a
    reasoning system for contradiction handling.
    """

    def __init__(self,
                 knowledge_store_file: str = "knowledge_store.json"):
        """
        :param knowledge_store_file: Path to a JSON file used for persisting and loading knowledge at startup.
        """
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.node_embeddings: Dict[str, np.ndarray] = {}

        self.integration_stats = {
            "nodes_added": 0,
            "relations_added": 0,
            "conflicts_resolved": 0,
            "knowledge_queries": 0,
            "integration_cycles": 0,
            "node_count": 0,
            "relation_count": 0,
            "node_types": {},
            "relation_types": {}
        }

        # Default config: you can customize or load from external config
        self.config = {
            "conflict_threshold": 0.7,
            "support_threshold": 0.7,
            "similarity_threshold": 0.8,  # for merges
            "decay_rate": 0.01,
            "integration_cycle_interval": 10,  # in minutes
            "enable_embeddings": True,
            "max_node_age_days": 30,
            "pruning_confidence_threshold": 0.3
        }

        # Timestamps and counters
        self.last_integration_cycle = datetime.now()
        self.next_node_id = 1

        # Caches
        self.integration_cache: Dict[str, Any] = {}
        self.query_cache: Dict[str, Any] = {}

        # System references (can be set via initialize())
        self.memory_system = None
        self.reasoning_system = None  # We'll use the SimpleReasoningSystem by default
        self.knowledge_store_file = knowledge_store_file

        # If real embeddings are enabled, try to load the model
        if self.config["enable_embeddings"] and _EMBEDDING_MODEL_AVAILABLE:
            try:
                # Use a standard SentenceTransformer model—change if you want a smaller/larger one
                self._embedding_model = SentenceTransformer('all-mpnet-base-v2')
                logger.info("Loaded embedding model (SentenceTransformer).")
            except Exception as e:
                logger.warning(f"Error loading embedding model: {e}")
                self._embedding_model = None
        else:
            self._embedding_model = None

    async def initialize(self,
                         system_references: Dict[str, Any]) -> None:
        """
        Initialize the knowledge integration system, load references to other subsystems,
        and attempt to load existing knowledge from disk.

        :param system_references: A dictionary that may contain 'memory_system' or 'reasoning_system' references.
        """
        if "memory_system" in system_references:
            self.memory_system = system_references["memory_system"]
        if "reasoning_system" in system_references:
            self.reasoning_system = system_references["reasoning_system"]
        else:
            # Provide a default simple reasoning system if none is provided
            self.reasoning_system = SimpleReasoningSystem()

        await self._load_knowledge()
        logger.info("Knowledge Core initialized.")

    async def _load_knowledge(self) -> None:
        """
        Load existing nodes/relations from a JSON file (if it exists) and populate
        the graph with them.
        """
        if not os.path.exists(self.knowledge_store_file):
            logger.info("No existing knowledge store found; starting fresh.")
            return

        try:
            with open(self.knowledge_store_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load nodes
            for node_data in data.get("nodes", []):
                node = KnowledgeNode.from_dict(node_data)
                self.nodes[node.id] = node
                self.graph.add_node(node.id, **node.to_dict())
                # Keep track of the highest numeric suffix for next_node_id
                try:
                    numeric_id = int(node.id.split("_")[-1])
                    if numeric_id >= self.next_node_id:
                        self.next_node_id = numeric_id + 1
                except:
                    pass

            # Load relations
            for rel_data in data.get("relations", []):
                rel = KnowledgeRelation.from_dict(rel_data)
                self.graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    **rel.to_dict()
                )
                # Also update supporting/conflicting references
                if rel.type == "supports":
                    self.nodes[rel.source_id].supporting_nodes.append(rel.target_id)
                elif rel.type == "contradicts":
                    self.nodes[rel.source_id].conflicting_nodes.append(rel.target_id)

            # Optionally rebuild embeddings
            if self.config["enable_embeddings"] and self._embedding_model:
                await self._rebuild_embeddings()

            logger.info(f"Knowledge store loaded with {len(self.nodes)} nodes and {self.graph.number_of_edges()} edges.")
        except Exception as e:
            logger.error(f"Error loading knowledge: {str(e)}")

    async def _rebuild_embeddings(self) -> None:
        """
        Rebuild embeddings for all nodes from scratch, e.g. after loading from disk.
        """
        for node in self.nodes.values():
            await self._update_node_embedding(node)

    async def save_knowledge(self) -> None:
        """
        Save all nodes/relations to a JSON file, to persist knowledge across restarts.
        """
        try:
            # Build list of node dicts
            nodes_data = [node.to_dict() for node in self.nodes.values()]

            # Build list of relation dicts from graph edges
            relations_data = []
            for source, target, data in self.graph.edges(data=True):
                # We only store the direct fields from KnowledgeRelation
                # that match the schema in KnowledgeRelation.to_dict().
                # Some fields like 'source_id' / 'target_id' might be duplicated,
                # so we just re-serialize them carefully.
                relation_dict = {
                    "source_id": source,
                    "target_id": target,
                    "type": data.get("type"),
                    "weight": data.get("weight", 1.0),
                    "timestamp": data.get("timestamp"),
                    "metadata": data.get("metadata", {})
                }
                # timestamp might be isoformat string in 'data'
                # be sure to handle it similarly to from_dict
                if isinstance(relation_dict["timestamp"], datetime):
                    relation_dict["timestamp"] = relation_dict["timestamp"].isoformat()
                relations_data.append(relation_dict)

            store_data = {
                "nodes": nodes_data,
                "relations": relations_data
            }

            with open(self.knowledge_store_file, "w", encoding="utf-8") as f:
                json.dump(store_data, f, indent=2)
            logger.info("Knowledge store saved successfully.")
        except Exception as e:
            logger.error(f"Error saving knowledge: {str(e)}")

    async def add_knowledge(self,
                            type: str,
                            content: Dict[str, Any],
                            source: str,
                            confidence: float = 0.5,
                            relations: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Add a new piece of knowledge (node) to the system.

        :param type: The node’s knowledge type (concept, fact, rule, etc.)
        :param content: Main data for the knowledge node.
        :param source: Where this knowledge came from (system name, user ID, etc.)
        :param confidence: Float in [0..1] for how reliable the knowledge is.
        :param relations: Optional list of dicts describing relations to other nodes, e.g.:
                          [{"target_id": "node_3", "type": "supports", "weight": 0.8}, ...]
        :return: The ID of the new or integrated node.
        """
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1

        node = KnowledgeNode(
            id=node_id,
            type=type,
            content=content,
            source=source,
            confidence=confidence
        )

        # Check if there's an existing node that's very similar
        similar_nodes = await self._find_similar_nodes(node)

        if similar_nodes:
            most_similar_id, similarity_val = similar_nodes[0]
            if similarity_val > self.config["similarity_threshold"]:
                # Merge into the existing node
                return await self._integrate_with_existing(node, most_similar_id, similarity_val)

        # Otherwise, create a brand-new node
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.to_dict())

        # Generate or update embedding if enabled
        if self.config["enable_embeddings"] and self._embedding_model:
            await self._add_node_embedding(node)

        # Add new relations if provided
        if relations:
            for r in relations:
                await self.add_relation(
                    source_id=node_id,
                    target_id=r["target_id"],
                    type=r["type"],
                    weight=r.get("weight", 1.0),
                    metadata=r.get("metadata", {})
                )

        # Optionally add “similar_to” edges for moderately similar nodes
        for sim_id, sim_val in similar_nodes:
            if sim_val > self.config["similarity_threshold"] * 0.6:
                await self.add_relation(
                    source_id=node_id,
                    target_id=sim_id,
                    type="similar_to",
                    weight=sim_val,
                    metadata={"auto_generated": True}
                )

        self.integration_stats["nodes_added"] += 1
        await self._check_integration_cycle()

        return node_id

    async def add_relation(self,
                           source_id: str,
                           target_id: str,
                           type: str,
                           weight: float = 1.0,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Connect two existing knowledge nodes with a directed relation (edge).

        :param source_id: ID of the source node
        :param target_id: ID of the target node
        :param type: e.g., "supports", "contradicts", "specializes", "similar_to"
        :param weight: Strength of the relation
        :param metadata: Additional fields to store about the relation
        :return: True if successful, False if either node doesn’t exist
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot add relation: node {source_id} or {target_id} does not exist.")
            return False

        rel = KnowledgeRelation(
            source_id=source_id,
            target_id=target_id,
            type=type,
            weight=weight,
            metadata=metadata or {}
        )

        self.graph.add_edge(source_id, target_id, **rel.to_dict())

        # Also update supporting/conflicting references in the source node
        if type == "supports":
            if target_id not in self.nodes[source_id].supporting_nodes:
                self.nodes[source_id].supporting_nodes.append(target_id)
        elif type == "contradicts":
            if target_id not in self.nodes[source_id].conflicting_nodes:
                self.nodes[source_id].conflicting_nodes.append(target_id)
            await self._handle_contradiction(source_id, target_id)

        self.integration_stats["relations_added"] += 1
        return True

    async def _integrate_with_existing(self,
                                       new_node: KnowledgeNode,
                                       existing_id: str,
                                       similarity: float) -> str:
        """
        Merge the content of a new node into an existing node if they're highly similar.

        :param new_node: The newly proposed KnowledgeNode
        :param existing_id: ID of the existing node
        :param similarity: A measure of how similar they are
        :return: The existing node’s ID (which now contains updated content)
        """
        existing_node = self.nodes[existing_id]
        existing_node.access()

        # Weighted combination of content and confidence
        if new_node.confidence > existing_node.confidence:
            # new_node is more reliable, use it as the basis
            updated_content = new_node.content.copy()
            # keep old content that’s missing in new
            for key, val in existing_node.content.items():
                if key not in updated_content:
                    updated_content[key] = val

            updated_conf = (new_node.confidence * 0.7 +
                            existing_node.confidence * 0.3)
            existing_node.update(updated_content, updated_conf,
                                 f"{existing_node.source}+{new_node.source}")
        else:
            # keep existing node as basis
            for key, val in new_node.content.items():
                if key not in existing_node.content:
                    existing_node.content[key] = val

            updated_conf = (existing_node.confidence * 0.7 +
                            new_node.confidence * 0.3)
            existing_node.confidence = updated_conf
            existing_node.last_accessed = datetime.now()

        # Update node data in the graph
        self.graph.nodes[existing_id].update(existing_node.to_dict())

        # Update embeddings if needed
        if self.config["enable_embeddings"] and self._embedding_model:
            await self._update_node_embedding(existing_node)

        logger.debug(f"Integrated node {new_node.id} into existing node {existing_id}.")
        return existing_id

    async def _handle_contradiction(self, node1_id: str, node2_id: str) -> None:
        """
        Called when two nodes have a 'contradicts' relation. Possibly attempt resolution via reasoning system.
        """
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]

        # If both are high confidence, we have a real conflict
        if node1.confidence > 0.7 and node2.confidence > 0.7:
            logger.info(f"Detected high-confidence conflict between {node1_id} and {node2_id}")
            if self.reasoning_system:
                try:
                    resolution = await self.reasoning_system.resolve_contradiction(node1.to_dict(),
                                                                                    node2.to_dict())
                    if resolution["resolved"]:
                        preferred = resolution["preferred_node"]
                        if preferred == node1_id:
                            node2.confidence *= 0.7
                        else:
                            node1.confidence *= 0.7

                        self.integration_stats["conflicts_resolved"] += 1
                        logger.info(f"Resolved conflict. Preferred node: {preferred}")
                except Exception as e:
                    logger.error(f"Error resolving contradiction: {str(e)}")

        # If there's a big confidence gap, just reduce the lower
        elif abs(node1.confidence - node2.confidence) > 0.3:
            if node1.confidence < node2.confidence:
                node1.confidence *= 0.9
            else:
                node2.confidence *= 0.9
            logger.debug(f"Adjusted confidence in conflict: {node1_id} vs {node2_id}.")

    async def _find_similar_nodes(self, node: KnowledgeNode) -> List[Tuple[str, float]]:
        """
        Check existing nodes for similarity to the incoming node.

        :return: List of (node_id, similarity_score) sorted descending by score.
        """
        if self.config["enable_embeddings"] and self._embedding_model:
            # Use real embeddings
            embedding = await self._generate_embedding(node)
            similar_list: List[Tuple[str, float]] = []
            for node_id, emb in self.node_embeddings.items():
                sim = self._calculate_embedding_similarity(embedding, emb)
                if sim > (self.config["similarity_threshold"] * 0.5):
                    similar_list.append((node_id, sim))
            similar_list.sort(key=lambda x: x[1], reverse=True)
            return similar_list
        else:
            # Fallback: do naive content-based matching
            results = []
            for node_id, existing_node in self.nodes.items():
                if existing_node.type == node.type:
                    content_sim = self._calculate_content_similarity(node.content, existing_node.content)
                    if content_sim > (self.config["similarity_threshold"] * 0.5):
                        results.append((node_id, content_sim))
            results.sort(key=lambda x: x[1], reverse=True)
            return results

    async def _add_node_embedding(self, node: KnowledgeNode) -> None:
        """
        Create and store an embedding for a newly added node.
        """
        emb = await self._generate_embedding(node)
        self.node_embeddings[node.id] = emb

    async def _update_node_embedding(self, node: KnowledgeNode) -> None:
        """
        Recompute embeddings for a node whose content changed.
        """
        emb = await self._generate_embedding(node)
        self.node_embeddings[node.id] = emb

    async def _generate_embedding(self, node: KnowledgeNode) -> np.ndarray:
        """
        Convert the node’s content to an embedding using a real (or fallback) model.
        """
        # If a real sentence-transformers model is loaded, do that:
        if self._embedding_model:
            # Convert content to a “document” string
            content_str = json.dumps(node.content, ensure_ascii=False)
            embedding = self._embedding_model.encode([content_str], show_progress_bar=False)
            # sentence-transformers returns a 2D array [ [vector], ]
            # We'll just return the 1D vector
            return embedding[0]

        # Otherwise fallback: naive numeric approach
        fallback_vector = np.zeros(128, dtype=np.float32)
        content_str = json.dumps(node.content)
        for i, c in enumerate(content_str):
            idx = i % 128
            fallback_vector[idx] += (ord(c) % 96) / 100.0
        norm = np.linalg.norm(fallback_vector)
        if norm > 0:
            fallback_vector = fallback_vector / norm
        return fallback_vector

    def _calculate_embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Standard cosine similarity between two vectors.
        """
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def _calculate_content_similarity(self, c1: Dict[str, Any], c2: Dict[str, Any]) -> float:
        """
        A naive Jaccard/overlap approach for dictionary content if no embeddings are available.
        """
        all_keys = set(c1.keys()).union(set(c2.keys()))
        if not all_keys:
            return 0.0
        matching_keys = set(c1.keys()).intersection(set(c2.keys()))
        matching_values = 0
        for k in matching_keys:
            if c1[k] == c2[k]:
                matching_values += 1
        key_similarity = len(matching_keys) / len(all_keys)
        value_similarity = matching_values / len(matching_keys) if matching_keys else 0.0
        # Weighted combination
        return 0.7 * key_similarity + 0.3 * value_similarity

    async def _check_integration_cycle(self) -> None:
        """
        See if we should run an integration cycle now (based on time).
        """
        now = datetime.now()
        elapsed = (now - self.last_integration_cycle).total_seconds()
        # If enough time has passed, do it
        if elapsed > self.config["integration_cycle_interval"] * 60:
            await self._run_integration_cycle()

    async def _run_integration_cycle(self) -> None:
        """
        Perform housekeeping tasks: pruning, conflict resolution, merging, inference, etc.
        """
        logger.info("Running knowledge integration cycle.")
        self.last_integration_cycle = datetime.now()
        self.integration_stats["integration_cycles"] += 1

        # 1. prune old or low-confidence nodes
        await self._prune_nodes()

        # 2. resolve conflicts
        await self._resolve_conflicts()

        # 3. consolidate similar nodes
        await self._consolidate_similar_nodes()

        # 4. infer new relations
        await self._infer_relations()

        # 5. apply confidence decay
        self._apply_confidence_decay()

        # 6. update stats
        await self._update_knowledge_stats()

        logger.info("Integration cycle completed.")
        # Optionally save knowledge
        await self.save_knowledge()

    async def _prune_nodes(self) -> None:
        """
        Remove old or low-confidence nodes that are rarely accessed.
        """
        now = datetime.now()
        to_prune = []
        for node_id, node in self.nodes.items():
            age_days = (now - node.timestamp).total_seconds() / (24 * 3600)
            if (age_days > self.config["max_node_age_days"] and
                    node.confidence < self.config["pruning_confidence_threshold"] and
                    node.access_count < 3):
                to_prune.append(node_id)

        for nid in to_prune:
            await self._remove_node(nid)

        if to_prune:
            logger.info(f"Pruned {len(to_prune)} node(s).")

    async def _remove_node(self, node_id: str) -> None:
        """
        Delete a node (and all edges from/to it) from memory.
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
        if node_id in self.graph:
            self.graph.remove_node(node_id)
        if node_id in self.node_embeddings:
            del self.node_embeddings[node_id]

        # Remove from any supporting/conflicting references
        for other_node in self.nodes.values():
            if node_id in other_node.supporting_nodes:
                other_node.supporting_nodes.remove(node_id)
            if node_id in other_node.conflicting_nodes:
                other_node.conflicting_nodes.remove(node_id)

    async def _resolve_conflicts(self) -> None:
        """
        Look for edges that have type=contradicts, handle them.
        """
        contradictions = []
        for s, t, data in self.graph.edges(data=True):
            if data.get("type") == "contradicts":
                contradictions.append((s, t))

        for (nid1, nid2) in contradictions:
            if nid1 in self.nodes and nid2 in self.nodes:
                await self._handle_contradiction(nid1, nid2)

    async def _consolidate_similar_nodes(self) -> None:
        """
        Merge nodes that are extremely similar to reduce duplication.
        """
        if not self.config["enable_embeddings"] or not self._embedding_model:
            # If embeddings are disabled or no model is loaded, skip
            return

        node_ids = list(self.nodes.keys())
        if len(node_ids) < 2:
            return

        to_merge = []
        # For performance, we limit to 1000 comparisons
        max_comparisons = min(1000, len(node_ids) * (len(node_ids) - 1) // 2)
        comparisons = 0

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                comparisons += 1
                if comparisons > max_comparisons:
                    break

                id1 = node_ids[i]
                id2 = node_ids[j]
                if id1 not in self.node_embeddings or id2 not in self.node_embeddings:
                    continue

                # Skip if they have a direct "contradicts" edge
                if id2 in self.nodes[id1].conflicting_nodes:
                    continue

                emb1 = self.node_embeddings[id1]
                emb2 = self.node_embeddings[id2]
                sim = self._calculate_embedding_similarity(emb1, emb2)
                if sim > self.config["similarity_threshold"]:
                    to_merge.append((id1, id2, sim))

        # Sort merges by descending similarity
        to_merge.sort(key=lambda x: x[2], reverse=True)
        merged_nodes = set()
        merges_done = 0

        for (id1, id2, sim) in to_merge:
            if id1 in merged_nodes or id2 in merged_nodes:
                continue
            if id1 not in self.nodes or id2 not in self.nodes:
                continue

            # pick the node with higher confidence as the target
            if self.nodes[id1].confidence >= self.nodes[id2].confidence:
                target_id, source_id = id1, id2
            else:
                target_id, source_id = id2, id1

            await self._merge_nodes(target_id, source_id)
            merged_nodes.add(source_id)
            merges_done += 1

        if merges_done > 0:
            logger.info(f"Consolidated {merges_done} similar node pairs.")

    async def _merge_nodes(self, target_id: str, source_id: str) -> None:
        """
        Integrate the content of 'source_id' into 'target_id', then remove 'source_id'.
        """
        if target_id not in self.nodes or source_id not in self.nodes:
            return

        target_node = self.nodes[target_id]
        source_node = self.nodes[source_id]

        # Merge content
        for key, val in source_node.content.items():
            if key not in target_node.content:
                target_node.content[key] = val

        # Merge metadata
        merged_from = target_node.metadata.get("merged_from", [])
        merged_from.append(source_id)
        target_node.metadata["merged_from"] = merged_from
        target_node.metadata["merged_timestamp"] = datetime.now().isoformat()

        # Weighted confidence
        new_conf = (target_node.confidence * 0.7 + source_node.confidence * 0.3)
        target_node.confidence = min(1.0, new_conf)

        # Combine sources
        if source_node.source not in target_node.source:
            target_node.source = f"{target_node.source}+{source_node.source}"

        # Transfer edges
        # Outgoing edges from source
        for succ in list(self.graph.successors(source_id)):
            if succ != target_id:
                edge_data = self.graph.get_edge_data(source_id, succ)
                self.graph.add_edge(target_id, succ, **edge_data)

        # Incoming edges to source
        for pred in list(self.graph.predecessors(source_id)):
            if pred != target_id:
                edge_data = self.graph.get_edge_data(pred, source_id)
                self.graph.add_edge(pred, target_id, **edge_data)

        # Merge supporting/conflicting
        for n_id in source_node.supporting_nodes:
            if n_id != target_id and n_id not in target_node.supporting_nodes:
                target_node.supporting_nodes.append(n_id)
        for n_id in source_node.conflicting_nodes:
            if n_id != target_id and n_id not in target_node.conflicting_nodes:
                target_node.conflicting_nodes.append(n_id)

        # Update graph data
        self.graph.nodes[target_id].update(target_node.to_dict())

        # Remove the source node
        await self._remove_node(source_id)

        # Possibly update embedding of the target node
        if self.config["enable_embeddings"] and self._embedding_model:
            await self._update_node_embedding(target_node)

    async def _infer_relations(self) -> None:
        """
        Demonstrate a simple “transitive” inference:
          A -> B, B -> C => maybe A -> C if certain relation types align
        """
        new_relations = []

        for n1 in self.nodes:
            for n2 in self.graph.successors(n1):
                data12 = self.graph.get_edge_data(n1, n2)
                rel1_type = data12.get("type")

                for n3 in self.graph.successors(n2):
                    if n3 == n1:
                        continue
                    data23 = self.graph.get_edge_data(n2, n3)
                    rel2_type = data23.get("type")

                    if self.graph.has_edge(n1, n3):
                        continue

                    # Infer a new relation type from (n1->n2, n2->n3)
                    inferred = self._infer_relation_type(rel1_type, rel2_type)
                    if inferred:
                        new_relations.append((n1, n3, inferred))

        # add them
        for (src, trg, rtype) in new_relations:
            if src in self.nodes and trg in self.nodes:
                await self.add_relation(src, trg, rtype, weight=0.7, metadata={"inferred": True})
        if new_relations:
            logger.info(f"Inferred {len(new_relations)} new relations.")

    def _infer_relation_type(self, t1: str, t2: str) -> Optional[str]:
        """
        Basic example composition rules:
          supports + supports -> supports
          supports + contradicts -> contradicts
          contradicts + supports -> contradicts
          contradicts + contradicts -> supports
          ... etc.
        """
        composition = {
            ("supports", "supports"): "supports",
            ("supports", "contradicts"): "contradicts",
            ("contradicts", "supports"): "contradicts",
            ("contradicts", "contradicts"): "supports",
            ("specializes", "specializes"): "specializes",
            ("has_property", "specializes"): "has_property"
        }
        return composition.get((t1, t2), None)

    def _apply_confidence_decay(self) -> None:
        """
        Decrease the confidence of nodes that have not been accessed for a long period.
        """
        now = datetime.now()
        for node in self.nodes.values():
            days_inactive = (now - node.last_accessed).total_seconds() / (24 * 3600)
            if days_inactive > 7:  # e.g. if not accessed in over a week
                factor = 1.0 - (self.config["decay_rate"] * min(days_inactive / 30.0, 1.0))
                node.confidence *= factor
                self.graph.nodes[node.id].update(node.to_dict())

    async def _update_knowledge_stats(self) -> None:
        """
        Refresh stats about the node/edge counts and types for reporting.
        """
        self.integration_stats["node_count"] = len(self.nodes)
        self.integration_stats["relation_count"] = self.graph.number_of_edges()

        # Tally node types
        node_types = {}
        for n in self.nodes.values():
            node_types[n.type] = node_types.get(n.type, 0) + 1
        self.integration_stats["node_types"] = node_types

        # Tally relation types
        relation_types = {}
        for s, t, data in self.graph.edges(data=True):
            rtype = data.get("type")
            if rtype:
                relation_types[rtype] = relation_types.get(rtype, 0) + 1
        self.integration_stats["relation_types"] = relation_types

    async def query_knowledge(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph for nodes matching certain criteria (type, content filter, relation filter, limit).
        Results are sorted by confidence descending.

        :param query: e.g.,
            {
                "type": "fact",
                "content_filter": {"domain": "medical"},
                "relation_filter": {"type": "supports", "node_id": "node_5", "direction": "outgoing"},
                "limit": 5
            }
        :return: A list of node dictionaries.
        """
        self.integration_stats["knowledge_queries"] += 1

        node_type = query.get("type")
        content_filter = query.get("content_filter", {})
        relation_filter = query.get("relation_filter", {})
        limit = query.get("limit", 10)

        # Build a cache key
        cache_key = json.dumps({
            "type": node_type,
            "content_filter": content_filter,
            "relation_filter": relation_filter,
            "limit": limit
        }, sort_keys=True)

        # If we already have an answer in our query_cache that’s <1 minute old, reuse
        now = datetime.now()
        if cache_key in self.query_cache:
            entry = self.query_cache[cache_key]
            cached_time = datetime.fromisoformat(entry["timestamp"])
            if (now - cached_time).total_seconds() < 60:
                return entry["results"]

        matching = []

        # Evaluate each node
        for nid, node in self.nodes.items():
            # type filter
            if node_type and node.type != node_type:
                continue

            # content filter
            content_ok = True
            for k, v in content_filter.items():
                if k not in node.content or node.content[k] != v:
                    content_ok = False
                    break
            if not content_ok:
                continue

            # relation filter
            relation_ok = True
            if relation_filter:
                rtype = relation_filter.get("type")
                other_id = relation_filter.get("node_id")
                direct = relation_filter.get("direction", "outgoing")
                if rtype and other_id:
                    if direct == "outgoing":
                        if not self.graph.has_edge(nid, other_id):
                            relation_ok = False
                        else:
                            edata = self.graph.get_edge_data(nid, other_id)
                            if edata["type"] != rtype:
                                relation_ok = False
                    else:  # incoming
                        if not self.graph.has_edge(other_id, nid):
                            relation_ok = False
                        else:
                            edata = self.graph.get_edge_data(other_id, nid)
                            if edata["type"] != rtype:
                                relation_ok = False

            if not relation_ok:
                continue

            # If this node passes all filters, return it
            node.access()
            self.graph.nodes[nid].update(node.to_dict())
            matching.append(node.to_dict())

        matching.sort(key=lambda x: x["confidence"], reverse=True)
        results = matching[:limit]

        self.query_cache[cache_key] = {
            "timestamp": now.isoformat(),
            "results": results
        }
        return results

    async def get_related_knowledge(self,
                                    node_id: str,
                                    relation_type: Optional[str] = None,
                                    direction: str = "both",
                                    limit: int = 10) -> List[Dict[str, Any]]:
        """
        Return nodes that have the specified relation to a given node, either incoming, outgoing, or both.

        :param node_id: The ID of the node around which we want to find neighbors.
        :param relation_type: Filter by a certain relation type or None for all.
        :param direction: 'incoming', 'outgoing', or 'both'
        :param limit: Max results
        :return: A list of dicts with "node" and "relation" fields describing the neighbor.
        """
        if node_id not in self.nodes:
            return []

        # update node usage
        self.nodes[node_id].access()

        neighbors = []

        # Outgoing edges
        if direction in ["outgoing", "both"]:
            for tgt in self.graph.successors(node_id):
                edata = self.graph.get_edge_data(node_id, tgt)
                if relation_type is None or edata["type"] == relation_type:
                    if tgt in self.nodes:
                        neighbors.append({
                            "node": self.nodes[tgt].to_dict(),
                            "relation": {
                                "type": edata["type"],
                                "direction": "outgoing",
                                "weight": edata.get("weight", 1.0)
                            }
                        })

        # Incoming edges
        if direction in ["incoming", "both"]:
            for src in self.graph.predecessors(node_id):
                edata = self.graph.get_edge_data(src, node_id)
                if relation_type is None or edata["type"] == relation_type:
                    if src in self.nodes:
                        neighbors.append({
                            "node": self.nodes[src].to_dict(),
                            "relation": {
                                "type": edata["type"],
                                "direction": "incoming",
                                "weight": edata.get("weight", 1.0)
                            }
                        })

        neighbors.sort(key=lambda x: x["relation"]["weight"], reverse=True)
        return neighbors[:limit]

    async def get_knowledge_statistics(self) -> Dict[str, Any]:
        """
        Return a summary of knowledge graph stats, plus a timestamp.
        """
        stats = {
            "timestamp": datetime.now().isoformat(),
            "node_count": len(self.nodes),
            "edge_count": self.graph.number_of_edges(),
            "node_types": self.integration_stats.get("node_types", {}),
            "relation_types": self.integration_stats.get("relation_types", {}),
            "nodes_added": self.integration_stats["nodes_added"],
            "relations_added": self.integration_stats["relations_added"],
            "conflicts_resolved": self.integration_stats["conflicts_resolved"],
            "knowledge_queries": self.integration_stats["knowledge_queries"],
            "integration_cycles": self.integration_stats["integration_cycles"]
        }
        return stats
