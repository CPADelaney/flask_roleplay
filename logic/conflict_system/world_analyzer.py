# logic/conflict_system/world_analyzer.py
"""
World state analysis for conflict generation
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

class WorldStateAnalyzer:
    """Analyzes world state to identify conflict opportunities"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    async def analyze_world_state(self) -> Dict[str, Any]:
        """Comprehensive world state analysis"""
        async with get_db_connection_context() as conn:
            analysis = {
                "timestamp": datetime.utcnow().isoformat(),
                "recent_events": await self._get_recent_events(conn),
                "relationship_tensions": await self._analyze_relationships(conn),
                "faction_dynamics": await self._analyze_factions(conn),
                "economic_stress": await self._analyze_economy(conn),
                "player_actions": await self._analyze_player_actions(conn)
            }
            
            # Calculate total pressure
            analysis["total_pressure"] = self._calculate_total_pressure(analysis)
            analysis["primary_driver"] = self._identify_primary_driver(analysis)
            analysis["recent_tensions"] = await self._get_recent_tensions(conn)
            
            return analysis
    
    async def _get_recent_events(self, conn) -> List[Dict[str, Any]]:
        """Get recent significant events"""
        events = await conn.fetch("""
            SELECT event_text, tags, significance, timestamp
            FROM CanonicalEvents
            WHERE user_id = $1 AND conversation_id = $2
                AND timestamp > $3
                AND significance >= 6
            ORDER BY timestamp DESC
            LIMIT 20
        """, self.user_id, self.conversation_id, 
        datetime.utcnow() - timedelta(days=7))
        
        return [dict(e) for e in events]
    
    async def _analyze_relationships(self, conn) -> Dict[str, Any]:
        """Analyze relationship tensions"""
        # High tensions
        tensions = await conn.fetch("""
            SELECT sl.*, n1.npc_name as name1, n2.npc_name as name2
            FROM SocialLinks sl
            JOIN NPCStats n1 ON sl.entity1_id = n1.npc_id
            JOIN NPCStats n2 ON sl.entity2_id = n2.npc_id
            WHERE sl.user_id = $1 AND sl.conversation_id = $2
                AND sl.entity1_type = 'npc' AND sl.entity2_type = 'npc'
                AND (sl.link_level < -50 OR 
                     (sl.dynamics->>'hostility')::int > 50)
            ORDER BY sl.link_level ASC
            LIMIT 10
        """, self.user_id, self.conversation_id)
        
        # Unresolved grudges
        grudges = await conn.fetch("""
            SELECT * FROM ConflictHistory
            WHERE user_id = $1 AND conversation_id = $2
                AND grudge_level > 50
                AND has_triggered_consequence = FALSE
            LIMIT 10
        """, self.user_id, self.conversation_id)
        
        score = len(tensions) * 10 + sum(g['grudge_level'] for g in grudges) / 10
        
        return {
            "high_tensions": [dict(t) for t in tensions],
            "unresolved_grudges": [dict(g) for g in grudges],
            "tension_score": score
        }
    
    async def _analyze_factions(self, conn) -> Dict[str, Any]:
        """Analyze faction dynamics"""
        # Power shifts
        shifts = await conn.fetch("""
            SELECT faction_name, SUM(change_amount) as total_change
            FROM FactionPowerShifts
            WHERE user_id = $1 AND conversation_id = $2
                AND created_at > $3
            GROUP BY faction_name
            HAVING ABS(SUM(change_amount)) > 3
        """, self.user_id, self.conversation_id,
        datetime.utcnow() - timedelta(days=14))
        
        instability = sum(abs(s['total_change']) for s in shifts)
        
        return {
            "power_shifts": [dict(s) for s in shifts],
            "instability_score": instability
        }
    
    async def _analyze_economy(self, conn) -> Dict[str, Any]:
        """Analyze economic factors"""
        # Resource scarcity
        scarcity = await conn.fetch("""
            SELECT resource_type, AVG(amount_changed) as avg_change
            FROM ResourceHistoryLog
            WHERE user_id = $1 AND conversation_id = $2
                AND timestamp > $3
                AND amount_changed < 0
            GROUP BY resource_type
            HAVING AVG(amount_changed) < -5
        """, self.user_id, self.conversation_id,
        datetime.utcnow() - timedelta(days=7))
        
        return {
            "resource_scarcity": [dict(s) for s in scarcity],
            "economic_stress": len(scarcity) * 20
        }
    
    async def _analyze_player_actions(self, conn) -> Dict[str, Any]:
        """Analyze recent player actions"""
        # This would analyze player choices that might trigger conflicts
        return {
            "aggressive_actions": 0,
            "diplomatic_actions": 0
        }
    
    async def _get_recent_tensions(self, conn) -> List[Dict[str, Any]]:
        """Get specific recent tension points"""
        # Combine various tension sources
        tensions = []
        
        # Recent negative relationship changes
        rel_changes = await conn.fetch("""
            SELECT * FROM RelationshipHistory
            WHERE user_id = $1 AND conversation_id = $2
                AND timestamp > $3
                AND change_amount < -10
            ORDER BY timestamp DESC
            LIMIT 5
        """, self.user_id, self.conversation_id,
        datetime.utcnow() - timedelta(days=3))
        
        for change in rel_changes:
            tensions.append({
                "type": "relationship",
                "description": f"Relationship deteriorated",
                "severity": abs(change['change_amount'])
            })
        
        return tensions
    
    def _calculate_total_pressure(self, analysis: Dict[str, Any]) -> float:
        """Calculate total conflict pressure"""
        return (
            analysis["relationship_tensions"]["tension_score"] +
            analysis["faction_dynamics"]["instability_score"] +
            analysis["economic_stress"]["economic_stress"]
        )
    
    def _identify_primary_driver(self, analysis: Dict[str, Any]) -> str:
        """Identify primary conflict driver"""
        drivers = {
            "relationships": analysis["relationship_tensions"]["tension_score"],
            "factions": analysis["faction_dynamics"]["instability_score"],
            "economics": analysis["economic_stress"]["economic_stress"]
        }
        
        return max(drivers.items(), key=lambda x: x[1])[0]
