"""
Live SDN Controller Simulator
Simulates OpenFlow-based SDN controller with real-time packet processing
Uses AI models to make eco-friendly routing choices in SDN environment
"""

import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
import random
import time
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class OpenFlowAction(Enum):
    """OpenFlow action types"""
    OUTPUT = "OUTPUT"
    DROP = "DROP"
    MODIFY = "MODIFY"


@dataclass
class OpenFlowMatch:
    """OpenFlow match fields"""
    in_port: Optional[int] = None
    eth_src: Optional[str] = None
    eth_dst: Optional[str] = None
    ip_src: Optional[int] = None
    ip_dst: Optional[int] = None
    
    def to_string(self) -> str:
        fields = []
        if self.in_port is not None:
            fields.append(f"in_port={self.in_port}")
        if self.ip_src is not None:
            fields.append(f"nw_src={self.ip_src}")
        if self.ip_dst is not None:
            fields.append(f"nw_dst={self.ip_dst}")
        return ", ".join(fields) if fields else "any"


@dataclass
class FlowEntry:
    """OpenFlow flow table entry"""
    priority: int
    match: OpenFlowMatch
    actions: List[Tuple[OpenFlowAction, Any]]
    idle_timeout: int
    hard_timeout: int
    cookie: int
    packet_count: int = 0
    byte_count: int = 0
    duration: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Packet:
    """Simulated network packet"""
    packet_id: str
    src_node: int
    dst_node: int
    in_port: int
    size: int
    timestamp: float
    
    def __str__(self):
        return f"PKT[{self.packet_id}] {self.src_node}->{self.dst_node}"


class LiveSDNController:
    """
    Simulates a live OpenFlow SDN controller with real-time packet processing
    Similar to Ryu controller functionality
    """
    
    def __init__(self, topology: nx.Graph, ai_model):
        """
        Initialize Live SDN controller
        
        Args:
            topology: NetworkX graph representing network topology
            ai_model: Trained ML model for routing decisions
        """
        self.topology = topology
        self.ai_model = ai_model
        self.flow_table = {}  # dpid -> list of FlowEntry
        self.packet_queue = []  # Incoming packets
        self.packet_history = []  # Processed packets
        self.event_log = []  # Controller events
        self.routing_stats = {
            'total_flows': 0,
            'eco_flows': 0,
            'latency_flows': 0,
            'packets_processed': 0,
            'packets_dropped': 0,
            'table_miss': 0,
            'flow_mod_count': 0
        }
        self.controller_state = "RUNNING"
        self.start_time = time.time()
        
        # Initialize flow tables for all switches (nodes)
        for node in topology.nodes():
            self.flow_table[node] = []
    
    def _log_event(self, event_type: str, message: str, details: Dict = None):
        """Log controller event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'message': message,
            'details': details or {}
        }
        self.event_log.append(event)
        if len(self.event_log) > 1000:  # Keep last 1000 events
            self.event_log = self.event_log[-1000:]
    
    def _lookup_flow(self, switch_id: int, packet: Packet) -> Optional[FlowEntry]:
        """
        Lookup matching flow entry in switch's flow table
        """
        for flow in self.flow_table.get(switch_id, []):
            match = flow.match
            # Check if packet matches flow entry
            if (match.ip_src is None or match.ip_src == packet.src_node) and \
               (match.ip_dst is None or match.ip_dst == packet.dst_node) and \
               (match.in_port is None or match.in_port == packet.in_port):
                return flow
        return None
    
    def packet_in_handler(self, packet: Packet) -> Dict:
        """
        Handle PACKET_IN event (OpenFlow packet_in message)
        Triggered when switch doesn't have matching flow entry
        """
        self._log_event("PACKET_IN", f"Received {packet}", {
            'src': packet.src_node,
            'dst': packet.dst_node,
            'port': packet.in_port
        })
        
        self.routing_stats['table_miss'] += 1
        
        # Compute paths using AI model
        routing_result = self.compute_ai_routing(
            packet.src_node, 
            packet.dst_node
        )
        
        if routing_result['success']:
            # Install flow entries along the path
            flow_cookie = self._install_path_flows(
                packet.src_node,
                packet.dst_node,
                routing_result['chosen_path'],
                routing_result
            )
            
            # Forward packet along computed path
            self._forward_packet(packet, routing_result['chosen_path'])
            
            return {
                'status': 'FLOW_INSTALLED',
                'flow_cookie': flow_cookie,
                'path': routing_result['chosen_path'],
                'decision': routing_result['decision']
            }
        else:
            # Drop packet if no path exists
            self._log_event("PACKET_DROP", f"No path for {packet}", {
                'reason': 'NO_PATH'
            })
            self.routing_stats['packets_dropped'] += 1
            return {
                'status': 'DROPPED',
                'reason': 'NO_PATH'
            }
    
    def _install_path_flows(self, src: int, dst: int, path: List[int], 
                           routing_info: Dict) -> int:
        """
        Install flow entries along the computed path
        Simulates FLOW_MOD messages to switches
        """
        flow_cookie = int(time.time() * 1000) % (2**32)
        
        # Install flows at each switch in the path
        for i in range(len(path) - 1):
            current_switch = path[i]
            next_hop = path[i + 1]
            
            # Create match for this flow
            match = OpenFlowMatch(
                ip_src=src,
                ip_dst=dst
            )
            
            # Create output action to next hop
            actions = [(OpenFlowAction.OUTPUT, next_hop)]
            
            # Create flow entry
            flow_entry = FlowEntry(
                priority=100,
                match=match,
                actions=actions,
                idle_timeout=30,  # 30 seconds
                hard_timeout=300,  # 5 minutes
                cookie=flow_cookie,
                metadata={
                    'decision': routing_info['decision'],
                    'confidence': routing_info['confidence'],
                    'carbon_saving': routing_info['metrics'].get('carbon_saving_pct', 0),
                    'latency_penalty': routing_info['metrics'].get('latency_penalty_pct', 0)
                }
            )
            
            # Add to flow table
            self.flow_table[current_switch].append(flow_entry)
            
            self._log_event("FLOW_MOD", 
                          f"Installed flow at switch {current_switch}",
                          {
                              'switch': current_switch,
                              'match': match.to_string(),
                              'output_port': next_hop,
                              'cookie': flow_cookie
                          })
            
            self.routing_stats['flow_mod_count'] += 1
        
        # Update routing statistics
        self.routing_stats['total_flows'] += 1
        if routing_info['decision'] == 'Eco-Friendly':
            self.routing_stats['eco_flows'] += 1
        else:
            self.routing_stats['latency_flows'] += 1
        
        return flow_cookie
    
    def _forward_packet(self, packet: Packet, path: List[int]):
        """Forward packet along the path"""
        self.routing_stats['packets_processed'] += 1
        self.packet_history.append({
            'packet_id': packet.packet_id,
            'src': packet.src_node,
            'dst': packet.dst_node,
            'path': path,
            'timestamp': packet.timestamp,
            'size': packet.size
        })
        
        self._log_event("PACKET_OUT", f"Forwarded {packet} via {path}")
    
    def compute_paths(self, source: int, destination: int) -> Dict[str, Any]:
        """Compute multiple paths between source and destination"""
        try:
            # Latency-optimized path
            latency_path = nx.shortest_path(
                self.topology, source, destination, weight='latency'
            )
            
            # Carbon-optimized path
            eco_path = nx.shortest_path(
                self.topology, source, destination, weight='carbon'
            )
            
            # Calculate path metrics
            lat_latency = self._path_cost(latency_path, 'latency')
            lat_carbon = self._path_cost(latency_path, 'carbon')
            eco_latency = self._path_cost(eco_path, 'latency')
            eco_carbon = self._path_cost(eco_path, 'carbon')
            
            return {
                'success': True,
                'latency_path': latency_path,
                'eco_path': eco_path,
                'lat_latency': lat_latency,
                'lat_carbon': lat_carbon,
                'eco_latency': eco_latency,
                'eco_carbon': eco_carbon
            }
            
        except nx.NetworkXNoPath:
            return {'success': False, 'error': 'No path found'}
    
    def _path_cost(self, path: List[int], weight_key: str) -> float:
        """Calculate total cost of a path"""
        if len(path) < 2:
            return 0
        return sum(
            self.topology[u][v][weight_key] 
            for u, v in zip(path[:-1], path[1:])
        )
    
    def compute_ai_routing(self, source: int, destination: int,
                          carbon_threshold: float = 20.0,
                          latency_threshold: float = 50.0) -> Dict:
        """
        Compute intelligent routing decision using AI model
        """
        # Compute paths
        paths_info = self.compute_paths(source, destination)
        
        if not paths_info.get('success', False):
            return {
                'success': False,
                'error': 'No path exists between source and destination'
            }
        
        # Extract metrics
        lat_latency = paths_info['lat_latency']
        lat_carbon = paths_info['lat_carbon']
        eco_latency = paths_info['eco_latency']
        eco_carbon = paths_info['eco_carbon']
        
        # AI model prediction
        features = [[lat_latency, lat_carbon, eco_latency, eco_carbon]]
        ai_decision = self.ai_model.predict(features)[0]
        decision_proba = self.ai_model.predict_proba(features)[0]
        
        # Calculate improvements
        carbon_saving = ((lat_carbon - eco_carbon) / lat_carbon * 100) if lat_carbon > 0 else 0
        latency_penalty = ((eco_latency - lat_latency) / lat_latency * 100) if lat_latency > 0 else 0
        
        # Apply threshold-based decision override
        threshold_decision = ai_decision
        if carbon_saving >= carbon_threshold and latency_penalty <= latency_threshold:
            threshold_decision = 1  # Choose eco path
        elif carbon_saving < carbon_threshold or latency_penalty > latency_threshold:
            threshold_decision = 0  # Choose latency path
        
        # Determine chosen path
        if threshold_decision == 1:
            chosen_path = paths_info['eco_path']
            decision_label = "Eco-Friendly"
            confidence = decision_proba[1] * 100
        else:
            chosen_path = paths_info['latency_path']
            decision_label = "Latency-Optimized"
            confidence = decision_proba[0] * 100
        
        return {
            'success': True,
            'decision': decision_label,
            'confidence': confidence,
            'chosen_path': chosen_path,
            'latency_path': paths_info['latency_path'],
            'eco_path': paths_info['eco_path'],
            'metrics': {
                'lat_latency': lat_latency,
                'lat_carbon': lat_carbon,
                'eco_latency': eco_latency,
                'eco_carbon': eco_carbon,
                'carbon_saving_pct': carbon_saving,
                'latency_penalty_pct': latency_penalty
            }
        }
    
    def _forward_packet_hopbyhop(self, packet: Packet) -> List[int]:
        """
        Forward packet hop-by-hop through the network
        Returns the actual path taken (for logging)
        """
        current_switch = packet.src_node
        path_taken = [current_switch]
        visited = {current_switch}
        
        while current_switch != packet.dst_node:
            # Look up flow entry at current switch
            matching_flow = self._lookup_flow(current_switch, packet)
            
            if matching_flow:
                # Flow hit - increment counters
                matching_flow.packet_count += 1
                matching_flow.byte_count += packet.size
                matching_flow.duration = time.time() - self.start_time
                
                # Get next hop from flow actions
                next_hop = None
                for action in matching_flow.actions:
                    if action[0] == OpenFlowAction.OUTPUT:
                        next_hop = action[1]
                        break
                
                if next_hop is None or next_hop in visited:
                    # Invalid next hop or loop detected
                    self._log_event("PACKET_DROP", f"Invalid forwarding for {packet}", {
                        'switch': current_switch,
                        'reason': 'INVALID_ACTION'
                    })
                    self.routing_stats['packets_dropped'] += 1
                    return path_taken
                
                self._log_event("FLOW_HIT", f"Flow match at switch {current_switch}", {
                    'packet': str(packet),
                    'cookie': matching_flow.cookie,
                    'next_hop': next_hop
                })
                
                current_switch = next_hop
                path_taken.append(current_switch)
                visited.add(current_switch)
                
            else:
                # Flow miss - trigger PACKET_IN to install flows
                self._log_event("FLOW_MISS", f"No flow at switch {current_switch}", {
                    'packet': str(packet)
                })
                self.routing_stats['table_miss'] += 1
                
                # Trigger PACKET_IN handler to install missing flows
                temp_packet = Packet(
                    packet_id=packet.packet_id + f"_hop{current_switch}",
                    src_node=packet.src_node,
                    dst_node=packet.dst_node,
                    in_port=current_switch,
                    size=packet.size,
                    timestamp=packet.timestamp
                )
                result = self.packet_in_handler(temp_packet)
                
                if result.get('status') == 'FLOW_INSTALLED':
                    # Flows installed, continue forwarding
                    # Retry lookup at current switch
                    matching_flow = self._lookup_flow(current_switch, packet)
                    if matching_flow:
                        matching_flow.packet_count += 1
                        matching_flow.byte_count += packet.size
                        
                        next_hop = None
                        for action in matching_flow.actions:
                            if action[0] == OpenFlowAction.OUTPUT:
                                next_hop = action[1]
                                break
                        
                        if next_hop and next_hop not in visited:
                            current_switch = next_hop
                            path_taken.append(current_switch)
                            visited.add(current_switch)
                            continue
                
                # Could not install flow or forward packet - drop
                self.routing_stats['packets_dropped'] += 1
                return path_taken
        
        # Packet reached destination
        return path_taken
    
    def inject_packet(self, src: int, dst: int, size: int = 1500) -> Packet:
        """
        Inject a packet into the network (simulate traffic)
        First checks flow table, only triggers PACKET_IN on miss
        """
        packet = Packet(
            packet_id=f"PKT_{len(self.packet_history):04d}",
            src_node=src,
            dst_node=dst,
            in_port=src,
            size=size,
            timestamp=time.time()
        )
        
        # Check if there's a matching flow entry at source
        matching_flow = self._lookup_flow(src, packet)
        
        if matching_flow:
            # Flow exists - forward packet hop-by-hop
            path_taken = self._forward_packet_hopbyhop(packet)
            
            if path_taken[-1] == dst:
                # Packet successfully delivered
                self._forward_packet(packet, path_taken)
                self._log_event("PACKET_DELIVERED", f"{packet} delivered", {
                    'path': path_taken
                })
            
        else:
            # Flow miss - trigger PACKET_IN handler to install flows
            result = self.packet_in_handler(packet)
        
        return packet
    
    def get_flow_table(self, switch_id: Optional[int] = None) -> Dict:
        """Get flow table entries"""
        if switch_id is not None:
            return {
                'switch_id': switch_id,
                'flows': [
                    {
                        'priority': flow.priority,
                        'match': flow.match.to_string(),
                        'actions': [f"{action[0].value}:{action[1]}" for action in flow.actions],
                        'cookie': flow.cookie,
                        'packets': flow.packet_count,
                        'metadata': flow.metadata
                    }
                    for flow in self.flow_table.get(switch_id, [])
                ]
            }
        else:
            return {
                switch: [
                    {
                        'priority': flow.priority,
                        'match': flow.match.to_string(),
                        'actions': [f"{action[0].value}:{action[1]}" for action in flow.actions],
                        'cookie': flow.cookie,
                        'metadata': flow.metadata
                    }
                    for flow in flows
                ]
                for switch, flows in self.flow_table.items()
            }
    
    def get_statistics(self) -> Dict:
        """Get controller statistics"""
        total = self.routing_stats['total_flows']
        eco_pct = (self.routing_stats['eco_flows'] / total * 100) if total > 0 else 0
        uptime = time.time() - self.start_time
        
        return {
            'controller_state': self.controller_state,
            'uptime_seconds': uptime,
            'total_flows': total,
            'eco_flows': self.routing_stats['eco_flows'],
            'latency_flows': self.routing_stats['latency_flows'],
            'eco_percentage': eco_pct,
            'packets_processed': self.routing_stats['packets_processed'],
            'packets_dropped': self.routing_stats['packets_dropped'],
            'table_misses': self.routing_stats['table_miss'],
            'flow_mods_sent': self.routing_stats['flow_mod_count'],
            'total_switches': len(self.flow_table)
        }
    
    def get_event_log(self, limit: int = 50) -> List[Dict]:
        """Get recent controller events"""
        return self.event_log[-limit:]
    
    def simulate_traffic(self, num_packets: int = 10) -> List[Dict]:
        """
        Simulate network traffic with random packet injections
        """
        nodes = list(self.topology.nodes())
        results = []
        
        for i in range(num_packets):
            src, dst = random.sample(nodes, 2)
            packet = self.inject_packet(src, dst, size=random.randint(64, 1500))
            
            results.append({
                'packet_id': packet.packet_id,
                'src': src,
                'dst': dst,
                'size': packet.size,
                'timestamp': packet.timestamp
            })
        
        return results
    
    def clear_flows(self):
        """Clear all flow entries (FLOW_MOD delete)"""
        for switch in self.flow_table:
            self.flow_table[switch] = []
        
        self._log_event("FLOW_CLEAR", "All flows cleared")
        self.routing_stats['flow_mod_count'] = 0


def create_live_sdn_controller(topology: nx.Graph, ai_model) -> LiveSDNController:
    """
    Factory function to create Live SDN controller instance
    
    Args:
        topology: NetworkX graph
        ai_model: Trained ML model
        
    Returns:
        LiveSDNController instance
    """
    return LiveSDNController(topology, ai_model)


# Legacy compatibility
class SDNController(LiveSDNController):
    """Legacy SDN Controller for backward compatibility"""
    
    def make_routing_decision(self, source: int, destination: int,
                             carbon_threshold: float = 20.0,
                             latency_threshold: float = 50.0) -> Dict:
        """Legacy method for backward compatibility"""
        routing_result = self.compute_ai_routing(
            source, destination, carbon_threshold, latency_threshold
        )
        
        if not routing_result['success']:
            return routing_result
        
        # Create flow ID for compatibility
        flow_id = f"flow_{source}_{destination}_{self.routing_stats['total_flows']}"
        
        # Install flow using new method
        packet = Packet(
            packet_id=f"LEGACY_{flow_id}",
            src_node=source,
            dst_node=destination,
            in_port=source,
            size=1500,
            timestamp=time.time()
        )
        
        flow_cookie = self._install_path_flows(
            source, destination, 
            routing_result['chosen_path'],
            routing_result
        )
        
        return {
            'success': True,
            'flow_id': flow_id,
            'flow_cookie': flow_cookie,
            'decision': routing_result['decision'],
            'confidence': routing_result['confidence'],
            'chosen_path': routing_result['chosen_path'],
            'latency_path': routing_result['latency_path'],
            'eco_path': routing_result['eco_path'],
            'metrics': routing_result['metrics'],
            'flow_entry': {
                'source': source,
                'destination': destination,
                'path': routing_result['chosen_path'],
                'decision': routing_result['decision'],
                'confidence': routing_result['confidence'],
                'match': f"ip_src={source}, ip_dst={destination}",
                'actions': [f"forward_to_port({node})" for node in routing_result['chosen_path'][1:]]
            }
        }


def create_sdn_controller(topology: nx.Graph, ai_model) -> SDNController:
    """Legacy factory function for backward compatibility"""
    return SDNController(topology, ai_model)
