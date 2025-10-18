import random
import heapq
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque
import networkx as nx

@dataclass
class Packet:
    """Represents a network packet"""
    packet_id: int
    source: int
    destination: int
    size: int  # bytes
    creation_time: float
    current_node: int
    path: List[int]
    path_index: int = 0
    arrival_time: Optional[float] = None
    total_latency: float = 0.0
    total_carbon: float = 0.0
    hops: int = 0

@dataclass
class Event:
    """Discrete event for simulation"""
    time: float
    event_type: str  # 'packet_arrival', 'packet_transmission'
    packet: Packet
    link: Optional[Tuple[int, int]] = None
    
    def __lt__(self, other):
        return self.time < other.time

class Link:
    """Represents a network link with bandwidth and queuing"""
    def __init__(self, source: int, dest: int, bandwidth_mbps: float, 
                 base_latency_ms: float, carbon_per_mb: float):
        self.source = source
        self.dest = dest
        self.bandwidth_mbps = bandwidth_mbps
        self.base_latency_ms = base_latency_ms
        self.carbon_per_mb = carbon_per_mb
        self.queue = deque()
        self.queue_size_bytes = 0
        self.max_queue_bytes = 1024 * 1024  # 1MB queue
        self.is_busy = False
        self.current_transmission = None
        self.bytes_transmitted = 0
        self.packets_dropped = 0
        
    def transmission_time(self, packet_size_bytes: int) -> float:
        """Calculate transmission time for a packet in seconds"""
        bits = packet_size_bytes * 8
        megabits = bits / 1_000_000
        return megabits / self.bandwidth_mbps
    
    def propagation_delay(self) -> float:
        """Get propagation delay in seconds"""
        return self.base_latency_ms / 1000.0
    
    def queuing_delay(self) -> float:
        """Estimate queuing delay based on queue size"""
        if self.queue_size_bytes == 0:
            return 0.0
        # Estimate time to clear current queue
        return self.transmission_time(self.queue_size_bytes)
    
    def carbon_emission(self, packet_size_bytes: int) -> float:
        """Calculate carbon emission for transmitting a packet"""
        megabytes = packet_size_bytes / (1024 * 1024)
        return megabytes * self.carbon_per_mb

class DiscreteEventSimulator:
    """Discrete-event network simulator for packet-level routing analysis"""
    
    def __init__(self, graph: nx.Graph, packet_size_bytes: int = 1500):
        self.graph = graph
        self.packet_size_bytes = packet_size_bytes
        self.event_queue = []
        self.current_time = 0.0
        self.links = {}
        self.packets_sent = 0
        self.packets_delivered = 0
        self.packets_in_flight = {}
        
        # Initialize links from graph
        self._initialize_links()
    
    def _initialize_links(self):
        """Create Link objects from NetworkX graph"""
        for u, v, data in self.graph.edges(data=True):
            base_latency = data.get('latency', 10.0)  # ms
            carbon = data.get('carbon', 0.01)  # g CO2
            
            # Infer bandwidth from latency (lower latency = higher bandwidth)
            # Model: 10ms -> 100 Mbps, 50ms -> 20 Mbps
            bandwidth_mbps = max(10.0, 1000.0 / base_latency)
            
            # Carbon per MB transmitted
            carbon_per_mb = carbon * 1000  # Scale carbon to per-MB
            
            # Create bidirectional links
            self.links[(u, v)] = Link(u, v, bandwidth_mbps, base_latency, carbon_per_mb)
            self.links[(v, u)] = Link(v, u, bandwidth_mbps, base_latency, carbon_per_mb)
    
    def send_packet(self, source: int, destination: int, path: List[int], 
                   start_time: float = 0.0) -> Packet:
        """Create and send a packet through the network"""
        packet = Packet(
            packet_id=self.packets_sent,
            source=source,
            destination=destination,
            size=self.packet_size_bytes,
            creation_time=start_time,
            current_node=source,
            path=path,
            path_index=0
        )
        self.packets_sent += 1
        self.packets_in_flight[packet.packet_id] = packet
        
        # Schedule first transmission
        self._schedule_transmission(packet, start_time)
        
        return packet
    
    def _schedule_transmission(self, packet: Packet, time: float):
        """Schedule packet transmission on next hop"""
        if packet.path_index >= len(packet.path) - 1:
            # Packet reached destination
            return
        
        current_node = packet.path[packet.path_index]
        next_node = packet.path[packet.path_index + 1]
        link = self.links.get((current_node, next_node))
        
        if link:
            event = Event(
                time=time,
                event_type='packet_transmission',
                packet=packet,
                link=(current_node, next_node)
            )
            heapq.heappush(self.event_queue, event)
    
    def _process_transmission(self, event: Event):
        """Process packet transmission event"""
        packet = event.packet
        link_key = event.link
        link = self.links[link_key]
        
        # Calculate transmission time and delays
        tx_time = link.transmission_time(packet.size)
        prop_delay = link.propagation_delay()
        queue_delay = link.queuing_delay()
        
        # Total time for this hop
        total_hop_time = tx_time + prop_delay + queue_delay
        
        # Add random jitter (0-10% of base latency)
        jitter = random.uniform(0, link.base_latency_ms / 1000.0 * 0.1)
        total_hop_time += jitter
        
        # Update packet metrics
        packet.total_latency += total_hop_time
        packet.total_carbon += link.carbon_emission(packet.size)
        packet.hops += 1
        
        # Move packet to next node
        packet.path_index += 1
        packet.current_node = packet.path[packet.path_index]
        
        # Update link statistics
        link.bytes_transmitted += packet.size
        
        # Check if packet reached destination
        if packet.path_index >= len(packet.path) - 1:
            packet.arrival_time = event.time + total_hop_time
            self.packets_delivered += 1
            if packet.packet_id in self.packets_in_flight:
                del self.packets_in_flight[packet.packet_id]
        else:
            # Schedule next hop
            next_event_time = event.time + total_hop_time
            self._schedule_transmission(packet, next_event_time)
    
    def simulate_flow(self, source: int, destination: int, path: List[int], 
                     num_packets: int = 10, flow_rate_mbps: float = 10.0) -> Dict:
        """Simulate a flow of packets along a path"""
        self.event_queue = []
        self.current_time = 0.0
        self.packets_sent = 0
        self.packets_delivered = 0
        self.packets_in_flight = {}
        
        # Calculate inter-packet interval to achieve desired flow rate
        packet_bits = self.packet_size_bytes * 8
        interval_seconds = packet_bits / (flow_rate_mbps * 1_000_000)
        
        # Generate packets
        packets = []
        for i in range(num_packets):
            start_time = i * interval_seconds
            packet = self.send_packet(source, destination, path, start_time)
            packets.append(packet)
        
        # Run simulation until all events processed
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            
            if event.event_type == 'packet_transmission':
                self._process_transmission(event)
        
        # Calculate statistics
        delivered_packets = [p for p in packets if p.arrival_time is not None]
        
        if not delivered_packets:
            return {
                'packets_sent': num_packets,
                'packets_delivered': 0,
                'packet_loss_rate': 1.0,
                'avg_latency_ms': 0.0,
                'total_carbon_g': 0.0,
                'min_latency_ms': 0.0,
                'max_latency_ms': 0.0,
                'jitter_ms': 0.0,
                'throughput_mbps': 0.0
            }
        
        latencies = [p.total_latency * 1000 for p in delivered_packets]  # Convert to ms
        carbons = [p.total_carbon for p in delivered_packets]
        
        # Calculate jitter (variation in latency)
        jitter = max(latencies) - min(latencies) if len(latencies) > 1 else 0.0
        
        # Calculate effective throughput
        if delivered_packets:
            total_time = max(p.arrival_time for p in delivered_packets)
            total_bytes = sum(p.size for p in delivered_packets)
            throughput_mbps = (total_bytes * 8) / (total_time * 1_000_000) if total_time > 0 else 0.0
        else:
            throughput_mbps = 0.0
        
        return {
            'packets_sent': num_packets,
            'packets_delivered': len(delivered_packets),
            'packet_loss_rate': (num_packets - len(delivered_packets)) / num_packets,
            'avg_latency_ms': sum(latencies) / len(latencies),
            'total_carbon_g': sum(carbons),
            'avg_carbon_per_packet_g': sum(carbons) / len(carbons),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'jitter_ms': jitter,
            'throughput_mbps': throughput_mbps,
            'path_length': len(path) - 1,
            'packets': delivered_packets
        }
    
    def compare_paths(self, source: int, destination: int, 
                     path1: List[int], path2: List[int],
                     num_packets: int = 20, flow_rate_mbps: float = 10.0) -> Dict:
        """Compare two routing paths with packet-level simulation"""
        
        # Simulate both paths
        results1 = self.simulate_flow(source, destination, path1, num_packets, flow_rate_mbps)
        results2 = self.simulate_flow(source, destination, path2, num_packets, flow_rate_mbps)
        
        return {
            'path1': results1,
            'path2': results2,
            'latency_improvement_pct': ((results1['avg_latency_ms'] - results2['avg_latency_ms']) 
                                       / results1['avg_latency_ms'] * 100) if results1['avg_latency_ms'] > 0 else 0,
            'carbon_improvement_pct': ((results1['total_carbon_g'] - results2['total_carbon_g']) 
                                      / results1['total_carbon_g'] * 100) if results1['total_carbon_g'] > 0 else 0,
            'jitter_comparison': {
                'path1_jitter_ms': results1['jitter_ms'],
                'path2_jitter_ms': results2['jitter_ms']
            },
            'throughput_comparison': {
                'path1_throughput_mbps': results1['throughput_mbps'],
                'path2_throughput_mbps': results2['throughput_mbps']
            }
        }

def run_packet_simulation(graph: nx.Graph, source: int, destination: int,
                         latency_path: List[int], eco_path: List[int],
                         num_packets: int = 20) -> Dict:
    """
    High-level function to run packet-level simulation comparing two paths.
    
    Args:
        graph: NetworkX graph with latency and carbon edge weights
        source: Source node
        destination: Destination node
        latency_path: Latency-optimized path
        eco_path: Eco-optimized path
        num_packets: Number of packets to simulate per flow
    
    Returns:
        Dictionary with comprehensive simulation results
    """
    simulator = DiscreteEventSimulator(graph, packet_size_bytes=1500)
    
    # Run comparison
    comparison = simulator.compare_paths(
        source, destination,
        latency_path, eco_path,
        num_packets=num_packets,
        flow_rate_mbps=10.0
    )
    
    return {
        'latency_path_stats': {
            'avg_latency_ms': comparison['path1']['avg_latency_ms'],
            'total_carbon_g': comparison['path1']['total_carbon_g'],
            'jitter_ms': comparison['path1']['jitter_ms'],
            'throughput_mbps': comparison['path1']['throughput_mbps'],
            'packet_loss_rate': comparison['path1']['packet_loss_rate'],
            'packets_delivered': comparison['path1']['packets_delivered']
        },
        'eco_path_stats': {
            'avg_latency_ms': comparison['path2']['avg_latency_ms'],
            'total_carbon_g': comparison['path2']['total_carbon_g'],
            'jitter_ms': comparison['path2']['jitter_ms'],
            'throughput_mbps': comparison['path2']['throughput_mbps'],
            'packet_loss_rate': comparison['path2']['packet_loss_rate'],
            'packets_delivered': comparison['path2']['packets_delivered']
        },
        'comparison': {
            'latency_improvement_pct': comparison['latency_improvement_pct'],
            'carbon_improvement_pct': comparison['carbon_improvement_pct']
        },
        'simulation_config': {
            'num_packets': num_packets,
            'packet_size_bytes': 1500,
            'flow_rate_mbps': 10.0
        }
    }
