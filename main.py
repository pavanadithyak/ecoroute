import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import joblib
import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from database import init_db, save_routing_decision, get_routing_history, get_routing_stats
from models import load_all_models, compare_models
from sdn_routing import create_sdn_controller, create_live_sdn_controller

# Set page config
st.set_page_config(
    page_title="EcoRoutingAI Dashboard",
    page_icon="🌱",
    layout="wide"
)

# Initialize database
@st.cache_resource
def initialize_database():
    """Initialize database on first run"""
    init_db()

initialize_database()

# Initialize session state
if 'graph_seed' not in st.session_state:
    st.session_state.graph_seed = random.randint(1, 1000)
if 'num_nodes' not in st.session_state:
    st.session_state.num_nodes = 6
if 'carbon_threshold' not in st.session_state:
    st.session_state.carbon_threshold = 20.0
if 'latency_threshold' not in st.session_state:
    st.session_state.latency_threshold = 50.0

# Load or create model
@st.cache_resource
def load_model():
    """Load pre-trained RandomForest model or create one if it doesn't exist"""
    try:
        model = joblib.load("model.pkl")
        return model
    except FileNotFoundError:
        # Create a simple RandomForest model for demonstration
        # Training data simulating decision patterns:
        # Features: [lat_latency, lat_carbon, eco_latency, eco_carbon]
        # Target: 1 for eco-path, 0 for latency-path
        
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        X_train = []
        y_train = []
        
        for _ in range(n_samples):
            lat_latency = np.random.uniform(10, 100)
            eco_latency = lat_latency * np.random.uniform(1.1, 2.0)  # eco path usually slower
            lat_carbon = np.random.uniform(0.01, 0.1)
            eco_carbon = lat_carbon * np.random.uniform(0.3, 0.8)   # eco path usually cleaner
            
            # Decision logic: choose eco if carbon savings > 30% and latency penalty < 50%
            carbon_saving = (lat_carbon - eco_carbon) / lat_carbon
            latency_penalty = (eco_latency - lat_latency) / lat_latency
            
            decision = 1 if (carbon_saving > 0.3 and latency_penalty < 0.5) else 0
            
            X_train.append([lat_latency, lat_carbon, eco_latency, eco_carbon])
            y_train.append(decision)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(model, "model.pkl")
        return model

def create_network(seed=None, num_nodes=6, custom_edges=None):
    """Create a network with configurable nodes and weighted edges"""
    if seed:
        random.seed(seed)
    
    G = nx.Graph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i, label=f"Node {i}")
    
    # Create edges to ensure connectivity
    if custom_edges:
        edges = custom_edges
    else:
        # Generate default edges based on number of nodes
        edges = []
        if num_nodes == 6:
            edges = [
                (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), 
                (2, 4), (3, 4), (3, 5), (4, 5), (1, 4)
            ]
        else:
            # Create a connected graph with random edges
            for i in range(num_nodes - 1):
                edges.append((i, i + 1))
            
            # Add additional random edges
            for i in range(num_nodes):
                for j in range(i + 2, min(i + 4, num_nodes)):
                    if random.random() < 0.5:
                        edges.append((i, j))
    
    for u, v in edges:
        if u < num_nodes and v < num_nodes:
            # Add some randomness to simulate dynamic traffic
            base_latency = random.uniform(5, 50)
            latency_variation = random.uniform(0.8, 1.2)
            latency = round(base_latency * latency_variation, 2)
            
            base_carbon = random.uniform(0.001, 0.05)
            carbon_variation = random.uniform(0.8, 1.2)
            carbon = round(base_carbon * carbon_variation, 4)
            
            G.add_edge(u, v, latency=latency, carbon=carbon)
    
    return G

def path_cost(G, path, weight_key):
    """Calculate total cost of a path"""
    if len(path) < 2:
        return 0
    return sum(G[u][v][weight_key] for u, v in zip(path[:-1], path[1:]))

def visualize_network(G, lat_path, eco_path, chosen_path, source, destination):
    """Visualize the network with highlighted paths"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create layout
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    # Draw all edges in gray first
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=2, alpha=0.5, ax=ax)
    
    # Highlight chosen path
    if chosen_path == "Eco-Friendly":
        path_edges = [(eco_path[i], eco_path[i+1]) for i in range(len(eco_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='green', 
                             width=4, alpha=0.8, ax=ax, label='Chosen Eco Path')
        
        # Draw latency path in red (not chosen)
        lat_edges = [(lat_path[i], lat_path[i+1]) for i in range(len(lat_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=lat_edges, edge_color='red', 
                             width=2, alpha=0.6, ax=ax, style='dashed', label='Latency Path')
    else:
        path_edges = [(lat_path[i], lat_path[i+1]) for i in range(len(lat_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', 
                             width=4, alpha=0.8, ax=ax, label='Chosen Latency Path')
        
        # Draw eco path in green (not chosen)
        eco_edges = [(eco_path[i], eco_path[i+1]) for i in range(len(eco_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=eco_edges, edge_color='green', 
                             width=2, alpha=0.6, ax=ax, style='dashed', label='Eco Path')
    
    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if node == source:
            node_colors.append('lightblue')
        elif node == destination:
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightgray')
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, 
                          alpha=0.9, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    # Add edge labels with weights
    edge_labels = {}
    for u, v in G.edges():
        latency = G[u][v]['latency']
        carbon = G[u][v]['carbon']
        edge_labels[(u, v)] = f"{latency}ms\n{carbon:.3f}g"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
    
    ax.set_title("Network Topology with Routing Paths", fontsize=16, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Load model
    model = load_model()
    
    # Title and header
    st.title("🌱 EcoRoutingAI - AI-Driven Eco-Friendly Routing Dashboard")
    st.markdown("*Intelligent SDN routing optimization balancing latency and carbon footprint*")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Network Configuration")
    
    # Number of nodes
    num_nodes = st.sidebar.slider("Number of Nodes", min_value=4, max_value=12, value=st.session_state.num_nodes, key="nodes_slider")
    if num_nodes != st.session_state.num_nodes:
        st.session_state.num_nodes = num_nodes
        st.session_state.graph_seed = random.randint(1, 1000)
    
    # Source and destination selection
    source = st.sidebar.selectbox("Source Node", options=list(range(num_nodes)), index=0)
    destination = st.sidebar.selectbox("Destination Node", options=list(range(num_nodes)), index=min(num_nodes-1, 5))
    
    # Threshold controls
    st.sidebar.header("🎯 Decision Thresholds")
    carbon_threshold = st.sidebar.slider(
        "Min Carbon Savings (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=st.session_state.carbon_threshold,
        help="Minimum carbon savings to consider eco path"
    )
    latency_threshold = st.sidebar.slider(
        "Max Latency Penalty (%)", 
        min_value=0.0, 
        max_value=200.0, 
        value=st.session_state.latency_threshold,
        help="Maximum acceptable latency penalty for eco path"
    )
    
    st.session_state.carbon_threshold = carbon_threshold
    st.session_state.latency_threshold = latency_threshold
    
    if source == destination:
        st.error("Source and destination must be different!")
        return
    
    # Create network
    G = create_network(st.session_state.graph_seed, num_nodes=num_nodes)
    
    try:
        # Compute paths
        lat_path = nx.shortest_path(G, source, destination, weight='latency')
        eco_path = nx.shortest_path(G, source, destination, weight='carbon')
        
        # Calculate metrics
        lat_latency = path_cost(G, lat_path, 'latency')
        lat_carbon = path_cost(G, lat_path, 'carbon')
        eco_latency = path_cost(G, eco_path, 'latency')
        eco_carbon = path_cost(G, eco_path, 'carbon')
        
        # AI decision
        features = [[lat_latency, lat_carbon, eco_latency, eco_carbon]]
        decision = model.predict(features)[0]
        decision_proba = model.predict_proba(features)[0]
        
        # Calculate improvements
        carbon_saving = ((lat_carbon - eco_carbon) / lat_carbon * 100) if lat_carbon > 0 else 0
        latency_penalty = ((eco_latency - lat_latency) / lat_latency * 100) if lat_latency > 0 else 0
        
        # Apply threshold-based decision override
        threshold_decision = decision
        if carbon_saving >= carbon_threshold and latency_penalty <= latency_threshold:
            threshold_decision = 1
        elif carbon_saving < carbon_threshold or latency_penalty > latency_threshold:
            threshold_decision = 0
        
        # Update chosen path based on threshold decision
        chosen_path = "Eco-Friendly" if threshold_decision == 1 else "Latency-Optimized"
        
        # Save decision to database
        network_config = json.dumps({
            'num_nodes': num_nodes,
            'edges': [(u, v, {'latency': G[u][v]['latency'], 'carbon': G[u][v]['carbon']}) for u, v in G.edges()]
        })
        
        try:
            # Convert numpy types to Python native types for database compatibility
            save_routing_decision(
                source, destination,
                lat_path, eco_path,
                float(lat_latency), float(lat_carbon),
                float(eco_latency), float(eco_carbon),
                chosen_path, int(threshold_decision),
                float(decision_proba[threshold_decision] * 100),
                float(carbon_saving), float(latency_penalty),
                network_config
            )
        except Exception as e:
            st.warning(f"Could not save to database: {str(e)}")
        
        # Main dashboard layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Network Visualization")
            fig = visualize_network(G, lat_path, eco_path, chosen_path, source, destination)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("AI Decision")
            
            # Decision display
            if threshold_decision == 1:
                st.success(f"🌱 **{chosen_path} Path** Selected")
                confidence = decision_proba[1] * 100
                if decision != threshold_decision:
                    st.info("ℹ️ Decision modified by threshold settings")
            else:
                st.info(f"⚡ **{chosen_path} Path** Selected")
                confidence = decision_proba[0] * 100
                if decision != threshold_decision:
                    st.info("ℹ️ Decision modified by threshold settings")
            
            st.write(f"Confidence: {confidence:.1f}%")
            
            # Path comparison
            st.subheader("Path Comparison")
            
            # Latency metrics
            st.metric(
                label="Latency (ms)",
                value=f"{lat_latency:.2f}",
                delta=f"{eco_latency:.2f}" if threshold_decision == 1 else None,
                help="Latency-optimal → Eco-optimal" if threshold_decision == 1 else "Current path latency"
            )
            
            # Carbon metrics
            st.metric(
                label="Carbon (g CO₂)",
                value=f"{lat_carbon:.4f}",
                delta=f"-{abs(eco_carbon-lat_carbon):.4f}" if threshold_decision == 1 else None,
                delta_color="inverse",
                help="Latency-optimal → Eco-optimal" if threshold_decision == 1 else "Current path carbon"
            )
            
            # Performance metrics
            st.subheader("Performance Metrics")
            
            if carbon_saving > 0:
                st.success(f"🌱 Carbon Savings: {carbon_saving:.1f}%")
            else:
                st.info(f"🌱 Carbon Impact: {abs(carbon_saving):.1f}%")
            
            if latency_penalty > 0:
                st.warning(f"⚡ Latency Penalty: +{latency_penalty:.1f}%")
            else:
                st.success(f"⚡ Latency Improvement: {abs(latency_penalty):.1f}%")
        
        # Detailed path information
        st.subheader("Detailed Path Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Latency-Optimized Path:**")
            st.write(" → ".join([f"Node {node}" for node in lat_path]))
            st.write(f"Total Latency: {lat_latency:.2f} ms")
            st.write(f"Total Carbon: {lat_carbon:.4f} g CO₂")
        
        with col2:
            st.write("**Eco-Friendly Path:**")
            st.write(" → ".join([f"Node {node}" for node in eco_path]))
            st.write(f"Total Latency: {eco_latency:.2f} ms")
            st.write(f"Total Carbon: {eco_carbon:.4f} g CO₂")
        
        # Control buttons
        st.subheader("Simulation Controls")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Simulate Next Packet", type="primary"):
                st.session_state.graph_seed = random.randint(1, 1000)
                st.rerun()
        
        with col2:
            if st.button("🎲 Randomize Network"):
                st.session_state.graph_seed = random.randint(1, 1000)
                st.rerun()
        
        with col3:
            # Export current decision
            export_data = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'source': source,
                'destination': destination,
                'latency_path': str(lat_path),
                'eco_path': str(eco_path),
                'latency_optimal_latency': lat_latency,
                'latency_optimal_carbon': lat_carbon,
                'eco_optimal_latency': eco_latency,
                'eco_optimal_carbon': eco_carbon,
                'decision': chosen_path,
                'carbon_saving_pct': carbon_saving,
                'latency_penalty_pct': latency_penalty,
                'confidence': decision_proba[threshold_decision] * 100
            }
            
            st.download_button(
                label="📥 Export Decision",
                data=json.dumps(export_data, indent=2),
                file_name=f"routing_decision_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # MATLAB Results Comparison
        st.subheader("📊 MATLAB Results Comparison")
        show_matlab = st.checkbox("Show MATLAB SimEvents Results")
        
        if show_matlab:
            uploaded_file = st.file_uploader(
                "Upload MATLAB results (.csv)",
                type=['csv'],
                help="Upload CSV file with columns: Time, Latency, Carbon"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if all(col in df.columns for col in ['Time', 'Latency', 'Carbon']):
                        st.write("**MATLAB SimEvents Results:**")
                        
                        # Display the chart
                        chart_data = df.set_index('Time')[['Latency', 'Carbon']]
                        st.line_chart(chart_data)
                        
                        # Show statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Latency Statistics:**")
                            st.write(f"Mean: {df['Latency'].mean():.2f} ms")
                            st.write(f"Std: {df['Latency'].std():.2f} ms")
                        
                        with col2:
                            st.write("**Carbon Statistics:**")
                            st.write(f"Mean: {df['Carbon'].mean():.4f} g CO₂")
                            st.write(f"Std: {df['Carbon'].std():.4f} g CO₂")
                        
                    else:
                        st.error("CSV must contain columns: Time, Latency, Carbon")
                        
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
            
            else:
                # Show sample data if no file uploaded
                if os.path.exists("sample_matlab_results.csv"):
                    st.write("**Sample MATLAB Results:**")
                    sample_df = pd.read_csv("sample_matlab_results.csv")
                    chart_data = sample_df.set_index('Time')[['Latency', 'Carbon']]
                    st.line_chart(chart_data)
        
        # AI Model Comparison
        st.subheader("🤖 AI Model Comparison")
        show_model_comparison = st.checkbox("Compare Multiple AI Models")
        
        if show_model_comparison:
            try:
                features = [lat_latency, lat_carbon, eco_latency, eco_carbon]
                model_results = compare_models(features)
                
                st.write("**Model Predictions & Confidence:**")
                
                comparison_df = pd.DataFrame([
                    {
                        'Model': name.title(),
                        'Decision': 'Eco-Friendly' if result['prediction'] == 1 else 'Latency-Optimized',
                        'Confidence': f"{result['confidence']:.1f}%",
                        'Eco Probability': f"{result['eco_probability']:.1f}%"
                    }
                    for name, result in model_results.items()
                ])
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualize model agreement
                eco_votes = sum(1 for r in model_results.values() if r['prediction'] == 1)
                lat_votes = len(model_results) - eco_votes
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Models Choosing Eco Path", f"{eco_votes}/{len(model_results)}")
                with col2:
                    st.metric("Models Choosing Latency Path", f"{lat_votes}/{len(model_results)}")
                
                # Export model comparison
                export_models = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export Model Comparison",
                    data=export_models,
                    file_name=f"model_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error comparing models: {str(e)}")
                st.info("Models will be created automatically on first use.")
        
        # SDN Controller Simulation
        st.subheader("🌐 SDN Controller Routing")
        show_sdn = st.checkbox("Enable SDN Controller Simulation")
        
        if show_sdn:
            st.info("Simulating Software-Defined Networking controller with AI-driven routing decisions...")
            
            try:
                # Create SDN controller
                sdn_controller = create_sdn_controller(G, model)
                
                # Make routing decision using SDN controller
                sdn_result = sdn_controller.make_routing_decision(
                    source, destination,
                    carbon_threshold=carbon_threshold,
                    latency_threshold=latency_threshold
                )
                
                if sdn_result['success']:
                    st.success(f"SDN Flow Installed: {sdn_result['flow_id']}")
                    
                    # Display SDN controller decision
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**SDN Controller Decision:**")
                        if sdn_result['decision'] == 'Eco-Friendly':
                            st.success(f"🌱 {sdn_result['decision']} Path Selected")
                        else:
                            st.info(f"⚡ {sdn_result['decision']} Path Selected")
                        
                        st.write(f"**Confidence:** {sdn_result['confidence']:.1f}%")
                        st.write(f"**Chosen Path:** {' → '.join([f'Node {n}' for n in sdn_result['chosen_path']])}")
                    
                    with col2:
                        st.write("**Flow Table Entry:**")
                        flow_entry = sdn_result['flow_entry']
                        st.code(f"""
Flow ID: {sdn_result['flow_id']}
Match: {flow_entry['match']}
Actions: forward to {' -> '.join([f'Node {n}' for n in sdn_result['chosen_path'][1:]])}
Decision: {flow_entry['decision']}
Confidence: {flow_entry['confidence']:.1f}%
                        """, language="text")
                    
                    # SDN vs Standard comparison
                    st.write("**SDN Controller Metrics:**")
                    metrics = sdn_result['metrics']
                    
                    comparison_data = {
                        'Metric': ['Latency (ms)', 'Carbon (g CO₂)', 'Carbon Savings (%)', 'Latency Penalty (%)'],
                        'Latency-Optimal': [
                            f"{metrics['lat_latency']:.2f}",
                            f"{metrics['lat_carbon']:.4f}",
                            "-",
                            "-"
                        ],
                        'Eco-Optimal': [
                            f"{metrics['eco_latency']:.2f}",
                            f"{metrics['eco_carbon']:.4f}",
                            f"{metrics['carbon_saving_pct']:.1f}%",
                            f"{metrics['latency_penalty_pct']:.1f}%"
                        ],
                        'SDN Chosen': [
                            f"{metrics['eco_latency'] if sdn_result['decision'] == 'Eco-Friendly' else metrics['lat_latency']:.2f}",
                            f"{metrics['eco_carbon'] if sdn_result['decision'] == 'Eco-Friendly' else metrics['lat_carbon']:.4f}",
                            f"{metrics['carbon_saving_pct']:.1f}%" if sdn_result['decision'] == 'Eco-Friendly' else "0%",
                            f"{metrics['latency_penalty_pct']:.1f}%" if sdn_result['decision'] == 'Eco-Friendly' else "0%"
                        ]
                    }
                    
                    sdn_df = pd.DataFrame(comparison_data)
                    st.dataframe(sdn_df, use_container_width=True)
                    
                    # SDN Controller Statistics
                    stats = sdn_controller.get_statistics()
                    st.write("**SDN Controller Statistics:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Flows", stats['total_flows'])
                    with col2:
                        st.metric("Eco Flows", stats['eco_flows'])
                    with col3:
                        st.metric("Eco Decision Rate", f"{stats['eco_percentage']:.1f}%")
                    
                    # Export SDN flow entry
                    export_sdn = json.dumps(sdn_result['flow_entry'], indent=2)
                    st.download_button(
                        label="📥 Export SDN Flow Entry",
                        data=export_sdn,
                        file_name=f"sdn_flow_{sdn_result['flow_id']}.json",
                        mime="application/json"
                    )
                else:
                    st.error(sdn_result['error'])
                    
            except Exception as e:
                st.error(f"Error in SDN controller: {str(e)}")
                st.info("SDN controller simulates intelligent flow installation based on AI predictions.")
        
        # Live SDN Controller with OpenFlow Simulation
        st.subheader("🔴 Live SDN Controller (OpenFlow)")
        show_live_sdn = st.checkbox("Enable Live SDN Controller with Packet Simulation")
        
        if show_live_sdn:
            st.info("Live OpenFlow-based SDN Controller - Simulates real-time packet processing and flow management")
            
            try:
                # Initialize or retrieve live SDN controller from session state
                if 'live_sdn_controller' not in st.session_state or st.session_state.get('controller_topology_seed') != st.session_state.graph_seed:
                    st.session_state.live_sdn_controller = create_live_sdn_controller(G, model)
                    st.session_state.controller_topology_seed = st.session_state.graph_seed
                
                live_controller = st.session_state.live_sdn_controller
                
                # Update topology if network changed
                live_controller.topology = G
                
                # Controller status
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Controller Status", "🟢 RUNNING")
                with col2:
                    stats = live_controller.get_statistics()
                    st.metric("Total Switches", stats['total_switches'])
                with col3:
                    st.metric("Uptime", f"{stats['uptime_seconds']:.1f}s")
                
                # Packet simulation controls
                st.write("**Packet Injection:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    packet_src = st.selectbox("Packet Source", options=list(range(num_nodes)), key="pkt_src")
                with col2:
                    packet_dst = st.selectbox("Packet Destination", options=list(range(num_nodes)), key="pkt_dst")
                with col3:
                    if st.button("📤 Inject Packet", type="primary"):
                        if packet_src != packet_dst:
                            packet = live_controller.inject_packet(packet_src, packet_dst)
                            st.success(f"Injected {packet.packet_id}: Node {packet_src} → Node {packet_dst}")
                        else:
                            st.warning("Source and destination must be different")
                
                # Simulate multiple packets
                if st.button("🔄 Simulate Traffic (10 packets)"):
                    results = live_controller.simulate_traffic(10)
                    st.success(f"Injected {len(results)} packets into the network")
                
                # Flow table display
                st.write("**OpenFlow Tables:**")
                flow_table = live_controller.get_flow_table()
                
                # Show flows per switch
                for switch_id in sorted(flow_table.keys()):
                    flows = flow_table[switch_id]
                    if flows:
                        with st.expander(f"Switch {switch_id} - {len(flows)} flow(s)"):
                            for i, flow in enumerate(flows):
                                st.write(f"**Flow {i+1}:**")
                                flow_info = f"""
                                - Priority: {flow['priority']}
                                - Match: {flow['match']}
                                - Actions: {', '.join(flow['actions'])}
                                - Cookie: {flow['cookie']}
                                - Packets: {flow.get('packets', 0)}
                                - Decision: {flow['metadata'].get('decision', 'N/A')}
                                - Confidence: {flow['metadata'].get('confidence', 0):.1f}%
                                - Carbon Savings: {flow['metadata'].get('carbon_saving', 0):.1f}%
                                """
                                st.text(flow_info)
                
                # Controller statistics
                st.write("**Controller Statistics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Packets Processed", stats['packets_processed'])
                with col2:
                    st.metric("Packets Dropped", stats['packets_dropped'])
                with col3:
                    st.metric("Table Misses", stats['table_misses'])
                with col4:
                    st.metric("Flow Mods Sent", stats['flow_mods_sent'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Eco Flows Installed", stats['eco_flows'])
                with col2:
                    st.metric("Latency Flows Installed", stats['latency_flows'])
                
                # Event log
                st.write("**Controller Event Log:**")
                events = live_controller.get_event_log(limit=20)
                if events:
                    event_df = pd.DataFrame(events)
                    st.dataframe(event_df[['timestamp', 'type', 'message']], use_container_width=True)
                else:
                    st.info("No events logged yet")
                
                # Clear flows button
                if st.button("🗑️ Clear All Flows"):
                    live_controller.clear_flows()
                    st.success("All flow entries cleared")
                    st.rerun()
                
                # Export controller state
                controller_state = {
                    'statistics': stats,
                    'flow_table': flow_table,
                    'events': events
                }
                st.download_button(
                    label="📥 Export Controller State",
                    data=json.dumps(controller_state, indent=2),
                    file_name=f"sdn_controller_state_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error in Live SDN Controller: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    except nx.NetworkXNoPath:
        st.error(f"No path exists between Node {source} and Node {destination}. Please select different nodes.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
