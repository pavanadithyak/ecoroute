# EcoRoutingAI Dashboard

## Overview

EcoRoutingAI is a Streamlit-based simulation dashboard that demonstrates intelligent, eco-friendly network path selection in software-defined networking (SDN) environments. The application simulates a network topology where routing decisions balance latency performance against carbon emissions, using machine learning models to make intelligent trade-offs between speed and environmental impact.

The system visualizes network paths in real-time, compares latency-optimized versus carbon-optimized routes, and employs AI models to decide when switching to eco-friendly paths provides sufficient carbon savings to justify performance penalties.

**Latest Updates (October 2025)**:
- Added historical tracking with persistent database storage for trend analysis
- Implemented configurable network topology (4-12 nodes with dynamic edge generation)
- Created multi-model AI comparison (RandomForest, GradientBoosting, DecisionTree, NeuralNetwork)
- Added real-time threshold controls for carbon/latency trade-off customization
- Integrated export functionality for routing decisions, history, and model comparisons

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology**: Streamlit web framework

**Problem Addressed**: Need for real-time, interactive visualization of network routing decisions and performance metrics

**Solution**: Streamlit provides a rapid development framework for data-driven dashboards with built-in state management and reactive UI components. The application uses Streamlit's caching mechanisms (`@st.cache_resource`) to optimize resource initialization and model loading.

**Key Design Decisions**:
- Session state management for persistent graph configuration (seed, node count, thresholds)
- Dynamic graph regeneration on user interaction to simulate network fluctuations
- Matplotlib integration for network topology visualization using NetworkX

**Pros**: Fast prototyping, automatic UI updates on data changes, minimal frontend code
**Cons**: Limited customization compared to full web frameworks, server-side rendering only

### Backend Architecture

**Network Simulation**: NetworkX graph-based topology with weighted edges

**Problem Addressed**: Realistic simulation of SDN network with multiple routing constraints (latency and carbon cost)

**Solution**: NetworkX directed graphs with dual edge weights representing latency (ms) and carbon emissions (g CO₂). The system computes shortest paths using different optimization criteria:
- Dijkstra's algorithm with latency weights for performance-optimal routes
- Dijkstra's algorithm with carbon weights for eco-optimal routes

**Key Components**:
- Dynamic edge weight generation with randomization to simulate network variability
- Path comparison logic calculating percentage improvements/penalties
- Dual-objective optimization framework

### Machine Learning Architecture

**Models**: Multiple classifier implementations for routing decision comparison

**Problem Addressed**: Intelligent decision-making for when to prioritize carbon reduction over latency performance

**Solution**: Ensemble of trained classification models that predict optimal routing choice based on path metrics:
- RandomForest (primary model)
- Gradient Boosting
- Decision Tree
- Support Vector Machine (SVC)
- Multi-Layer Perceptron (MLP)

**Input Features** (4-dimensional):
- Latency-optimal path latency (ms)
- Latency-optimal path carbon cost (g CO₂)
- Eco-optimal path latency (ms)
- Eco-optimal path carbon cost (g CO₂)

**Output**: Binary classification (0 = latency-optimal, 1 = eco-optimal)

**Training Data Generation**: Synthetic data (2000 samples) with realistic decision rules:
- Choose eco-path when carbon savings > 60% (regardless of latency)
- Choose eco-path when carbon savings > 40% AND latency penalty < 100%
- Choose eco-path when carbon savings > 20% AND latency penalty < 60%
- 10% noise injection for model robustness

**Model Persistence**: Serialized using joblib for fast loading

**Pros**: Multiple model comparison enables architecture evaluation, synthetic data allows controlled experimentation
**Cons**: Synthetic training data may not reflect real-world network behavior patterns

### Data Storage

**Technology**: SQLAlchemy ORM with configurable database backend

**Problem Addressed**: Persistent storage of routing decisions for historical analysis and model refinement

**Solution**: SQLAlchemy provides database-agnostic abstraction layer supporting both SQLite (development) and PostgreSQL (production)

**Schema Design**:

**RoutingDecision Table**:
- `id`: Primary key
- `timestamp`: Decision timestamp (UTC)
- `source`, `destination`: Network node identifiers
- `lat_path`, `eco_path`: Serialized path representations
- `lat_latency`, `lat_carbon`: Latency-optimal path metrics
- `eco_latency`, `eco_carbon`: Eco-optimal path metrics
- `decision`: Human-readable decision string
- `ai_decision`: Binary model prediction
- `confidence`: Model confidence score
- `carbon_saving_pct`: Percentage carbon reduction
- `latency_penalty_pct`: Percentage latency increase
- `network_config`: Serialized network configuration snapshot

**Database Selection**:
- Development: SQLite (file-based, zero configuration)
- Production: PostgreSQL via `DATABASE_URL` environment variable

**Pros**: ORM abstraction enables easy database switching, comprehensive decision logging for analytics
**Cons**: Potential performance overhead from ORM layer for high-frequency routing decisions

### Application State Management

**Problem Addressed**: Maintaining consistent application state across Streamlit reruns

**Solution**: Streamlit session state for persistent configuration:
- `graph_seed`: Random seed for reproducible network topology
- `num_nodes`: Network size configuration
- `carbon_threshold`: Decision threshold for carbon savings
- `latency_threshold`: Decision threshold for latency penalty

**Design Pattern**: Initialization checks ensure state variables exist before access, preventing key errors on first run

## External Dependencies

### Core Python Libraries

- **streamlit**: Web application framework for data dashboards
- **networkx**: Graph data structure and shortest path algorithms
- **matplotlib**: Network topology visualization and plotting
- **scikit-learn**: Machine learning models (RandomForest, GradientBoosting, DecisionTree, SVC, MLPClassifier)
- **pandas**: Data manipulation for analytics and CSV handling
- **numpy**: Numerical computing for synthetic data generation
- **joblib**: Model serialization and deserialization
- **sqlalchemy**: Database ORM for routing decision persistence

### Database

- **SQLite**: Default embedded database (development)
- **PostgreSQL**: Optional production database via `DATABASE_URL` environment variable

### Optional Integrations

**MATLAB SimEvents Integration** (mentioned in requirements):
- CSV import capability for comparing simulation results
- Latency vs carbon analysis plotting
- Cross-validation between Python simulation and MATLAB models

**Purpose**: Validate synthetic network simulation against established network simulation tools

### Environment Configuration

- `DATABASE_URL`: Optional environment variable for production database connection string
- Model file: `model.pkl` (pre-trained RandomForest classifier)

### Data Formats

- **Model persistence**: Pickle format via joblib
- **Network configuration**: JSON serialization for database storage
- **Path representation**: String serialization of node sequences
- **External data**: CSV format for MATLAB integration