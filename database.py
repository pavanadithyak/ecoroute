import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class RoutingDecision(Base):
    __tablename__ = 'routing_decisions'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(Integer)
    destination = Column(Integer)
    
    lat_path = Column(String)
    eco_path = Column(String)
    
    lat_latency = Column(Float)
    lat_carbon = Column(Float)
    eco_latency = Column(Float)
    eco_carbon = Column(Float)
    
    decision = Column(String)
    ai_decision = Column(Integer)
    confidence = Column(Float)
    
    carbon_saving_pct = Column(Float)
    latency_penalty_pct = Column(Float)
    
    network_config = Column(String)

def get_database_url():
    """Get database URL from environment"""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        db_url = 'sqlite:///ecorouting.db'
    return db_url

def init_db():
    """Initialize database tables"""
    engine = create_engine(get_database_url())
    Base.metadata.create_all(engine)
    return engine

def get_session():
    """Get database session"""
    engine = create_engine(get_database_url())
    Session = sessionmaker(bind=engine)
    return Session()

def save_routing_decision(source, dest, lat_path, eco_path, lat_latency, lat_carbon, 
                          eco_latency, eco_carbon, decision, ai_decision, confidence,
                          carbon_saving, latency_penalty, network_config):
    """Save a routing decision to database"""
    session = get_session()
    
    routing = RoutingDecision(
        source=source,
        destination=dest,
        lat_path=str(lat_path),
        eco_path=str(eco_path),
        lat_latency=lat_latency,
        lat_carbon=lat_carbon,
        eco_latency=eco_latency,
        eco_carbon=eco_carbon,
        decision=decision,
        ai_decision=ai_decision,
        confidence=confidence,
        carbon_saving_pct=carbon_saving,
        latency_penalty_pct=latency_penalty,
        network_config=network_config
    )
    
    session.add(routing)
    session.commit()
    
    # Get the ID before closing the session
    routing_id = routing.id
    session.close()
    
    return routing_id

def get_routing_history(limit=100):
    """Get routing decision history"""
    session = get_session()
    results = session.query(RoutingDecision).order_by(RoutingDecision.timestamp.desc()).limit(limit).all()
    session.close()
    return results

def get_routing_stats():
    """Get aggregated routing statistics"""
    session = get_session()
    
    total_decisions = session.query(RoutingDecision).count()
    eco_decisions = session.query(RoutingDecision).filter(RoutingDecision.ai_decision == 1).count()
    
    avg_carbon_saving = session.query(RoutingDecision).filter(
        RoutingDecision.ai_decision == 1
    ).with_entities(RoutingDecision.carbon_saving_pct).all()
    
    avg_latency_penalty = session.query(RoutingDecision).filter(
        RoutingDecision.ai_decision == 1
    ).with_entities(RoutingDecision.latency_penalty_pct).all()
    
    session.close()
    
    avg_carbon = sum([x[0] for x in avg_carbon_saving]) / len(avg_carbon_saving) if avg_carbon_saving else 0
    avg_latency = sum([x[0] for x in avg_latency_penalty]) / len(avg_latency_penalty) if avg_latency_penalty else 0
    
    return {
        'total_decisions': total_decisions,
        'eco_decisions': eco_decisions,
        'eco_percentage': (eco_decisions / total_decisions * 100) if total_decisions > 0 else 0,
        'avg_carbon_saving': avg_carbon,
        'avg_latency_penalty': avg_latency
    }
