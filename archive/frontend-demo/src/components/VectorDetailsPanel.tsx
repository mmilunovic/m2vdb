import React, { useEffect, useState } from 'react';
import axios from 'axios';

// Local type definitions (duplicated from VectorVisualization for now)
interface VectorData {
  id: number;
  text: string;
  vector: number[];
  metadata: {
    id: string;
    name: string;
    category: string;
    subcategory: string;
    brand: string;
    price_usd: number | null;
    source: string;
  };
}

interface NeighborData {
  id: number;
  distance: number;
  text: string;
  metadata: {
    id: string;
    name: string;
    category: string;
    subcategory: string;
    brand: string;
    price_usd: number | null;
    source: string;
  };
}

interface VectorDetailsPanelProps {
  selectedVector: VectorData;
  setSelectedVector: React.Dispatch<React.SetStateAction<VectorData | null>>;
}

const VectorDetailsPanel: React.FC<VectorDetailsPanelProps> = ({ selectedVector, setSelectedVector }) => {
  const [neighbors, setNeighbors] = useState<NeighborData[]>([]);
  const [isNeighborsExpanded, setIsNeighborsExpanded] = useState(true);

  // Fetch neighbors whenever a new vector is selected
  useEffect(() => {
    const fetchNeighbors = async () => {
      try {
        const response = await axios.get<{ neighbors: NeighborData[] }>(`http://localhost:8000/neighbors/${selectedVector.id}`);
        setNeighbors(response.data.neighbors || []);
      } catch (error) {
        console.error('Error fetching neighbors:', error);
        setNeighbors([]);
      }
    };

    fetchNeighbors();
  }, [selectedVector]);

  return (
    <div
      style={{
        width: '30%',
        padding: '32px',
        boxSizing: 'border-box',
        backgroundColor: '#ffffff',
        overflowY: 'auto',
        color: '#37352f',
        borderLeft: '1px solid rgba(0, 0, 0, 0.1)',
        position: 'fixed',
        right: 0,
        top: '92px',
        height: 'calc(100vh - 92px)'
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '32px'
        }}
      >
        <div style={{ fontSize: '24px', fontWeight: 600 }}>Vector Details</div>
        <button
          aria-label="Close details"
          onClick={() => setSelectedVector(null)}
          style={{
            width: '24px',
            height: '24px',
            background: 'none',
            border: 'none',
            padding: 0,
            marginLeft: '8px',
            cursor: 'pointer',
            color: '#b3b3b1',
            borderRadius: '3px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '20px',
            transition: 'background-color 0.2s ease, color 0.2s ease'
          }}
          onMouseOver={e => {
            e.currentTarget.style.backgroundColor = '#f1f1ef';
            e.currentTarget.style.color = '#37352f';
          }}
          onMouseOut={e => {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.color = '#b3b3b1';
          }}
        >
          ✕
        </button>
      </div>

      {/* Basic metadata */}
      <div style={{ marginBottom: '32px' }}>
        <div
          style={{
            fontSize: '32px',
            fontWeight: 700,
            color: '#37352f',
            marginBottom: '16px',
            fontFamily: 'ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace'
          }}
        >
          {selectedVector.metadata.id}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {/* Name */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', color: '#787774' }}>
            <span>📦 Name:</span>
            <span
              style={{ backgroundColor: '#f1f1ef', padding: '2px 8px', borderRadius: '3px', color: '#37352f' }}
            >
              {selectedVector.metadata.name}
            </span>
          </div>
          {/* Category */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', color: '#787774' }}>
            <span>🏷️ Category:</span>
            <span
              style={{ backgroundColor: '#f1f1ef', padding: '2px 8px', borderRadius: '3px', color: '#37352f' }}
            >
              {selectedVector.metadata.category}
            </span>
          </div>
          {/* Subcategory */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', color: '#787774' }}>
            <span>📑 Subcategory:</span>
            <span
              style={{ backgroundColor: '#f1f1ef', padding: '2px 8px', borderRadius: '3px', color: '#37352f' }}
            >
              {selectedVector.metadata.subcategory}
            </span>
          </div>
          {/* Brand */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', color: '#787774' }}>
            <span>🏭 Brand:</span>
            <span
              style={{ backgroundColor: '#f1f1ef', padding: '2px 8px', borderRadius: '3px', color: '#37352f' }}
            >
              {selectedVector.metadata.brand}
            </span>
          </div>
          {/* Price */}
          {selectedVector.metadata.price_usd !== null && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', color: '#787774' }}>
              <span>💰 Price:</span>
              <span
                style={{ backgroundColor: '#f1f1ef', padding: '2px 8px', borderRadius: '3px', color: '#37352f' }}
              >
                ${selectedVector.metadata.price_usd}
              </span>
            </div>
          )}
          {/* Source */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', color: '#787774' }}>
            <span>📚 Source:</span>
            <span
              style={{ backgroundColor: '#f1f1ef', padding: '2px 8px', borderRadius: '3px', color: '#37352f' }}
            >
              {selectedVector.metadata.source}
            </span>
          </div>
        </div>
      </div>

      {/* Content */}
      <div style={{ marginBottom: '32px' }}>
        <div
          style={{
            fontSize: '13px',
            color: '#787774',
            marginBottom: '8px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <span>📝</span> Content
        </div>
        <div
          style={{
            backgroundColor: '#f7f6f3',
            padding: '16px',
            borderRadius: '4px',
            fontSize: '13px',
            lineHeight: '1.5',
            color: '#37352f'
          }}
        >
          {selectedVector.text}
        </div>
      </div>

      {/* Vector values */}
      <div style={{ marginBottom: '48px' }}>
        <div
          style={{
            fontSize: '13px',
            color: '#787774',
            marginBottom: '8px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <span>🔢</span> Vector Values
        </div>
        <div
          style={{
            backgroundColor: '#f7f6f3',
            padding: '16px',
            borderRadius: '4px',
            fontSize: '13px',
            lineHeight: '1.5',
            color: '#37352f',
            fontFamily:
              'ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace'
          }}
        >
          <div style={{ marginBottom: '8px', color: '#787774' }}>
            Dimension: {selectedVector.vector.length}
          </div>
          <div
            style={{
              backgroundColor: '#ffffff',
              padding: '12px',
              borderRadius: '3px',
              border: '1px solid rgba(0, 0, 0, 0.1)'
            }}
          >
            [
            {selectedVector.vector
              .slice(0, 3)
              .map(v => v.toFixed(4))
              .join(', ')}{' '}
            ...{' '}
            {selectedVector.vector
              .slice(-2)
              .map(v => v.toFixed(4))
              .join(', ')}
            ]
          </div>
        </div>
      </div>

      {/* Nearest neighbors */}
      <div style={{ marginBottom: '32px' }}>
        <div
          style={{
            fontSize: '13px',
            color: '#787774',
            marginBottom: '8px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            cursor: 'pointer',
            userSelect: 'none'
          }}
          onClick={() => setIsNeighborsExpanded(!isNeighborsExpanded)}
        >
          <span>🔍</span> Nearest Neighbors
          <span
            style={{
              marginLeft: 'auto',
              transition: 'transform 0.2s ease',
              transform: isNeighborsExpanded ? 'rotate(180deg)' : 'rotate(0deg)'
            }}
          >
            ▼
          </span>
        </div>
        <div
          style={{
            backgroundColor: '#f7f6f3',
            borderRadius: '4px',
            fontSize: '13px',
            lineHeight: '1.5',
            color: '#37352f',
            maxHeight: isNeighborsExpanded ? '1000px' : '0',
            overflow: 'hidden',
            transition: 'max-height 0.3s ease, padding 0.3s ease',
            padding: isNeighborsExpanded ? '16px' : '0 16px'
          }}
        >
          {neighbors.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {neighbors.map(neighbor => (
                <div
                  key={neighbor.id}
                  style={{
                    backgroundColor: '#ffffff',
                    padding: '12px',
                    borderRadius: '3px',
                    border: '1px solid rgba(0, 0, 0, 0.1)'
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: '8px'
                    }}
                  >
                    <div
                      style={{
                        fontFamily:
                          'ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace',
                        fontWeight: 600,
                        fontSize: '13px'
                      }}
                    >
                      #{neighbor.id}
                    </div>
                    <div
                      style={{
                        fontSize: '11px',
                        color: '#787774',
                        backgroundColor: '#f1f1ef',
                        padding: '2px 8px',
                        borderRadius: '3px'
                      }}
                    >
                      {neighbor.distance.toFixed(4)}
                    </div>
                  </div>
                  <div style={{ fontSize: '12px', color: '#37352f', marginBottom: '8px' }}>
                    {neighbor.text}
                  </div>
                  <div style={{ display: 'flex', gap: '8px', fontSize: '11px' }}>
                    <span
                      style={{
                        backgroundColor: '#f1f1ef',
                        padding: '2px 8px',
                        borderRadius: '3px',
                        color: '#787774'
                      }}
                    >
                      🏷️ {neighbor.metadata.category}
                    </span>
                    <span
                      style={{
                        backgroundColor: '#f1f1ef',
                        padding: '2px 8px',
                        borderRadius: '3px',
                        color: '#787774'
                      }}
                    >
                      📑 {neighbor.metadata.subcategory}
                    </span>
                    <span
                      style={{
                        backgroundColor: '#f1f1ef',
                        padding: '2px 8px',
                        borderRadius: '3px',
                        color: '#787774'
                      }}
                    >
                      🏭 {neighbor.metadata.brand}
                    </span>
                    {neighbor.metadata.price_usd !== null && (
                      <span
                        style={{
                          backgroundColor: '#f1f1ef',
                          padding: '2px 8px',
                          borderRadius: '3px',
                          color: '#787774'
                        }}
                      >
                        💰 ${neighbor.metadata.price_usd}
                      </span>
                    )}
                    <span
                      style={{
                        backgroundColor: '#f1f1ef',
                        padding: '2px 8px',
                        borderRadius: '3px',
                        color: '#787774'
                      }}
                    >
                      📚 {neighbor.metadata.source}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: '#787774', textAlign: 'center' }}>No neighbors found</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VectorDetailsPanel; 