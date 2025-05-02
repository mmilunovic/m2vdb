import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import * as Plotly from 'plotly.js-dist-min';
import type { Data, Layout } from 'plotly.js';

interface VectorData {
  id: number;
  text: string;  // Changed from optional to required since we know it exists
  vector: number[];
  metadata: {
    source: string;
    category: string;
    [key: string]: any;  // Allow for additional metadata fields
  };
}

interface NeighborData {
  id: number;
  distance: number;
  text: string;
  metadata: {
    source: string;
    category: string;
  };
}

interface SearchResult {
  id: number;
  text: string;
  similarity_score: number;
  metadata: {
    source: string;
    category: string;
  };
}

interface VectorVisualizationProps {
  vectors: VectorData[];
  setVectors: React.Dispatch<React.SetStateAction<VectorData[]>>;
  selectedVector: VectorData | null;
  setSelectedVector: React.Dispatch<React.SetStateAction<VectorData | null>>;
  searchResults: SearchResult[];
  setSearchResults: React.Dispatch<React.SetStateAction<SearchResult[]>>;
  searchQuery: string;
}

const VectorVisualization: React.FC<VectorVisualizationProps> = ({
  vectors,
  setVectors,
  selectedVector,
  setSelectedVector,
  searchResults,
  setSearchResults,
  searchQuery,
}) => {
  const plotRef = useRef<HTMLDivElement>(null);
  const [neighbors, setNeighbors] = useState<NeighborData[]>([]);
  const [isNeighborsExpanded, setIsNeighborsExpanded] = useState(true);
  const plotInitialized = useRef(false);

  useEffect(() => {
    const fetchVectors = async () => {
      try {
        const response = await axios.get('http://localhost:8000/vectors');
        console.log('Raw response data:', response.data);
        const vectorsData = response.data.vectors || response.data;
        console.log('Processed vectors data:', vectorsData);
        setVectors(Array.isArray(vectorsData) ? vectorsData : []);
      } catch (error) {
        console.error('Error fetching vectors:', error);
      }
    };

    fetchVectors();
  }, []);

  useEffect(() => {
    if (!plotRef.current || vectors.length === 0) {
      console.log('No vectors to visualize:', vectors);
      return;
    }

    console.log('Preparing to visualize vectors:', vectors);

    // Simple PCA implementation
    const performPCA = (vectors: number[][], dimensions: number = 3) => {
      const n = vectors.length;
      if (n === 0) return [];

      console.log('Input vectors for PCA:', vectors);
      console.log('Vector dimensions:', vectors[0].length);

      // Center the data
      const mean = vectors[0].map((_, i) => 
        vectors.reduce((sum, v) => sum + v[i], 0) / n
      );
      
      const centered = vectors.map(v => 
        v.map((val, i) => val - mean[i])
      );

      // Compute covariance matrix
      const cov = centered[0].map((_, i) =>
        centered[0].map((_, j) =>
          centered.reduce((sum, v) => sum + v[i] * v[j], 0) / (n - 1)
        )
      );

      // Power iteration for eigenvectors
      const getEigenvectors = (matrix: number[][], numComponents: number) => {
        const n = matrix.length;
        const eigenvectors = Array(numComponents).fill(null).map(() => 
          Array(n).fill(0).map(() => Math.random())
        );

        for (let i = 0; i < 10; i++) {
          eigenvectors.forEach((eigenvector, idx) => {
            const newVector = matrix.map(row =>
              row.reduce((sum, val, j) => sum + val * eigenvector[j], 0)
            );
            
            const norm = Math.sqrt(newVector.reduce((sum, val) => sum + val * val, 0));
            eigenvectors[idx] = newVector.map(val => val / norm);
          });
        }

        return eigenvectors;
      };

      const eigenvectors = getEigenvectors(cov, dimensions);
      
      // Project data onto principal components
      return centered.map(v => 
        eigenvectors.map(eigenvector =>
          v.reduce((sum, val, i) => sum + val * eigenvector[i], 0)
        )
      );
    };

    try {
      const vectorArrays = vectors.map(v => v.vector);
      console.log('Vector arrays for PCA:', vectorArrays);
      const pcaVectors = performPCA(vectorArrays);
      console.log('PCA result:', pcaVectors);

      const data: Data[] = [{
        type: 'scatter3d',
        mode: 'markers',
        x: pcaVectors.map(v => v[0]),
        y: pcaVectors.map(v => v[1]),
        z: pcaVectors.map(v => v[2]),
        marker: {
          size: 8,
          color: vectors.map((_, i) => i),
          colorscale: 'Viridis',
          opacity: 0.8
        },
        text: vectors.map(v => `ID: ${v.id}`),
        hoverinfo: 'text'
      }];

      const layout: Partial<Layout> = {
        title: 'Vector Visualization (PCA)',
        scene: {
          aspectmode: 'cube',
          xaxis: { title: 'PC1' },
          yaxis: { title: 'PC2' },
          zaxis: { title: 'PC3' },
          camera: {
            eye: { x: 1.5, y: 1.5, z: 1.5 },
            center: { x: 0, y: 0, z: 0 }
          }
        },
        margin: { l: 0, r: 0, b: 0, t: 40 },
        width: window.innerWidth,
        height: window.innerHeight,
        uirevision: 'constant'
      };

      const plotElement = plotRef.current;
      if (plotElement) {
        if (!plotInitialized.current) {
          Plotly.newPlot(plotElement, data, layout, {
            responsive: true,
            displayModeBar: true
          });
          plotInitialized.current = true;

          // Add click handler
          (plotElement as any).on('plotly_click', (eventData: any) => {
            if (eventData && eventData.points && eventData.points[0]) {
              const point = eventData.points[0];
              const pointIndex = point.pointNumber;
              setSelectedVector(vectors[pointIndex]);
            }
          });
        } else {
          // Update only the data
          Plotly.newPlot(plotElement, data, layout, {
            responsive: true,
            displayModeBar: true
          });
        }
      }

    } catch (error) {
      console.error('Error during visualization:', error);
    }

    return () => {
      if (plotRef.current) {
        Plotly.purge(plotRef.current);
        plotInitialized.current = false;
      }
    };
  }, [vectors]);

  // Handle container width changes separately
  useEffect(() => {
    if (plotRef.current && plotInitialized.current) {
      const width = selectedVector ? window.innerWidth * 0.7 : window.innerWidth;
      const layout = {
        width,
        uirevision: 'constant'
      };
      (plotRef.current as any).style.width = `${width}px`;
    }
  }, [selectedVector]);

  // Add new effect to fetch neighbors when a vector is selected
  useEffect(() => {
    const fetchNeighbors = async () => {
      if (!selectedVector) {
        setNeighbors([]);
        return;
      }

      try {
        const response = await axios.get(`http://localhost:8000/neighbors/${selectedVector.id}`);
        setNeighbors(response.data.neighbors || []);
      } catch (error) {
        console.error('Error fetching neighbors:', error);
        setNeighbors([]);
      }
    };

    fetchNeighbors();
  }, [selectedVector]);

  return (
    <div style={{ 
      display: 'flex', 
      width: '100%', 
      height: '100vh',
      backgroundColor: '#ffffff',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div 
        ref={plotRef} 
        style={{ 
          width: selectedVector || searchResults.length > 0 ? '70%' : '100%',
          height: '100vh',
          border: 'none',
          borderRadius: '0',
          overflow: 'hidden',
          transition: 'width 0.3s ease',
          boxShadow: 'none',
          position: 'fixed',
          left: 0,
          top: 0
        }} 
      />
      {(selectedVector || searchResults.length > 0) && (
        <div style={{
          width: '30%',
          padding: '32px',
          backgroundColor: '#ffffff',
          overflowY: 'auto',
          color: '#37352f',
          borderLeft: '1px solid rgba(0, 0, 0, 0.1)',
          position: 'fixed',
          right: 0,
          top: '92px',
          height: 'calc(100vh - 92px)'
        }}>
          {selectedVector ? (
            // Show selected vector info
            <>
              <div style={{ 
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '32px',
              }}>
                <div style={{ 
                  fontSize: '24px',
                  fontWeight: '600',
                  color: '#37352f',
                }}>
                  Vector Details
                </div>
                <button
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
                    transition: 'background-color 0.2s ease, color 0.2s ease',
                  }}
                  onMouseOver={e => {
                    e.currentTarget.style.backgroundColor = '#f1f1ef';
                    e.currentTarget.style.color = '#37352f';
                  }}
                  onMouseOut={e => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                    e.currentTarget.style.color = '#b3b3b1';
                  }}
                  aria-label="Close details"
                >
                  ✕
                </button>
              </div>
              <div style={{ marginBottom: '32px' }}>
                <div style={{ 
                  fontSize: '32px',
                  fontWeight: '700',
                  color: '#37352f',
                  marginBottom: '16px',
                  fontFamily: 'ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace'
                }}>
                  #{selectedVector.id}
                </div>
                <div style={{ 
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px'
                }}>
                  <div style={{ 
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    fontSize: '14px',
                    color: '#787774'
                  }}>
                    <span>🏷️ Category:</span>
                    <span style={{ 
                      backgroundColor: '#f1f1ef',
                      padding: '2px 8px',
                      borderRadius: '3px',
                      color: '#37352f'
                    }}>
                      {selectedVector.metadata.category}
                    </span>
                  </div>
                  <div style={{ 
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    fontSize: '14px',
                    color: '#787774'
                  }}>
                    <span>📚 Source:</span>
                    <span style={{ 
                      backgroundColor: '#f1f1ef',
                      padding: '2px 8px',
                      borderRadius: '3px',
                      color: '#37352f'
                    }}>
                      {selectedVector.metadata.source}
                    </span>
                  </div>
                </div>
              </div>

              <div style={{ marginBottom: '32px' }}>
                <div style={{ 
                  fontSize: '13px',
                  color: '#787774',
                  marginBottom: '8px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  <span>📝</span> Content
                </div>
                <div style={{ 
                  backgroundColor: '#f7f6f3', 
                  padding: '16px', 
                  borderRadius: '4px',
                  fontSize: '13px', 
                  lineHeight: '1.5',
                  color: '#37352f'
                }}>
                  {selectedVector.text}
                </div>
              </div>

              <div style={{ marginBottom: '48px' }}>
                <div style={{ 
                  fontSize: '13px',
                  color: '#787774',
                  marginBottom: '8px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  <span>🔢</span> Vector Values
                </div>
                <div style={{ 
                  backgroundColor: '#f7f6f3', 
                  padding: '16px', 
                  borderRadius: '4px',
                  fontSize: '13px',
                  lineHeight: '1.5',
                  color: '#37352f',
                  fontFamily: 'ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace'
                }}>
                  <div style={{ marginBottom: '8px', color: '#787774' }}>
                    Dimension: {selectedVector.vector.length}
                  </div>
                  <div style={{ 
                    backgroundColor: '#ffffff',
                    padding: '12px',
                    borderRadius: '3px',
                    border: '1px solid rgba(0, 0, 0, 0.1)'
                  }}>
                    [{selectedVector.vector.slice(0, 3).map(v => v.toFixed(4)).join(', ')} ... {selectedVector.vector.slice(-2).map(v => v.toFixed(4)).join(', ')}]
                  </div>
                </div>
              </div>

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
                  <span style={{ 
                    marginLeft: 'auto',
                    transition: 'transform 0.2s ease',
                    transform: isNeighborsExpanded ? 'rotate(180deg)' : 'rotate(0deg)'
                  }}>
                    ▼
                  </span>
                </div>
                <div style={{ 
                  backgroundColor: '#f7f6f3', 
                  borderRadius: '4px',
                  fontSize: '13px',
                  lineHeight: '1.5',
                  color: '#37352f',
                  maxHeight: isNeighborsExpanded ? '1000px' : '0',
                  overflow: 'hidden',
                  transition: 'max-height 0.3s ease, padding 0.3s ease',
                  padding: isNeighborsExpanded ? '16px' : '0 16px'
                }}>
                  {neighbors.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      {neighbors.map((neighbor, index) => (
                        <div 
                          key={neighbor.id}
                          style={{
                            backgroundColor: '#ffffff',
                            padding: '12px',
                            borderRadius: '3px',
                            border: '1px solid rgba(0, 0, 0, 0.1)'
                          }}
                        >
                          <div style={{ 
                            display: 'flex', 
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            marginBottom: '8px'
                          }}>
                            <div style={{ 
                              fontFamily: 'ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace',
                              fontWeight: '600',
                              fontSize: '13px'
                            }}>
                              #{neighbor.id}
                            </div>
                            <div style={{ 
                              fontSize: '11px',
                              color: '#787774',
                              backgroundColor: '#f1f1ef',
                              padding: '2px 8px',
                              borderRadius: '3px'
                            }}>
                              {neighbor.distance.toFixed(4)}
                            </div>
                          </div>
                          <div style={{ 
                            fontSize: '12px',
                            color: '#37352f',
                            marginBottom: '8px'
                          }}>
                            {neighbor.text}
                          </div>
                          <div style={{ 
                            display: 'flex',
                            gap: '8px',
                            fontSize: '11px'
                          }}>
                            <span style={{ 
                              backgroundColor: '#f1f1ef',
                              padding: '2px 8px',
                              borderRadius: '3px',
                              color: '#787774'
                            }}>
                              {neighbor.metadata.category}
                            </span>
                            <span style={{ 
                              backgroundColor: '#f1f1ef',
                              padding: '2px 8px',
                              borderRadius: '3px',
                              color: '#787774'
                            }}>
                              {neighbor.metadata.source}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ color: '#787774', textAlign: 'center' }}>
                      No neighbors found
                    </div>
                  )}
                </div>
              </div>
            </>
          ) : (
            // Show search results
            <>
              <div style={{ 
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '32px',
              }}>
                <div style={{ 
                  fontSize: '24px',
                  fontWeight: '600',
                  color: '#37352f',
                }}>
                  Search Results
                </div>
                <button
                  onClick={() => {
                    setSelectedVector(null);
                    setSearchResults([]);
                  }}
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
                    transition: 'background-color 0.2s ease, color 0.2s ease',
                  }}
                  onMouseOver={e => {
                    e.currentTarget.style.backgroundColor = '#f1f1ef';
                    e.currentTarget.style.color = '#37352f';
                  }}
                  onMouseOut={e => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                    e.currentTarget.style.color = '#b3b3b1';
                  }}
                  aria-label="Close search results"
                >
                  ✕
                </button>
              </div>
              <div style={{ 
                fontSize: '16px',
                color: '#37352f',
                marginBottom: '16px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <span>🔍</span>
                <span style={{ fontWeight: '500' }}>Query:</span>
                <span style={{ color: '#787774' }}>&quot;{searchQuery}&quot;</span>
              </div>
              <div style={{ marginBottom: '32px' }}>
                <div style={{ 
                  fontSize: '13px',
                  color: '#787774',
                  marginBottom: '8px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  <span>📋</span> Results
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {searchResults.map((result) => (
                    <div 
                      key={result.id}
                      style={{
                        backgroundColor: '#f7f6f3',
                        padding: '16px',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        transition: 'background-color 0.2s ease',
                      }}
                      onMouseOver={e => e.currentTarget.style.backgroundColor = '#f1f1ef'}
                      onMouseOut={e => e.currentTarget.style.backgroundColor = '#f7f6f3'}
                      onClick={() => {
                        const vector = vectors.find(v => v.id === result.id);
                        if (vector) {
                          setSelectedVector(vector);
                        }
                      }}
                    >
                      <div style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: '8px'
                      }}>
                        <div style={{ 
                          fontFamily: 'ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace',
                          fontWeight: '600',
                          fontSize: '13px'
                        }}>
                          #{result.id}
                        </div>
                        <div style={{ 
                          fontSize: '11px',
                          color: '#787774',
                          backgroundColor: '#ffffff',
                          padding: '2px 8px',
                          borderRadius: '3px'
                        }}>
                          {result.similarity_score.toFixed(4)}
                        </div>
                      </div>
                      <div style={{ 
                        fontSize: '13px',
                        color: '#37352f',
                        marginBottom: '8px',
                        lineHeight: '1.5'
                      }}>
                        {result.text}
                      </div>
                      <div style={{ 
                        display: 'flex',
                        gap: '8px',
                        fontSize: '11px'
                      }}>
                        <span style={{ 
                          backgroundColor: '#ffffff',
                          padding: '2px 8px',
                          borderRadius: '3px',
                          color: '#787774'
                        }}>
                          {result.metadata.category}
                        </span>
                        <span style={{ 
                          backgroundColor: '#ffffff',
                          padding: '2px 8px',
                          borderRadius: '3px',
                          color: '#787774'
                        }}>
                          {result.metadata.source}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default VectorVisualization; 