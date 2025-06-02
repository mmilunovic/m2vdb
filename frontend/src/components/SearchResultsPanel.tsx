import React from 'react';

// Local type definitions (duplicated for now)
interface Metadata {
  id: string;
  name: string;
  category: string;
  subcategory: string;
  brand: string;
  price_usd: number | null;
  source: string;
}

interface VectorData {
  id: number;
  text: string;
  vector: number[];
  metadata: Metadata;
}

interface SearchResult {
  id: number;
  text: string;
  similarity_score: number;
  metadata: Metadata;
}

interface SearchResultsPanelProps {
  vectors: VectorData[];
  searchResults: SearchResult[];
  searchQuery: string;
  setSelectedVector: React.Dispatch<React.SetStateAction<VectorData | null>>;
  setSearchResults: React.Dispatch<React.SetStateAction<SearchResult[]>>;
}

const SearchResultsPanel: React.FC<SearchResultsPanelProps> = ({
  vectors,
  searchResults,
  searchQuery,
  setSelectedVector,
  setSearchResults
}) => {
  return (
    <div
      style={{
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
        <div style={{ fontSize: '24px', fontWeight: 600 }}>Search Results</div>
        <button
          aria-label="Close search results"
          onClick={() => setSearchResults([])}
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

      {/* Query info */}
      <div
        style={{
          fontSize: '16px',
          color: '#37352f',
          marginBottom: '16px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}
      >
        <span>🔍</span>
        <span style={{ fontWeight: 500 }}>Query:</span>
        <span style={{ color: '#787774' }}>&quot;{searchQuery}&quot;</span>
      </div>

      {/* Results list */}
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
          <span>📋</span> Results
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {searchResults.map(result => (
            <div
              key={result.id}
              onClick={() => {
                const vector = vectors.find(v => v.id === result.id);
                if (vector) {
                  setSelectedVector(vector);
                }
              }}
              style={{
                backgroundColor: '#f7f6f3',
                padding: '16px',
                borderRadius: '4px',
                cursor: 'pointer',
                transition: 'background-color 0.2s ease'
              }}
              onMouseOver={e => {
                e.currentTarget.style.backgroundColor = '#f1f1ef';
              }}
              onMouseOut={e => {
                e.currentTarget.style.backgroundColor = '#f7f6f3';
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
                  #{result.id}
                </div>
                <div
                  style={{
                    fontSize: '11px',
                    color: '#787774',
                    backgroundColor: '#ffffff',
                    padding: '2px 8px',
                    borderRadius: '3px'
                  }}
                >
                  {result.similarity_score.toFixed(4)}
                </div>
              </div>
              <div
                style={{
                  fontSize: '13px',
                  color: '#37352f',
                  marginBottom: '8px',
                  lineHeight: '1.5'
                }}
              >
                {result.text}
              </div>
              <div style={{ display: 'flex', gap: '8px', fontSize: '11px' }}>
                <span
                  style={{
                    backgroundColor: '#ffffff',
                    padding: '2px 8px',
                    borderRadius: '3px',
                    color: '#787774'
                  }}
                >
                  🏷️ {result.metadata.category}
                </span>
                <span
                  style={{
                    backgroundColor: '#ffffff',
                    padding: '2px 8px',
                    borderRadius: '3px',
                    color: '#787774'
                  }}
                >
                  📑 {result.metadata.subcategory}
                </span>
                <span
                  style={{
                    backgroundColor: '#ffffff',
                    padding: '2px 8px',
                    borderRadius: '3px',
                    color: '#787774'
                  }}
                >
                  🏭 {result.metadata.brand}
                </span>
                {result.metadata.price_usd !== null && (
                  <span
                    style={{
                      backgroundColor: '#ffffff',
                      padding: '2px 8px',
                      borderRadius: '3px',
                      color: '#787774'
                    }}
                  >
                    💰 ${result.metadata.price_usd}
                  </span>
                )}
                <span
                  style={{
                    backgroundColor: '#ffffff',
                    padding: '2px 8px',
                    borderRadius: '3px',
                    color: '#787774'
                  }}
                >
                  📚 {result.metadata.source}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SearchResultsPanel; 